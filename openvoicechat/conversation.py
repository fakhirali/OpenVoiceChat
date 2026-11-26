"""The listen -> transcribe -> LLM -> speak loop, in one file. Read it in order:
:meth:`Conversation.run` is the call, :meth:`Conversation.utterances` is the
endpointing state machine, :meth:`Conversation.respond` is one turn, and
:class:`Speaker` is the one background task (TTS overlapping the LLM stream).
"""

import asyncio
import contextlib
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Self

import pysbd
from openai import AsyncOpenAI

from .config import Config
from .vad import VAD

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnLatency:
    """Per-turn timings in ms, from the last user speech frame. ``voice_to_voice_ms``
    ends when the first audio chunk is queued; :meth:`Conversation.audio_written`
    reports when a transport actually wrote it."""

    voice_to_voice_ms: float | None
    endpoint_ms: float
    stt_ms: float
    llm_ttft_ms: float | None
    first_sentence_ms: float | None
    tts_first_ms: float | None
    turn_ms: float
    completed: bool


class Conversation:
    """One voice conversation over input/output PCM16 queues. Put 16 kHz mono
    frames into ``input``, read reply audio from ``output``; that is all a
    transport has to do (see ``main.py``)."""

    def __init__(
        self,
        speech,
        *,
        system_prompt: str = "",
        config: Config | None = None,
        client: AsyncOpenAI | None = None,
        on_turn=None,
        on_event=None,
        vad=None,
        verbose: bool = False,
    ):
        self.speech = speech
        self.config = config or Config()
        self.system_prompt = system_prompt
        self.on_turn = on_turn
        self.on_event = on_event
        self.verbose = verbose
        self.last_latency: TurnLatency | None = None
        self._last_speech_at = 0.0
        self._first_audio_ref: float | None = None
        self._owns_client = client is None
        if client is None:
            if not self.config.llm_api_key:
                raise ValueError("LLM_API_KEY (or OPENAI_API_KEY) is required")
            client = AsyncOpenAI(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url or None,
                timeout=self.config.llm_timeout_s,
                max_retries=1,
            )
        self.client = client
        self.vad = vad or VAD(self.config.vad_threshold, self.config.sample_rate)

        self.input: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)
        self.output: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self.agent_speaking = False
        self.messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        self._frame_ms = 1000 * self.config.frame_samples / self.config.sample_rate
        self._chunk_bytes = int(self.config.sample_rate * self.config.output_chunk_ms / 1000) * 2
        self._stop = asyncio.Event()
        self._turn_task: asyncio.Task | None = None

    # -- the call ---------------------------------------------------------

    async def run(self) -> None:
        """Listen for utterances and answer each one, until ``stop()``."""
        self._event(
            "call_start",
            sample_rate=self.config.sample_rate,
            endpoint_silence_ms=self.config.endpoint_silence_ms,
        )
        try:
            async for utterance, speech_end_at in self.utterances():
                self._turn_task = asyncio.create_task(
                    self.respond(utterance, speech_end_at), name="respond"
                )
        finally:
            self._stop.set()
            task, self._turn_task = self._turn_task, None
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            self._event("call_end")
            self.agent_speaking = False
            self._first_audio_ref = None
            _drain(self.output)
            if self._owns_client:
                with contextlib.suppress(Exception):
                    await self.client.close()

    async def utterances(self):
        """Yield ``(pcm16, last_speech_at)`` once per finished user utterance.

        Endpointing only: two voiced frames open an utterance (one can be a
        click), ``pre_roll_ms`` is kept ahead of it, barge-in cuts the agent
        off, silence or the ``max_utterance_ms`` cap closes it, and anything
        shorter than ``min_speech_ms`` is reported as dropped.
        """
        config = self.config
        frame_ms = self._frame_ms
        pre_roll: deque[bytes] = deque(maxlen=math.ceil(config.pre_roll_ms / frame_ms))
        onset: list[bytes] = []
        utterance: deque[bytes] = deque()
        voiced_run = 0
        in_speech = False
        capped = False
        speech_ms = silence_ms = 0.0

        while not self._stop.is_set():
            frame = await self.input.get()
            if frame is None or self._stop.is_set():
                return
            if not frame:
                continue

            # A turn whose audio has played out is no longer interruptible.
            if self._turn_task and self._turn_task.done() and self.output.empty():
                self._turn_task = None
                self.agent_speaking = False

            voiced = bool(self.vad(frame))
            if voiced:
                self._last_speech_at = time.monotonic()

            if voiced and not in_speech and (self._turn_task or self.agent_speaking):
                await self.interrupt()

            if not in_speech:
                if not voiced:
                    voiced_run = 0
                    onset.clear()
                    if pre_roll.maxlen:
                        pre_roll.append(frame)
                    continue
                voiced_run += 1
                onset.append(frame)
                if voiced_run < 2:
                    continue
                in_speech = True
                speech_ms = voiced_run * frame_ms
                pre_roll_ms = len(pre_roll) * frame_ms
                utterance.extend(pre_roll)
                utterance.extend(onset)
                pre_roll.clear()
                onset.clear()
                self._log(f"speech start | pre-roll={pre_roll_ms:.0f}ms")
                self._event("speech_start", pre_roll_ms=round(pre_roll_ms, 1))
                continue

            utterance.append(frame)
            utterance_ms = len(utterance) * frame_ms

            if voiced:
                speech_ms += frame_ms
                silence_ms = 0.0
                if utterance_ms >= config.max_utterance_ms:
                    # Still talking past the cap: keep the most recent window and
                    # wait for silence. Ending the turn mid-speech makes the next
                    # voiced frame look like barge-in and cancels it.
                    while len(utterance) > 1 and utterance_ms > config.max_utterance_ms:
                        utterance.popleft()
                        utterance_ms = len(utterance) * frame_ms
                    if not capped:
                        capped = True
                        self._log(f"utterance capped | keeping last {utterance_ms:.0f}ms")
                continue

            silence_ms += frame_ms
            if silence_ms < config.endpoint_silence_ms and utterance_ms < config.max_utterance_ms:
                continue

            reason = "max-utterance" if utterance_ms >= config.max_utterance_ms else "silence"
            self._log(
                f"endpoint ({reason}) | silence={silence_ms:.0f}ms "
                f"utterance={utterance_ms:.0f}ms speech={speech_ms:.0f}ms"
            )
            if speech_ms >= config.min_speech_ms and utterance:
                self._event(
                    "endpoint",
                    reason=reason,
                    silence_ms=round(silence_ms, 1),
                    speech_ms=round(speech_ms, 1),
                    utterance_ms=round(utterance_ms, 1),
                )
                yield b"".join(utterance), self._last_speech_at
            else:
                self._event("dropped", speech_ms=round(speech_ms, 1))
                self._log(f"drop short utterance | speech={speech_ms:.0f}ms")

            # Reset for the next utterance; this frame is its first pre-roll.
            in_speech = False
            voiced_run = 0
            capped = False
            speech_ms = silence_ms = 0.0
            utterance.clear()
            onset.clear()
            pre_roll.clear()
            self.vad.reset()
            if pre_roll.maxlen:
                pre_roll.append(frame)

    async def respond(self, utterance: bytes, speech_end_at: float) -> None:
        """One turn: transcribe the utterance, stream the reply, speak it."""
        started = time.monotonic()
        metrics: dict[str, float | None] = {
            "endpoint_ms": 1000 * (started - speech_end_at),
            "stt_ms": 0.0,
            "llm_ttft_ms": None,
            "first_sentence_ms": None,
            "tts_first_ms": None,
            "voice_to_voice_ms": None,
        }
        text = ""
        response = ""
        speaker: Speaker | None = None
        completed = False
        self._log(
            f"turn start | pcm={len(utterance) / 2 / self.config.sample_rate * 1000:.0f}ms"
        )

        try:
            stt_started = time.monotonic()
            text = (await self.speech.transcribe(utterance)).strip()
            metrics["stt_ms"] = 1000 * (time.monotonic() - stt_started)
            self._log(f"STT ({metrics['stt_ms']:.0f}ms): {text!r}")
            self._event("stt", ms=round(metrics["stt_ms"], 1), text=text)
            if not text:
                return

            self.messages.append({"role": "user", "content": text})
            self._emit("user", text)

            buffer = ""
            async with Speaker(self, speech_end_at) as speaker:
                stream = await self.client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=self.messages,
                    stream=True,
                )
                llm_started = time.monotonic()
                async for chunk in stream:
                    for choice in getattr(chunk, "choices", None) or []:
                        token = getattr(getattr(choice, "delta", None), "content", None)
                        if not token:
                            continue
                        if metrics["llm_ttft_ms"] is None:
                            ms = 1000 * (time.monotonic() - llm_started)
                            metrics["llm_ttft_ms"] = ms
                            self._log(f"LLM first token ({ms:.0f}ms)")
                            self._event("llm_first_token", ms=round(ms, 1))
                        response += token
                        buffer, complete = split_sentences(buffer, token)
                        for sentence in complete:
                            await speaker.say(sentence)
                if buffer.strip():
                    await speaker.say(buffer.strip())

            if response.strip():
                self.messages.append({"role": "assistant", "content": response})
                self._emit("assistant", response)
                completed = True
        except asyncio.CancelledError:
            self._log(f"turn cancelled | spoken_sentences={len(speaker.spoken) if speaker else 0}")
            raise
        except Exception as exc:
            self._log(f"turn error: {exc!r}")
            log.exception("conversation turn failed")
            self._emit("error", str(exc))
        finally:
            # A cut-off turn keeps what the user already heard.
            spoken = " ".join(speaker.spoken).strip() if speaker else ""
            if spoken and not completed:
                self.messages.append({"role": "assistant", "content": spoken})
                self._emit("assistant", spoken)
            self._log(f"turn finished ({1000 * (time.monotonic() - started):.0f}ms)")
            self._trim_history()
            if speaker is not None:
                metrics["first_sentence_ms"] = speaker.first_sentence_ms
                metrics["tts_first_ms"] = speaker.first_synthesis_ms
                metrics["voice_to_voice_ms"] = speaker.first_audio_ms
            self._record_latency(
                metrics, speech_end_at, completed, text, response.strip() or spoken
            )

    # -- transport side ---------------------------------------------------

    async def interrupt(self) -> None:
        """Barge-in: cancel the in-flight turn and drop its queued audio."""
        self._log(f"barge-in | cancelling turn, output_chunks={self.output.qsize()}")
        self._event("barge_in")
        task, self._turn_task = self._turn_task, None
        self.agent_speaking = False
        self._first_audio_ref = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        _drain(self.output)

    def audio_written(self) -> None:
        """Call when a transport writes its first output frame of a reply."""
        if self._first_audio_ref is None:
            return
        ref, self._first_audio_ref = self._first_audio_ref, None
        elapsed_ms = 1000 * (time.monotonic() - ref)
        self._log(f"first audio written ({elapsed_ms:.0f}ms after last speech)")
        self._event("first_audio_written", ms=round(elapsed_ms, 1))

    def stop(self) -> None:
        """End the call: cancel the turn, drop queued audio, wake the loop."""
        self._stop.set()
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
        self._first_audio_ref = None
        _drain(self.output)
        with contextlib.suppress(asyncio.QueueFull):
            self.input.put_nowait(None)

    # -- bookkeeping ------------------------------------------------------

    def _record_latency(
        self,
        metrics: dict[str, float | None],
        speech_end_at: float,
        completed: bool,
        user_text: str,
        assistant_text: str,
    ) -> None:
        latency = TurnLatency(
            voice_to_voice_ms=metrics["voice_to_voice_ms"],
            endpoint_ms=metrics["endpoint_ms"] or 0.0,
            stt_ms=metrics["stt_ms"] or 0.0,
            llm_ttft_ms=metrics["llm_ttft_ms"],
            first_sentence_ms=metrics["first_sentence_ms"],
            tts_first_ms=metrics["tts_first_ms"],
            turn_ms=1000 * (time.monotonic() - speech_end_at),
            completed=completed,
        )
        self.last_latency = latency
        self._log(f"latency | {_format_latency(latency)}")
        self._event(
            "turn_done",
            user=user_text,
            assistant=assistant_text,
            completed=completed,
            voice_to_voice_ms=latency.voice_to_voice_ms,
            endpoint_ms=round(latency.endpoint_ms, 1),
            stt_ms=round(latency.stt_ms, 1),
            llm_ttft_ms=latency.llm_ttft_ms,
            first_sentence_ms=latency.first_sentence_ms,
            tts_first_ms=latency.tts_first_ms,
            turn_ms=round(latency.turn_ms, 1),
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[conversation] {message}", flush=True)

    def _event(self, name: str, **fields) -> None:
        if self.on_event is None:
            return
        record = {"event": name, "t": round(time.time(), 6), **fields}
        try:
            self.on_event(record)
        except Exception:
            log.exception("on_event callback failed")

    def _emit(self, role: str, text: str) -> None:
        if self.on_turn is None:
            return
        try:
            self.on_turn(role, text)
        except Exception:
            log.exception("on_turn callback failed")

    def _trim_history(self) -> None:
        offset = 1 if self.system_prompt else 0
        if len(self.messages) > offset + 20:
            self.messages = self.messages[:offset] + self.messages[-20:]


class Speaker:
    """Speaks one turn's sentences while the LLM keeps streaming.

    The only concurrency in the pipeline: the turn hands finished sentences to
    :meth:`say`, this task synthesizes each one and queues its PCM16 on
    ``conversation.output``, so the first sentence is heard while the model is
    still writing the rest. ``say`` blocks only when synthesis falls
    ``QUEUE_DEPTH`` sentences behind, and re-raises a synthesis failure so the
    turn stops instead of feeding a dead speaker.
    """

    QUEUE_DEPTH = 16

    def __init__(self, conversation: Conversation, speech_end_at: float):
        self._conversation = conversation
        self._speech_end_at = speech_end_at
        self._sentences: asyncio.Queue[str | None] = asyncio.Queue(self.QUEUE_DEPTH)
        self._task: asyncio.Task | None = None
        self.spoken: list[str] = []
        self.first_sentence_ms: float | None = None
        self.first_synthesis_ms: float | None = None
        self.first_audio_ms: float | None = None

    def _since_speech_ms(self) -> float:
        return 1000 * (time.monotonic() - self._speech_end_at)

    async def __aenter__(self) -> Self:
        self._task = asyncio.create_task(self._run(), name="speak")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if exc is None:
            if self._task.done():
                self._task.result()  # synthesis already failed; surface it
            await self._sentences.put(None)  # no more sentences: speak what is queued
            await self._task
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task

    async def say(self, sentence: str) -> None:
        """Queue one sentence; block only if the speaker is that far behind."""
        if self._task.done():
            self._task.result()
        if self.first_sentence_ms is None:
            self.first_sentence_ms = self._since_speech_ms()
        try:
            self._sentences.put_nowait(sentence)
        except asyncio.QueueFull:
            put = asyncio.create_task(self._sentences.put(sentence))
            await asyncio.wait((put, self._task), return_when=asyncio.FIRST_COMPLETED)
            if not put.done():
                put.cancel()
                await asyncio.gather(put, return_exceptions=True)
                self._task.result()

    async def _run(self) -> None:
        conversation = self._conversation
        chunk = conversation._chunk_bytes
        rate = conversation.config.sample_rate
        while True:
            sentence = await self._sentences.get()
            if sentence is None:
                return
            started = time.monotonic()
            audio = await conversation.speech.synthesize(sentence)
            elapsed_ms = 1000 * (time.monotonic() - started)
            if self.first_synthesis_ms is None:
                self.first_synthesis_ms = elapsed_ms
            conversation._log(f"TTS ({elapsed_ms:.0f}ms, {len(audio)} bytes): {sentence!r}")
            conversation._event(
                "tts",
                ms=round(elapsed_ms, 1),
                bytes=len(audio),
                audio_ms=round(len(audio) / 2 / rate * 1000, 1),
                text=sentence,
            )
            if not audio:
                continue
            self.spoken.append(sentence)
            conversation.agent_speaking = True
            if self.first_audio_ms is None:
                self.first_audio_ms = self._since_speech_ms()
                conversation._first_audio_ref = self._speech_end_at
                conversation._log(
                    f"voice-to-voice ({self.first_audio_ms:.0f}ms) | "
                    "last speech -> first audio queued"
                )
            for start in range(0, len(audio), chunk):
                await conversation.output.put(audio[start : start + chunk])


SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def split_sentences(buffer: str, token: str) -> tuple[str, list[str]]:
    """Append one LLM token; return the pending buffer and any complete sentences.

    A period is not a boundary on its own (the model quotes text), so pysbd
    decides and the text after the last boundary stays pending.
    """
    buffer += token
    if not any(mark in token for mark in ".!?\n"):
        return buffer, []
    parts = SEGMENTER.segment(buffer)
    if len(parts) < 2:
        return buffer, []
    return parts[-1], [text for text in (part.strip() for part in parts[:-1]) if text]


def _drain(queue: asyncio.Queue) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return


def _format_latency(latency: TurnLatency) -> str:
    def ms(value: float | None) -> str:
        return "-" if value is None else f"{value:.0f}ms"

    return (
        f"v2v={ms(latency.voice_to_voice_ms)} endpoint={ms(latency.endpoint_ms)} "
        f"stt={ms(latency.stt_ms)} llm_ttft={ms(latency.llm_ttft_ms)} "
        f"sentence={ms(latency.first_sentence_ms)} tts={ms(latency.tts_first_ms)} "
        f"turn={ms(latency.turn_ms)}"
    )
