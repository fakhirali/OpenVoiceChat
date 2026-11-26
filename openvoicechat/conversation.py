"""The async listen -> transcribe -> LLM -> speak pipeline."""

import asyncio
import contextlib
import logging
import math
import time
from collections import deque
from dataclasses import dataclass

import pysbd
from openai import AsyncOpenAI

from .config import Config
from .vad import VAD

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnLatency:
    """Per-turn timings in milliseconds, measured from the last user speech frame.

    ``voice_to_voice_ms`` is the number that matters: it ends when the first
    audio chunk is queued for the transport. Call :meth:`Conversation.audio_written`
    from a transport to also log when that chunk actually reached the device.
    """

    voice_to_voice_ms: float | None
    endpoint_ms: float
    stt_ms: float
    llm_ttft_ms: float | None
    first_sentence_ms: float | None
    tts_first_ms: float | None
    turn_ms: float
    completed: bool


class Conversation:
    """Owns one voice conversation over input/output PCM16 queues."""

    def __init__(
        self,
        speech,
        *,
        system_prompt: str = "",
        config: Config | None = None,
        client: AsyncOpenAI | None = None,
        on_turn=None,
        on_latency=None,
        on_event=None,
        vad=None,
        verbose: bool = False,
    ):
        self.speech = speech
        self.config = config or Config()
        self.system_prompt = system_prompt
        self.on_turn = on_turn
        self.on_latency = on_latency
        self.on_event = on_event
        self.verbose = verbose
        self.last_latency: TurnLatency | None = None
        self._speech_end_at = 0.0
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
        self.messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        self._frame_ms = 1000 * self.config.frame_samples / self.config.sample_rate
        self._chunk_bytes = (
            int(self.config.sample_rate * self.config.output_chunk_ms / 1000) * 2
        )
        self._stop = asyncio.Event()
        self._turn_task: asyncio.Task | None = None

    async def run(self) -> None:
        self._event(
            "call_start",
            sample_rate=self.config.sample_rate,
            endpoint_silence_ms=self.config.endpoint_silence_ms,
        )
        pre_roll: deque[bytes] = deque(
            maxlen=math.ceil(self.config.pre_roll_ms / self._frame_ms)
        )
        onset: list[bytes] = []
        utterance: deque[bytes] = deque()
        utterance_frames = 0
        voiced_run = 0
        in_speech = False
        capped = False
        speech_ms = silence_ms = 0.0
        endpoint_ms = self.config.endpoint_silence_ms
        min_speech_ms = self.config.min_speech_ms
        max_utterance_ms = self.config.max_utterance_ms

        def reset() -> None:
            nonlocal in_speech, voiced_run, capped, speech_ms, silence_ms, utterance_frames
            in_speech = False
            voiced_run = 0
            capped = False
            speech_ms = silence_ms = 0.0
            utterance_frames = 0
            utterance.clear()
            onset.clear()
            pre_roll.clear()
            self.vad.reset()

        try:
            while not self._stop.is_set():
                frame = await self.input.get()
                if frame is None or self._stop.is_set():
                    break
                if not frame:
                    continue

                if (
                    self._turn_task is not None
                    and self._turn_task.done()
                    and self.output.empty()
                ):
                    self._turn_task = None
                    self.agent_speaking = False

                voiced = bool(self.vad(frame))
                if voiced:
                    self._speech_end_at = time.monotonic()

                if (
                    voiced
                    and not in_speech
                    and (self._turn_task is not None or self.agent_speaking)
                ):
                    await self._interrupt()

                if not in_speech:
                    if voiced:
                        voiced_run += 1
                        onset.append(frame)
                        if voiced_run >= 2:
                            in_speech = True
                            speech_ms = voiced_run * self._frame_ms
                            pre_roll_ms = len(pre_roll) * self._frame_ms
                            utterance.extend(pre_roll)
                            utterance.extend(onset)
                            utterance_frames = len(pre_roll) + len(onset)
                            pre_roll.clear()
                            onset.clear()
                            self._log(
                                f"speech start | pre-roll={pre_roll_ms:.0f}ms"
                            )
                            self._event(
                                "speech_start", pre_roll_ms=round(pre_roll_ms, 1)
                            )
                    else:
                        voiced_run = 0
                        onset.clear()
                        if pre_roll.maxlen:
                            pre_roll.append(frame)
                    continue

                utterance.append(frame)
                utterance_frames += 1
                utterance_ms = utterance_frames * self._frame_ms

                if voiced:
                    speech_ms += self._frame_ms
                    silence_ms = 0.0
                    if utterance_ms >= max_utterance_ms:
                        # Still talking past the cap: keep the most recent window and
                        # wait for silence. Ending the turn mid-speech makes the next
                        # voiced frame look like barge-in and cancels it.
                        while utterance_frames > 1 and utterance_ms > max_utterance_ms:
                            utterance.popleft()
                            utterance_frames -= 1
                            utterance_ms = utterance_frames * self._frame_ms
                        if not capped:
                            capped = True
                            self._log(
                                f"utterance capped | keeping last {utterance_ms:.0f}ms"
                            )
                    continue

                silence_ms += self._frame_ms
                if silence_ms < endpoint_ms and utterance_ms < max_utterance_ms:
                    continue

                reason = "max-utterance" if utterance_ms >= max_utterance_ms else "silence"
                self._log(
                    f"endpoint ({reason}) | silence={silence_ms:.0f}ms "
                    f"utterance={utterance_ms:.0f}ms speech={speech_ms:.0f}ms"
                )
                if speech_ms >= min_speech_ms and utterance:
                    self._event(
                        "endpoint",
                        reason=reason,
                        silence_ms=round(silence_ms, 1),
                        speech_ms=round(speech_ms, 1),
                        utterance_ms=round(utterance_ms, 1),
                    )
                    self._turn_task = asyncio.create_task(
                        self._turn(b"".join(utterance), self._speech_end_at)
                    )
                else:
                    self._event("dropped", speech_ms=round(speech_ms, 1))
                    self._log(f"drop short utterance | speech={speech_ms:.0f}ms")
                reset()
                if pre_roll.maxlen:
                    pre_roll.append(frame)
        finally:
            self._stop.set()
            task, self._turn_task = self._turn_task, None
            if task is not None:
                if not task.done():
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

    def stop(self) -> None:
        self._stop.set()
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
        self._first_audio_ref = None
        _drain(self.output)
        with contextlib.suppress(asyncio.QueueFull):
            self.input.put_nowait(None)

    async def _turn(self, pcm: bytes, speech_end_at: float) -> None:
        spoken: list[str] = []
        sentences: asyncio.Queue[str | None] = asyncio.Queue(maxsize=16)
        tts_task: asyncio.Task | None = None
        response = ""
        text = ""
        completed = False
        started = time.monotonic()
        metrics: dict[str, float | None] = {
            "endpoint_ms": 1000 * (started - speech_end_at),
            "stt_ms": 0.0,
            "llm_ttft_ms": None,
            "first_sentence_ms": None,
            "tts_first_ms": None,
            "voice_to_voice_ms": None,
        }
        self._log(
            f"turn start | pcm={len(pcm) / 2 / self.config.sample_rate * 1000:.0f}ms"
        )

        async def feed_sentence(sentence: str) -> None:
            if metrics["first_sentence_ms"] is None:
                metrics["first_sentence_ms"] = 1000 * (
                    time.monotonic() - speech_end_at
                )
            await self._feed(sentences, sentence, tts_task)

        try:
            stt_started = time.monotonic()
            text = (await self.speech.transcribe(pcm)).strip()
            metrics["stt_ms"] = 1000 * (time.monotonic() - stt_started)
            self._log(f"STT ({metrics['stt_ms']:.0f}ms): {text!r}")
            self._event("stt", ms=round(metrics["stt_ms"], 1), text=text)
            if not text:
                return
            self.messages.append({"role": "user", "content": text})
            self._emit("user", text)

            tts_task = asyncio.create_task(
                self._speak(sentences, spoken, metrics, speech_end_at)
            )
            segmenter = pysbd.Segmenter(language="en", clean=False)
            buffer = ""
            llm_started = time.monotonic()
            first_token = True
            stream = await self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=self.messages,
                stream=True,
            )
            async for chunk in stream:
                for choice in getattr(chunk, "choices", None) or []:
                    token = getattr(getattr(choice, "delta", None), "content", None)
                    if not token:
                        continue
                    if first_token:
                        first_token = False
                        metrics["llm_ttft_ms"] = 1000 * (
                            time.monotonic() - llm_started
                        )
                        self._log(
                            f"LLM first token ({metrics['llm_ttft_ms']:.0f}ms)"
                        )
                        self._event(
                            "llm_first_token",
                            ms=round(metrics["llm_ttft_ms"], 1),
                        )
                    response += token
                    buffer += token
                    if any(mark in buffer for mark in ".!?\n"):
                        parts = segmenter.segment(buffer)
                        if len(parts) > 1:
                            for sentence in parts[:-1]:
                                sentence = sentence.strip()
                                if sentence:
                                    await feed_sentence(sentence)
                            buffer = parts[-1]

            if buffer.strip():
                await feed_sentence(buffer.strip())
            await self._feed(sentences, None, tts_task)
            await tts_task

            if response.strip():
                self.messages.append({"role": "assistant", "content": response})
                self._emit("assistant", response)
                completed = True
        except asyncio.CancelledError:
            self._log(f"turn cancelled | spoken_sentences={len(spoken)}")
            partial = " ".join(spoken).strip()
            if partial and not completed:
                self.messages.append({"role": "assistant", "content": partial})
                self._emit("assistant", partial)
            raise
        except Exception as exc:
            self._log(f"turn error: {exc!r}")
            log.exception("conversation turn failed")
            partial = " ".join(spoken).strip()
            if partial and not completed:
                self.messages.append({"role": "assistant", "content": partial})
                self._emit("assistant", partial)
            self._emit("error", str(exc))
        finally:
            self._log(f"turn finished ({1000 * (time.monotonic() - started):.0f}ms)")
            self._trim_history()
            if tts_task is not None:
                if not tts_task.done():
                    tts_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await tts_task
            self._record_latency(
                metrics,
                speech_end_at,
                completed,
                text,
                response.strip() or " ".join(spoken).strip(),
            )

    async def _speak(
        self,
        sentences: asyncio.Queue,
        spoken: list[str],
        metrics: dict[str, float | None],
        speech_end_at: float,
    ) -> None:
        while True:
            sentence = await sentences.get()
            if sentence is None:
                return
            tts_started = time.monotonic()
            audio = await self.speech.synthesize(sentence)
            elapsed_ms = 1000 * (time.monotonic() - tts_started)
            if metrics["tts_first_ms"] is None:
                metrics["tts_first_ms"] = elapsed_ms
            self._log(
                f"TTS ({elapsed_ms:.0f}ms, {len(audio)} bytes): {sentence!r}"
            )
            self._event(
                "tts",
                ms=round(elapsed_ms, 1),
                bytes=len(audio),
                audio_ms=round(len(audio) / 2 / self.config.sample_rate * 1000, 1),
                text=sentence,
            )
            if not audio:
                continue
            spoken.append(sentence)
            self.agent_speaking = True
            if metrics["voice_to_voice_ms"] is None:
                metrics["voice_to_voice_ms"] = 1000 * (
                    time.monotonic() - speech_end_at
                )
                self._first_audio_ref = speech_end_at
                self._log(
                    f"voice-to-voice ({metrics['voice_to_voice_ms']:.0f}ms) | "
                    "last speech -> first audio queued"
                )
            for start in range(0, len(audio), self._chunk_bytes):
                await self.output.put(audio[start : start + self._chunk_bytes])

    def audio_written(self) -> None:
        """Call when a transport writes its first output frame of a reply."""
        if self._first_audio_ref is None:
            return
        ref, self._first_audio_ref = self._first_audio_ref, None
        elapsed_ms = 1000 * (time.monotonic() - ref)
        self._log(
            f"first audio written ({elapsed_ms:.0f}ms after last speech)"
        )
        self._event("first_audio_written", ms=round(elapsed_ms, 1))

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
        if self.on_latency is None:
            return
        try:
            self.on_latency(latency)
        except Exception:
            log.exception("on_latency callback failed")

    @staticmethod
    async def _feed(queue: asyncio.Queue, item, tts_task: asyncio.Task) -> None:
        put = asyncio.create_task(queue.put(item))
        try:
            done, _ = await asyncio.wait((put, tts_task), return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            put.cancel()
            await asyncio.gather(put, return_exceptions=True)
            raise
        if put in done:
            put.result()
            return
        put.cancel()
        await asyncio.gather(put, return_exceptions=True)
        tts_task.result()

    async def _interrupt(self) -> None:
        self._log(
            f"barge-in | cancelling turn, output_chunks={self.output.qsize()}"
        )
        self._event("barge_in")
        task, self._turn_task = self._turn_task, None
        self.agent_speaking = False
        self._first_audio_ref = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        _drain(self.output)

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
