"""The async listen -> transcribe -> LLM -> speak pipeline."""

import asyncio
import contextlib
import logging
import math
import time
from collections import deque

import pysbd
from openai import AsyncOpenAI

from .config import Config
from .vad import VAD

log = logging.getLogger(__name__)


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
        vad=None,
        verbose: bool = False,
    ):
        self.speech = speech
        self.config = config or Config()
        self.system_prompt = system_prompt
        self.on_turn = on_turn
        self.verbose = verbose
        self._owns_client = client is None
        if client is None:
            if not self.config.llm_api_key:
                raise ValueError("LLM_API_KEY (or OPENAI_API_KEY) is required")
            client = AsyncOpenAI(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url or None,
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
        pre_roll: deque[bytes] = deque(
            maxlen=math.ceil(self.config.pre_roll_ms / self._frame_ms)
        )
        onset = bytearray()
        utterance = bytearray()
        voiced_run = 0
        in_speech = False
        speech_ms = silence_ms = utterance_ms = 0.0
        endpoint_ms = self.config.endpoint_silence_ms
        min_speech_ms = self.config.min_speech_ms
        max_utterance_ms = self.config.max_utterance_ms

        def reset() -> None:
            nonlocal in_speech, voiced_run, speech_ms, silence_ms, utterance_ms
            in_speech = False
            voiced_run = 0
            speech_ms = silence_ms = utterance_ms = 0.0
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

                if (
                    voiced
                    and not in_speech
                    and (self._turn_task is not None or self.agent_speaking)
                ):
                    await self._interrupt()

                if not in_speech:
                    if voiced:
                        voiced_run += 1
                        onset += frame
                        if voiced_run >= 2:
                            in_speech = True
                            speech_ms = voiced_run * self._frame_ms
                            pre_roll_ms = len(pre_roll) * self._frame_ms
                            utterance = bytearray(b"".join(pre_roll)) + onset
                            utterance_ms = (len(pre_roll) + voiced_run) * self._frame_ms
                            pre_roll.clear()
                            onset.clear()
                            self._log(
                                f"speech start | pre-roll={pre_roll_ms:.0f}ms"
                            )
                    else:
                        voiced_run = 0
                        onset.clear()
                        if pre_roll.maxlen:
                            pre_roll.append(frame)
                    continue

                utterance += frame
                utterance_ms += self._frame_ms
                if voiced:
                    speech_ms += self._frame_ms
                    silence_ms = 0.0
                else:
                    silence_ms += self._frame_ms

                if silence_ms < endpoint_ms and utterance_ms < max_utterance_ms:
                    continue

                reason = "max-utterance" if utterance_ms >= max_utterance_ms else "silence"
                self._log(
                    f"endpoint ({reason}) | silence={silence_ms:.0f}ms "
                    f"utterance={utterance_ms:.0f}ms speech={speech_ms:.0f}ms"
                )
                if speech_ms >= min_speech_ms and utterance:
                    self._turn_task = asyncio.create_task(self._turn(bytes(utterance)))
                else:
                    self._log(f"drop short utterance | speech={speech_ms:.0f}ms")
                reset()
                if not voiced and pre_roll.maxlen:
                    pre_roll.append(frame)
        finally:
            self._stop.set()
            task, self._turn_task = self._turn_task, None
            if task is not None:
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            self.agent_speaking = False
            _drain(self.output)
            if self._owns_client:
                with contextlib.suppress(Exception):
                    await self.client.close()

    def stop(self) -> None:
        self._stop.set()
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
        _drain(self.output)
        with contextlib.suppress(asyncio.QueueFull):
            self.input.put_nowait(None)

    async def _turn(self, pcm: bytes) -> None:
        spoken: list[str] = []
        sentences: asyncio.Queue[str | None] = asyncio.Queue(maxsize=16)
        tts_task: asyncio.Task | None = None
        response = ""
        completed = False
        started = time.monotonic()
        self._log(
            f"turn start | pcm={len(pcm) / 2 / self.config.sample_rate * 1000:.0f}ms"
        )
        try:
            stt_started = time.monotonic()
            text = (await self.speech.transcribe(pcm)).strip()
            self._log(
                f"STT ({1000 * (time.monotonic() - stt_started):.0f}ms): {text!r}"
            )
            if not text:
                return
            self.messages.append({"role": "user", "content": text})
            self._emit("user", text)

            tts_task = asyncio.create_task(self._speak(sentences, spoken))
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
                        self._log(
                            f"LLM first token ({1000 * (time.monotonic() - llm_started):.0f}ms)"
                        )
                    response += token
                    buffer += token
                    if any(mark in buffer for mark in ".!?\n"):
                        parts = segmenter.segment(buffer)
                        if len(parts) > 1:
                            for sentence in parts[:-1]:
                                sentence = sentence.strip()
                                if sentence:
                                    await self._feed(sentences, sentence, tts_task)
                            buffer = parts[-1]

            if buffer.strip():
                await self._feed(sentences, buffer.strip(), tts_task)
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

    async def _speak(self, sentences: asyncio.Queue, spoken: list[str]) -> None:
        while True:
            sentence = await sentences.get()
            if sentence is None:
                return
            tts_started = time.monotonic()
            audio = await self.speech.synthesize(sentence)
            self._log(
                f"TTS ({1000 * (time.monotonic() - tts_started):.0f}ms, "
                f"{len(audio)} bytes): {sentence!r}"
            )
            if not audio:
                continue
            spoken.append(sentence)
            self.agent_speaking = True
            for start in range(0, len(audio), self._chunk_bytes):
                await self.output.put(audio[start : start + self._chunk_bytes])

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
        task, self._turn_task = self._turn_task, None
        self.agent_speaking = False
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        _drain(self.output)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[conversation] {message}", flush=True)

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
