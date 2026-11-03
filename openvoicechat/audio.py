"""Local microphone/speaker I/O and PCM16 resampling."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .conversation import Conversation

SPEECH_RATE = 16000
FRAME_SAMPLES = 512
FRAME_SECONDS = FRAME_SAMPLES / SPEECH_RATE
SPEECH_FRAME_BYTES = FRAME_SAMPLES * 2


class Pcm16Resampler:
    """Stateful PCM16 resampler; avoids clicks at chunk boundaries."""

    def __init__(self, source: int, target: int):
        self._identity = source == target
        if not self._identity:
            import soxr

            self._stream = soxr.ResampleStream(
                source, target, 1, dtype="int16", quality="HQ"
            )

    def process(self, pcm: bytes) -> bytes:
        if not pcm or self._identity:
            return pcm
        data = np.frombuffer(pcm, dtype=np.int16)
        return self._stream.resample_chunk(data, last=False).tobytes()


def _default_rate(sounddevice, kind: str, device=None) -> int:
    try:
        return round(
            float(sounddevice.query_devices(device, kind=kind)["default_samplerate"])
        )
    except Exception:  # noqa: BLE001 - PortAudio raises its own exception types
        return 48000


async def microphone(conversation: Conversation, stream, rate: int) -> None:
    """Read device audio, resample to 16 kHz, and feed conversation.input."""
    frames = max(1, int(rate * FRAME_SECONDS))
    resampler = Pcm16Resampler(rate, SPEECH_RATE)
    buffer = bytearray()
    while True:
        data, overflowed = await asyncio.to_thread(stream.read, frames)
        if overflowed:
            print("input overflow", file=sys.stderr, flush=True)
        if not data:
            await asyncio.sleep(0.01)
            continue
        buffer.extend(resampler.process(bytes(data)))
        while len(buffer) >= SPEECH_FRAME_BYTES:
            chunk = bytes(buffer[:SPEECH_FRAME_BYTES])
            del buffer[:SPEECH_FRAME_BYTES]
            await conversation.input.put(chunk)


async def speaker(conversation: Conversation, stream, rate: int) -> None:
    """Drain conversation.output and write it at the device rate."""
    resampler = Pcm16Resampler(SPEECH_RATE, rate)
    while True:
        pcm = await conversation.output.get()
        data = resampler.process(pcm)
        if data:
            await asyncio.to_thread(stream.write, data)


async def run_local(
    conversation: Conversation,
    *,
    input_device=None,
    output_device=None,
) -> None:
    """Run one conversation against the default microphone and speaker."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise ImportError(
            'Missing audio dependencies. Install with: pip install -e ".[local]"'
        ) from exc

    input_rate = _default_rate(sd, "input", input_device)
    output_rate = _default_rate(sd, "output", output_device)
    print(
        f"Audio: mic {input_rate} Hz, speaker {output_rate} Hz, speech {SPEECH_RATE} Hz",
        flush=True,
    )
    with sd.RawInputStream(
        samplerate=input_rate,
        channels=1,
        dtype="int16",
        blocksize=0,
        device=input_device,
    ) as input_stream, sd.RawOutputStream(
        samplerate=output_rate,
        channels=1,
        dtype="int16",
        blocksize=0,
        device=output_device,
    ) as output_stream:
        print("Listening. Ctrl-C to stop.", flush=True)
        tasks = [
            asyncio.create_task(conversation.run(), name="conversation"),
            asyncio.create_task(
                microphone(conversation, input_stream, input_rate), name="microphone"
            ),
            asyncio.create_task(
                speaker(conversation, output_stream, output_rate), name="speaker"
            ),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.cancelled():
                    continue
                if task.exception() is not None:
                    print(
                        f"{task.get_name()} failed: {task.exception()}",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            conversation.stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
