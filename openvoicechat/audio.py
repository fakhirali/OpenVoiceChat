"""PCM16 plumbing: resampling, and the microphone/speaker pumps.

These are the two ends of the pipeline. Each one owns one direction of the
conversation's queues and nothing else; wiring them to streams is ``main.py``.
"""

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


def default_rate(sounddevice, kind: str, device=None) -> int:
    """The device's preferred sample rate, or 48 kHz when it will not say."""
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
            conversation.audio_written()
