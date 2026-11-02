"""Run the full local voice loop: microphone -> VAD -> STT -> LLM -> TTS -> speaker."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import numpy as np

from openvoicechat.config import Config
from openvoicechat.conversation import Conversation
from openvoicechat.speech import Speech

DEFAULT_PROMPT = "You are a concise, friendly voice assistant."
SPEECH_RATE = 16000
FRAME_SECONDS = 0.032


def _resample(pcm: bytes, source: int, target: int) -> bytes:
    if not pcm or source == target:
        return pcm
    import soxr

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    audio = soxr.resample(audio, source, target)
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _default_rate(sounddevice, kind: str) -> int:
    try:
        return round(float(sounddevice.query_devices(kind=kind)["default_samplerate"]))
    except Exception:  # noqa: BLE001 - PortAudio raises its own exception types
        return 48000


def _device(value: str):
    try:
        return int(value)
    except ValueError:
        return value


async def _microphone(conversation: Conversation, stream, rate: int) -> None:
    frames = max(1, int(rate * FRAME_SECONDS))
    while True:
        data, overflowed = await asyncio.to_thread(stream.read, frames)
        if overflowed:
            print("input overflow", file=sys.stderr, flush=True)
        pcm = _resample(bytes(data), rate, SPEECH_RATE)
        if pcm:
            await conversation.input.put(pcm)


async def _speaker(conversation: Conversation, stream, rate: int) -> None:
    while True:
        pcm = await conversation.output.get()
        if pcm is None:
            return
        data = _resample(pcm, SPEECH_RATE, rate)
        if data:
            await asyncio.to_thread(stream.write, data)


async def run(args: argparse.Namespace) -> None:
    try:
        import sounddevice as sd
        import soxr  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            'Missing local audio dependencies. Install with: pip install -e ".[local]"'
        ) from exc

    speech = Speech(Config())
    try:
        print("Loading VAD model...", flush=True)
        conversation = Conversation(
            speech,
            system_prompt=args.prompt,
            on_turn=lambda role, text: print(f"{role.upper()}: {text}", flush=True),
        )
        input_rate = _default_rate(sd, "input")
        output_rate = _default_rate(sd, "output")
        print(
            f"Audio: mic {input_rate} Hz, speaker {output_rate} Hz, speech {SPEECH_RATE} Hz",
            flush=True,
        )
        with sd.RawInputStream(
            samplerate=input_rate,
            channels=1,
            dtype="int16",
            blocksize=0,
            device=args.input_device,
        ) as input_stream, sd.RawOutputStream(
            samplerate=output_rate,
            channels=1,
            dtype="int16",
            blocksize=0,
            device=args.output_device,
        ) as output_stream:
            print("Listening. Ctrl-C to stop.", flush=True)
            tasks = [
                asyncio.create_task(conversation.run(), name="conversation"),
                asyncio.create_task(
                    _microphone(conversation, input_stream, input_rate), name="microphone"
                ),
                asyncio.create_task(
                    _speaker(conversation, output_stream, output_rate), name="speaker"
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
    finally:
        await speech.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full local voice loop: microphone -> STT -> LLM -> TTS -> speaker"
    )
    parser.add_argument(
        "--prompt",
        default=os.getenv("SYSTEM_PROMPT", DEFAULT_PROMPT),
        help="system prompt",
    )
    parser.add_argument("--input-device", type=_device, default=None, help="sounddevice input device")
    parser.add_argument(
        "--output-device", type=_device, default=None, help="sounddevice output device"
    )
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)


if __name__ == "__main__":
    main()
