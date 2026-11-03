"""Run the local voice loop: microphone -> VAD -> Parakeet -> LLM -> Kokoro -> speaker."""

from __future__ import annotations

import argparse
import asyncio
import os

from openvoicechat.audio import run_local
from openvoicechat.conversation import Conversation

DEFAULT_PROMPT = "You are a concise, friendly voice assistant."


def _device(value: str):
    try:
        return int(value)
    except ValueError:
        return value


async def run(args: argparse.Namespace) -> None:
    try:
        from openvoicechat.local_speech import LocalSpeech
    except ImportError as exc:
        raise SystemExit(
            "Missing local model dependencies. Install with: "
            'pip install -e ".[models]"'
        ) from exc

    speech = LocalSpeech()
    try:
        print("Loading VAD model...", flush=True)
        conversation = Conversation(
            speech,
            system_prompt=args.prompt,
            on_turn=lambda role, text: print(f"{role.upper()}: {text}", flush=True),
            verbose=args.verbose,
        )
        try:
            await run_local(
                conversation,
                input_device=args.input_device,
                output_device=args.output_device,
            )
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc
    finally:
        await speech.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local voice loop: microphone -> Parakeet -> LLM -> Kokoro"
    )
    parser.add_argument(
        "--prompt",
        default=os.getenv("SYSTEM_PROMPT", DEFAULT_PROMPT),
        help="system prompt",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="print VAD, endpointing, STT, LLM, and TTS timing details",
    )
    parser.add_argument(
        "--input-device", type=_device, default=None, help="sounddevice input device"
    )
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
