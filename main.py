"""Run the local voice loop: microphone -> VAD -> Parakeet -> LLM -> Kokoro -> speaker."""

from __future__ import annotations

import argparse
import asyncio

from openvoicechat.audio import run_local
from openvoicechat.config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, Config
from openvoicechat.conversation import Conversation
from openvoicechat.local_speech import DEFAULT_VOICE, LocalSpeech

DEFAULT_PROMPT = "You are a concise, friendly voice assistant."


def _device(value: str):
    try:
        return int(value)
    except ValueError:
        return value


async def run(args: argparse.Namespace) -> None:
    try:
        speech = LocalSpeech(voice=args.voice)
    except ImportError as exc:
        raise SystemExit(
            "Missing local model dependencies. Install with: "
            'pip install -e ".[models]"'
        ) from exc

    config = Config(llm_model=args.model, llm_base_url=args.base_url)
    try:
        print("Loading VAD model...", flush=True)
        conversation = Conversation(
            speech,
            system_prompt=args.prompt,
            config=config,
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
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="system prompt")
    parser.add_argument(
        "--model", default=DEFAULT_LLM_MODEL, help="LLM model name"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_LLM_BASE_URL,
        help="OpenAI-compatible base URL (empty means OpenAI)",
    )
    parser.add_argument(
        "--voice", default=DEFAULT_VOICE, help="Kokoro voice name"
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
