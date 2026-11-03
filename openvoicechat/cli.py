"""The ``ovc`` command line.

Only ``doctor`` exists so far. ``init``, ``models``, ``run``, ``bench``, and
``serve`` land as the runtime is built out.

Imports stay lazy: ``ovc --help`` and ``ovc --version`` must not load numpy,
onnxruntime, torch, or any model code.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("openvoicechat")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.4.0"

SCHEMA_VERSION = 1
_LABEL = {"ok": "PASS", "warn": "WARN", "missing": "MISSING", "fail": "FAIL"}


@dataclass
class Check:
    name: str
    status: str  # ok | warn | missing | fail
    detail: str


def _python() -> Check:
    return Check("python", "ok", f"{platform.python_version()} ({platform.system()})")


def _torch() -> Check:
    try:
        import torch
    except ImportError:
        return Check("torch", "missing", 'pip install "openvoicechat[models]"')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1024**3
        return Check("torch", "ok", f"{torch.__version__} | CUDA: {props.name} ({vram:.1f} GB)")
    return Check("torch", "warn", f"{torch.__version__} | CPU only; Kokoro will be slow")


def _onnxruntime() -> Check:
    try:
        import onnxruntime as ort
    except ImportError:
        return Check("onnxruntime", "fail", "not installed; reinstall openvoicechat")
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return Check("onnxruntime", "ok", f"{ort.__version__} | CUDAExecutionProvider")
    joined = ", ".join(providers) or "none"
    return Check("onnxruntime", "warn", f"{ort.__version__} | CPU only ({joined})")


def _module(name: str, check: str, hint: str) -> Check:
    try:
        __import__(name)
    except ImportError:
        return Check(check, "missing", hint)
    return Check(check, "ok", f"import {name}")


def _audio() -> Check:
    try:
        import sounddevice as sd
    except ImportError:
        return Check("audio", "missing", 'pip install "openvoicechat[local]"')
    try:
        devices = sd.query_devices()
    except Exception as exc:  # noqa: BLE001 - PortAudio raises its own types
        return Check("audio", "warn", f"sounddevice present, no device list: {exc}")
    inputs = sum(1 for device in devices if device["max_input_channels"] > 0)
    outputs = sum(1 for device in devices if device["max_output_channels"] > 0)
    status = "ok" if inputs and outputs else "warn"
    return Check("audio", status, f"{inputs} input / {outputs} output device(s)")


def _silero() -> Check:
    from .vad import MODEL_PATH

    if MODEL_PATH.exists():
        return Check("silero", "ok", f"{MODEL_PATH} ({MODEL_PATH.stat().st_size // 1024} KB)")
    return Check("silero", "warn", "downloads on first run (~2 MB)")


def _llm() -> Check:
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        return Check("llm", "fail", "set LLM_API_KEY (any value works for local servers)")
    base = os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
    model = os.getenv("LLM_MODEL", "gpt-4.1-nano")
    return Check("llm", "ok", f"{model} @ {base}")


def collect_checks() -> list[Check]:
    return [
        _python(),
        _torch(),
        _onnxruntime(),
        _module("onnx_asr", "parakeet", 'pip install "openvoicechat[models]"'),
        _module("kokoro", "kokoro", 'pip install "openvoicechat[models]"'),
        _audio(),
        _silero(),
        _llm(),
    ]


def doctor(as_json: bool) -> int:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    checks = collect_checks()
    ok = not any(check.status in ("missing", "fail") for check in checks)
    if as_json:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "ok": ok,
            "checks": [asdict(check) for check in checks],
        }
        print(json.dumps(payload, indent=2))
    else:
        for check in checks:
            print(f"{_LABEL[check.status]:<8} {check.name:<12} {check.detail}")
        print()
        print("Ready." if ok else "Not ready: fix the missing or failed checks above.")
    return 0 if ok else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ovc",
        description="OpenVoiceChat: local voice runtime for Python.",
    )
    parser.add_argument("--version", action="version", version=f"ovc {__version__}")
    subcommands = parser.add_subparsers(dest="command")
    check = subcommands.add_parser("doctor", help="check hardware, models, and configuration")
    check.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor(args.json)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
