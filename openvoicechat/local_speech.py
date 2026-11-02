"""Local Parakeet transcription and Qwen3-TTS synthesis."""

from __future__ import annotations

import asyncio
import os

import numpy as np
import soxr

STT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
SAMPLE_RATE = 16000
# Passed to Qwen; the 0.6B CustomVoice checkpoint currently ignores it.
STYLE = "Speak in a warm, conversational tone."

# qwen-tts pins Transformers 4.57, which predates Parakeet TDT. onnx-asr runs
# the same NVIDIA checkpoint through ONNX Runtime.
_ONNX_STT_MODEL = "nemo-parakeet-tdt-0.6b-v3"


class LocalSpeech:
    """Load Parakeet and Qwen3-TTS once, then use them from the local loop."""

    def __init__(self) -> None:
        import onnx_asr
        import onnxruntime
        import torch
        from qwen_tts import Qwen3TTSModel

        self._on_cuda = torch.cuda.is_available()
        available = onnxruntime.get_available_providers()
        stt_device = "CUDA" if "CUDAExecutionProvider" in available else "CPU"
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if stt_device == "CUDA"
            else ["CPUExecutionProvider"]
        )
        print(f"Loading Parakeet ({STT_MODEL})... ONNX Runtime, {stt_device}", flush=True)
        self._stt = onnx_asr.load_model(_ONNX_STT_MODEL, providers=providers)
        tts_device = "CUDA" if self._on_cuda else "CPU"
        print(f"Loading Qwen3-TTS ({TTS_MODEL})... {tts_device}", flush=True)
        kwargs = {
            "device_map": "cuda:0" if self._on_cuda else "cpu",
            "dtype": torch.bfloat16 if self._on_cuda else torch.float32,
        }
        try:
            self._tts = Qwen3TTSModel.from_pretrained(
                TTS_MODEL, attn_implementation="sdpa", **kwargs
            )
        except TypeError:
            self._tts = Qwen3TTSModel.from_pretrained(TTS_MODEL, **kwargs)
        self._voice = os.getenv("LOCAL_TTS_VOICE", "Aiden")
        self._language = os.getenv("LOCAL_TTS_LANGUAGE", "English")

    async def transcribe(self, pcm16: bytes) -> str:
        if not pcm16:
            return ""
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        text = await asyncio.to_thread(
            self._stt.recognize, audio, sample_rate=SAMPLE_RATE
        )
        return (text or "").strip()

    async def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""
        wavs, sample_rate = await asyncio.to_thread(
            self._tts.generate_custom_voice,
            text=text,
            language=self._language,
            speaker=self._voice,
            instruct=STYLE,
        )
        wav = wavs[0]
        if hasattr(wav, "detach"):
            wav = wav.detach().float().cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE:
            wav = soxr.resample(wav, sample_rate, SAMPLE_RATE)
        wav = np.clip(wav, -1.0, 1.0)
        return (wav * 32767.0).astype(np.int16).tobytes()

    async def aclose(self) -> None:
        return None
