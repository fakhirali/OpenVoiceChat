"""Local Parakeet transcription and Kokoro speech synthesis."""

from __future__ import annotations

import asyncio
import os
import warnings

import numpy as np
import soxr

STT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
TTS_MODEL = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"

# onnx-asr runs the same NVIDIA Parakeet checkpoint through ONNX Runtime.
_ONNX_STT_MODEL = "nemo-parakeet-tdt-0.6b-v3"


class LocalSpeech:
    """Load Parakeet and Kokoro once, then use them from the local loop."""

    def __init__(self) -> None:
        import onnx_asr
        import onnxruntime
        import torch

        # ORT logs a CUDA memcpy notice at warning level; it is not actionable
        # here because the pipeline does not use CUDA graphs. Keep errors only.
        onnxruntime.set_default_logger_severity(3)

        self._on_cuda = torch.cuda.is_available()
        available = onnxruntime.get_available_providers()
        stt_device = (
            "CUDA"
            if self._on_cuda and "CUDAExecutionProvider" in available
            else "CPU"
        )
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if stt_device == "CUDA"
            else ["CPUExecutionProvider"]
        )
        print(f"Loading Parakeet ({STT_MODEL})... ONNX Runtime, {stt_device}", flush=True)
        self._stt = onnx_asr.load_model(_ONNX_STT_MODEL, providers=providers)
        tts_device = "CUDA" if self._on_cuda else "CPU"
        print(f"Loading Kokoro ({TTS_MODEL})... {tts_device}", flush=True)
        with warnings.catch_warnings():
            # Kokoro's model code trips three torch deprecation warnings that we
            # cannot fix from here. Keep the rest of the warning stream intact.
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.filterwarnings(
                "ignore",
                message=r".*dropout option adds dropout.*",
                category=UserWarning,
            )
            from kokoro import KPipeline

            self._tts = KPipeline(
                lang_code="a",
                repo_id=TTS_MODEL,
                device="cuda" if self._on_cuda else "cpu",
            )
        self._voice = os.getenv("LOCAL_TTS_VOICE", DEFAULT_VOICE)

    async def transcribe(self, pcm16: bytes) -> str:
        if not pcm16:
            return ""
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        text = await asyncio.to_thread(
            self._stt.recognize, audio, sample_rate=SAMPLE_RATE
        )
        return (text or "").strip()

    def _render(self, text: str) -> bytes:
        chunks = [result.audio for result in self._tts(text, voice=self._voice, speed=1.0)]
        if not chunks:
            return b""
        arrays = [
            chunk.detach().float().cpu().numpy()
            if hasattr(chunk, "detach")
            else np.asarray(chunk, dtype=np.float32)
            for chunk in chunks
        ]
        wav = np.concatenate([array.reshape(-1) for array in arrays])
        if TTS_SAMPLE_RATE != SAMPLE_RATE:
            wav = soxr.resample(wav, TTS_SAMPLE_RATE, SAMPLE_RATE)
        wav = np.clip(wav, -1.0, 1.0)
        return (wav * 32767.0).astype(np.int16).tobytes()

    async def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""
        return await asyncio.to_thread(self._render, text)

    async def aclose(self) -> None:
        return None
