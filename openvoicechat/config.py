"""Environment-backed configuration."""

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # LLM: any OpenAI-compatible endpoint. Local by preference, hosted by configuration.
    llm_api_key: str = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4.1-nano")
    llm_timeout_s: float = _float("LLM_TIMEOUT", 15.0)

    # Daily agent server (optional extra).
    agent_api_token: str = os.getenv("AGENT_API_TOKEN", "")
    max_sessions: int = _int("MAX_SESSIONS", 5)
    data_dir: str = os.getenv("DATA_DIR", "data")

    # Audio and turn taking.
    sample_rate: int = _int("SAMPLE_RATE", 16000)
    frame_samples: int = _int("FRAME_SAMPLES", 512)
    output_chunk_ms: int = _int("OUTPUT_CHUNK_MS", 100)
    vad_threshold: float = _float("VAD_THRESHOLD", 0.6)
    endpoint_silence_ms: int = _int("ENDPOINT_SILENCE_MS", 600)
    pre_roll_ms: int = _int("PRE_ROLL_MS", 300)
    min_speech_ms: int = _int("MIN_SPEECH_MS", 250)
    max_utterance_ms: int = _int("MAX_UTTERANCE_MS", 30000)
