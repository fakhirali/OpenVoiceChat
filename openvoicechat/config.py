"""Environment-backed configuration."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


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
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    deepgram_model: str = os.getenv("DEEPGRAM_MODEL", "nova-3")
    deepgram_url: str = os.getenv("DEEPGRAM_URL", "https://api.deepgram.com/v1/listen")
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "")
    elevenlabs_model: str = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
    elevenlabs_url: str = os.getenv(
        "ELEVENLABS_URL", "https://api.elevenlabs.io/v1/text-to-speech"
    )
    llm_api_key: str = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4.1-nano")
    agent_api_token: str = os.getenv("AGENT_API_TOKEN", "")
    max_sessions: int = _int("MAX_SESSIONS", 5)
    data_dir: str = os.getenv("DATA_DIR", "data")
    sample_rate: int = _int("SAMPLE_RATE", 16000)
    frame_samples: int = _int("FRAME_SAMPLES", 512)
    output_chunk_ms: int = _int("OUTPUT_CHUNK_MS", 100)
    vad_threshold: float = _float("VAD_THRESHOLD", 0.6)
    endpoint_silence_ms: int = _int("ENDPOINT_SILENCE_MS", 600)
    pre_roll_ms: int = _int("PRE_ROLL_MS", 300)
    min_speech_ms: int = _int("MIN_SPEECH_MS", 250)
    max_utterance_ms: int = _int("MAX_UTTERANCE_MS", 30000)
