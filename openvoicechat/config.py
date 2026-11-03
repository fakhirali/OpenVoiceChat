"""Configuration.

Only secrets are read from the environment (or a local ``.env``). Everything
else is a plain dataclass field, set in code or overridden per instance:

    Config(endpoint_silence_ms=400, llm_model="qwen3:4b")
"""

import os
from dataclasses import dataclass, field

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

DEFAULT_LLM_BASE_URL = ""  # empty means https://api.openai.com/v1
DEFAULT_LLM_MODEL = "gpt-4.1-nano"


@dataclass(frozen=True)
class Config:
    # Secrets: environment only.
    llm_api_key: str = field(
        default_factory=lambda: (
            os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        )
    )
    agent_api_token: str = field(
        default_factory=lambda: os.getenv("AGENT_API_TOKEN", "")
    )

    # LLM endpoint.
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model: str = DEFAULT_LLM_MODEL
    llm_timeout_s: float = 15.0

    # Runtime and turn taking.
    max_sessions: int = 5
    data_dir: str = "data"
    sample_rate: int = 16000
    frame_samples: int = 512
    output_chunk_ms: int = 100
    vad_threshold: float = 0.6
    endpoint_silence_ms: int = 600
    pre_roll_ms: int = 300
    min_speech_ms: int = 250
    max_utterance_ms: int = 30000
