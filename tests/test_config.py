"""Config: secrets from the environment, everything else from code."""

from openvoicechat.config import Config


def test_tuning_is_not_read_from_env(monkeypatch):
    monkeypatch.setenv("ENDPOINT_SILENCE_MS", "123")
    monkeypatch.setenv("VAD_THRESHOLD", "0.1")
    monkeypatch.setenv("LLM_MODEL", "written-in-env")
    monkeypatch.setenv("LLM_BASE_URL", "http://env.invalid/v1")

    config = Config()

    assert config.endpoint_silence_ms == 600
    assert config.vad_threshold == 0.6
    assert config.llm_model == "gpt-6-luna"
    assert config.llm_base_url == ""


def test_code_can_override_any_field():
    config = Config(endpoint_silence_ms=400, llm_model="qwen3:4b")
    assert config.endpoint_silence_ms == 400
    assert config.llm_model == "qwen3:4b"


def test_secrets_are_read_from_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")

    config = Config()

    assert config.llm_api_key == "sk-test"
