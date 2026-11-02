"""Hosted speech-to-text and text-to-speech over plain HTTP."""

import httpx

from .config import Config


class Speech:
    """Deepgram transcription and ElevenLabs synthesis."""

    def __init__(self, config: Config | None = None, client: httpx.AsyncClient | None = None):
        self.config = config or Config()
        missing = [
            name
            for name, value in (
                ("DEEPGRAM_API_KEY", self.config.deepgram_api_key),
                ("ELEVENLABS_API_KEY", self.config.elevenlabs_api_key),
                ("ELEVENLABS_VOICE_ID", self.config.elevenlabs_voice_id),
            )
            if not value
        ]
        if missing:
            raise ValueError("Missing required configuration: " + ", ".join(missing))
        self._client = client or httpx.AsyncClient(timeout=60.0)
        self._owns_client = client is None

    async def transcribe(self, pcm16: bytes) -> str:
        response = await self._client.post(
            self.config.deepgram_url,
            params={
                "encoding": "linear16",
                "sample_rate": self.config.sample_rate,
                "channels": 1,
                "model": self.config.deepgram_model,
                "smart_format": "true",
                "punctuate": "true",
            },
            headers={
                "Authorization": f"Token {self.config.deepgram_api_key}",
                "Content-Type": "audio/raw",
            },
            content=pcm16,
        )
        _raise_for_status(response, "Deepgram")
        try:
            data = response.json()
            return (
                data.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
                or ""
            )
        except (ValueError, AttributeError, IndexError, TypeError):
            return ""

    async def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""
        response = await self._client.post(
            f"{self.config.elevenlabs_url.rstrip('/')}/{self.config.elevenlabs_voice_id}",
            params={"output_format": "pcm_16000"},
            headers={"xi-api-key": self.config.elevenlabs_api_key},
            json={
                "text": text,
                "model_id": self.config.elevenlabs_model,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
        )
        _raise_for_status(response, "ElevenLabs")
        return response.content[: len(response.content) // 2 * 2]

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _raise_for_status(response: httpx.Response, provider: str) -> None:
    if response.is_error:
        raise RuntimeError(
            f"{provider} request failed with status {response.status_code}: {response.text[:300]}"
        )
