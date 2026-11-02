"""Tests for the hosted STT/TTS client."""

import json

import httpx
import pytest

from openvoicechat.config import Config
from openvoicechat.speech import Speech


def make_config() -> Config:
    return Config(
        deepgram_api_key="d",
        elevenlabs_api_key="e",
        elevenlabs_voice_id="v",
    )


async def test_transcribe_request_and_response():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "results": {
                    "channels": [
                        {"alternatives": [{"transcript": "hello world", "confidence": 0.99}]}
                    ]
                }
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    speech = Speech(make_config(), client=client)
    result = await speech.transcribe(b"\x01\x02\x03\x04")
    await client.aclose()

    assert result == "hello world"
    request = captured[0]
    assert request.url.scheme == "https"
    assert request.url.host == "api.deepgram.com"
    assert request.url.path == "/v1/listen"
    params = request.url.params
    assert params["encoding"] == "linear16"
    assert params["sample_rate"] == "16000"
    assert params["channels"] == "1"
    assert params["model"] == "nova-3"
    assert params["smart_format"] == "true"
    assert params["punctuate"] == "true"
    assert request.headers["authorization"] == "Token d"
    assert request.headers["content-type"] == "audio/raw"
    assert request.content == b"\x01\x02\x03\x04"


async def test_transcribe_missing_transcript_is_empty():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    )
    speech = Speech(make_config(), client=client)
    assert await speech.transcribe(b"\x00\x00") == ""
    await client.aclose()


async def test_synthesize_request_and_pcm():
    captured: list[httpx.Request] = []
    payload = b"\x01\x02" * 10 + b"\x03"

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, content=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    speech = Speech(make_config(), client=client)
    result = await speech.synthesize("Hi there.")
    await client.aclose()

    assert result == payload[:-1]
    request = captured[0]
    assert request.url.scheme == "https"
    assert request.url.host == "api.elevenlabs.io"
    assert request.url.path == "/v1/text-to-speech/v"
    assert request.url.params["output_format"] == "pcm_16000"
    assert request.headers["xi-api-key"] == "e"
    body = json.loads(request.content)
    assert body["text"] == "Hi there."
    assert body["model_id"] == "eleven_v4_turbo"
    assert body["voice_settings"] == {"stability": 0.5, "similarity_boost": 0.75}


async def test_empty_text_does_not_call_elevenlabs():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    speech = Speech(make_config(), client=client)
    assert await speech.synthesize("   ") == b""
    await client.aclose()


async def test_http_error_is_clear():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, text="upstream unavailable")
        )
    )
    speech = Speech(make_config(), client=client)
    with pytest.raises(RuntimeError, match=r"Deepgram.*503.*upstream unavailable"):
        await speech.transcribe(b"\x00\x00")
    await client.aclose()


def test_missing_credentials():
    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
        Speech(Config(deepgram_api_key="", elevenlabs_api_key="", elevenlabs_voice_id=""))
