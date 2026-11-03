# OpenVoiceChat

**The local voice runtime for Python.** Parakeet STT, Kokoro TTS, Silero VAD —
one GPU, many calls, no cloud voice APIs, no provider zoo, no platform.

> Status: **0.4.0, alpha.** The local pipeline, mic loop, and `ovc doctor` work.
> The shared multi-call runtime, Twilio/Daily transports, and `ovc init/run/bench`
> are the next steps. See [Roadmap](#roadmap).

## What it is

A small library for embedding an interruptible voice agent in a Python process:

- **STT:** NVIDIA Parakeet TDT 0.6B v3 via `onnx-asr` (ONNX Runtime)
- **TTS:** Kokoro-82M
- **VAD / endpointing:** Silero VAD v5
- **LLM:** any OpenAI-compatible API; `LLM_BASE_URL` points at OpenAI, Ollama,
  llama.cpp, vLLM, or your own gateway

It is not a platform, a model zoo, or a cloud speech wrapper. There is no
dashboard, auth service, database, or provider registry.

## Architecture

```text
microphone ──► resample to 16 kHz ──► Conversation.input
                                           │
                                    Silero VAD + endpointing
                                           │ utterance PCM16
                                           ▼
                                   Parakeet (local STT)
                                           │ text
                                           ▼
                              OpenAI-compatible LLM (stream)
                                           │ sentences
                                           ▼
                                    Kokoro (local TTS)
                                           │ 16 kHz PCM16
                                           ▼
speaker ◄── resample to device rate ◄── Conversation.output
```

`Conversation` is the whole engine: endpointing, barge-in, sentence-level TTS,
and history. Transports only move PCM16 in and out.

## Quickstart

Python 3.11+ is required.

```bash
pip install -e ".[local,models]"
cp .env.example .env
python main.py --verbose
```

The first run downloads Parakeet (~2 GB), Kokoro-82M (~330 MB), and Silero VAD
(~2 MB). `--verbose` prints speech start/end, endpointing, STT, LLM first-token,
and TTS timings. `Ctrl-C` stops it.

On Linux you may need PortAudio: `sudo apt install libportaudio2`.

`--prompt`, `--input-device`, and `--output-device` are available. Audio is
resampled between the device rate and 16 kHz internally. **Use headphones** —
there is no echo cancellation, and speaker output will feed back into the VAD.

## Models and hardware

| Component | Model | Notes |
|---|---|---|
| STT | `nvidia/parakeet-tdt-0.6b-v3` | ONNX Runtime; CUDA where available, CPU otherwise |
| TTS | `hexgrad/Kokoro-82M` | 24 kHz, resampled to 16 kHz for the pipeline |
| VAD | Silero VAD v5 | Always CPU, ~0.4% of one core |

On Linux x86_64 and Windows AMD64 the core dependency is `onnxruntime-gpu`, so
Parakeet uses CUDA when present and falls back to CPU. macOS and non-x86 Linux
use the CPU `onnxruntime` build. Kokoro uses CUDA when `torch` reports it.

Measured on one RTX 3080 Ti (12 GB), Python 3.12:

| Metric | Value |
|---|---|
| Model load (Parakeet + Kokoro) | ~6.9 s |
| Kokoro warm RTF | ~0.01 |
| Parakeet steady transcription | ~0.019 s per utterance |
| Peak VRAM | ~5.2 GB |

These are single-session numbers. Concurrent-call benchmarks are not published
yet; that measurement is part of the next milestone.

Change the voice with `LOCAL_TTS_VOICE` (default `af_heart`, see the
[Kokoro voice list](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)).

### Startup warnings

Startup prints one ONNX Runtime CUDA memcpy notice and three torch deprecation
warnings from Kokoro. They are expected and do not affect output:

- Kokoro runs in `eval()` mode and its dropout calls pass `training=False`.
- The dropout warning is PyTorch noting an unused `dropout` argument on a
  single-layer LSTM, where dropout is ignored by definition.
- The other two are genuine deprecations inside Kokoro's model code; they will
  disappear when Kokoro updates.

They print once per process. The library does not hide them.

## LLM

The LLM is any OpenAI-compatible API. The default is OpenAI; set `LLM_BASE_URL`
to point somewhere else. The voice stack stays local either way.

```dotenv
LLM_API_KEY=...
LLM_MODEL=gpt-4.1-nano
# LLM_BASE_URL=        # empty means https://api.openai.com/v1
```

Local servers use the same shape:

```bash
ollama pull qwen3:4b
```

```dotenv
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:4b
LLM_API_KEY=local
```

llama.cpp (`http://localhost:8080/v1`) and vLLM (`http://localhost:8000/v1`)
work the same way. `LLM_TIMEOUT` defaults to 15 seconds with one retry, so a
stalled model cannot freeze a turn.

## Check your machine

```bash
ovc doctor
ovc doctor --json
```

`doctor` checks Python, `torch`/CUDA, ONNX Runtime providers, Parakeet and Kokoro
imports, audio devices, the Silero cache, and LLM configuration. It exits `0`
when ready and `2` when something required is missing. `--json` emits a stable
schema for agents:

```json
{
  "schema_version": 1,
  "ok": true,
  "checks": [{"name": "onnxruntime", "status": "ok", "detail": "1.31.1 | CUDAExecutionProvider"}]
}
```

## Turn taking

| Variable | Default | Effect |
|---|---:|---|
| `ENDPOINT_SILENCE_MS` | `600` | Silence before an utterance is sent to STT |
| `PRE_ROLL_MS` | `300` | Audio kept before speech onset |
| `VAD_THRESHOLD` | `0.6` | Silero speech probability threshold |
| `MIN_SPEECH_MS` | `250` | Utterances shorter than this are dropped |
| `MAX_UTTERANCE_MS` | `30000` | Rolling window for one uninterrupted utterance |
| `OUTPUT_CHUNK_MS` | `100` | TTS chunk size written to the transport |

`MAX_UTTERANCE_MS` does not cut a speaker off mid-sentence. Past the cap the
engine keeps the most recent window and waits for silence, so a long monologue
costs bounded memory instead of cancelling its own turn.

## Daily (optional)

The Daily transport lives in `openvoicechat/daily_agent/`. It is a thin call client,
but it still starts **one process per meeting**, so each meeting loads its own
models. Moving it onto a shared runtime is the next milestone.

```bash
pip install -e ".[daily,models]"
python scripts/daily_test.py --open
```

The script creates a room, starts the agent server, joins the agent, tails the
transcript, and cleans up on `Ctrl-C`.

Server:

```bash
uvicorn openvoicechat.daily_agent.server:app --host 0.0.0.0 --port 8001
```

`GET /health` is public; all other routes require
`Authorization: Bearer $AGENT_API_TOKEN`. `POST /join_meeting` takes
`meeting_url`, `prompt`, and `conversation_id`.

## Tests

```bash
pytest -q
ruff check .
```

Tests use fake providers and no network or GPU.

## Roadmap

| Milestone | Content |
|---|---|
| **0.4** | Local-only core, `ovc doctor`, P1 turn-taking fixes |
| **0.5** | `VoiceRuntime`: load models once, serve many sessions; `ovc init/run/bench` |
| **0.6** | Twilio Media Streams transport and scaffold |
| **0.7** | Shared-runtime Daily transport, `ovc eval`, concurrency benchmarks |
| **1.0** | Stable API, published comparisons, extension docs |

Full browser STT + TTS is a separate later package, not part of the Python core.

## License and attribution

Code is Apache-2.0. Model licenses are upstream: Parakeet is CC-BY-4.0, Kokoro is
Apache-2.0, and Silero VAD is MIT. Check each model card before redistribution.
