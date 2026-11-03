# OpenVoiceChat

A tiny async voice-conversation engine and a Daily.co agent service.

It listens to meeting audio, detects speech with Silero VAD, transcribes with
Deepgram, streams an OpenAI-compatible LLM response, and speaks with ElevenLabs.
One process joins the meeting as an agent; a second records the meeting to WAV.

There is no web platform, WebRTC demo, or tool calling in this repository.

## Architecture

```
Daily audio ──► Conversation.input ──► Silero VAD / endpointing
                                            │ utterance PCM16
                                            ▼
                                   Deepgram REST (STT)
                                            │ text
                                            ▼
                              OpenAI-compatible LLM (stream)
                                            │ sentences
                                            ▼
                                  ElevenLabs REST (TTS)
                                            │ 16 kHz PCM16 chunks
                                            ▼
Daily audio ◄── Conversation.output ◄────────┘
```

`agent/worker.py` runs the conversation in a meeting. `agent/recorder.py` records
the meeting independently. `agent/server.py` starts both, one pair per meeting,
and exposes downloads for the recording and transcript.

## Install

Python 3.11+ is required.

```bash
pip install -e ".[agent]"
```

For development:

```bash
pip install -e ".[dev]"
```

## Local loop (no Daily)

To test the whole loop with your microphone and speakers:

```bash
pip install -e ".[local]"
python main.py
```

On Linux you may need PortAudio: `sudo apt install libportaudio2`. Use
`--prompt`, `--input-device`, `--output-device`, and `--verbose` as needed.
`--verbose` prints speech start/end, endpointing, STT, LLM first-token, and TTS
timings. `Ctrl-C` stops it. Audio is resampled between the device rate and 16 kHz
internally.

## Local models (Parakeet + Kokoro-82M)

```bash
pip install -e ".[local,models]"
python main_local.py --verbose
```

`main.py --local` is equivalent. The first run downloads Parakeet (~2 GB) and
Kokoro-82M (~330 MB). Kokoro runs on the GPU when available. Parakeet uses ONNX
Runtime's CUDA provider when `onnxruntime-gpu` is installed, and otherwise falls
back to CPU. Default voice is `af_heart`; set `LOCAL_TTS_VOICE` to another
[Kokoro voice](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)
to change it.

Parakeet TDT 0.6B v3 runs through `onnx-asr` (ONNX Runtime). Kokoro uses
`espeak-ng` for out-of-dictionary words via the bundled `espeakng-loader`
package, so no system package is required. The LLM still uses the
OpenAI-compatible endpoint configured in `.env`, so `LLM_API_KEY` is required;
only STT and TTS are local.

### GPU Parakeet (optional)

The default `onnxruntime` dependency is the CPU build. To run Parakeet on an
NVIDIA GPU, replace it with `onnxruntime-gpu` in the same environment:

```bash
uv pip uninstall onnxruntime
uv pip install "onnxruntime-gpu"
```

Both packages install the same `onnxruntime` Python module, so the CPU package
must be removed first. Choose the wheel that matches your CUDA major version:
`onnxruntime-gpu` 1.27 and newer target CUDA 13.0 with cuDNN 9, while 1.26
targets CUDA 12.8. PyTorch 2.4+ already ships the CUDA and cuDNN libraries, so
no system CUDA toolkit is needed as long as `torch` is imported before the
session is created (the local speech loop does this). Without PyTorch, install
`onnxruntime-gpu[cuda,cudnn]` to pull the NVIDIA runtime libraries from PyPI.

`LocalSpeech` checks `onnxruntime.get_available_providers()` and uses
`CUDAExecutionProvider` when it is present, otherwise `CPUExecutionProvider`;
there is nothing to configure. Reinstalling the project or running `uv sync`
will restore the CPU `onnxruntime` dependency, so repeat the swap afterward.
This is optional: the hosted path only uses CPU ONNX Runtime for Silero VAD.

## Configure

```bash
cp .env.example .env
```

Fill in `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`,
`LLM_API_KEY`, and set `AGENT_API_TOKEN` to a secret. Any OpenAI-compatible API
works; for example Cerebras:

```dotenv
LLM_BASE_URL=https://api.cerebras.ai/v1
LLM_MODEL=llama3.1-8b
```

The Silero VAD ONNX model is downloaded once to
`~/.cache/openvoicechat/silero_vad.onnx`.

## One-command Daily test

This creates a public Daily room, starts the agent server, joins the agent, tails
the transcript, and cleans up the room/server when you press `Ctrl-C`:

```bash
python scripts/daily_test.py --open
```

Options: `--prompt`, `--conversation-id`, `--port`, `--keep-room`, and `--open`
(open the room in your browser). The script checks that the required keys are in
`.env` and prints the transcript, agent log, and recording paths.

## Run the agent server

```bash
uvicorn openvoicechat.agent.server:app --host 0.0.0.0 --port 8001
```

or:

```bash
python -m openvoicechat.agent.server
```

`GET /health` is public. All other routes require
`Authorization: Bearer $AGENT_API_TOKEN`.

### Join a meeting

```bash
curl -X POST http://localhost:8001/join_meeting \
  -H "Authorization: Bearer $AGENT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "meeting_url": "https://your-domain.daily.co/room",
    "prompt": "You are a concise, friendly voice assistant.",
    "conversation_id": "demo-1"
  }'
```

The response includes the agent and recorder PIDs and the recording path.
Transcripts are written to `data/logs/demo-1.log`; process output goes to
`data/logs/demo-1.agent.log` and `data/logs/demo-1.recorder.log`.

### Download results

```bash
curl -H "Authorization: Bearer $AGENT_API_TOKEN" \
  "http://localhost:8001/download_recording?conversation_id=demo-1" -o demo-1.wav

curl -H "Authorization: Bearer $AGENT_API_TOKEN" \
  "http://localhost:8001/download_logs?conversation_id=demo-1" -o demo-1.log
```

## Run a worker directly

```bash
python -m openvoicechat.agent.worker <meeting_url> "<prompt>" <conversation_id>
python -m openvoicechat.agent.recorder <meeting_url> --output data/recordings/meeting.wav
```

## Latency knobs

| Variable | Default | Effect |
| --- | --- | --- |
| `ENDPOINT_SILENCE_MS` | `600` | Silence before a user turn is sent to STT |
| `PRE_ROLL_MS` | `300` | Audio kept before speech onset |
| `VAD_THRESHOLD` | `0.6` | Silero speech probability threshold |
| `MIN_SPEECH_MS` | `250` | Utterances shorter than this are dropped |
| `MAX_UTTERANCE_MS` | `30000` | Hard cap on one utterance |
| `OUTPUT_CHUNK_MS` | `100` | TTS chunk size written to Daily |

`DEEPGRAM_MODEL`, `ELEVENLABS_MODEL`, and `LLM_MODEL` also affect latency;
`eleven_v4_turbo` (or the cheaper `eleven_flash_v2_5`) and small LLMs give the
quickest responses.

## Tests

```bash
pytest -q
ruff check .
```
