# OpenVoiceChat

A tiny async voice-conversation engine and a Daily.co agent service.

It listens to meeting audio, detects speech with Silero VAD, transcribes with
Deepgram, streams an OpenAI-compatible LLM response, and speaks with ElevenLabs.
One process joins the meeting as an agent; a second records the meeting to WAV.

There is no web platform, local microphone demo, WebRTC demo, tool calling, or
local model runtime in this repository.

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
`--prompt`, `--input-device`, and `--output-device` as needed. `Ctrl-C` stops it.
Audio is resampled between the device rate and 16 kHz internally.

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
