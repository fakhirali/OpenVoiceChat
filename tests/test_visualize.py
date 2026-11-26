"""End-to-end test for scripts/visualize_call.py."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "visualize_call.py"


def sample_events() -> list[dict]:
    base = 1_700_000_000.0
    return [
        {"event": "call_start", "t": base, "sample_rate": 16000},
        {"event": "speech_start", "t": base + 1.0, "pre_roll_ms": 300},
        {
            "event": "endpoint",
            "t": base + 2.0,
            "reason": "silence",
            "silence_ms": 400,
            "speech_ms": 800,
            "utterance_ms": 1200,
        },
        {"event": "stt", "t": base + 2.1, "ms": 80, "text": "hello there"},
        {"event": "llm_first_token", "t": base + 3.2, "ms": 1100},
        {
            "event": "tts",
            "t": base + 3.4,
            "ms": 200,
            "bytes": 32000,
            "audio_ms": 1000,
            "text": "Hi.",
        },
        {"event": "first_audio_written", "t": base + 3.45, "ms": 1450},
        {
            "event": "turn_done",
            "t": base + 4.2,
            "user": "hello there",
            "assistant": "Hi.",
            "completed": True,
            "voice_to_voice_ms": 1450,
            "endpoint_ms": 400,
            "stt_ms": 80,
            "llm_ttft_ms": 1100,
            "first_sentence_ms": 1400,
            "tts_first_ms": 200,
            "turn_ms": 2200,
        },
        {"event": "call_end", "t": base + 4.5},
    ]


def test_visualize_call_writes_html(tmp_path: Path) -> None:
    log = tmp_path / "call.jsonl"
    log.write_text(
        "\n".join(json.dumps(event) for event in sample_events()) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "call.html"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(log), "-o", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    html = output.read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert "Call timeline" in html
    assert "<svg" in html
    assert "hello there" in html
    assert "Hi." in html
    assert "1.45 s" in html
    assert "None" not in html
