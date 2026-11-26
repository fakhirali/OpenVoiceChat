#!/usr/bin/env python3
"""Render a JSONL call log as a self-contained HTML report.

Capture a log from the local loop:

    python main.py --log calls/latest.jsonl

Then:

    python scripts/visualize_call.py calls/latest.jsonl

The report has aggregate numbers, a per-turn latency breakdown, and a
timeline of user speech, pipeline stages, and agent audio. It is one HTML
file with inline CSS and SVG, so it works offline and can be shared as-is.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import webbrowser
from datetime import datetime
from pathlib import Path

SPEECH_RATE = 16000

COLORS = {
    "user": "#3b82f6",
    "endpoint": "#f59e0b",
    "stt": "#6366f1",
    "llm": "#8b5cf6",
    "wait": "#cbd5e1",
    "tts": "#10b981",
    "agent": "#14b8a6",
    "barge": "#ef4444",
    "ink": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
}


# --------------------------------------------------------------------------
# Data


def load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number}: not JSON ({exc.msg})") from None
        if isinstance(event, dict) and "event" in event and "t" in event:
            events.append(event)
    if not events:
        raise SystemExit(f"{path}: no call events found")
    return events


def build_turns(events: list[dict]) -> list[dict]:
    turns: list[dict] = []
    pending_speech = None
    current = None
    for event in events:
        name = event["event"]
        if name == "speech_start":
            pending_speech = event
        elif name == "endpoint":
            current = {
                "endpoint": event,
                "speech_start": pending_speech,
                "stt": None,
                "llm": None,
                "tts": [],
                "written": None,
                "done": None,
                "barge_in": None,
            }
            turns.append(current)
            pending_speech = None
        elif current is None:
            continue
        elif name == "stt":
            current["stt"] = event
        elif name == "llm_first_token":
            current["llm"] = event
        elif name == "tts":
            current["tts"].append(event)
        elif name == "first_audio_written":
            current["written"] = event
        elif name == "turn_done":
            current["done"] = event
        elif name == "barge_in":
            current["barge_in"] = event
    return turns


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * weight


def values_of(done: list[dict], key: str) -> list[float]:
    return [
        float(item[key])
        for item in done
        if item.get(key) is not None
    ]


def agent_audio_seconds(turn: dict) -> float:
    cursor = 0.0
    total = 0.0
    barge_at = float(turn["barge_in"]["t"]) if turn["barge_in"] else None
    for sentence in turn["tts"]:
        audio_s = float(sentence.get("audio_ms", 0.0)) / 1000
        if audio_s <= 0:
            continue
        start = max(float(sentence["t"]), cursor)
        cursor = start + audio_s
        end = cursor if barge_at is None else min(cursor, barge_at)
        total += max(0.0, end - start)
        if barge_at is not None and barge_at <= cursor:
            break
    return total


def collect_stats(events: list[dict], turns: list[dict]) -> dict:
    completed = [turn["done"] for turn in turns if turn["done"] and turn["done"].get("completed")]
    v2v = values_of(completed, "voice_to_voice_ms")
    v2v_written = [
        float(turn["written"]["ms"])
        for turn in turns
        if turn["written"] and turn["written"].get("ms") is not None
    ]
    endpoint = values_of(completed, "endpoint_ms")
    stt = values_of(completed, "stt_ms")
    llm = values_of(completed, "llm_ttft_ms")
    first_sentence = values_of(completed, "first_sentence_ms")
    tts = values_of(completed, "tts_first_ms")
    sentence_wait = [
        max(0.0, first - end - transcribe - think)
        for first, end, transcribe, think in zip(
            first_sentence, endpoint, stt, llm, strict=True
        )
    ]

    t0 = min(event["t"] for event in events)
    t1 = max(event["t"] for event in events)
    user_text = " ".join(turn["done"].get("user", "") for turn in turns if turn["done"])
    assistant_text = " ".join(
        turn["done"].get("assistant", "") for turn in turns if turn["done"]
    )

    return {
        "t0": t0,
        "t1": t1,
        "duration_s": max(0.0, t1 - t0),
        "turns": len(turns),
        "completed": len(completed),
        "barge_ins": sum(1 for event in events if event["event"] == "barge_in"),
        "dropped": sum(1 for event in events if event["event"] == "dropped"),
        "v2v_mean": statistics.fmean(v2v) if v2v else None,
        "v2v_median": percentile(v2v, 0.5),
        "v2v_p95": percentile(v2v, 0.95),
        "v2v_min": min(v2v) if v2v else None,
        "v2v_max": max(v2v) if v2v else None,
        "v2v_written": statistics.fmean(v2v_written) if v2v_written else None,
        "endpoint_ms": statistics.fmean(endpoint) if endpoint else None,
        "stt_ms": statistics.fmean(stt) if stt else None,
        "llm_ttft_ms": statistics.fmean(llm) if llm else None,
        "sentence_wait_ms": statistics.fmean(sentence_wait) if sentence_wait else None,
        "tts_first_ms": statistics.fmean(tts) if tts else None,
        "user_speech_s": sum(
            float(turn["endpoint"].get("speech_ms", 0.0)) for turn in turns
        )
        / 1000,
        "agent_speech_s": sum(agent_audio_seconds(turn) for turn in turns),
        "user_words": len(user_text.split()),
        "assistant_words": len(assistant_text.split()),
    }


# --------------------------------------------------------------------------
# Formatting


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    return f"{value:.0f} ms"


def fmt_seconds(value: float) -> str:
    if value >= 60:
        minutes, seconds = divmod(value, 60)
        return f"{int(minutes)}m {seconds:04.1f}s"
    return f"{value:.1f}s"


def esc(value) -> str:
    return html.escape(str(value), quote=True)


# --------------------------------------------------------------------------
# Timeline


def nice_step(raw: float) -> float:
    if raw <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw))
    base = raw / 10**exponent
    for candidate in (1, 2, 5, 10):
        if base <= candidate:
            return candidate * 10**exponent
    return 10 ** (exponent + 1)


def render_timeline(turns: list[dict], stats: dict, width: int = 1180) -> str:
    gutter, pad = 116, 22
    lane_y = {"user": 34, "pipeline": 80, "agent": 126}
    bar_h = 22
    axis_y = 170
    height = 196
    span = max(stats["t1"] - stats["t0"], 0.5)
    scale = (width - gutter - pad) / span
    t0 = stats["t0"]

    def x(t: float) -> float:
        return gutter + (t - t0) * scale

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            'role="img" aria-label="Call timeline">'
        ),
        (
            "<defs><style>"
            ".lane-label{font:600 11px ui-sans-serif,system-ui,sans-serif;fill:#475569}"
            ".block-label{font:600 10px ui-sans-serif,system-ui,sans-serif;fill:#ffffff}"
            ".tick{font:10px ui-monospace,monospace;fill:#94a3b8}"
            ".axis{stroke:#cbd5e1;stroke-width:1}"
            "</style></defs>"
        ),
    ]

    step = nice_step(span / 8)
    tick = 0.0
    while tick <= span + 1e-9:
        tick_x = x(t0 + tick)
        label = f"+{tick:.0f}s" if span >= 8 else f"+{tick:.1f}s"
        parts.append(
            f'<line x1="{tick_x:.1f}" y1="20" x2="{tick_x:.1f}" y2="{axis_y}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text class="tick" x="{tick_x:.1f}" y="{axis_y + 16}" '
            f'text-anchor="middle">{label}</text>'
        )
        tick += step

    for name, label in (
        ("user", "User speech"),
        ("pipeline", "Pipeline"),
        ("agent", "Agent audio"),
    ):
        parts.append(
            f'<text class="lane-label" x="10" y="{lane_y[name] + bar_h / 2 + 4}">{label}</text>'
        )

    def block(
        start: float,
        end: float,
        lane: str,
        color: str,
        label: str,
        tooltip: str,
        *,
        opacity: float = 1.0,
        stroke: str = "none",
    ) -> None:
        left = x(start)
        right = x(end)
        if right - left < 1.5:
            right = left + 1.5
        y = lane_y[lane]
        parts.append(
            f'<g><title>{esc(tooltip)}</title>'
            f'<rect x="{left:.1f}" y="{y}" width="{right - left:.1f}" height="{bar_h}" '
            f'rx="4" fill="{color}" fill-opacity="{opacity}" stroke="{stroke}"/>'
        )
        if right - left >= 30:
            parts.append(
                f'<text class="block-label" x="{(left + right) / 2:.1f}" '
                f'y="{y + 15}" text-anchor="middle">{esc(label)}</text>'
            )
        parts.append("</g>")

    for index, turn in enumerate(turns, 1):
        endpoint = turn["endpoint"]
        speech_end = endpoint["t"] - float(endpoint.get("silence_ms", 0.0)) / 1000
        speech_start = (
            turn["speech_start"]["t"]
            if turn["speech_start"]
            else speech_end - float(endpoint.get("speech_ms", 0.0)) / 1000
        )

        parts.append(
            f'<line x1="{x(endpoint["t"]):.1f}" y1="18" x2="{x(endpoint["t"]):.1f}" '
            f'y2="{axis_y}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3 4"/>'
        )

        block(
            speech_start,
            speech_end,
            "user",
            COLORS["user"],
            f"{speech_end - speech_start:.1f}s",
            f"Turn {index} — user spoke {speech_end - speech_start:.2f}s",
        )

        block(
            speech_end,
            endpoint["t"],
            "pipeline",
            COLORS["endpoint"],
            "endpoint",
            f"Turn {index} — endpoint wait "
            f"{float(endpoint.get('silence_ms', 0.0)):.0f}ms",
        )

        stt = turn["stt"]
        if stt:
            block(
                stt["t"] - float(stt.get("ms", 0.0)) / 1000,
                stt["t"],
                "pipeline",
                COLORS["stt"],
                "STT",
                f"Turn {index} — STT {float(stt.get('ms', 0)):.0f}ms: {stt.get('text', '')}",
            )

        llm = turn["llm"]
        if llm and stt:
            block(
                stt["t"],
                llm["t"],
                "pipeline",
                COLORS["llm"],
                "LLM",
                f"Turn {index} — first token {float(llm.get('ms', 0)):.0f}ms",
            )

        first_tts = turn["tts"][0] if turn["tts"] else None
        if llm and first_tts:
            synth_start = first_tts["t"] - float(first_tts.get("ms", 0.0)) / 1000
            if synth_start > llm["t"]:
                block(
                    llm["t"],
                    synth_start,
                    "pipeline",
                    COLORS["wait"],
                    "wait",
                    f"Turn {index} — waiting for a complete sentence",
                )

        for sentence in turn["tts"]:
            block(
                sentence["t"] - float(sentence.get("ms", 0.0)) / 1000,
                sentence["t"],
                "pipeline",
                COLORS["tts"],
                "TTS",
                f"Turn {index} — TTS {float(sentence.get('ms', 0)):.0f}ms: {sentence.get('text', '')}",
            )

        cursor = 0.0
        barge_at = float(turn["barge_in"]["t"]) if turn["barge_in"] else None
        for sentence in turn["tts"]:
            audio_s = float(sentence.get("audio_ms", 0.0)) / 1000
            if audio_s <= 0:
                continue
            start = max(float(sentence["t"]), cursor)
            cursor = start + audio_s
            end = cursor if barge_at is None else min(cursor, barge_at)
            if end <= start:
                break
            block(
                start,
                end,
                "agent",
                COLORS["agent"],
                f"{end - start:.1f}s",
                f"Turn {index} — agent audio ≈{end - start:.2f}s: {sentence.get('text', '')}"
                + (" (cut by barge-in)" if barge_at is not None else ""),
                opacity=0.85 if barge_at is None else 0.55,
            )
            if barge_at is not None and barge_at <= cursor:
                break

        if turn["barge_in"]:
            barge_x = x(turn["barge_in"]["t"])
            parts.append(
                f'<line x1="{barge_x:.1f}" y1="20" x2="{barge_x:.1f}" y2="{axis_y}" '
                f'stroke="{COLORS["barge"]}" stroke-width="2" stroke-dasharray="4 3"/>'
            )
            parts.append(
                f'<text x="{barge_x + 4:.1f}" y="16" fill="{COLORS["barge"]}" '
                'style="font:600 10px ui-sans-serif,system-ui,sans-serif">barge-in</text>'
            )

        if turn["written"]:
            mark_x = x(turn["written"]["t"])
            parts.append(
                f'<g><title>Turn {index} — first audio written '
                f'{float(turn["written"].get("ms", 0)):.0f}ms after last speech</title>'
                f'<path d="M {mark_x:.1f} {lane_y["pipeline"] - 7} l 5 5 l -5 5 l -5 -5 z" '
                f'fill="{COLORS["ink"]}"/></g>'
            )

    parts.append(
        f'<line class="axis" x1="{gutter}" y1="{axis_y}" x2="{width - pad}" y2="{axis_y}"/>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# HTML


def card(label: str, value: str, hint: str = "") -> str:
    note = f'<span class="hint">{esc(hint)}</span>' if hint else ""
    return (
        '<div class="card">'
        f'<span class="label">{esc(label)}</span>'
        f'<strong>{esc(value)}</strong>{note}'
        "</div>"
    )


def render_cards(stats: dict) -> str:
    return "\n".join(
        (
            card("Turns", str(stats["turns"]), f'{stats["completed"]} completed'),
            card("Call length", fmt_seconds(stats["duration_s"])),
            card("Median voice→voice", fmt_ms(stats["v2v_median"])),
            card("p95 voice→voice", fmt_ms(stats["v2v_p95"])),
            card("Barge-ins", str(stats["barge_ins"]), f'{stats["dropped"]} dropped'),
            card(
                "Speech",
                f'{stats["user_speech_s"]:.1f}s in',
                f'{stats["agent_speech_s"]:.1f}s out',
            ),
        )
    )


def render_breakdown(stats: dict) -> str:
    rows = [
        ("Endpoint wait", stats["endpoint_ms"]),
        ("STT", stats["stt_ms"]),
        ("LLM first token", stats["llm_ttft_ms"]),
        ("Waiting for sentence", stats["sentence_wait_ms"]),
        ("TTS first sentence", stats["tts_first_ms"]),
        ("Voice→voice (mean)", stats["v2v_mean"]),
    ]
    known = [value for _, value in rows if value is not None]
    maximum = max(known) if known else 1.0
    parts = []
    for label, value in rows:
        width = 0.0 if value is None else 100 * value / maximum
        parts.append(
            '<div class="bar-row">'
            f'<span class="bar-label">{esc(label)}</span>'
            f'<div class="bar"><i style="width:{width:.1f}%"></i></div>'
            f'<b>{esc(fmt_ms(value))}</b>'
            "</div>"
        )
    return "\n".join(parts)


def render_turns(turns: list[dict]) -> str:
    parts = []
    for index, turn in enumerate(turns, 1):
        done = turn["done"] or {}
        user = done.get("user", "")
        assistant = done.get("assistant", "")
        status = "completed" if done.get("completed") else "interrupted"
        if turn["barge_in"] and not done.get("completed"):
            status = "barge-in"
        summary = (
            f'<summary><b>Turn {index}</b>'
            f'<span class="pill">{esc(status)}</span>'
            f'<span class="mono">{esc(fmt_ms(done.get("voice_to_voice_ms")))}</span>'
            f'<span class="quote">{esc(user[:90] or "(no transcript)")}</span></summary>'
        )
        metrics = [
            ("Endpoint", fmt_ms(done.get("endpoint_ms"))),
            ("STT", fmt_ms(done.get("stt_ms"))),
            ("LLM first token", fmt_ms(done.get("llm_ttft_ms"))),
            ("First sentence", fmt_ms(done.get("first_sentence_ms"))),
            ("First TTS", fmt_ms(done.get("tts_first_ms"))),
            ("Voice→voice", fmt_ms(done.get("voice_to_voice_ms"))),
        ]
        if turn["written"]:
            metrics.append(("Audio written", fmt_ms(turn["written"].get("ms"))))
        grid = "".join(
            f'<div><span>{esc(label)}</span><b>{esc(value)}</b></div>'
            for label, value in metrics
        )
        parts.append(
            f'<details class="turn">{summary}'
            f'<div class="turn-body"><div class="metrics">{grid}</div>'
            f'<p class="utterance"><span>User</span>{esc(user) or "—"}</p>'
            f'<p class="utterance"><span>Agent</span>{esc(assistant) or "—"}</p>'
            "</div></details>"
        )
    return "\n".join(parts)


def render_html(
    *,
    source: Path,
    turns: list[dict],
    stats: dict,
    timeline: str,
    breakdown: str,
    cards: str,
    turn_rows: str,
) -> str:
    started = datetime.fromtimestamp(stats["t0"]).strftime("%Y-%m-%d %H:%M:%S")
    words = (
        f'{stats["user_words"]} user words → {stats["assistant_words"]} agent words'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Call timeline · {esc(source.name)}</title>
<style>
:root {{
  color-scheme: light;
  --ink: {COLORS["ink"]};
  --muted: {COLORS["muted"]};
  --line: #e2e8f0;
  --bg: #f8fafc;
  --card: #ffffff;
  --accent: #2563eb;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 32px 28px 64px;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}}
main {{ max-width: 1220px; margin: 0 auto; }}
header h1 {{ margin: 0 0 4px; font-size: 24px; letter-spacing: -0.02em; }}
header .sub {{ margin: 0; color: var(--muted); font-size: 13px; }}
header .sub b {{ color: var(--ink); font-weight: 600; }}
h2 {{ margin: 34px 0 12px; font-size: 15px; letter-spacing: 0.01em; }}
.cards {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-top: 22px; }}
.card {{
  background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 14px; display: flex; flex-direction: column; gap: 2px;
}}
.card .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }}
.card strong {{ font-size: 20px; font-weight: 650; letter-spacing: -0.01em; }}
.card .hint {{ color: var(--muted); font-size: 11px; }}
.panel {{
  background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px; overflow-x: auto;
}}
svg {{ display: block; width: 100%; height: auto; min-width: 860px; }}
.bar-row {{ display: grid; grid-template-columns: 170px 1fr 76px; gap: 12px; align-items: center; padding: 5px 0; }}
.bar-label {{ color: var(--muted); }}
.bar {{ height: 8px; background: #f1f5f9; border-radius: 99px; overflow: hidden; }}
.bar i {{ display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg, #60a5fa, #2563eb); }}
.bar-row b {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }}
.turn {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; margin-bottom: 8px; }}
.turn summary {{
  list-style: none; cursor: pointer; display: flex; gap: 12px; align-items: center;
  padding: 10px 14px;
}}
.turn summary::-webkit-details-marker {{ display: none; }}
.turn summary::before {{ content: "▸"; color: var(--muted); }}
.turn[open] summary::before {{ content: "▾"; }}
.turn summary .pill {{
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em;
  background: #f1f5f9; color: #475569; border-radius: 99px; padding: 2px 8px;
}}
.turn summary .mono {{ font: 600 12px ui-monospace, monospace; color: var(--accent); }}
.turn summary .quote {{ color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.turn-body {{ padding: 0 14px 14px; }}
.metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 14px; }}
.metrics div {{ background: var(--bg); border-radius: 7px; padding: 7px 9px; }}
.metrics span {{ display: block; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em; }}
.metrics b {{ font-variant-numeric: tabular-nums; }}
.utterance {{ margin: 8px 0; }}
.utterance span {{ display: inline-block; width: 54px; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin: 10px 0 0; color: var(--muted); font-size: 12px; }}
.legend i {{ display: inline-block; width: 9px; height: 9px; border-radius: 3px; margin-right: 5px; vertical-align: 0; }}
@media (max-width: 900px) {{
  body {{ padding: 22px 14px 48px; }}
  .cards {{ grid-template-columns: repeat(2, 1fr); }}
  .metrics {{ grid-template-columns: repeat(2, 1fr); }}
}}
</style>
</head>
<body>
<main>
<header>
  <h1>Call timeline</h1>
  <p class="sub">
    <b>{esc(source.name)}</b> · started {esc(started)} · {esc(fmt_seconds(stats["duration_s"]))} ·
    {esc(words)}
  </p>
</header>

<section class="cards">
{cards}
</section>

<section>
  <h2>Timeline</h2>
  <div class="panel">
    {timeline}
    <div class="legend">
      <span><i style="background:{COLORS["user"]}"></i>user speech</span>
      <span><i style="background:{COLORS["endpoint"]}"></i>endpoint wait</span>
      <span><i style="background:{COLORS["stt"]}"></i>STT</span>
      <span><i style="background:{COLORS["llm"]}"></i>LLM first token</span>
      <span><i style="background:{COLORS["wait"]}"></i>sentence wait</span>
      <span><i style="background:{COLORS["tts"]}"></i>TTS</span>
      <span><i style="background:{COLORS["agent"]}"></i>agent audio (estimated)</span>
      <span><i style="background:{COLORS["barge"]}"></i>barge-in</span>
    </div>
  </div>
</section>

<section>
  <h2>Where the time goes</h2>
  <div class="panel">
    {breakdown}
  </div>
</section>

<section>
  <h2>Turns</h2>
  {turn_rows}
</section>
</main>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a JSONL call log from openvoicechat as an HTML report."
    )
    parser.add_argument("log", type=Path, help="JSONL file written by main.py --log")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="HTML output path (default: the log path with .html)",
    )
    parser.add_argument(
        "--open", action="store_true", help="open the report in a browser"
    )
    args = parser.parse_args(argv)

    events = load_events(args.log)
    turns = build_turns(events)
    if not turns:
        raise SystemExit(f"{args.log}: no completed turns found")
    stats = collect_stats(events, turns)

    output = args.output or args.log.with_suffix(".html")
    output.write_text(
        render_html(
            source=args.log,
            turns=turns,
            stats=stats,
            timeline=render_timeline(turns, stats),
            breakdown=render_breakdown(stats),
            cards=render_cards(stats),
            turn_rows=render_turns(turns),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output} ({stats['turns']} turns)")
    if args.open:
        webbrowser.open(output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
