"""Create a Daily room, start the agent server, and join one test conversation."""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import signal
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DAILY_API = "https://api.daily.co/v1"


def _handle_stop(signum, frame):
    raise KeyboardInterrupt


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set in .env")
    return value


def _create_room(client: httpx.Client, key: str) -> dict:
    response = client.post(
        f"{DAILY_API}/rooms",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "privacy": "public",
            "properties": {
                "exp": int(time.time()) + 3600,
                "eject_at_room_exp": True,
            },
        },
    )
    if response.is_error:
        raise SystemExit(f"Daily room creation failed: {response.status_code} {response.text[:300]}")
    return response.json()


def _delete_room(client: httpx.Client, key: str, name: str) -> None:
    try:
        client.delete(f"{DAILY_API}/rooms/{name}", headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError:
        pass


def _start_server(port: int) -> tuple[subprocess.Popen, object, Path]:
    log_path = ROOT / "data" / "logs" / "daily_test_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "openvoicechat.agent.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_file, log_path


def _wait_for_server(client: httpx.Client, port: int, server: subprocess.Popen, log_path: Path) -> None:
    deadline = time.time() + 20
    while time.time() < deadline:
        if server.poll() is not None:
            raise SystemExit(f"Agent server exited early. See {log_path}")
        try:
            response = client.get(f"http://127.0.0.1:{port}/health", timeout=1)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    raise SystemExit(f"Agent server did not become healthy. See {log_path}")


def _tail(path: Path, offset: int) -> int:
    if not path.exists():
        return offset
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        chunk = handle.read()
        offset = handle.tell()
    if chunk:
        print(chunk, end="", flush=True)
    return offset


def _stop(server: subprocess.Popen) -> None:
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(server.pid, signal.SIGTERM)
    elif server.poll() is None:
        server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGKILL)
        else:
            server.kill()
        server.wait(timeout=5)
    if os.name == "posix":
        # The server may exit while its worker/recorder children are still alive.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(server.pid, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser(description="One-command Daily room + agent test")
    parser.add_argument(
        "--prompt",
        default="You are a concise, friendly voice assistant.",
        help="system prompt for the agent",
    )
    parser.add_argument(
        "--conversation-id",
        default=f"test-{datetime.now():%Y%m%d-%H%M%S}",
        help="conversation id (letters, digits, dash, underscore)",
    )
    parser.add_argument("--port", type=int, default=8001, help="agent server port")
    parser.add_argument("--open", action="store_true", help="open the room URL in a browser")
    parser.add_argument("--keep-room", action="store_true", help="do not delete the room on exit")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.conversation_id):
        raise SystemExit("conversation-id must match [A-Za-z0-9_-]{1,64}")

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, _handle_stop)

    try:
        import uvicorn  # noqa: F401
    except ImportError as exc:
        raise SystemExit('Install the agent extras: pip install -e ".[agent]"') from exc

    load_dotenv(ROOT / ".env")
    _required("DAILY_API_KEY")
    _required("AGENT_API_TOKEN")
    _required("DEEPGRAM_API_KEY")
    _required("ELEVENLABS_API_KEY")
    _required("ELEVENLABS_VOICE_ID")
    _required("LLM_API_KEY")

    token = os.environ["AGENT_API_TOKEN"]
    daily_key = os.environ["DAILY_API_KEY"]
    server = None
    log_file = None
    room = None

    with httpx.Client(timeout=15) as client:
        try:
            print("Creating Daily room...", flush=True)
            room = _create_room(client, daily_key)
            room_url = room["url"]
            print(f"Room: {room_url}")

            print("Starting agent server...", flush=True)
            server, log_file, server_log = _start_server(args.port)
            _wait_for_server(client, args.port, server, server_log)
            print(f"Server log: {server_log}")

            response = client.post(
                f"http://127.0.0.1:{args.port}/join_meeting",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "meeting_url": room_url,
                    "prompt": args.prompt,
                    "conversation_id": args.conversation_id,
                },
            )
            if response.is_error:
                raise SystemExit(
                    f"join_meeting failed: {response.status_code} {response.text[:300]}"
                )

            print(f"Conversation: {args.conversation_id}")
            print(f"Transcript: {ROOT / 'data' / 'logs' / (args.conversation_id + '.log')}")
            print(f"Agent log:  {ROOT / 'data' / 'logs' / (args.conversation_id + '.agent.log')}")
            print(f"Recording:  {ROOT / 'data' / 'recordings' / (args.conversation_id + '.wav')}")
            print()
            print("Join the room and talk. Ctrl-C to stop.")
            if args.open:
                webbrowser.open(room_url)

            transcript = ROOT / "data" / "logs" / f"{args.conversation_id}.log"
            offset = 0
            joined_at = time.time()
            empty_checks = 0
            while True:
                offset = _tail(transcript, offset)
                try:
                    health = client.get(
                        f"http://127.0.0.1:{args.port}/health", timeout=2
                    ).json()
                    if health.get("active_sessions") == 0 and time.time() - joined_at > 5:
                        empty_checks += 1
                        if empty_checks >= 2:
                            print("\nSession ended.")
                            break
                    else:
                        empty_checks = 0
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping...", flush=True)
        finally:
            print("Cleaning up...", flush=True)
            if room is not None and not args.keep_room:
                _delete_room(client, daily_key, room["name"])
            if server is not None:
                _stop(server)
            if log_file is not None:
                log_file.close()

    recording = ROOT / "data" / "recordings" / f"{args.conversation_id}.wav"
    if recording.exists():
        print(f"Recording saved: {recording}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
