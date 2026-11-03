"""FastAPI service that spawns one agent and one recorder per meeting."""

import asyncio
import contextlib
import os
import re
import secrets
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from openvoicechat.config import Config

config = Config()
root = Path(config.data_dir)
logs_dir = root / "logs"
recordings_dir = root / "recordings"
conversation_id_pattern = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logs_dir.mkdir(parents=True, exist_ok=True)
    recordings_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="OpenVoiceChat Agent", lifespan=lifespan)
bearer = HTTPBearer(auto_error=False)
sessions: dict[str, dict] = {}
sessions_lock = asyncio.Lock()


class JoinMeetingRequest(BaseModel):
    meeting_url: str
    prompt: str
    conversation_id: str


def require_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),  # noqa: B008
) -> None:
    if not config.agent_api_token:
        raise HTTPException(status_code=503, detail="AGENT_API_TOKEN is not configured")
    if credentials is None or not secrets.compare_digest(
        credentials.credentials, config.agent_api_token
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.post("/join_meeting")
async def join_meeting(body: JoinMeetingRequest, _: None = Depends(require_token)):
    _validate_id(body.conversation_id)
    conversation_id = body.conversation_id
    async with sessions_lock:
        if conversation_id in sessions:
            raise HTTPException(status_code=409, detail="Conversation is already active")
        if len(sessions) >= config.max_sessions:
            raise HTTPException(status_code=429, detail="Maximum number of sessions reached")
        sessions[conversation_id] = {}

    recording_path = recordings_dir / f"{conversation_id}.wav"
    agent_log = recorder_log = None
    worker = recorder = None
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        recordings_dir.mkdir(parents=True, exist_ok=True)
        agent_log = open(  # noqa: ASYNC230, SIM115
            logs_dir / f"{conversation_id}.agent.log", "ab"
        )
        recorder_log = open(  # noqa: ASYNC230, SIM115
            logs_dir / f"{conversation_id}.recorder.log", "ab"
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        worker = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "openvoicechat.daily_agent.worker",
            body.meeting_url,
            body.prompt,
            conversation_id,
            stdout=agent_log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        recorder = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "openvoicechat.daily_agent.recorder",
            body.meeting_url,
            "--output",
            str(recording_path),
            stdout=recorder_log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        session = {
            "worker": worker,
            "recorder": recorder,
            "agent_log": agent_log,
            "recorder_log": recorder_log,
        }
        async with sessions_lock:
            sessions[conversation_id] = session
        session["watcher"] = asyncio.create_task(_watch(conversation_id, session))
    except Exception:  # noqa: BLE001 - cleanup must cover any spawn failure
        for process in (worker, recorder):
            if process is not None:
                await _stop_process(process)
        for handle in (agent_log, recorder_log):
            if handle is not None:
                handle.close()
        async with sessions_lock:
            sessions.pop(conversation_id, None)
        raise HTTPException(status_code=500, detail="Failed to start agent processes")

    return {
        "status": "success",
        "message": "Agent and recording service started",
        "url": body.meeting_url,
        "recording_file": str(recording_path),
        "agent_pid": worker.pid,
        "recording_pid": recorder.pid,
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "active_sessions": len(sessions)}


@app.get("/download_recording")
async def download_recording(
    conversation_id: str, _: None = Depends(require_token)
):
    _validate_id(conversation_id)
    path = recordings_dir / f"{conversation_id}.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Recording not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.get("/download_logs")
async def download_logs(conversation_id: str, _: None = Depends(require_token)):
    _validate_id(conversation_id)
    path = logs_dir / f"{conversation_id}.log"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Logs not found")
    return FileResponse(path, media_type="text/plain", filename=path.name)


def _validate_id(conversation_id: str) -> None:
    if not conversation_id_pattern.fullmatch(conversation_id):
        raise HTTPException(status_code=400, detail="Invalid conversation_id")


async def _watch(conversation_id: str, session: dict) -> None:
    try:
        await session["worker"].wait()
    finally:
        with contextlib.suppress(Exception):
            await _stop_process(session["recorder"])
        with contextlib.suppress(Exception):
            await _stop_process(session["worker"])
        session["agent_log"].close()
        session["recorder_log"].close()
        async with sessions_lock:
            sessions.pop(conversation_id, None)


async def _stop_process(process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        process.kill()
        await process.wait()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("openvoicechat.daily_agent.server:app", host="0.0.0.0", port=8001)
