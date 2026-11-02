"""Join a Daily meeting and run a voice conversation."""

import argparse
import asyncio
import contextlib
import signal
import sys
from pathlib import Path

from daily import CallClient, Daily, EventHandler

from openvoicechat.config import Config
from openvoicechat.conversation import Conversation
from openvoicechat.speech import Speech

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
SPEAKER = "ovc-speaker"
MICROPHONE = "ovc-mic"


class AgentEventHandler(EventHandler):
    def __init__(self, loop: asyncio.AbstractEventLoop, stop: asyncio.Event):
        self.loop = loop
        self.stop = stop

    def on_participant_left(self, participant, reason):
        print(f"Participant left: {reason}", flush=True)
        self.loop.call_soon_threadsafe(self.stop.set)


async def main(meeting_url: str, prompt: str, conversation_id: str) -> None:
    loop = asyncio.get_running_loop()
    config = Config()
    speech = Speech(config)
    log_dir = Path(config.data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(  # noqa: ASYNC230, SIM115
        log_dir / f"{conversation_id}.log", "a", buffering=1, encoding="utf-8"
    )

    def on_turn(role: str, text: str) -> None:
        log_file.write(f"{role.upper()}: {text}\n")

    stop = asyncio.Event()
    joined = asyncio.Event()
    join_errors: list = []

    def on_joined(data, error):
        if error:
            join_errors.append(error)
        loop.call_soon_threadsafe(joined.set)

    Daily.init()
    speaker = Daily.create_speaker_device(SPEAKER, sample_rate=SAMPLE_RATE, channels=1)
    microphone = Daily.create_microphone_device(
        MICROPHONE, sample_rate=SAMPLE_RATE, channels=1
    )
    Daily.select_speaker_device(SPEAKER)
    client = CallClient(AgentEventHandler(loop, stop))
    client.update_subscription_profiles(
        {"base": {"camera": "unsubscribed", "microphone": "subscribed"}}
    )
    client.set_user_name("OpenVoiceChat")

    conversation = None
    tasks: list[asyncio.Task] = []
    try:
        conversation = Conversation(
            speech, system_prompt=prompt, config=config, on_turn=on_turn
        )
        client.join(
            meeting_url,
            client_settings={
                "inputs": {
                    "camera": False,
                    "microphone": {
                        "isEnabled": True,
                        "settings": {"deviceId": MICROPHONE},
                    },
                }
            },
            completion=on_joined,
        )
        try:
            await asyncio.wait_for(joined.wait(), timeout=15)
        except TimeoutError as exc:
            raise RuntimeError("Timed out after 15s joining the Daily meeting") from exc
        if join_errors:
            raise RuntimeError(f"Unable to join the Daily meeting: {join_errors[0]}")

        async def receive_audio() -> None:
            while not stop.is_set():
                data = await loop.run_in_executor(None, speaker.read_frames, FRAME_SAMPLES)
                if data:
                    await conversation.input.put(bytes(data))

        async def send_audio() -> None:
            while not stop.is_set():
                chunk = await conversation.output.get()
                await loop.run_in_executor(None, microphone.write_frames, chunk)

        tasks = [
            asyncio.create_task(conversation.run(), name="conversation"),
            asyncio.create_task(receive_audio(), name="receive-audio"),
            asyncio.create_task(send_audio(), name="send-audio"),
        ]
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, stop.set)

        waiter = asyncio.create_task(stop.wait(), name="stop")
        try:
            await asyncio.wait((waiter, tasks[0]), return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
    finally:
        stop.set()
        if conversation is not None:
            conversation.stop()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            client.leave()
        with contextlib.suppress(Exception):
            client.release()
        await speech.aclose()
        log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a voice agent in a Daily meeting")
    parser.add_argument("meeting_url")
    parser.add_argument("prompt")
    parser.add_argument("conversation_id")
    args = parser.parse_args()
    asyncio.run(main(args.meeting_url, args.prompt, args.conversation_id))
    sys.exit(0)
