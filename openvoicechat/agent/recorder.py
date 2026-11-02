"""Record a Daily meeting to a WAV file."""

import argparse
import contextlib
import signal
import sys
import threading
import time
import wave
from pathlib import Path

from daily import CallClient, Daily, EventHandler

SAMPLE_RATE = 16000
SPEAKER = "ovc-recorder-speaker"


class RecorderEventHandler(EventHandler):
    def __init__(self, finished: threading.Event):
        self.finished = finished

    def on_participant_left(self, participant, reason):
        print(f"Participant left: {reason}", flush=True)
        self.finished.set()


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a Daily meeting to WAV")
    parser.add_argument("meeting_url")
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Daily.init()
    speaker = Daily.create_speaker_device(
        SPEAKER, sample_rate=SAMPLE_RATE, channels=1, non_blocking=True
    )
    Daily.select_speaker_device(SPEAKER)

    joined = threading.Event()
    finished = threading.Event()
    errors: list = []

    def stop_recording(signum, frame):
        finished.set()

    signal.signal(signal.SIGINT, stop_recording)
    signal.signal(signal.SIGTERM, stop_recording)

    def on_joined(data, error):
        if error:
            errors.append(error)
        joined.set()

    client = CallClient(RecorderEventHandler(finished))
    client.update_subscription_profiles(
        {"base": {"camera": "unsubscribed", "microphone": "subscribed"}}
    )
    client.set_user_name("OpenVoiceChat Recorder")

    try:
        with wave.open(str(output), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            client.join(args.meeting_url, completion=on_joined)
            if not joined.wait(15):
                print("Timed out after 15s joining the Daily meeting", file=sys.stderr)
                return 1
            if errors:
                print(f"Unable to join the Daily meeting: {errors[0]}", file=sys.stderr)
                return 1
            while not finished.is_set():
                data = speaker.read_frames(1600)
                if data:
                    wav.writeframes(data)
                else:
                    time.sleep(0.02)
            return 0
    finally:
        with contextlib.suppress(Exception):
            client.leave()
        with contextlib.suppress(Exception):
            client.release()


if __name__ == "__main__":
    sys.exit(main())
