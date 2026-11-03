"""Tests for the async conversation pipeline."""

import asyncio
from types import SimpleNamespace

from openvoicechat.config import Config
from openvoicechat.conversation import Conversation


class FakeVAD:
    def __call__(self, frame: bytes) -> bool:
        return frame == b"\x01"

    def reset(self) -> None:
        pass


class FakeSpeech:
    def __init__(self, chunks=1):
        self.chunks = chunks
        self.transcribe_calls = 0

    async def transcribe(self, pcm16: bytes) -> str:
        self.transcribe_calls += 1
        return "hello"

    async def synthesize(self, text: str) -> bytes:
        return b"\x02\x00" * 1600 * self.chunks

    async def aclose(self) -> None:
        pass


class FakeCompletions:
    async def create(self, **kwargs):
        async def stream():
            for token in ("Hi ", "there."):
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=token))]
                )

        return stream()


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def make_config() -> Config:
    return Config(
        endpoint_silence_ms=64,
        min_speech_ms=32,
        pre_roll_ms=0,
        max_utterance_ms=5000,
        output_chunk_ms=100,
    )


def make_conversation(events, speech=None) -> Conversation:
    return Conversation(
        speech or FakeSpeech(),
        system_prompt="You are helpful.",
        config=make_config(),
        client=FakeClient(),
        on_turn=lambda role, text: events.append((role, text)),
        vad=FakeVAD(),
    )


async def feed_utterance(conv: Conversation) -> asyncio.Task:
    task = asyncio.create_task(conv.run())
    for frame in (b"\x01", b"\x01", b"\x00", b"\x00"):
        await conv.input.put(frame)
    return task


async def wait_for_role(events, role: str, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not any(event_role == role for event_role, _ in events):
        if loop.time() > deadline:
            raise AssertionError(f"timed out waiting for {role!r} event; got {events!r}")
        await asyncio.sleep(0.005)


async def stop(conv: Conversation, run_task: asyncio.Task) -> None:
    conv.stop()
    await asyncio.wait_for(run_task, timeout=2)


async def test_conversation_produces_audio_and_turns():
    events = []
    conv = make_conversation(events)
    run_task = await feed_utterance(conv)

    chunk = await asyncio.wait_for(conv.output.get(), timeout=2)
    assert len(chunk) == 3200
    assert chunk == b"\x02\x00" * 1600

    await wait_for_role(events, "assistant")
    assert ("user", "hello") in events
    assert ("assistant", "Hi there.") in events

    await stop(conv, run_task)


async def test_history_ends_with_user_and_assistant():
    events = []
    conv = make_conversation(events)
    run_task = await feed_utterance(conv)

    await wait_for_role(events, "assistant")
    assert conv.messages[0] == {"role": "system", "content": "You are helpful."}
    assert [message["role"] for message in conv.messages[-2:]] == ["user", "assistant"]

    await stop(conv, run_task)


async def test_max_utterance_waits_for_silence_instead_of_self_interrupting():
    events = []
    speech = FakeSpeech()
    conv = Conversation(
        speech,
        config=Config(
            endpoint_silence_ms=64,
            min_speech_ms=32,
            pre_roll_ms=0,
            max_utterance_ms=96,
            output_chunk_ms=100,
        ),
        client=FakeClient(),
        on_turn=lambda role, text: events.append((role, text)),
        vad=FakeVAD(),
    )
    run_task = asyncio.create_task(conv.run())
    for _ in range(5):  # keep talking past the 96 ms cap
        await conv.input.put(b"\x01")
    for _ in range(2):  # then stop
        await conv.input.put(b"\x00")

    await wait_for_role(events, "assistant")
    assert speech.transcribe_calls == 1

    await stop(conv, run_task)


async def test_interrupt_cancels_turn_and_drains_output():
    events = []
    conv = make_conversation(events)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def long_turn() -> None:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    conv._turn_task = asyncio.create_task(long_turn())
    await started.wait()
    await conv.output.put(b"stale audio")

    await conv._interrupt()

    assert cancelled.is_set()
    assert conv._turn_task is None
    assert conv.output.empty()
    assert conv.agent_speaking is False
