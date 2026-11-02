"""Tests for the local audio loop helpers."""

import numpy as np
import pytest

pytest.importorskip("soxr")

from main import SPEECH_RATE, _Pcm16Resampler


def test_streaming_resampler_has_no_large_boundary_jumps():
    rate = 48000
    time = np.arange(rate) / rate
    audio = (np.sin(2 * np.pi * 440 * time) * 0.5 * 32767).astype(np.int16)
    frame = int(rate * 0.032)
    resampler = _Pcm16Resampler(rate, SPEECH_RATE)

    output = b"".join(
        resampler.process(audio[start : start + frame].tobytes())
        for start in range(0, len(audio), frame)
    )
    samples = np.frombuffer(output, dtype=np.int16).astype(np.int32)

    assert len(samples) > SPEECH_RATE * 0.9
    assert np.max(np.abs(np.diff(samples))) < 20000
