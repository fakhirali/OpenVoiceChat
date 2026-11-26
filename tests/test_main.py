"""Tests for the local runner helpers."""

from pathlib import Path

import main


def test_auto_log_path_is_unique_and_under_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    first = main._auto_log_path()
    second = main._auto_log_path()

    assert first.parent == Path("logs")
    assert first.suffix == ".jsonl"
    assert first != second
    assert len(first.stem.rsplit("-", 1)[-1]) == 8
