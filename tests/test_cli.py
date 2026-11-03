"""Tests for the ovc command line."""

import json
import subprocess
import sys

import pytest

from openvoicechat import cli


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "ovc" in capsys.readouterr().out


def test_doctor_json_schema(capsys):
    code = cli.main(["doctor", "--json"])
    assert code in (0, 2)
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == cli.SCHEMA_VERSION
    assert isinstance(payload["ok"], bool)
    assert payload["checks"]
    for check in payload["checks"]:
        assert set(check) == {"name", "status", "detail"}
        assert check["status"] in {"ok", "warn", "missing", "fail"}


def test_package_import_stays_lazy():
    code = "import sys, openvoicechat; print('openai' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
