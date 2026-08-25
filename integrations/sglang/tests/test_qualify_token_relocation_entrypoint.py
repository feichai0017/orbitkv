from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = INTEGRATION_ROOT / "qualify_token_relocation.py"
sys.path.insert(0, str(INTEGRATION_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "qualify_token_relocation_entrypoint", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def test_main_forwards_argv_once_and_returns_legacy_result(monkeypatch):
    calls = []
    argv = ["verify-seal", "/tmp/sealed-qualification"]

    def legacy_main(received):
        calls.append(received)
        return 73

    monkeypatch.setattr(qualification._legacy, "main", legacy_main)

    assert qualification.main(argv) == 73
    assert calls == [argv]
    assert calls[0] is argv


@pytest.mark.parametrize("interpreter_flags", [[], ["-S"]])
def test_help_runs_directly_without_site_packages(interpreter_flags):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, *interpreter_flags, str(MODULE_PATH), "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "usage:" in completed.stdout
    for action in ("preflight", "run", "component", "seal", "verify-seal"):
        assert action in completed.stdout
    assert all(
        marker not in completed.stdout.lower()
        for marker in ("h20", "qwen", "gpt-oss", "v0.5.17", "v0517")
    )
