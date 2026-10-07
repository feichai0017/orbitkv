"""Failed release gates must retain the original error and cleanup evidence."""

import json
import os
import sys
from types import SimpleNamespace

import pytest
import requests

from tests.support import installed_serving


def test_startup_exit_keeps_original_failure_and_cleanup_record(monkeypatch, tmp_path):
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("not ready")

    monkeypatch.setattr(installed_serving.requests, "get", unavailable)
    with (
        pytest.raises(pytest.fail.Exception, match="failed-engine exited"),
        installed_serving.service(
            [sys.executable, "-c", "raise SystemExit(7)"],
            "http://unused",
            dict(os.environ),
            tmp_path,
            "failed-engine",
        ),
    ):
        pytest.fail("startup exit should not yield a service")
    cleanup = json.loads((tmp_path / "failed-engine-cleanup.json").read_text())
    assert cleanup["exit_code"] == 7
    assert cleanup["errors"] == ["unexpected exit code 7"]
    assert cleanup["primary_failure"]["type"] == "Failed"
    assert not cleanup["remaining_processes"] and not cleanup["forced_kill"]


@pytest.mark.parametrize("body_failure", [True, False])
def test_unexpected_exit_retained_without_masking_body_failure(monkeypatch, tmp_path, body_failure):
    monkeypatch.setattr(
        installed_serving.requests, "get", lambda *args, **kwargs: SimpleNamespace(ok=True)
    )
    error_type = ValueError if body_failure else AssertionError
    message = "output mismatch" if body_failure else "cleanup failed"
    with (
        pytest.raises(error_type, match=message),
        installed_serving.service(
            [sys.executable, "-c", "import time; time.sleep(0.05); raise SystemExit(7)"],
            "http://unused",
            dict(os.environ),
            tmp_path,
            "failed-engine",
        ) as process,
    ):
        process.wait(timeout=5)
        if body_failure:
            raise ValueError("output mismatch")
    cleanup = json.loads((tmp_path / "failed-engine-cleanup.json").read_text())
    assert cleanup["errors"] == ["unexpected exit code 7"]
    if body_failure:
        assert cleanup["primary_failure"] == {"type": "ValueError", "message": "output mismatch"}
    else:
        assert cleanup["primary_failure"] is None
    assert not cleanup["remaining_processes"] and not cleanup["forced_kill"]
