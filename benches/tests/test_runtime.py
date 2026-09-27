"""Service teardown must retain caller-owned GPU exports until actual exit."""

import contextlib
import signal
import subprocess
from types import SimpleNamespace

import pytest

from benches import runtime


@pytest.mark.parametrize(
    ("steps", "body_failure", "expected_error", "broken_stderr"),
    [
        (["exit"], False, None, False),
        (["timeout", "timeout", "exit"], True, RuntimeError, False),
        (["timeout", "timeout", "exit"], True, RuntimeError, True),
        (["signal", "interrupt", "signal", "exit"], False, KeyboardInterrupt, False),
    ],
)
def test_server_reaps_before_caller_releases_exports_even_after_kill_timeouts_and_interrupts(
    monkeypatch, tmp_path, steps, body_failure, expected_error, broken_stderr
):
    state = {"owner_held": True, "reaped": False}
    pending = iter(steps)
    sent = []
    handler = {"current": signal.default_int_handler}

    class Process:
        pid = 123456

        def poll(self):
            return None

        def wait(self, timeout):
            assert state["owner_held"], "caller released its export before service exit"
            step = next(pending)
            if step == "signal":
                handler["current"](signal.SIGINT, None)
                raise subprocess.TimeoutExpired("manager", timeout)
            if step == "timeout":
                raise subprocess.TimeoutExpired("manager", timeout)
            if step == "interrupt":
                raise KeyboardInterrupt
            state["reaped"] = True
            return 0

    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(runtime.requests, "get", lambda *a, **kw: SimpleNamespace(ok=True))
    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(runtime.signal, "getsignal", lambda _: handler["current"])
    monkeypatch.setattr(runtime.signal, "signal", lambda _, value: handler.update(current=value))
    if broken_stderr:

        def fail_write(_text):
            raise BrokenPipeError("log reader closed")

        monkeypatch.setattr(runtime.sys, "stderr", SimpleNamespace(write=fail_write))
    expectation = pytest.raises(expected_error) if expected_error else contextlib.nullcontext()
    with expectation:
        try:
            with runtime.server(["manager"], {}, "http://manager", tmp_path / "manager.log"):
                if body_failure:
                    raise RuntimeError("GPU completion was not established")
        finally:
            assert state["reaped"], "teardown unwound with a live service"
            state["owner_held"] = False

    assert list(pending) == []
    assert sent[0] == (Process.pid, signal.SIGTERM)
    assert all(item == (Process.pid, signal.SIGKILL) for item in sent[1:])
    assert handler["current"] is signal.default_int_handler
