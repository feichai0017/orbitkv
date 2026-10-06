"""CPU-only checks for the bounded S2.10 observer process protocol."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from benches.analyze_observer_isolation import _bootstrap_geomean, _overlaps
from benches.observer_polling import (
    MISSED_DEADLINE_POLICY,
    OBSERVER_PROTOCOL,
    ObserverSession,
)


def test_pair_bootstrap_resamples_runs_and_overlap_includes_post_return_boundary():
    summary = _bootstrap_geomean([0.5, 0.6, 0.7, 0.8], seed=20261006, draws=1000)
    assert summary["pairs"] == 4
    assert 0.5 < summary["geometric_mean"] < 0.8
    assert summary["bootstrap_95_ci"][0] <= summary["geometric_mean"]
    assert summary["bootstrap_95_ci"][1] >= summary["geometric_mean"]
    assert _overlaps([100, 300], [150, 350], 140, 200)
    assert _overlaps([100, 300], [150, 350], 150, 150)
    assert not _overlaps([100, 300], [150, 350], 151, 299)


@contextmanager
def owner_server(owner: dict):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/cache/metadata/owners?limit=128":
                self.send_error(404)
                return
            payload = json.dumps([owner]).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _run_session(tmp_path, endpoint, mode):
    output = tmp_path / mode
    output.mkdir()
    session = ObserverSession(
        mode=mode,
        endpoint=endpoint,
        incarnation="owner-1",
        expected_view="view-1",
        output=output,
        cadence_ns=10_000_000,
        poll_count=4,
    )
    ready = session.launch(timeout_seconds=5)
    epoch = time.monotonic_ns() + 50_000_000
    start = session.start(
        epoch_mono_ns=epoch,
        observer_phase_offset_ns=2_000_000,
        foreground_phase_offset_ns=7_000_000,
    )
    result = session.finish(timeout_seconds=5)
    return output, ready, start, result


def test_in_process_and_helper_use_identical_bounded_schedule_and_sample_schema(tmp_path):
    owner = {
        "owner": "owner-1",
        "view_id": "view-1",
        "fresh": True,
        "applied_sequence": 7,
        "installed_mono_ns": time.monotonic_ns(),
    }
    with owner_server(owner) as endpoint:
        first = _run_session(tmp_path, endpoint, "in-process")
        second = _run_session(tmp_path, endpoint, "helper-process")

    for output, ready, start, result in (first, second):
        assert ready["contract"] == OBSERVER_PROTOCOL
        assert start["missed_deadline_policy"] == MISSED_DEADLINE_POLICY
        assert result["status"] == "exited"
        assert result["samples"] == 4
        assert (output / "observer-stop.json").is_file()
        assert not (output / "observer-error.json").exists()
        rows = [
            json.loads(line)
            for line in (output / "observer-samples.jsonl").read_text().splitlines()
        ]
        assert [row["index"] for row in rows] == list(range(4))
        assert [
            row["scheduled_poll_start_mono_ns"] - rows[0]["scheduled_poll_start_mono_ns"]
            for row in rows
        ] == [0, 10_000_000, 20_000_000, 30_000_000]
        assert all(row["response_bytes"] > 0 for row in rows)
        assert all(row["scheduling_deviation_ns"] >= 0 for row in rows)
    first_rows = [
        json.loads(line) for line in (first[0] / "observer-samples.jsonl").read_text().splitlines()
    ]
    second_rows = [
        json.loads(line) for line in (second[0] / "observer-samples.jsonl").read_text().splitlines()
    ]
    assert first_rows[0].keys() == second_rows[0].keys()
    assert second[1]["gpu_or_native_modules_loaded"] == []


def test_helper_failure_is_terminal_and_writes_error_protocol(tmp_path):
    owner = {
        "owner": "owner-1",
        "view_id": "unexpected-view",
        "fresh": True,
        "applied_sequence": 7,
        "installed_mono_ns": time.monotonic_ns(),
    }
    output = tmp_path / "failed-helper"
    output.mkdir()
    with owner_server(owner) as endpoint:
        session = ObserverSession(
            mode="helper-process",
            endpoint=endpoint,
            incarnation="owner-1",
            expected_view="view-1",
            output=output,
            cadence_ns=5_000_000,
            poll_count=3,
        )
        session.launch(timeout_seconds=5)
        session.start(
            epoch_mono_ns=time.monotonic_ns() + 20_000_000,
            observer_phase_offset_ns=0,
            foreground_phase_offset_ns=2_500_000,
        )
        with pytest.raises(RuntimeError, match="observer"):
            session.finish(timeout_seconds=5)
        session.abort()
    error = json.loads((output / "observer-error.json").read_text())
    assert error["error_type"] == "RuntimeError"
    assert "unexpected-view" in error["error"]
    assert not (output / "observer-complete.json").exists()


def test_sample_digest_mismatch_invalidates_completed_helper(tmp_path):
    owner = {
        "owner": "owner-1",
        "view_id": "view-1",
        "fresh": True,
        "applied_sequence": 7,
        "installed_mono_ns": time.monotonic_ns(),
    }
    output = tmp_path / "corrupt-helper"
    output.mkdir()
    with owner_server(owner) as endpoint:
        session = ObserverSession(
            mode="helper-process",
            endpoint=endpoint,
            incarnation="owner-1",
            expected_view="view-1",
            output=output,
            cadence_ns=5_000_000,
            poll_count=2,
        )
        session.launch(timeout_seconds=5)
        session.start(
            epoch_mono_ns=time.monotonic_ns() + 20_000_000,
            observer_phase_offset_ns=0,
            foreground_phase_offset_ns=2_500_000,
        )
        deadline = time.monotonic() + 5
        while not (output / "observer-complete.json").exists():
            assert time.monotonic() < deadline
            session.check()
            time.sleep(0.005)
        with (output / "observer-samples.jsonl").open("a") as samples:
            samples.write("{}\n")
        with pytest.raises(RuntimeError, match="wrote 3/2 samples"):
            session.finish(timeout_seconds=5)
        session.abort()
