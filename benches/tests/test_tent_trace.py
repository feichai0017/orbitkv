"""Reject incomplete native evidence and preserve concurrent stage timing."""

import copy
import json

import pytest

from benches import tent_trace


def trace():
    events = [
        {"stage": stage, "begin_mono_ns": 100, "end_mono_ns": 300, "tid": index + 1, "status": 0}
        for index, stage in enumerate(sorted(tent_trace.STAGES))
    ]
    events.append({**events[0], "begin_mono_ns": 200, "end_mono_ns": 400, "tid": 5})
    return {
        "schema": "tent.native-trace.v1",
        "clock": "CLOCK_MONOTONIC",
        "pid": 42,
        "limit": 4096,
        "seen": len(events),
        "overflow": False,
        "events": events,
    }


def test_stage_union_does_not_add_concurrent_bootstrap_calls():
    sample = dict(
        zip(
            ("begin_mono_ns", "submit_return_mono_ns", "terminal_mono_ns", "freed_mono_ns"),
            (150, 160, 350, 360),
            strict=True,
        )
    )
    summary = tent_trace.summarize(trace(), "consumer", 42, sample)
    assert summary["stages"]["bootstrap_rpc"]["overlap_union_ms"] == 300 / 1e6
    cold = summary["cold_read"]["stages"]["bootstrap_rpc"]
    assert cold["calls"] == 2
    assert cold["overlap_union_ms"] == 210 / 1e6
    assert cold["duration_max_ms"] == 160 / 1e6


@pytest.mark.parametrize(
    ("field", "value"),
    [("pid", 43), ("clock", "foreign clock"), ("overflow", True), ("seen", 4), ("limit", 8192)],
)
def test_rejects_wrong_process_clock_overflow_and_missing_events(field, value):
    evidence = trace()
    evidence[field] = value
    with pytest.raises(ValueError):
        tent_trace.summarize(evidence, "consumer", 42)


@pytest.mark.parametrize(
    "mutation",
    [
        {"stage": "unknown"},
        {"status": 1},
        {"begin_mono_ns": 301},
        {"end_mono_ns": 300.0},
        {"tid": 0},
    ],
)
def test_rejects_failed_or_malformed_native_intervals(mutation):
    evidence = trace()
    evidence["events"][1].update(mutation)
    with pytest.raises(ValueError):
        tent_trace.summarize(evidence, "consumer", 42)


def test_source_cannot_borrow_consumer_clock_or_claim_full_decomposition():
    evidence = trace()
    evidence["events"] = [row for row in evidence["events"] if row["stage"] == "endpoint_construct"]
    evidence["seen"] = len(evidence["events"])
    assert (
        tent_trace.summarize(evidence, "source", 42)["stages"]["endpoint_construct"]["calls"] == 1
    )
    with pytest.raises(ValueError, match="required stage"):
        tent_trace.summarize(evidence, "consumer", 42)
    sample = {
        "begin_mono_ns": 0,
        "submit_return_mono_ns": 1,
        "terminal_mono_ns": 2,
        "freed_mono_ns": 3,
    }
    with pytest.raises(ValueError, match="source trace"):
        tent_trace.summarize(evidence, "source", 42, sample)
    with pytest.raises(ValueError, match="do not overlap"):
        tent_trace.summarize(copy.deepcopy(trace()), "consumer", 42, sample)


@pytest.mark.parametrize("exit_code", [0, 1, -15])
def test_cli_requires_actual_zero_exit_even_with_a_drained_stop(tmp_path, monkeypatch, exit_code):
    native = tmp_path / "native.json"
    native.write_text(json.dumps(trace()))
    stdout = tmp_path / "endpoint.stdout"
    sample = {
        "begin_mono_ns": 0,
        "submit_return_mono_ns": 1,
        "terminal_mono_ns": 500,
        "freed_mono_ns": 501,
    }
    records = [
        {"event": "ready", "pid": 42},
        {"event": "read_complete", "samples": [sample]},
        {"event": "stopped", "registered_regions": 0, "active_batches": 0},
    ]
    stdout.write_text("\n".join(json.dumps({"tent_stage_probe": True, **row}) for row in records))
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tent_trace",
            "--trace",
            str(native),
            "--endpoint-stdout",
            str(stdout),
            "--endpoint-exit-code",
            str(exit_code),
            "--role",
            "consumer",
            "--output",
            str(output),
        ],
    )
    if exit_code:
        with pytest.raises(ValueError, match="must exit zero"):
            tent_trace.main()
        assert not output.exists()
    else:
        tent_trace.main()
        assert json.loads(output.read_text())["pid"] == 42
