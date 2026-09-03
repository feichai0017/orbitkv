from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_verify_qualification import _record, _write_pair, verifier


@pytest.mark.parametrize("failure", ("counter-decrease", "failure", "drain"))
def test_stream_gate_rejects_counter_decrease_failure_or_incomplete_drain(
    tmp_path: Path, failure: str
) -> None:
    manager = _record("manager")
    snapshots = manager["manager"]["snapshots"]
    if failure == "counter-decrease":
        snapshots[2]["batch_counters"]["prepare_batch_calls"] = 0
        with pytest.raises(RuntimeError, match="counters decreased"):
            verifier.validate_record(manager)
        return
    if failure == "failure":
        snapshots[1]["batch_counters"]["fail_stops"] = 1
        snapshots[2]["batch_counters"]["fail_stops"] = 1
    else:
        snapshots[2]["manager_stats"]["active_requests"] = 1

    path = _write_pair(tmp_path, failure, _record("stock"), manager)
    pair = verifier.verify_pair(path)
    assert pair["gates"]["stream_event_qualified"]["qualified"] is False
    assert pair["gates"]["stream_event_qualified"]["reasons"]


def test_stored_derived_pair_tamper_is_rejected(tmp_path: Path) -> None:
    path = _write_pair(tmp_path, "pair", _record("stock"), _record("manager"))
    pair = json.loads(path.read_text(encoding="utf-8"))
    pair["gates"]["capacity_qualified"] = {"qualified": True, "reasons": []}
    path.write_text(json.dumps(pair), encoding="utf-8")

    with pytest.raises(RuntimeError, match="stored pair differs"):
        verifier.verify_pair(path)


def test_stored_derived_summary_tamper_is_rejected(tmp_path: Path) -> None:
    paths = [
        _write_pair(tmp_path, f"epoch-{epoch}", _record("stock"), _record("manager"))
        for epoch in range(3)
    ]
    summary = verifier.build_summary(tmp_path, [path.name for path in paths])
    summary["gates"]["throughput_go"] = {"qualified": False, "reasons": ["forged"]}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(RuntimeError, match="stored summary differs"):
        verifier.verify_summary(path)
