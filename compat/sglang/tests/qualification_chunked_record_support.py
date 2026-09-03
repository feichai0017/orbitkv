"""Assertions for the frozen exact-Chunked qualification record."""

from __future__ import annotations

import uuid
from typing import Any


def assert_chunked_record_shape(
    bench: Any, record: dict[str, object], mode: str, case: str, *, failed: bool = False
) -> None:
    assert set(record) == bench.TOP_LEVEL_KEYS
    assert record["schema"] == bench.RECORD_SCHEMA
    assert record["mode"] == mode
    for payload, digest in (
        ("command", "command_sha256"),
        ("environment", "environment_sha256"),
        ("source_identity", "source_identity_sha256"),
        ("checkpoint", "checkpoint_identity_sha256"),
    ):
        assert record[digest] == bench.canonical_digest(record[payload])
    run_id = record["runtime_identity"]["run_id"]
    assert uuid.UUID(run_id).version == 4
    assert set(record["checkpoint"]) == {"identity", "config"}
    assert set(record["workload"]) == {
        "case", "requests", "prompt_tokens", "decode_tokens",
        "materialized_kv_tokens_per_request", "iterations", "seed",
        "fresh_prompts", "input_token_digest_sha256",
        "input_token_digests_by_iteration_sha256", "chunk_geometry",
    }
    assert record["workload"]["case"] == case
    assert set(record["workload"]["chunk_geometry"]) == {
        "page_tokens", "chunk_tokens", "blocks_per_epoch",
        "chunk_epoch_count_per_request", "epoch_end_crossings_per_request",
    }
    assert len(record["timings"]["iteration_seconds"]) == (0 if failed else 3)
    expected_capacity = 256 if case == "roomy" else 64
    if failed:
        assert record["outputs"] == {
            "iterations": [], "aggregate_sha256": bench.canonical_digest([]),
        }
        assert record["server_capacity"]["status"] == "failed"
        assert record["server_capacity"]["requested_tokens"] == expected_capacity
        assert record["server_capacity"]["failure"]["type"] == "capacity_exhausted"
    else:
        assert record["server_capacity"] == {
            "status": "observed", "requested_tokens": expected_capacity,
            "available_tokens": expected_capacity, "failure": None,
        }
    stages = [item["stage"] for item in record["gpu_snapshots"]]
    expected = ["before_engine", "after_load", "after_workload", "after_shutdown"]
    if failed:
        assert stages[0] == "before_engine" and stages[-1] == "after_shutdown"
        assert len(stages) == len(set(stages))
    else:
        assert stages == expected
    assert all(
        set(gate) == {"qualified", "reasons"} and not gate["qualified"]
        and gate["reasons"] for gate in record["claims"].values()
    )
