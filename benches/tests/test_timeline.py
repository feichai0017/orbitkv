"""Timeline evidence must belong to measured requests and one local clock."""

import json

from benches.timeline import collect


def test_timeline_does_not_pair_distinct_process_clocks_or_count_seed_requests(tmp_path):
    events = [
        {"request_id": "cmpl-abc-0-a1b2c3", "stage": "queued", "pid": 1, "monotonic_ns": 0},
        {
            "request_id": "cmpl-abc-0-a1b2c3",
            "stage": "restore_submit",
            "pid": 2,
            "monotonic_ns": 10,
        },
        {
            "request_id": "cmpl-abc-0-a1b2c3",
            "stage": "gpu_ready",
            "pid": 2,
            "monotonic_ns": 1000010,
        },
        {
            "request_id": "cmpl-abc-0-a1b2c3",
            "stage": "first_use",
            "pid": 1,
            "monotonic_ns": 3000000,
        },
        {"request_id": "seed", "stage": "queued", "pid": 1, "monotonic_ns": 0},
    ]
    (tmp_path / "engine.log").write_text(
        "\n".join("prefix cache_timeline " + json.dumps(e) for e in events)
    )
    event = {
        "request_id": "cmpl-abc-0-a1b2c3",
        "stage": "source_ready",
        "pid": 3,
        "warmup": True,
        "elapsed_us": 1500,
    }
    (tmp_path / "manager.log").write_text("cache_timeline " + json.dumps(event))
    summary = collect(tmp_path, [{"request_id": "cmpl-abc"}, {"request_id": "untraced"}])
    assert summary["measured_requests"] == 2
    assert summary["traced_requests"] == 1
    assert summary["stage_counts"]["queued"] == 1
    assert summary["intervals"]["queued_to_first_use_ms"]["p50"] == 3
    assert summary["intervals"]["restore_ms"]["p50"] == 1
    assert summary["intervals"]["warmup_prepare_ms"]["p50"] == 1.5


def test_batched_completion_is_linked_without_counting_each_request_as_a_transfer(tmp_path):
    (tmp_path / "engine.log").write_text(
        "\n".join(
            "cache_timeline "
            + json.dumps({"request_id": rid, "stage": "restore_link", "restore_key": "manager:1:2"})
            for rid in ("a", "b")
        )
    )
    (tmp_path / "manager.log").write_text(
        "\n".join(
            "cache_timeline "
            + json.dumps({"stage": "restore_delivered", "restore_key": key, "elapsed_us": elapsed})
            for key, elapsed in (("manager:1:2", 3500), ("manager:2:2", 999999))
        )
    )
    summary = collect(tmp_path, [{"request_id": "a"}, {"request_id": "b"}])
    assert summary["traced_requests"] == 2
    assert summary["intervals"]["completion_delivery_ms"] == {
        "count": 1,
        "p50": 3.5,
        "p95": 3.5,
        "p99": 3.5,
    }
