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
        "stage": "host_ready",
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
