import pytest

from benches.metrics import REMOTE_STAGES
from benches.shared_cache_profile import sample_evidence, summarize


def boundaries():
    before = {
        "orbitkv_remote_fetch_bytes_total": 64,
        "orbitkv_load_bytes_total": 128,
        "orbitkv_remote_fetch_duration_seconds_count": 2,
        "orbitkv_remote_fetch_duration_seconds_sum": 0.02,
    }
    after = {
        **before,
        "orbitkv_remote_fetch_bytes_total": 128,
        "orbitkv_load_bytes_total": 192,
        "orbitkv_remote_fetch_duration_seconds_count": 3,
        "orbitkv_remote_fetch_duration_seconds_sum": 0.021,
    }
    for stage in REMOTE_STAGES:
        count = f"orbitkv_remote_stage_duration_seconds_count_{stage}"
        total = f"orbitkv_remote_stage_duration_seconds_sum_{stage}"
        before.update({count: 2, total: 0.02})
        after.update({count: 3, total: 0.0202})
    restored = {"checked_payload_bytes": 64, "sha256": ["a"] * 4, "timing": {"query_ms": 2}}
    return before, after, restored


@pytest.mark.parametrize(
    "field,value",
    [
        ("orbitkv_remote_fetch_bytes_total", 64),
        ("orbitkv_load_bytes_total", 128),
        ("orbitkv_remote_stage_duration_seconds_count_release", 2),
        ("orbitkv_remote_stage_duration_seconds_count_read", 4),
        ("orbitkv_remote_stage_duration_seconds_sum_read", float("nan")),
        ("orbitkv_remote_fetch_duration_seconds_sum", float("inf")),
    ],
    ids=["local-hit", "no-gpu-copy", "no-release", "multiple-reads", "nan-stage", "infinite-fetch"],
)
def test_profile_rejects_invalid_remote_sample(field, value):
    before, after, restored = boundaries()
    after[field] = value
    with pytest.raises(AssertionError):
        sample_evidence(before, after, restored, 64)


def test_profile_uses_deltas_and_separates_cold_and_warmup():
    before, after, restored = boundaries()
    evidence = sample_evidence(before, after, restored, 64)
    assert evidence["remote_bytes"] == evidence["h2d_bytes"] == 64
    assert evidence["remote_stages_ms"]["read"] == pytest.approx(0.2)
    rows = []
    for phase, duration in [("cold", 100), ("warmup", 50), ("measured", 1), ("measured", 3)]:
        rows.append({**evidence, "phase": phase, "manager_fetch_ms": duration})
    summary = summarize(rows)
    assert summary["measured_samples"] == 2
    assert summary["statistics"]["manager_fetch_ms"] == {
        "p50": 2,
        "p95_nearest_rank": 3,
        "min": 1,
        "max": 3,
    }
