import json

import pytest

from benches.live_store_measurements import _installation_sample, _process_sample
from benches.summarize_live_store_isolation import summarize


def test_installation_sample_uses_owner_commit_timestamps_not_http_completion():
    targets = {
        "a": {
            "node": "a",
            "incarnation": "owner-a",
            "view_id": "view-a",
            "sequence": 9,
            "published_mono_ns": 1_000_000,
            "save_start_mono_ns": 500_000,
        },
        "b": {
            "node": "b",
            "incarnation": "owner-b",
            "view_id": "view-b",
            "sequence": 11,
            "published_mono_ns": 2_000_000,
            "save_start_mono_ns": 1_250_000,
        },
    }
    installed = {
        "a": {
            "applied_sequence": 9,
            "view_id": "view-a",
            "installed_mono_ns": 4_000_000,
        },
        "b": {
            "applied_sequence": 11,
            "view_id": "view-b",
            "installed_mono_ns": 5_500_000,
        },
    }
    sample = _installation_sample(targets, installed, 8_000_000)
    assert sample["publication_to_install_ms"] == 3.5
    assert sample["save_start_to_install_ms"] == 4.25
    assert sample["install_to_harness_ms"] == 4.0
    assert sample["owners"][0]["installed_sequence"] == 9
    assert sample["owners"][1]["installed_sequence"] == 11
    installed["b"]["applied_sequence"] = 12
    with pytest.raises(AssertionError):
        _installation_sample(targets, installed, 8_000_000)


def _run(condition, medium, seed, order, p99):
    summary = {"samples": 20, "p50": p99 / 2, "p95": p99 * 0.9, "p99": p99, "max": p99}
    return {
        "measurement_contract": "s2.10-performance-v2",
        "status": "passed",
        "condition": condition,
        "medium": medium,
        "seed": seed,
        "samples": 20,
        "warmup_rounds": 2,
        "cadence_ms": 1000,
        "achieved_samples_per_second": 1.0,
        "local_namespace": "local",
        "pressure_namespace": "pressure",
        "pressure_keys": 32,
        "pressure_active_keys": 24,
        "pressure_window_shift": 4,
        "artifacts": {"manager": {"sha256": "same"}},
        "order_index": order,
        "budgets": {},
        "key_oracle_sha256": "keys",
        "payload_oracle_sha256": "payloads",
        "latencies": {
            "manager_save_submit_ms": summary,
            "local_query_ms": summary,
        },
        "process_before": {
            "thread_context_switches": {},
            "cpu_ticks": 1,
            "voluntary_context_switches": 1,
            "nonvoluntary_context_switches": 1,
        },
        "process_after": {
            "thread_context_switches": {},
            "cpu_ticks": 2,
            "voluntary_context_switches": 2,
            "nonvoluntary_context_switches": 2,
        },
        "pressure_result": {
            "mode": condition,
            "changes": 22 * 4 * 2 if condition == "pressure" else 0,
        },
    }


def test_isolation_summary_pairs_runs_and_never_uses_requests_as_bootstrap_units(tmp_path):
    seeds = ["one", "two"]
    for pair, seed in enumerate(seeds):
        order = (pair * 2, pair * 2 + 1)
        if pair % 2:
            order = order[::-1]
        for condition, index in zip(("quiet", "pressure"), order, strict=True):
            directory = tmp_path / f"dram-{seed}-{condition}"
            directory.mkdir()
            p99 = 1.0 if condition == "quiet" else 1.04
            (directory / "isolation-result.json").write_text(
                json.dumps(_run(condition, "dram", seed, index, p99))
            )
    result = summarize(tmp_path, ["dram"], seeds, False)
    assert result["statistical_unit"] == "independent matched run pair"
    assert result["request_samples_are_not_bootstrap_units"]
    assert result["media"]["dram"]["metrics"]["local_query_ms"]["every_pair_at_most_1_05"]
    assert result["status"] == "preexperiment"
    assert not result["media"]["dram"]["isolation_qualified"]


def test_process_sample_reports_context_switch_and_affinity_fields():
    sample = _process_sample([__import__("os").getpid()])
    assert sample["processes"] == 1
    assert sample["cpu_affinity"]
    assert sample["memory_affinity"]
    assert sample["voluntary_context_switches"] >= 0
    assert sample["nonvoluntary_context_switches"] >= 0
