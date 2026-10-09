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


@pytest.mark.parametrize("corruption", ["unrelated", "missing_previous", "nonfinite", "negative"])
def test_drain_rejects_invalid_or_disappearing_gauges(monkeypatch, corruption):
    snapshots = [
        {"orbitkv_query_reserved_bytes": 0, "orbitkv_ssd_read_pinned_bytes": 4096},
        {"orbitkv_query_reserved_bytes": 0, "orbitkv_ssd_read_pinned_bytes": 0},
    ]
    if corruption == "unrelated":
        snapshots = [{"unrelated_metric": 0}]
    elif corruption == "missing_previous":
        snapshots[1].pop("orbitkv_ssd_read_pinned_bytes")
    elif corruption == "nonfinite":
        snapshots[0]["orbitkv_query_reserved_bytes"] = float("nan")
    else:
        snapshots[0]["orbitkv_ssd_read_pinned_bytes"] = -1
    values = iter(snapshots)
    monkeypatch.setattr(installed_serving, "fetch_orbitkv_metrics", lambda _: next(values))
    monkeypatch.setattr(installed_serving.time, "sleep", lambda _: None)
    with pytest.raises(AssertionError, match="resource-drain gauges"):
        installed_serving.wait_for_drain(1)


def test_drain_preserves_lazy_unused_route_metrics(monkeypatch):
    snapshots = [
        {"orbitkv_query_reserved_bytes": 0, "orbitkv_inflight_bytes": 4096},
        {"orbitkv_query_reserved_bytes": 0, "orbitkv_inflight_bytes": 0},
    ]
    values = iter(snapshots)
    monkeypatch.setattr(installed_serving, "fetch_orbitkv_metrics", lambda _: next(values))
    monkeypatch.setattr(installed_serving.time, "sleep", lambda _: None)
    assert installed_serving.wait_for_drain(1) == snapshots[-1]


@pytest.mark.parametrize(
    "engine,text,expected",
    [
        ("vllm", "Capturing CUDA graphs (FULL) 100% - Mode: FULL", 0),
        ("vllm", "| 1 | 1 | 0 | NONE | 7 |", 0),
        ("vllm", "| 769 | 769 | 0 | FULL | 9 |", 0),
        ("vllm", "| 1 | 1 | 0 | FULL | 7 |\n| 1 | 2 | 1 | FULL | 8 |", 15),
        ("sglang", 'sglang:cuda_graph_passes_total{mode="prefill_cuda_graph"} 9', 0),
        ("sglang", 'sglang:cuda_graph_passes_total{mode="decode_none"} 9', 0),
        ("sglang", 'sglang:cuda_graph_passes_total{rank="0",mode="decode_cuda_graph"} 7', 7),
        ("sglang", "", 0),
    ],
)
def test_native_decode_graph_observations_distinguish_replays(engine, text, expected):
    assert installed_serving.native_runtime_graph_count(engine, text) == expected


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("value", ["NaN", "+Inf", "-1", "0.5"])
def test_native_decode_graph_rejects_corrupt_counters(engine, value):
    text = (
        f"| 1 | 1 | 0 | FULL | {value} |"
        if engine == "vllm"
        else f'sglang:cuda_graph_passes_total{{mode="decode_cuda_graph"}} {value}'
    )
    with pytest.raises(AssertionError, match="Invalid native graph count"):
        installed_serving.native_runtime_graph_count(engine, text)
