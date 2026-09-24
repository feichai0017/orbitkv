"""Codec controls and evidence must stay usable without native/GPU imports."""

import json
from argparse import ArgumentTypeError, Namespace
from types import SimpleNamespace

import pytest

from benches.launch import STORAGE_CODECS, configure, storage_codec_budget
from benches.metrics import codec_summary, delta, measure, metrics
from benches.report import compare_outputs


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("codec", STORAGE_CODECS)
@pytest.mark.parametrize("ssd_gib", [0, 4])
def test_codec_launch_uses_same_engine_capacity_and_explicit_manager_policy(
    tmp_path, monkeypatch, engine, codec, ssd_gib
):
    monkeypatch.setattr("benches.launch.free_port", lambda: 23456)
    monkeypatch.setenv("ORBITKV_CACHE_MANAGER_BINARY", "/prebuilt/manager")
    monkeypatch.setenv("ORBITKV_NVCOMP_LIBRARY", "/runtime/libnvcomp.so.5")
    args = Namespace(
        engine=engine,
        backend="orbitkv",
        model=tmp_path / "model",
        output=tmp_path,
        host_gib=1,
        ssd_gib=ssd_gib,
        ssd_path=tmp_path / "private/cache.bin",
        ssd_backend="uring",
        ssd_write_policy="all",
        storage_codec=codec,
        storage_codec_budget=64 * 1024**2,
        cache_protected_percent=0,
        query_budget_gib=None,
        queue_warmup="off",
        prepare_requests="off",
        trace_transfers=False,
        read_batch_mib=0,
        read_timeout_ms=0,
        read_max_batches=0,
        workload="sustained",
        lengths=[4096],
        gpu_tokens=8192,
        prefill_tokens=8192,
        orbitkv_transfer_backend=None,
    )
    launch = configure(args, 147456)
    command = launch.manager_command
    assert command[0] == "/prebuilt/manager"
    assert command[command.index("--storage-codec") + 1] == codec
    assert command[command.index("--storage-codec-budget") + 1] == str(64 * 1024**2)
    assert launch.env["ORBITKV_NVCOMP_LIBRARY"] == "/runtime/libnvcomp.so.5"
    assert launch.backend_configuration["storage_codec"] == codec
    assert ("--ssd-cache-path" in command) == bool(ssd_gib)
    if ssd_gib:
        assert command[command.index("--ssd-cache-path") + 1] == str(args.ssd_path)
    capacity_flag = "--kv-cache-memory-bytes" if engine == "vllm" else "--max-total-tokens"
    capacity = 147456 * 8192 if engine == "vllm" else 8192
    assert launch.command[launch.command.index(capacity_flag) + 1] == str(capacity)


def test_codec_budget_accepts_binary_units_and_rejects_out_of_range():
    for value, expected in (("64mb", 67108864), ("4096", 4096), ("1.5 GB", 1610612736)):
        assert storage_codec_budget(value) == expected
    for value in ("0", "4095", "4gb", "-1", "nan", "inf", "64mib"):
        with pytest.raises(ArgumentTypeError):
            storage_codec_budget(value)


def test_codec_labels_batch_histograms_and_retained_workspace_are_not_combined(monkeypatch):
    monkeypatch.setattr(
        "benches.metrics.requests.get",
        lambda *a, **kw: SimpleNamespace(
            text="\n".join(
                [
                    'orbitkv_storage_codec_bytes_total{representation="logical"} 4096',
                    'orbitkv_storage_codec_bytes_total{representation="stored"} 1024',
                    'orbitkv_storage_codec_transfer_bytes_total{direction="d2h"} 2048',
                    'orbitkv_storage_codec_transfer_bytes_total{direction="h2d"} 1024',
                    'orbitkv_storage_codec_duration_seconds_sum{operation="encode"} 0.2',
                    'orbitkv_storage_codec_duration_seconds_sum{operation="decode"} 0.1',
                    'orbitkv_storage_codec_batches_total{operation="encode"} 2',
                    'orbitkv_storage_codec_batches_total{operation="decode"} 3',
                    'orbitkv_storage_codec_batch_segments_sum{operation="encode"} 64',
                    'orbitkv_storage_codec_batch_segments_count{operation="encode"} 2',
                    'orbitkv_storage_codec_batch_segments_bucket{operation="encode",le="64"} 2',
                    'orbitkv_storage_codec_skips_total{reason="range_ratio_or_budget"} 4',
                    "orbitkv_storage_codec_workspace_bytes 67108864",
                    "orbitkv_storage_codec_workspace_allocations_total 1",
                    "orbitkv_storage_codec_reserved_bytes 0",
                ]
            ),
            raise_for_status=lambda: None,
        ),
    )
    after = metrics("http://manager")
    before = {"orbitkv_storage_codec_workspace_allocations_total": 1}
    row = codec_summary(
        [
            {
                "manager_before": before,
                "manager_after": after,
                "manager_delta": delta(before, after),
                "sampled_peak_bytes": {"orbitkv_storage_codec_reserved_bytes": 134217728},
            }
        ]
    )
    assert row["encoded_publication_stored_fraction"] == 0.25
    assert "orbitkv_storage_codec_bytes_total" not in row
    assert row["orbitkv_storage_codec_duration_seconds_sum_encode"] == 0.2
    assert row["orbitkv_storage_codec_duration_seconds_sum_decode"] == 0.1
    assert row["orbitkv_storage_codec_batches_total_encode"] == 2
    assert row["orbitkv_storage_codec_batches_total_decode"] == 3
    assert row["orbitkv_storage_codec_batch_segments_sum_encode"] == 64
    assert row["orbitkv_storage_codec_batch_segments_count_encode"] == 2
    assert row["orbitkv_storage_codec_workspace_allocations_total"] == 0
    assert row["sampled_peak_codec_reserved_bytes"] == 134217728
    assert row["max_codec_reserved_bytes_after"] == 0
    assert row["max_codec_workspace_bytes_after"] == 67108864
    assert not any("bucket" in key for key in row)
    old = codec_summary([{"manager_delta": {"orbitkv_storage_codec_bytes_total_logical": 4096}}])
    assert old["encoded_publication_stored_fraction"] is None
    assert old["sampled_peak_codec_workspace_bytes"] is None
    assert "orbitkv_storage_codec_workspace_allocations_total" not in old
    assert "orbitkv_storage_codec_batches_total_encode" not in old


def test_none_reference_retains_output_drift_and_requires_verified_inputs(tmp_path):
    args = {"engine": "vllm", "workload": "sustained", "seed": 42, "storage_codec": "none"}
    control = tmp_path / "none"
    control.mkdir()
    samples = [
        {"concurrency": 4, "index": i, "length": 64, "prompt_sha256": str(i), "text": "same"}
        for i in range(2)
    ]
    (control / "samples.jsonl").write_text("\n".join(map(json.dumps, samples)))
    reference = {"directory": str(control), "manifest": {"arguments": args}}
    run = {
        "directory": str(tmp_path),
        "manifest": {"arguments": {**args, "storage_codec": "fp8"}},
        "summary": [{"concurrency": 4}],
    }
    samples[1]["text"] = "changed"
    samples.append({**samples[0], "index": 2})
    (tmp_path / "samples.jsonl").write_text("\n".join(map(json.dumps, samples)))
    comparison = compare_outputs(run, reference)
    assert comparison["compared_requests"] == 2
    assert comparison["output_mismatches"] == 1
    assert comparison["run_uncompared_requests"] == 1
    assert run["summary"][0]["reference_output_mismatches"] == 1
    del samples[0]["prompt_sha256"]
    (tmp_path / "samples.jsonl").write_text("\n".join(map(json.dumps, samples)))
    assert compare_outputs(run, reference)["unverified_input_requests"] == 1
    samples[1]["prompt_sha256"] = "different"
    (tmp_path / "samples.jsonl").write_text("\n".join(map(json.dumps, samples)))
    with pytest.raises(ValueError, match="prompt differs"):
        compare_outputs(run, reference)
    run["manifest"]["arguments"]["seed"] = 43
    with pytest.raises(ValueError, match="different seed"):
        compare_outputs(run, reference)


def test_idle_retained_codec_workspace_does_not_prevent_measurement_drain(monkeypatch):
    observed = {
        "orbitkv_storage_codec_workspace_bytes": 67108864,
        "orbitkv_storage_codec_reserved_bytes": 0,
    }
    monkeypatch.setattr("benches.metrics.metrics", lambda url: observed)
    monkeypatch.setattr("benches.metrics.time.sleep", lambda seconds: None)
    ticks = iter(range(0, 1000, 10))
    monkeypatch.setattr("benches.metrics.time.monotonic", lambda: next(ticks))
    with measure("engine", "manager", 0) as measurement:
        pass
    assert measurement["manager_after"] == observed
    summary = codec_summary([measurement])
    assert summary["max_codec_reserved_bytes_after"] == 0
    assert summary["max_codec_workspace_bytes_after"] == 67108864
