"""Released-engine cache qualification using only an installed, non-editable wheel."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pytest
import requests

from tests.support.cache_manager import evict_dram_after_ssd_writes, find_available_port
from tests.support.installed_artifacts import isolated_environment
from tests.support.installed_serving import (
    cached_tokens,
    compare_output,
    engine_command,
    manager_command,
    native_runtime_graph_count,
    probe_installation,
    service,
    wait_for_drain,
)
from tests.support.installed_serving import (
    complete as generate,
)
from tests.support.metrics import fetch_orbitkv_metrics, fetch_vllm_prefix_cache_hits

pytestmark = [pytest.mark.release_smoke, pytest.mark.gpu]


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("tier", ["dram", "ssd"])
def test_installed_wheel_recovers_after_engine_restart(
    engine, tier, model, tmp_path, orbitkv_transfer_backend, request
):
    assert Path(model).is_dir(), "release gate requires --model with a local dense model"
    env = isolated_environment(dict(os.environ), tmp_path)
    env["ORBITKV_TRANSFER_BACKEND"] = orbitkv_transfer_backend
    before = probe_installation(
        sys.executable, engine, env, tmp_path, "installed-before", native=True
    )
    cuda_graph = request.config.getoption("--release-cuda-graph")
    query_control = request.config.getoption("--release-query-control")
    try:
        run_cache_plan(
            engine, tier, model, tmp_path, env, cuda_graph=cuda_graph, query_control=query_control
        )
    finally:
        after = probe_installation(sys.executable, engine, env, tmp_path, "installed-after")
        assert before["distributions"] == after["distributions"], (
            "Installed engine or OrbitKV files changed during qualification"
        )


def run_cache_plan(engine, tier, model, directory, env, *, cuda_graph=False, query_control=False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    fragment = tokenizer.encode("An installed cache retains deterministic reusable context. ")
    tokens = (fragment * (769 // len(fragment) + 1))[:769]
    suffix = tokenizer.encode(" Fresh unseen context extends the saved prefix. " * 32)[:192]
    extended = tokens + suffix
    port, http_port, engine_port = (find_available_port() for _ in range(3))
    manager_url = f"http://127.0.0.1:{http_port}"
    engine_url = f"http://127.0.0.1:{engine_port}"
    env["VLLM_SERVER_DEV_MODE"] = "1"
    if cuda_graph:
        env["VLLM_LOG_STATS_INTERVAL"] = "1"
        env["VLLM_LOGGING_LEVEL"] = "DEBUG"
    command, cache_options = engine_command(
        engine,
        sys.executable,
        model,
        engine_port,
        env["ORBITKV_TRANSFER_BACKEND"],
        cuda_graph=cuda_graph,
        query_control=query_control,
    )
    manager_args = manager_command(sys.executable, port, http_port, tier, directory)
    if query_control:
        manager_args.append("--enable-query-control")
    phases, graph_observations = {}, {}
    active_log = directory / f"{engine}-native.log"

    def graph_snapshot():
        if engine == "vllm":
            text = active_log.read_text()
        else:
            response = requests.get(f"{engine_url}/metrics", timeout=5)
            response.raise_for_status()
            text = response.text
        return {
            "count": native_runtime_graph_count(engine, text),
            "observation_kind": (
                "vllm_one_token_full_runtime" if engine == "vllm" else "sglang_decode_graph_passes"
            ),
            "raw": text,
        }

    def complete(label, prompt):
        before = graph_snapshot() if cuda_graph else None
        result = generate(engine, engine_url, model, prompt)
        phases[label] = result
        (directory / "responses.json").write_text(json.dumps(phases, indent=2) + "\n")
        if cuda_graph:
            samples = [before]
            graph_observations[label] = samples
            deadline = time.monotonic() + 15
            while True:
                after = graph_snapshot()
                samples.append(after)
                (directory / "graph-observations.json").write_text(
                    json.dumps(graph_observations, indent=2) + "\n"
                )
                assert after["count"] >= before["count"], "Native graph counter regressed"
                if after["count"] - before["count"] >= 7:
                    break
                assert time.monotonic() < deadline, (
                    f"Fewer than seven native runtime graph observations for {label}",
                    samples,
                )
                time.sleep(0.1)
        return result

    with service(command, engine_url, env, directory, f"{engine}-native"):
        baseline_cold = complete("native-cold", tokens)
        baseline_warm = complete("native-hbm", tokens)
        baseline_partial = complete("native-partial", extended)
    env["ORBITKV_PORT"] = str(port)
    env["ORBITKV_SGLANG_ENDPOINT"] = f"unix:///tmp/orbitkv-{port}.sock"
    snapshots = {}
    with service(manager_args, manager_url, env, directory, "manager"):
        active_log = directory / f"{engine}-cold.log"
        with service(command + cache_options, engine_url, env, directory, f"{engine}-cold"):
            cold = complete("cache-cold", tokens)
            compare_output(engine, cold, baseline_cold)
            assert cached_tokens(engine, cold) == 0, cold
            snapshots["after_cold"] = wait_for_drain(http_port)
            assert snapshots["after_cold"].get("orbitkv_save_bytes_total", 0) > 0
            assert snapshots["after_cold"].get("orbitkv_hll_total_requests", 0) > 0
            before_native_hits = (
                fetch_vllm_prefix_cache_hits(engine_port) if engine == "vllm" else 0
            )
            native = complete("cache-native-hbm", tokens)
            compare_output(engine, native, baseline_warm)
            assert cached_tokens(engine, native) >= 704, native
            snapshots["after_native_hbm"] = wait_for_drain(http_port)
            for counter in ("orbitkv_load_bytes_total", "orbitkv_hll_total_requests"):
                assert snapshots["after_cold"].get(counter, 0) == snapshots["after_native_hbm"].get(
                    counter, 0
                ), (f"Native HBM reuse performed external cache work: {counter}", snapshots)
            if engine == "vllm":
                assert fetch_vllm_prefix_cache_hits(engine_port) > before_native_hits
        if tier == "ssd":
            evict_dram_after_ssd_writes(http_port)
            assert fetch_orbitkv_metrics(http_port).get("orbitkv_cache_resident_bytes", 0) == 0
        snapshots["before_restart"] = wait_for_drain(http_port)
        active_log = directory / f"{engine}-restart.log"
        with service(command + cache_options, engine_url, env, directory, f"{engine}-restart"):
            full = complete("cache-full-after-restart", tokens)
            compare_output(engine, full, baseline_warm)
            assert cached_tokens(engine, full) >= 704, full
            snapshots["after_full"] = wait_for_drain(http_port)
            assert snapshots["after_full"].get("orbitkv_load_bytes_total", 0) > snapshots[
                "before_restart"
            ].get("orbitkv_load_bytes_total", 0), snapshots
            reset_path = "/reset_prefix_cache" if engine == "vllm" else "/flush_cache?timeout=30"
            deadline = time.monotonic() + 30
            while True:
                reset = requests.post(f"{engine_url}{reset_path}", timeout=40)
                reset.raise_for_status()
                if engine != "vllm" or reset.json()["success"]:
                    break
                assert time.monotonic() < deadline, reset.text
                time.sleep(0.1)
            wait_for_drain(http_port)
            if tier == "ssd" and fetch_orbitkv_metrics(http_port).get(
                "orbitkv_cache_resident_bytes", 0
            ):
                evict_dram_after_ssd_writes(http_port)
            snapshots["before_partial"] = wait_for_drain(http_port)
            partial = complete("cache-partial", extended)
            compare_output(engine, partial, baseline_partial)
            assert 64 <= cached_tokens(engine, partial) < len(extended) - 64, partial
            snapshots["after_partial"] = wait_for_drain(http_port)
            assert snapshots["after_partial"].get("orbitkv_load_bytes_total", 0) > snapshots[
                "before_partial"
            ].get("orbitkv_load_bytes_total", 0), snapshots
            if engine == "vllm":
                deadline = time.monotonic() + 30
                while True:
                    reset = requests.post(
                        f"{engine_url}/reset_prefix_cache?reset_external=true", timeout=40
                    )
                    reset.raise_for_status()
                    if reset.json()["success"]:
                        break
                    assert time.monotonic() < deadline, reset.text
                    time.sleep(0.1)
                snapshots["before_external_reset_request"] = wait_for_drain(http_port)
                invalidated = complete("cache-after-external-reset", tokens)
                compare_output(engine, invalidated, baseline_cold)
                assert cached_tokens(engine, invalidated) == 0, invalidated
                snapshots["after_external_reset_request"] = wait_for_drain(http_port)
                assert (
                    snapshots["after_external_reset_request"]["orbitkv_load_bytes_total"]
                    == (snapshots["before_external_reset_request"]["orbitkv_load_bytes_total"])
                ), snapshots
                assert (
                    snapshots["after_external_reset_request"]["orbitkv_save_bytes_total"]
                    > (snapshots["before_external_reset_request"]["orbitkv_save_bytes_total"])
                ), snapshots
                reset = requests.post(f"{engine_url}/reset_prefix_cache", timeout=40)
                reset.raise_for_status()
                assert reset.json()["success"], reset.text
                if tier == "ssd" and fetch_orbitkv_metrics(http_port).get(
                    "orbitkv_cache_resident_bytes", 0
                ):
                    evict_dram_after_ssd_writes(http_port)
                snapshots["before_reset_generation_reuse"] = wait_for_drain(http_port)
                reused = complete("cache-reset-generation-reuse", tokens)
                compare_output(engine, reused, baseline_warm)
                assert cached_tokens(engine, reused) >= 704, reused
                snapshots["after_reset_generation_reuse"] = wait_for_drain(http_port)
                assert (
                    snapshots["after_reset_generation_reuse"]["orbitkv_load_bytes_total"]
                    > (snapshots["before_reset_generation_reuse"]["orbitkv_load_bytes_total"])
                ), snapshots
                if tier == "ssd":
                    assert (
                        snapshots["after_reset_generation_reuse"][
                            "orbitkv_ssd_prefetch_bytes_total"
                        ]
                        > snapshots["before_reset_generation_reuse"][
                            "orbitkv_ssd_prefetch_bytes_total"
                        ]
                    ), snapshots
        final = wait_for_drain(http_port)
        assert (
            f"backend={env['ORBITKV_TRANSFER_BACKEND']}" in (directory / "manager.log").read_text()
        )
        assert final.get("orbitkv_load_failures_total", 0) == 0, final
        if tier == "ssd":
            assert final.get("orbitkv_ssd_write_bytes_total", 0) > 0, final
            for start, end in (
                ("before_restart", "after_full"),
                ("before_partial", "after_partial"),
            ):
                assert snapshots[end].get("orbitkv_ssd_prefetch_bytes_total", 0) > snapshots[
                    start
                ].get("orbitkv_ssd_prefetch_bytes_total", 0), snapshots
                assert snapshots[end].get("orbitkv_ssd_cufile_read_bytes_total", 0) == snapshots[
                    start
                ].get("orbitkv_ssd_cufile_read_bytes_total", 0), snapshots
        if query_control:
            (directory / "query-control-final.json").write_text(
                json.dumps({"snapshots": snapshots, "final": final}, indent=2) + "\n"
            )
            assert final["orbitkv_shard_query_submissions_total"] > 0
            assert (
                "registered query handshake configured: nodes=1 ranks=1"
                in (directory / "vllm-restart.log").read_text()
            )
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "engine": engine,
                    "tier": tier,
                    "transfer_backend": env["ORBITKV_TRANSFER_BACKEND"],
                    "model": model,
                    "profile": f"dense TP=1 PP=1 {'FULL decode graph' if cuda_graph else 'eager'} same-host",
                    "graph_runtime_required": cuda_graph,
                    "native_query_control_required": query_control,
                    "native_graph_progress": {
                        label: samples[-1]["count"] - samples[0]["count"]
                        for label, samples in graph_observations.items()
                    },
                    "model_config_sha256": hashlib.sha256(
                        (Path(model) / "config.json").read_bytes()
                    ).hexdigest(),
                    "snapshots": snapshots,
                    "final": final,
                },
                indent=2,
            )
            + "\n"
        )
    assert not Path(f"/tmp/orbitkv-{port}.sock").exists(), "Manager left its UDS behind"
