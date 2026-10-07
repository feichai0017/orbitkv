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
    engine, tier, model, tmp_path, orbitkv_transfer_backend
):
    assert Path(model).is_dir(), "release gate requires --model with a local dense model"
    env = isolated_environment(dict(os.environ), tmp_path)
    env["ORBITKV_TRANSFER_BACKEND"] = orbitkv_transfer_backend
    before = probe_installation(
        sys.executable, engine, env, tmp_path, "installed-before", native=True
    )
    try:
        run_cache_plan(engine, tier, model, tmp_path, env)
    finally:
        after = probe_installation(sys.executable, engine, env, tmp_path, "installed-after")
        assert before["distributions"] == after["distributions"], (
            "Installed engine or OrbitKV files changed during qualification"
        )


def run_cache_plan(engine, tier, model, directory, env):
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
    command, cache_options = engine_command(
        engine, sys.executable, model, engine_port, env["ORBITKV_TRANSFER_BACKEND"]
    )
    manager_args = manager_command(sys.executable, port, http_port, tier, directory)
    phases = {}

    def complete(label, prompt):
        result = generate(engine, engine_url, model, prompt)
        phases[label] = result
        (directory / "responses.json").write_text(json.dumps(phases, indent=2) + "\n")
        return result

    with service(command, engine_url, env, directory, f"{engine}-native"):
        baseline_cold = complete("native-cold", tokens)
        baseline_warm = complete("native-hbm", tokens)
        baseline_partial = complete("native-partial", extended)
    env["ORBITKV_PORT"] = str(port)
    env["ORBITKV_SGLANG_ENDPOINT"] = f"unix:///tmp/orbitkv-{port}.sock"
    snapshots = {}
    with service(manager_args, manager_url, env, directory, "manager"):
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
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "engine": engine,
                    "tier": tier,
                    "transfer_backend": env["ORBITKV_TRANSFER_BACKEND"],
                    "model": model,
                    "profile": "dense TP=1 PP=1 eager same-host",
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
