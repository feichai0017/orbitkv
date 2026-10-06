"""Released-engine cache qualification using only an installed, non-editable wheel."""

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from tests.support.cache_manager import evict_dram_after_ssd_writes, find_available_port
from tests.support.installed_artifacts import isolated_environment
from tests.support.metrics import fetch_orbitkv_metrics, fetch_vllm_prefix_cache_hits

pytestmark = [pytest.mark.release_smoke, pytest.mark.gpu]
DRAIN_GAUGES = (
    "orbitkv_query_reserved_bytes",
    "orbitkv_inflight_bytes",
    "orbitkv_ssd_write_inflight",
    "orbitkv_ssd_write_queue_pending",
    "orbitkv_ssd_prefetch_inflight",
    "orbitkv_ssd_read_pinned_bytes",
    "orbitkv_ssd_gpu_staging_bytes",
)


def process_group_members(group: int) -> list[int]:
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group:
                members.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return sorted(members)


@contextlib.contextmanager
def service(command, url, env, directory, name):
    log_path = directory / f"{name}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(f"{name} exited: {log_path.read_text()[-8000:]}")
            try:
                if requests.get(f"{url}/health", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        else:
            pytest.fail(f"{name} startup timed out: {log_path.read_text()[-8000:]}")
        yield
    finally:
        stop_started = time.monotonic()
        with contextlib.suppress(ProcessLookupError):
            process.send_signal(signal.SIGTERM)
        deadline = stop_started + 90
        while time.monotonic() < deadline:
            process.poll()
            if not process_group_members(process.pid):
                break
            time.sleep(0.1)
        remaining = process_group_members(process.pid)
        if remaining:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
        cleanup = {
            "command": command,
            "pid": process.pid,
            "exit_code": process.returncode,
            "sigterm_target": process.pid,
            "shutdown_seconds": time.monotonic() - stop_started,
            "forced_kill": bool(remaining),
            "before_forced_kill": remaining,
            "remaining_processes": process_group_members(process.pid),
        }
        (directory / f"{name}-cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")
        assert not remaining, f"{name} required forced cleanup: {cleanup}"
        assert not cleanup["remaining_processes"], cleanup
        if name == "manager":
            assert process.returncode == 0, cleanup
        else:
            assert process.returncode in (0, -signal.SIGTERM), cleanup


def wait_for_drain(http_port: int) -> dict[str, float]:
    deadline = time.monotonic() + 30
    while True:
        metrics = fetch_orbitkv_metrics(http_port)
        if not any(metrics.get(name, 0) for name in DRAIN_GAUGES):
            return metrics
        assert time.monotonic() < deadline, metrics
        time.sleep(0.1)


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("tier", ["dram", "ssd"])
def test_installed_wheel_recovers_after_engine_restart(engine, tier, model, tmp_path):
    assert Path(model).is_dir(), "release gate requires --model with a local dense model"
    env = isolated_environment(dict(os.environ), tmp_path)
    probe_script = Path(__file__).parents[1] / "support" / "installed_artifacts.py"

    def snapshot(name, native=False):
        command = [sys.executable, "-I", str(probe_script), engine]
        if native:
            command.append("--native")
        probe = subprocess.run(
            command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60
        )
        (tmp_path / f"{name}-probe.log").write_text(probe.stdout + probe.stderr)
        assert probe.returncode == 0, probe.stdout + probe.stderr
        result = json.loads(probe.stdout)
        (tmp_path / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    before = snapshot("installed-before", native=True)
    try:
        run_cache_plan(engine, tier, model, tmp_path, env)
    finally:
        after = snapshot("installed-after")
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
    manager = str(Path(sys.executable).parent / "orbitkv-cache-manager")
    subprocess.run([manager, "--help"], cwd=directory, env=env, check=True, capture_output=True)
    port, http_port, engine_port = (find_available_port() for _ in range(3))
    manager_url = f"http://127.0.0.1:{http_port}"
    engine_url = f"http://127.0.0.1:{engine_port}"
    manager_command = [
        manager,
        "--addr",
        f"127.0.0.1:{port}",
        "--http-addr",
        f"127.0.0.1:{http_port}",
        "--pool-size",
        "1gb",
    ]
    if tier == "ssd":
        manager_command += [
            "--ssd-cache-path",
            str(directory / "cache.bin"),
            "--ssd-cache-capacity",
            "2gb",
            "--ssd-backend",
            "uring",
            "--ssd-read-path",
            "uring",
        ]
    if engine == "vllm":
        env["VLLM_SERVER_DEV_MODE"] = "1"
        command = [
            sys.executable,
            "-I",
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(engine_port),
            "--max-model-len",
            "2048",
            "--max-num-seqs",
            "4",
            "--gpu-memory-utilization",
            "0.4",
            "--enforce-eager",
            "--seed",
            "42",
            "--enable-prefix-caching",
            "--enable-prompt-tokens-details",
            "--attention-backend",
            "FLASH_ATTN",
            "--generation-config",
            "vllm",
        ]
        cache_options = [
            "--kv-transfer-config",
            json.dumps(
                {
                    "kv_connector": "OrbitKVConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "orbitkv.vllm",
                }
            ),
        ]
    else:
        command = [
            sys.executable,
            "-I",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(engine_port),
            "--nccl-port",
            str(find_available_port()),
            "--context-length",
            "2048",
            "--max-total-tokens",
            "4096",
            "--mem-fraction-static",
            "0.4",
            "--page-size",
            "64",
            "--disable-cuda-graph",
            "--random-seed",
            "42",
            "--enable-deterministic-inference",
        ]
        cache_options = [
            "--enable-unified-cache-external-linker",
            "--radix-cache-backend",
            "orbitkv",
        ]
    phases = {}

    def complete(label, prompt):
        if engine == "vllm":
            response = requests.post(
                f"{engine_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 8,
                    "temperature": 0,
                    "ignore_eos": True,
                    "logprobs": 1,
                },
                timeout=90,
            )
        else:
            response = requests.post(
                f"{engine_url}/generate",
                json={
                    "input_ids": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 8,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                },
                timeout=90,
            )
        response.raise_for_status()
        result = response.json()
        phases[label] = result
        (directory / "responses.json").write_text(json.dumps(phases, indent=2) + "\n")
        return result

    def cached_tokens(result):
        if engine == "vllm":
            return result["usage"]["prompt_tokens_details"]["cached_tokens"]
        return result["meta_info"]["cached_tokens"]

    def compare(actual, expected):
        if engine == "vllm":
            assert actual["choices"][0]["text"] == expected["choices"][0]["text"]
            assert (
                actual["choices"][0]["logprobs"]["tokens"]
                == expected["choices"][0]["logprobs"]["tokens"]
            )
        else:
            assert actual["text"] == expected["text"]
            assert actual["output_ids"] == expected["output_ids"]

    with service(command, engine_url, env, directory, f"{engine}-native"):
        baseline_cold = complete("native-cold", tokens)
        baseline_warm = complete("native-hbm", tokens)
        baseline_partial = complete("native-partial", extended)
    env["ORBITKV_PORT"] = str(port)
    env["ORBITKV_SGLANG_ENDPOINT"] = f"unix:///tmp/orbitkv-{port}.sock"
    snapshots = {}
    with service(manager_command, manager_url, env, directory, "manager"):
        with service(command + cache_options, engine_url, env, directory, f"{engine}-cold"):
            cold = complete("cache-cold", tokens)
            compare(cold, baseline_cold)
            assert cached_tokens(cold) == 0, cold
            snapshots["after_cold"] = wait_for_drain(http_port)
            assert snapshots["after_cold"].get("orbitkv_save_bytes_total", 0) > 0
            assert snapshots["after_cold"].get("orbitkv_hll_total_requests", 0) > 0
            before_native_hits = (
                fetch_vllm_prefix_cache_hits(engine_port) if engine == "vllm" else 0
            )
            native = complete("cache-native-hbm", tokens)
            compare(native, baseline_warm)
            assert cached_tokens(native) >= 704, native
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
            compare(full, baseline_warm)
            assert cached_tokens(full) >= 704, full
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
            compare(partial, baseline_partial)
            assert 64 <= cached_tokens(partial) < len(extended) - 64, partial
            snapshots["after_partial"] = wait_for_drain(http_port)
            assert snapshots["after_partial"].get("orbitkv_load_bytes_total", 0) > snapshots[
                "before_partial"
            ].get("orbitkv_load_bytes_total", 0), snapshots
        final = wait_for_drain(http_port)
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
