"""Installed engine processes, native controls and resource-drain oracles."""

import contextlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
import requests

from tests.support.cache_manager import find_available_port
from tests.support.metrics import fetch_orbitkv_metrics

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
        yield process
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
        if name.startswith("manager"):
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


def engine_command(engine, python, model, port, transfer_backend="direct"):
    if engine == "vllm":
        command = [
            str(python),
            "-I",
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
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
                    "kv_connector_extra_config": {"orbitkv.transfer_backend": transfer_backend},
                }
            ),
        ]
    else:
        command = [
            str(python),
            "-I",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
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
    return command, cache_options


def manager_command(python, port, http_port, tier, directory):
    command = [
        str(Path(python).parent / "orbitkv-cache-manager"),
        "--addr",
        f"127.0.0.1:{port}",
        "--http-addr",
        f"127.0.0.1:{http_port}",
        "--pool-size",
        "1gb",
    ]
    if tier == "ssd":
        command += [
            "--ssd-cache-path",
            str(directory / "cache.bin"),
            "--ssd-cache-capacity",
            "2gb",
            "--ssd-backend",
            "uring",
            "--ssd-read-path",
            "uring",
        ]
    return command


def probe_installation(python, engine, env, directory, name, native=False):
    script = Path(__file__).with_name("installed_artifacts.py")
    command = [str(python), "-I", str(script), engine]
    if native:
        command.append("--native")
    probe = subprocess.run(
        command, cwd=directory, env=env, text=True, capture_output=True, timeout=60
    )
    (directory / f"{name}-probe.log").write_text(probe.stdout + probe.stderr)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    result = json.loads(probe.stdout)
    (directory / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def complete(engine, url, model, prompt):
    if engine == "vllm":
        path = "/v1/completions"
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": 0,
            "ignore_eos": True,
            "logprobs": 1,
        }
    else:
        path = "/generate"
        body = {
            "input_ids": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
            "return_logprob": True,
        }
    response = requests.post(f"{url}{path}", json=body, timeout=90)
    response.raise_for_status()
    return response.json()


def cached_tokens(engine, result):
    if engine == "vllm":
        return result["usage"]["prompt_tokens_details"]["cached_tokens"]
    return result["meta_info"]["cached_tokens"]


def compare_output(engine, actual, expected):
    if engine == "vllm":
        assert actual["choices"][0]["text"] == expected["choices"][0]["text"]
        assert (
            actual["choices"][0]["logprobs"]["tokens"]
            == expected["choices"][0]["logprobs"]["tokens"]
        )
    else:
        assert actual["text"] == expected["text"]
        assert actual["output_ids"] == expected["output_ids"]
