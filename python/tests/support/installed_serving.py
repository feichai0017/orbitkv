"""Installed engine processes, native controls and resource-drain oracles."""

import contextlib
import json
import math
import os
import re
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
    "orbitkv_query_control_interests",
    "orbitkv_shard_query_active",
    "orbitkv_shard_query_holds",
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
    primary_error = None
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
    except BaseException as error:
        primary_error = error
        raise
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
        errors = []
        if remaining:
            errors.append("forced cleanup required")
        if cleanup["remaining_processes"]:
            errors.append("owned processes remain")
        accepted = (0,) if name.startswith("manager") else (0, -signal.SIGTERM)
        if process.returncode not in accepted:
            errors.append(f"unexpected exit code {process.returncode}")
        cleanup["errors"] = errors
        cleanup["primary_failure"] = (
            {"type": type(primary_error).__name__, "message": str(primary_error)}
            if primary_error is not None
            else None
        )
        (directory / f"{name}-cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")
        if errors and primary_error is None:
            raise AssertionError(f"{name} cleanup failed: {cleanup}")


def wait_for_drain(http_port: int) -> dict[str, float]:
    deadline = time.monotonic() + 30
    required = {"orbitkv_query_reserved_bytes"}
    while True:
        metrics = fetch_orbitkv_metrics(http_port)
        required.update(name for name in DRAIN_GAUGES if name in metrics)
        missing = required.difference(metrics)
        assert not missing, f"Missing resource-drain gauges: {sorted(missing)}"
        assert all(math.isfinite(metrics[name]) and metrics[name] >= 0 for name in required), (
            f"Invalid resource-drain gauges: {metrics}"
        )
        if all(metrics[name] == 0 for name in required):
            return metrics
        assert time.monotonic() < deadline, metrics
        time.sleep(0.1)


def engine_command(
    engine, python, model, port, transfer_backend="direct", *, cuda_graph=False, query_control=False
):
    if query_control and engine != "vllm":
        raise ValueError("registered worker query control requires the vLLM profile")
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
                    "kv_connector_extra_config": {
                        "orbitkv.transfer_backend": transfer_backend,
                        **({"orbitkv.query_control": True} if query_control else {}),
                    },
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
            "--random-seed",
            "42",
            "--enable-deterministic-inference",
        ]
        cache_options = [
            "--enable-unified-cache-external-linker",
            "--radix-cache-backend",
            "orbitkv",
        ]
    if engine == "vllm":
        command += ["--shutdown-timeout", "30"]
        if cuda_graph:
            command += [
                "--compilation-config",
                json.dumps(
                    {"mode": 0, "cudagraph_mode": "FULL", "cudagraph_capture_sizes": [1, 2, 4]}
                ),
                "--cudagraph-metrics",
            ]
        else:
            command.append("--enforce-eager")
    elif cuda_graph:
        command += [
            "--cuda-graph-backend-decode",
            "full",
            "--cuda-graph-max-bs-decode",
            "4",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--enable-metrics",
        ]
    else:
        command.append("--disable-cuda-graph")
    return command, cache_options


def native_runtime_graph_count(engine, text):
    if engine == "vllm":
        rows = re.findall(
            r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*FULL\s*\|\s*(\S+)\s*\|",
            text,
        )
        count = 0
        for unpadded, padded, padding, value in rows:
            value = float(value)
            assert math.isfinite(value) and value >= 0 and value.is_integer(), (
                "Invalid native graph count",
                value,
            )
            assert int(padded) - int(unpadded) == int(padding), "Invalid native graph padding"
            # FULL token counts do not distinguish decode from one-token prefill.
            # Exclude long prefill; report these as one-token runtime observations.
            if int(unpadded) == 1:
                count += int(value)
        return count
    count = 0
    for labels, value in re.findall(
        r"^sglang:cuda_graph_passes_total\{([^}]*)\}\s+(\S+)\s*$", text, re.MULTILINE
    ):
        labels = dict(re.findall(r'(\w+)="([^"]*)"', labels))
        if labels.get("mode") != "decode_cuda_graph":
            continue
        value = float(value)
        assert math.isfinite(value) and value >= 0 and value.is_integer(), (
            "Invalid native graph count",
            value,
        )
        count += int(value)
    return count


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
