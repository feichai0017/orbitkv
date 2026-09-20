"""SGLang restores GPU KV through OrbitKV's CUDA IPC data path."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from tests.support.cache_manager import find_available_port
from tests.support.metrics import fetch_orbitkv_metrics
from tests.support.paths import PYTHON_ROOT

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


def test_sglang_direct_gpu_cache_recovery(channel_server, request, tmp_path):
    pytest.importorskip("sglang")
    model = request.config.getoption("--model")
    if not Path(model).exists():
        pytest.skip("pass --model with a local model path")

    # A source checkout has no installed entry-point metadata. Publish only the
    # test plugin metadata into PYTHONPATH so every SGLang subprocess discovers it.
    plugin_dir = tmp_path / "orbitkv_source_plugin-0.0.dist-info"
    plugin_dir.mkdir()
    (plugin_dir / "METADATA").write_text("Name: orbitkv-source-plugin\nVersion: 0.0\n")
    (plugin_dir / "entry_points.txt").write_text(
        "[sglang.srt.plugins]\norbitkv = orbitkv.sglang.plugin:register\n"
    )

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--context-length",
        "2048",
        "--max-total-tokens",
        "4096",
        "--max-prefill-tokens",
        "2048",
        "--random-seed",
        "42",
        "--enable-deterministic-inference",
        "--page-size",
        "64",
        "--radix-cache-backend",
        "orbitkv",
        "--enable-unified-cache-external-linker",
        "--enable-cache-report",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PYTHON_ROOT), str(tmp_path), env.get("PYTHONPATH", "")]
    )
    env["ORBITKV_SGLANG_ENDPOINT"] = f"unix://{channel_server.bootstrap_socket}"
    env["FLASHINFER_WORKSPACE_BASE"] = str(tmp_path / "flashinfer")
    log_path = tmp_path / "sglang-direct.log"

    def start_server(fingerprint: str | None = None) -> tuple[subprocess.Popen, str]:
        port = find_available_port()
        base_url = f"http://127.0.0.1:{port}"
        launch_cmd = list(cmd)
        launch_cmd[launch_cmd.index("--port") + 1] = str(port)
        launch_env = dict(env)
        if fingerprint is not None:
            launch_env["ORBITKV_MODEL_FINGERPRINT"] = fingerprint
        with log_path.open("a") as log_file:
            process = subprocess.Popen(
                launch_cmd,
                env=launch_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(f"SGLang exited during startup:\n{log_path.read_text()[-8000:]}")
                try:
                    if requests.get(f"{base_url}/health", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                pytest.fail(f"SGLang startup timed out:\n{log_path.read_text()[-8000:]}")
            return process, base_url
        except BaseException:
            stop_server(process)
            raise

    def stop_server(process: subprocess.Popen) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    prompt = "A CUDA IPC cache correctness test for a long context. " * 36
    payload = {
        "text": prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
    }
    process, base_url = start_server()
    try:
        first = requests.post(f"{base_url}/generate", json=payload, timeout=90)
        first.raise_for_status()
        time.sleep(3)
        flushed = requests.post(f"{base_url}/flush_cache?timeout=30", timeout=40)
        flushed.raise_for_status()
        second = requests.post(f"{base_url}/generate", json=payload, timeout=90)
        second.raise_for_status()
        assert first.json()["text"] == second.json()["text"]
        assert second.json()["meta_info"]["cached_tokens"] >= 64
    finally:
        stop_server(process)

    # Keep the Cache Manager alive while SGLang's HBM prefix tree disappears.
    before_restart_load = fetch_orbitkv_metrics(channel_server.http_port).get(
        "orbitkv_load_bytes_total", 0
    )
    process, base_url = start_server()
    try:
        third = requests.post(f"{base_url}/generate", json=payload, timeout=90)
        third.raise_for_status()
        assert third.json()["meta_info"]["cached_tokens"] >= 64
        after_restart_load = fetch_orbitkv_metrics(channel_server.http_port).get(
            "orbitkv_load_bytes_total", 0
        )
        assert after_restart_load > before_restart_load, "Cache Manager did not restore GPU KV"
    finally:
        stop_server(process)

    # A changed model identity must miss even with identical page hashes and the
    # same model, sampling seed, and Cache Manager process.
    process, base_url = start_server(fingerprint="9" * 64)
    try:
        cold = requests.post(f"{base_url}/generate", json=payload, timeout=90)
        cold.raise_for_status()
        assert cold.json()["meta_info"]["cached_tokens"] == 0
        outputs = [response.json()["text"] for response in (first, second, third, cold)]
        assert len(set(outputs)) == 1, outputs
        assert log_path.read_text().count("OrbitKV direct GPU linker registered") >= 3
    finally:
        stop_server(process)
