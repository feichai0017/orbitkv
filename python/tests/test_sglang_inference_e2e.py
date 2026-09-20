"""SGLang 0.5.20 server recovery from OrbitKV after flushing HiCache L1/L2."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from .conftest import find_available_port

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


def test_sglang_restores_prompt_after_hicache_flush(local_control_server, request, tmp_path):
    pytest.importorskip("sglang")
    model = request.config.getoption("--model")
    if not Path(model).exists():
        pytest.skip("pass --model with a local model path")
    port = find_available_port()
    base_url = f"http://127.0.0.1:{port}"
    extra_config = {
        "backend_name": "orbitkv",
        "module_path": "orbitkv.sglang.storage",
        "class_name": "OrbitKVHiCacheStorage",
        "endpoint": f"unix://{local_control_server.local_bootstrap_socket}",
        "allocator": "shm",
        "namespace": f"sglang-e2e:{model}",
    }
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--context-length",
        "2048",
        "--max-total-tokens",
        "4096",
        "--max-prefill-tokens",
        "2048",
        "--enable-hierarchical-cache",
        "--hicache-size",
        "1",
        "--page-size",
        "64",
        "--hicache-write-policy",
        "write_through",
        "--hicache-storage-prefetch-policy",
        "wait_complete",
        "--hicache-storage-backend",
        "dynamic",
        "--hicache-storage-backend-extra-config",
        json.dumps(extra_config),
        "--enable-cache-report",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parents[1]), env.get("PYTHONPATH", "")])
    log_path = tmp_path / "sglang.log"
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            cmd,
            env=env,
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

            prompt = (
                "A cache correctness test for long language model context. " * 28
                + "Conclude with one sentence about why checksums matter."
            )
            payload = {
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
            }
            first = requests.post(f"{base_url}/generate", json=payload, timeout=90)
            first.raise_for_status()
            # Wait for the write-through worker before clearing SGLang's L1/L2.
            time.sleep(3)
            flushed = requests.post(f"{base_url}/flush_cache?timeout=30", timeout=40)
            flushed.raise_for_status()
            second = requests.post(f"{base_url}/generate", json=payload, timeout=90)
            second.raise_for_status()
            first_json, second_json = first.json(), second.json()
            assert first_json["text"] == second_json["text"]
            assert second_json["meta_info"]["cached_tokens"] >= 64
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
