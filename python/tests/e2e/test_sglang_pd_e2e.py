"""SGLang P/D correctness through OrbitKV's Rust/TENT payload engine."""

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
from tests.support.paths import PYTHON_ROOT

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


def test_sglang_pd_tent_matches_monolithic(request, tmp_path):
    pytest.importorskip("sglang")
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("SGLang P/D qualification requires two visible GPUs")

    model = Path(request.config.getoption("--model"))
    if not model.exists():
        pytest.skip("pass --model with a local model path")

    plugin_dir = tmp_path / "orbitkv_source_plugin-0.0.dist-info"
    plugin_dir.mkdir()
    (plugin_dir / "METADATA").write_text("Name: orbitkv-source-plugin\nVersion: 0.0\n")
    (plugin_dir / "entry_points.txt").write_text(
        "[sglang.srt.plugins]\norbitkv = orbitkv.sglang.plugin:register\n"
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PYTHON_ROOT), str(tmp_path), env.get("PYTHONPATH", "")]
    )
    env["ORBITKV_SGLANG_TENT"] = "1"
    env["MC_FORCE_TCP"] = "1"

    prefill_port = find_available_port()
    decode_port = find_available_port()
    router_port = find_available_port()
    bootstrap_port = find_available_port()
    logs = {
        "prefill": tmp_path / "sglang-pd-prefill.log",
        "decode": tmp_path / "sglang-pd-decode.log",
        "router": tmp_path / "sglang-pd-router.log",
        "monolithic": tmp_path / "sglang-monolithic.log",
    }
    processes: list[subprocess.Popen] = []

    common = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model),
        "--trust-remote-code",
        "--load-format",
        request.config.getoption("--sglang-load-format"),
        "--host",
        "127.0.0.1",
        "--context-length",
        "2048",
        "--max-total-tokens",
        "4096",
        "--max-prefill-tokens",
        "2048",
        "--random-seed",
        "42",
        "--enable-deterministic-inference",
    ]

    def launch(
        name: str,
        command: list[str],
        launch_env: dict[str, str] | None = None,
    ):
        log_file = logs[name].open("w")
        try:
            process = subprocess.Popen(
                command,
                env=env if launch_env is None else launch_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_file.close()
        processes.append(process)
        return process

    def wait_ready(name: str, process: subprocess.Popen, port: int) -> None:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(f"{name} exited during startup:\n{logs[name].read_text()[-8000:]}")
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        pytest.fail(f"{name} startup timed out:\n{logs[name].read_text()[-8000:]}")

    def stop_all() -> None:
        while processes:
            process = processes.pop()
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

    payload = {
        "text": (
            "A deterministic prefill/decode cache transfer must preserve every attention "
            "state byte before decode begins. Explain the invariant in one sentence."
        ),
        "sampling_params": {"temperature": 0, "max_new_tokens": 16},
    }

    try:
        pd_args = [
            "--disaggregation-transfer-backend",
            "mooncake",
            "--disaggregation-bootstrap-port",
            str(bootstrap_port),
        ]
        prefill = launch(
            "prefill",
            common
            + [
                "--port",
                str(prefill_port),
                "--nccl-port",
                str(find_available_port()),
                "--base-gpu-id",
                "0",
                "--disaggregation-mode",
                "prefill",
            ]
            + pd_args,
        )
        decode = launch(
            "decode",
            common
            + [
                "--port",
                str(decode_port),
                "--nccl-port",
                str(find_available_port()),
                "--base-gpu-id",
                "1",
                "--disaggregation-mode",
                "decode",
            ]
            + pd_args,
        )
        wait_ready("prefill", prefill, prefill_port)
        wait_ready("decode", decode, decode_port)

        router = launch(
            "router",
            [
                sys.executable,
                "-m",
                "sglang_router.launch_router",
                "--pd-disaggregation",
                "--mini-lb",
                "--prefill",
                f"http://127.0.0.1:{prefill_port}",
                "--decode",
                f"http://127.0.0.1:{decode_port}",
                "--host",
                "127.0.0.1",
                "--port",
                str(router_port),
            ],
        )
        wait_ready("router", router, router_port)
        pd_response = requests.post(
            f"http://127.0.0.1:{router_port}/generate",
            json=payload,
            timeout=180,
        )
        pd_response.raise_for_status()
        pd_text = pd_response.json()["text"]

        for name in ("prefill", "decode"):
            assert "OrbitKV installed the Rust TENT payload engine" in logs[name].read_text()
        stop_all()

        monolithic_port = find_available_port()
        monolithic_env = dict(env)
        monolithic_env.pop("ORBITKV_SGLANG_TENT")
        monolithic = launch(
            "monolithic",
            common
            + [
                "--port",
                str(monolithic_port),
                "--nccl-port",
                str(find_available_port()),
                "--base-gpu-id",
                "0",
            ],
            monolithic_env,
        )
        wait_ready("monolithic", monolithic, monolithic_port)
        control_response = requests.post(
            f"http://127.0.0.1:{monolithic_port}/generate",
            json=payload,
            timeout=180,
        )
        control_response.raise_for_status()
        assert pd_text == control_response.json()["text"]
    finally:
        stop_all()
