"""SGLang P/D correctness through OrbitKV's Rust/TENT payload engine."""

from __future__ import annotations

import concurrent.futures
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

from tests.support.cache_manager import find_available_port
from tests.support.metrics import fetch_orbitkv_metrics
from tests.support.paths import PYTHON_ROOT

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


@pytest.mark.parametrize("channel_server", [{"tier": "dram", "pool_size": "512mb"}], indirect=True)
def test_sglang_pd_tent_and_external_cache_match_monolithic(channel_server, request, tmp_path):
    pytest.importorskip("sglang")
    torch = pytest.importorskip("torch")
    device_count = torch.cuda.device_count()
    if device_count < 1:
        pytest.skip("SGLang P/D requires a visible GPU")
    decode_device = 1 if device_count > 1 else 0
    request.node.user_properties.append(
        ("pd_topology", "two-gpu-tcp" if decode_device else "same-gpu-tcp")
    )

    model = Path(request.config.getoption("--model"))
    if not model.exists():
        pytest.skip("pass --model with a local model path")

    plugin_dir = tmp_path / "orbitkv_source_plugin-0.0.dist-info"
    plugin_dir.mkdir()
    (plugin_dir / "METADATA").write_text("Name: orbitkv-source-plugin\nVersion: 0.0\n")
    (plugin_dir / "entry_points.txt").write_text(
        "[sglang.srt.plugins]\norbitkv_fault_gate = tests.support.pd_faults:register_sglang\n"
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PYTHON_ROOT), str(tmp_path), env.get("PYTHONPATH", "")]
    )
    env["SGLANG_MOONCAKE_TRANSFER_ENGINE"] = "orbitkv"
    env["SGLANG_PLUGINS"] = "orbitkv_fault_gate"
    env["ORBITKV_SGLANG_ENDPOINT"] = f"unix://{channel_server.bootstrap_socket}"
    env["ORBITKV_TRANSFER_BACKEND"] = request.config.getoption("--orbitkv-transfer-backend")
    env["MC_FORCE_TCP"] = "1"
    fault_root = tmp_path / "faults"
    fault_root.mkdir()
    env["ORBITKV_TEST_PD_FAULT_DIR"] = str(fault_root)
    env["SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT"] = "1"

    logs = {
        "prefill": tmp_path / "sglang-pd-prefill.log",
        "decode": tmp_path / "sglang-pd-decode.log",
        "router": tmp_path / "sglang-pd-router.log",
        "monolithic": tmp_path / "sglang-monolithic.log",
    }
    processes: list[subprocess.Popen] = []
    serving_ports: dict[str, int] = {}

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
        "--page-size",
        "64",
        "--enable-cache-report",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.8",
    ]

    def launch(
        name: str,
        command: list[str],
        launch_env: dict[str, str] | None = None,
    ):
        log_file = logs[name].open("a")
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

    def start_pd() -> int:
        prefill_port = find_available_port()
        decode_port = find_available_port()
        serving_ports["decode"] = decode_port
        router_port = find_available_port()
        bootstrap_port = find_available_port()
        pd_args = [
            "--disaggregation-transfer-backend",
            "mooncake",
            "--disaggregation-bootstrap-port",
            str(bootstrap_port),
            "--radix-cache-backend",
            "orbitkv",
            "--enable-unified-cache-external-linker",
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
                str(decode_device),
                "--disaggregation-mode",
                "decode",
                "--disaggregation-decode-enable-radix-cache",
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
                str(bootstrap_port),
                "--decode",
                f"http://127.0.0.1:{decode_port}",
                "--host",
                "127.0.0.1",
                "--port",
                str(router_port),
            ],
        )
        wait_ready("router", router, router_port)
        return router_port

    def request_generation(port: int, payload: dict) -> dict:
        response = requests.post(
            f"http://127.0.0.1:{port}/generate",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    def wait_for_saved_bytes(previous: float) -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            current = fetch_orbitkv_metrics(channel_server.http_port).get(
                "orbitkv_save_bytes_total", 0
            )
            if current > previous:
                return
            time.sleep(0.2)
        pytest.fail(
            "P/D workers did not publish cache state:\n" + channel_server.read_logs()[-8000:]
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    fragment = tokenizer.encode("A TENT P/D cache composition correctness sequence. ")
    prompt_tokens = (fragment * (513 // len(fragment) + 1))[:513]
    first_payload = {
        "input_ids": prompt_tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "ignore_eos": True,
        },
    }

    try:
        save_before = fetch_orbitkv_metrics(channel_server.http_port).get(
            "orbitkv_save_bytes_total", 0
        )
        router_port = start_pd()

        def events_since(start):
            path = fault_root / "events.jsonl"
            if not path.exists():
                return []
            complete = [
                json.loads(line)
                for line in path.read_text().splitlines(keepends=True)
                if line.endswith("\n")
            ]
            return complete[start:]

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as caller:
            for case in ("delayed", "partial"):
                case_root = fault_root / case
                case_root.mkdir()
                event_start = len(events_since(0))
                (fault_root / "arm.json").write_text(json.dumps({"name": case}))
                pending = caller.submit(
                    request_generation,
                    router_port,
                    {
                        **first_payload,
                        "rid": f"fault-{case}",
                    },
                )
                try:
                    deadline = time.monotonic() + 20
                    while not (case_root / "first-write.json").exists():
                        assert not pending.done(), pending.result() if pending.done() else None
                        assert time.monotonic() < deadline, (
                            "native P/D did not reach the GPU WRITE gate"
                        )
                        time.sleep(0.02)
                    response = requests.post(
                        f"http://127.0.0.1:{serving_ports['decode']}/abort_request",
                        json={"abort_all": True},
                        timeout=5,
                    )
                    response.raise_for_status()
                    while not any(
                        event["phase"] == "quarantined" for event in events_since(event_start)
                    ):
                        assert time.monotonic() < deadline, (
                            "abort did not retain its destination pages"
                        )
                        time.sleep(0.02)
                    held = events_since(event_start)
                    assert not any(
                        event["phase"] in {"completed", "cancelled", "failed"} for event in held
                    )
                    (case_root / "held.json").write_text(json.dumps(held, indent=2))
                finally:
                    (case_root / "release").touch()
                deadline = time.monotonic() + 20
                while not any(event["phase"] == "cancelled" for event in events_since(event_start)):
                    assert time.monotonic() < deadline, (
                        "late drain ACK did not release the cancelled pages"
                    )
                    time.sleep(0.02)
                terminal = [
                    event
                    for event in events_since(event_start)
                    if event["phase"] in {"completed", "cancelled", "failed"}
                ]
                assert len(terminal) == 1 and terminal[0]["phase"] == "cancelled"
                try:
                    result = pending.result(timeout=30)
                except requests.HTTPError as error:
                    result = {
                        "status_code": error.response.status_code,
                        "body": error.response.text,
                    }
                (case_root / "result.json").write_text(json.dumps(result, indent=2))

        # A partial transport failure quarantines the native peer session.
        # Recreate both workers before admitting new writes to that peer.
        stop_all()
        router_port = start_pd()
        decode_url = f"http://127.0.0.1:{serving_ports['decode']}"
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as caller:
            pending = caller.submit(request_generation, router_port, first_payload)
            deadline = time.monotonic() + 30
            while True:
                response = requests.get(decode_url + "/v1/loads?include=core", timeout=5)
                response.raise_for_status()
                if any(load["num_running_reqs"] for load in response.json()["loads"]):
                    break
                assert not pending.done(), "generation finished before native retraction"
                assert time.monotonic() < deadline, "decode request never became running"
                time.sleep(0.02)
            try:
                response = requests.post(
                    decode_url + "/pause_generation", json={"mode": "retract"}, timeout=15
                )
                response.raise_for_status()
                assert not pending.done(), "retracted request completed while paused"
            finally:
                response = requests.post(decode_url + "/continue_generation", json={}, timeout=15)
                response.raise_for_status()
            first = pending.result(timeout=60)
        output_ids = first["output_ids"]
        assert len(output_ids) == 64
        assert first["meta_info"]["num_retractions"] > 0, first
        wait_for_saved_bytes(save_before)
        (tmp_path / "first-output.json").write_text(json.dumps(first, indent=2))
        (tmp_path / "after-first-metrics.json").write_text(
            json.dumps(fetch_orbitkv_metrics(channel_server.http_port), indent=2)
        )

        for name in ("prefill", "decode"):
            assert "SGLang P/D TENT ready:" in logs[name].read_text()
        stop_all()

        before_restart = fetch_orbitkv_metrics(channel_server.http_port)
        follow_tokens = (
            prompt_tokens
            + output_ids
            + tokenizer.encode(" Continue with one more fact.", add_special_tokens=False)
        )
        prompt_boundary = ((len(prompt_tokens) - 1) // 64) * 64
        decode_boundary = ((len(prompt_tokens) + len(output_ids) - 1) // 64) * 64
        assert decode_boundary > prompt_boundary
        follow_payload = {
            "input_ids": follow_tokens,
            "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
        }

        router_port = start_pd()
        follow = request_generation(router_port, follow_payload)
        after_restart = fetch_orbitkv_metrics(channel_server.http_port)
        (tmp_path / "restart-evidence.json").write_text(
            json.dumps(
                {
                    "before": before_restart,
                    "after": after_restart,
                    "output": follow,
                    "required_cache_boundary": decode_boundary,
                },
                indent=2,
            )
        )
        assert follow["meta_info"]["cached_tokens"] >= decode_boundary, follow
        assert after_restart.get("orbitkv_load_bytes_total", 0) > before_restart.get(
            "orbitkv_load_bytes_total", 0
        )
        stop_all()

        monolithic_port = find_available_port()
        monolithic_env = dict(env)
        monolithic_env.pop("SGLANG_MOONCAKE_TRANSFER_ENGINE")
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
        control_first = request_generation(monolithic_port, first_payload)
        control_follow = request_generation(monolithic_port, follow_payload)
        assert first["output_ids"] == control_first["output_ids"]
        assert first["text"] == control_first["text"]
        assert follow["output_ids"] == control_follow["output_ids"]
        assert follow["text"] == control_follow["text"]
    finally:
        stop_all()
