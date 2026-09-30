"""Native Mooncake P/D + TENT + upstream Router; requires the S5.4 engine patches.

Two TP=1 processes share one GPU over TCP. This gate does not qualify RDMA,
parallelism changes, failure recovery, or composition with external cache loads.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import requests

from tests.support.cache_manager import find_available_port
from tests.support.vllm_helpers import VLLMServer

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


def test_native_pd_tent_matches_monolithic(model, tmp_path):
    pytest.importorskip("vllm")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not Path(model).is_dir():
        pytest.skip("requires one GPU and --model pointing to a local dense model")
    router_binary = shutil.which("vllm-router")
    assert router_binary, "install the S5.4 pinned vllm-router in PATH"

    common = {
        "model": model,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.4,
        "prefix_caching": False,
        "env_overrides": {"MC_FORCE_TCP": "1", "VLLM_LOG_STATS_INTERVAL": "1"},
        "extra_args": [
            "--host",
            "127.0.0.1",
            "--dtype",
            "bfloat16",
            "--block-size",
            "64",
            "--kv-cache-memory-bytes",
            str(576 * 1024**2),
            "--max-num-seqs",
            "4",
            "--max-num-batched-tokens",
            "512",
            "--seed",
            "42",
            "--attention-config.flash_attn_version",
            "2",
            "--enforce-eager",
        ],
    }
    bodies = [
        {
            "model": model,
            "prompt": [100 + i % 17 for i in range(length)],
            "max_tokens": 16,
            "temperature": 0,
            "seed": 42,
            "logprobs": 1,
        }
        for length in (129, 257, 769)
    ]

    def launch(stack, name, config=None, bootstrap=None):
        options = dict(common)
        if bootstrap is not None:
            options["env_overrides"] = {
                **common["env_overrides"],
                "VLLM_MOONCAKE_BOOTSTRAP_PORT": str(bootstrap),
            }
        server = VLLMServer(
            port=find_available_port(),
            log_file=tmp_path / f"{name}.log",
            kv_transfer_config=config,
            server_label=name,
            **options,
        )
        stack.callback(server.__exit__, None, None, None)
        server.__enter__()
        return f"http://127.0.0.1:{server.port}"

    def generate(url, body):
        response = requests.post(url + "/v1/completions", json=body, timeout=120)
        response.raise_for_status()
        return response.json()

    with contextlib.ExitStack() as stack:
        native = launch(stack, "monolithic")
        controls = [generate(native, body) for body in bodies]
        (tmp_path / "monolithic.json").write_text(json.dumps(controls, indent=2))

    with contextlib.ExitStack() as stack:
        bootstrap = find_available_port()
        urls = {}
        for role in ("producer", "consumer"):
            urls[role] = launch(
                stack,
                role,
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": f"kv_{role}",
                    "engine_id": role,
                    "kv_connector_extra_config": {
                        "mooncake_protocol": "tcp",
                        "transfer_engine_factory": "orbitkv.vllm.transport.TentTransferEngine",
                    },
                },
                bootstrap,
            )

        router_port = find_available_port()
        command = [
            router_binary,
            "--vllm-pd-disaggregation",
            "--kv-connector",
            "mooncake",
            "--prefill",
            urls["producer"],
            str(bootstrap),
            "--decode",
            urls["consumer"],
            "--host",
            "127.0.0.1",
            "--port",
            str(router_port),
        ]
        (tmp_path / "router-command.json").write_text(json.dumps(command, indent=2))
        with (tmp_path / "router.log").open("w") as log:
            router = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()
            )

        def stop_router():
            if router.poll() is None:
                router.terminate()
                try:
                    router.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    router.kill()
                    router.wait(timeout=5)

        stack.callback(stop_router)
        router_url = f"http://127.0.0.1:{router_port}"
        deadline = time.monotonic() + 60
        while True:
            assert router.poll() is None, (tmp_path / "router.log").read_text()[-6000:]
            try:
                if requests.get(router_url + "/health", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            assert time.monotonic() < deadline, "native Router did not become ready"
            time.sleep(0.2)

        outputs = [generate(router_url, body) for body in bodies]
        (tmp_path / "pd.json").write_text(json.dumps(outputs, indent=2))
        for control, output in zip(controls, outputs, strict=True):
            actual, expected = output["choices"][0], control["choices"][0]
            assert actual["text"] == expected["text"]
            assert actual["logprobs"]["tokens"] == expected["logprobs"]["tokens"]
            assert actual["finish_reason"] == expected["finish_reason"]

        deadline = time.monotonic() + 30
        while True:
            log = (tmp_path / "producer.log").read_text()
            completed = sum(
                int(value) for value in re.findall(r"Num successful transfers=(\d+)", log)
            )
            if completed >= len(bodies):
                break
            assert time.monotonic() < deadline, log[-6000:]
            time.sleep(0.2)
        assert any(float(value) > 0 for value in re.findall(r"Avg MB per transfer=([\d.]+)", log))
        for role in ("producer", "consumer"):
            role_log = (tmp_path / f"{role}.log").read_text()
            assert "vLLM native P/D TENT ready:" in role_log
            assert all(
                int(value) == 0
                for value in re.findall(r"Num failed (?:transfers|recvs)=(\d+)", role_log)
            )
