"""Trigger: vLLM P/D callback, layout, sender, waiter or request-lifetime changes.

Two TP=1 workers share one GPU over TCP. This gate compares greedy outputs with
native execution and checks payload/completion/drain counters, including chunked
prefill. It does not qualify cross-GPU RDMA or distributed parallelism.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from pathlib import Path

import pytest
import requests

from tests.support.cache_manager import find_available_port
from tests.support.vllm_helpers import VLLMServer

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


def test_vllm_pd_matches_native_outputs_and_drains(model, tmp_path):
    pytest.importorskip("vllm")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires one GPU with capacity for two model copies")
    if not Path(model).exists():
        pytest.skip("pass --model with a local dense model path")

    from orbitkv.vllm.pd.proxy import ProxyConfig, build_pd_proxy_request

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

    def launch(stack, name, config=None):
        server = VLLMServer(
            port=find_available_port(),
            log_file=tmp_path / f"{name}.log",
            kv_transfer_config=config,
            server_label=name,
            **common,
        )
        # Register cleanup before startup so a failed __enter__ also gets reaped.
        stack.callback(server.__exit__, None, None, None)
        server.__enter__()
        return f"http://127.0.0.1:{server.port}"

    def generate(url, body):
        response = requests.post(url + "/v1/completions", json=body, timeout=120)
        response.raise_for_status()
        return response.json()

    def metrics(url):
        response = requests.get(url + "/metrics", timeout=5)
        response.raise_for_status()
        values = {}
        for line in response.text.splitlines():
            match = re.match(r"^(vllm:orbitkv_pd_\w+)(?:\{[^}]*\})?\s+(\S+)$", line)
            if match:
                key, value = match.groups()
                values[key] = values.get(key, 0) + float(value)
        return values

    with contextlib.ExitStack() as stack:
        native = launch(stack, "native")
        controls = [generate(native, body) for body in bodies]
        (tmp_path / "native-outputs.json").write_text(json.dumps(controls, indent=2))

    with contextlib.ExitStack() as stack:
        urls = {}
        for role in ("decode", "prefill"):
            urls[role] = launch(
                stack,
                role,
                {
                    "kv_connector": f"Pd{role.title()}Connector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "orbitkv.vllm.pd",
                    "engine_id": role,
                    "kv_connector_extra_config": {"orbitkv.pd.mooncake.bind_host": "127.0.0.1"},
                },
            )
        route = ProxyConfig(
            prefill_url=urls["prefill"],
            decode_url=urls["decode"],
            timeout_s=120,
            prefill_max_tokens=1,
        )
        outputs = []
        for body in bodies:
            request = build_pd_proxy_request(body, route)
            outputs.append(generate(request.decode_url, request.decode_body))
        (tmp_path / "pd-outputs.json").write_text(json.dumps(outputs, indent=2))
        for control, output in zip(controls, outputs, strict=True):
            actual, expected = output["choices"][0], control["choices"][0]
            assert actual["text"] == expected["text"]
            assert actual["logprobs"]["tokens"] == expected["logprobs"]["tokens"]
            assert actual["finish_reason"] == expected["finish_reason"]

        deadline = time.monotonic() + 30
        while True:
            prefill, decode = metrics(urls["prefill"]), metrics(urls["decode"])
            gauges = (
                "prefill_active_pushes",
                "prefill_inflight_push_tasks",
                "prefill_inflight_finalize_tasks",
            )
            drained = all(prefill.get("vllm:orbitkv_pd_" + name) == 0 for name in gauges)
            drained &= decode.get("vllm:orbitkv_pd_decode_active_waits") == 0
            completed = prefill.get("vllm:orbitkv_pd_prefill_push_success_total") == len(bodies)
            completed &= decode.get("vllm:orbitkv_pd_load_success_total") == len(bodies)
            if completed and drained:
                break
            assert time.monotonic() < deadline, (prefill, decode)
            time.sleep(0.2)
        (tmp_path / "pd-metrics.json").write_text(
            json.dumps({"prefill": prefill, "decode": decode}, indent=2)
        )
        assert prefill["vllm:orbitkv_pd_prefill_push_bytes_sum"] > 0
