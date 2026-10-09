"""Installed static LoRA cache domains across real engine restart and weight replacement."""

import json
import math
import os
import shutil
import sys
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

pytestmark = [pytest.mark.release_smoke, pytest.mark.gpu]


def write_adapter(directory, model_config, seed):
    import torch
    from safetensors.torch import save_file

    directory.mkdir(exist_ok=True)
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0,
        "bias": "none",
        "target_modules": ["q_proj", "v_proj"],
        "inference_mode": True,
    }
    (directory / "adapter_config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = model_config["hidden_size"]
    head_dim = model_config.get("head_dim", hidden // model_config["num_attention_heads"])
    weights = {}
    for layer in range(model_config["num_hidden_layers"]):
        for module, heads in (
            ("q_proj", model_config["num_attention_heads"]),
            ("v_proj", model_config["num_key_value_heads"]),
        ):
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{module}"
            weights[f"{prefix}.lora_A.weight"] = (
                torch.randn((8, hidden), generator=generator) * 0.015
            ).to(torch.bfloat16)
            weights[f"{prefix}.lora_B.weight"] = (
                torch.randn((heads * head_dim, 8), generator=generator) * 0.015
            ).to(torch.bfloat16)
    save_file(weights, directory / "adapter_model.safetensors", metadata={"format": "pt"})


def output_logprobs(engine, result):
    if engine == "vllm":
        return result["choices"][0]["logprobs"]["token_logprobs"]
    return [entry[0] for entry in result["meta_info"]["output_token_logprobs"]]


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_static_adapters_keep_native_hits_and_separate_changed_contents(engine, model, tmp_path):
    from transformers import AutoTokenizer

    config = json.loads((Path(model) / "config.json").read_text())
    assert config["model_type"] == "qwen3", "deterministic LoRA fixture targets dense Qwen3"
    adapters = {name: tmp_path / name for name in ("static-a", "static-b")}
    for name, seed in (("static-a", 17), ("static-b", 29)):
        write_adapter(adapters[name], config, seed)
    original = tmp_path / "adapter-a-original"
    shutil.copytree(adapters["static-a"], original)
    env = isolated_environment(dict(os.environ), tmp_path)
    env["ORBITKV_CACHE_SCOPE"] = tmp_path.name
    env["ORBITKV_STATIC_LORA"] = "1"
    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "0"
    before = probe_installation(
        sys.executable, engine, env, tmp_path, "installed-before", native=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    fragment = tokenizer.encode("Static adapter context remains tied to immutable weight bytes. ")
    prompt = (fragment * (769 // len(fragment) + 1))[:769]
    port, http_port, engine_port = (find_available_port() for _ in range(3))
    url = f"http://127.0.0.1:{engine_port}"
    command, options = engine_command(engine, sys.executable, model, engine_port, "direct")
    if engine == "vllm":
        command += [
            "--enable-lora",
            "--max-loras",
            "2",
            "--max-lora-rank",
            "8",
            "--shutdown-timeout",
            "30",
            "--lora-modules",
            *[f"{name}={path}" for name, path in adapters.items()],
        ]
        connector = json.loads(options[1])
        connector["kv_connector_extra_config"]["orbitkv.static_lora_adapters"] = [
            {"name": name, "path": str(path)} for name, path in adapters.items()
        ]
        options[1] = json.dumps(connector)
    else:
        command += [
            "--lora-paths",
            *[f"{name}={path}" for name, path in adapters.items()],
            "--lora-backend",
            "triton",
            "--max-lora-rank",
            "8",
        ]
    env["ORBITKV_PORT"] = str(port)
    env["ORBITKV_SGLANG_ENDPOINT"] = f"unix:///tmp/orbitkv-{port}.sock"
    responses, snapshots = {}, {}
    selections = (None, *adapters)

    def generate(phase, adapter):
        body = {
            "input_ids": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
            "return_logprob": True,
        }
        path = "/generate"
        if engine == "vllm":
            path = "/v1/completions"
            body = {
                "model": adapter or model,
                "prompt": prompt,
                "max_tokens": 8,
                "temperature": 0,
                "ignore_eos": True,
                "logprobs": 1,
                "return_tokens_as_token_ids": True,
            }
        elif adapter is not None:
            body["lora_path"] = adapter
        response = requests.post(url + path, json=body, timeout=90)
        response.raise_for_status()
        result = response.json()
        responses[f"{phase}-{adapter or 'base'}"] = result
        (tmp_path / "responses.json").write_text(json.dumps(responses, indent=2) + "\n")
        return result

    def snapshot(label):
        metrics = wait_for_drain(http_port)
        snapshots[label] = metrics
        (tmp_path / "metrics.json").write_text(json.dumps(snapshots, indent=2) + "\n")
        return metrics

    def compare(actual, expected):
        compare_output(engine, actual, expected)
        assert output_logprobs(engine, actual) == pytest.approx(
            output_logprobs(engine, expected), abs=0.005, rel=0
        )

    try:
        with service(command, url, env, tmp_path, f"{engine}-native-original"):
            baseline = {name: generate("native-original", name) for name in selections}
        base_probs = output_logprobs(engine, baseline[None])
        for name in adapters:
            assert (
                max(
                    abs(a - b)
                    for a, b in zip(
                        output_logprobs(engine, baseline[name]), base_probs, strict=True
                    )
                )
                > 0.001
            ), "LoRA fixture did not measurably affect native computation"
        with service(
            manager_command(sys.executable, port, http_port, "ssd", tmp_path),
            f"http://127.0.0.1:{http_port}",
            env,
            tmp_path,
            "manager",
        ):
            with service(command + options, url, env, tmp_path, f"{engine}-cache-original"):
                for name in selections:
                    cold = generate("cold-original", name)
                    compare(cold, baseline[name])
                    assert cached_tokens(engine, cold) == 0, cold
                    cold_metrics = snapshot(f"cold-{name}")
                    hot = generate("native-hbm", name)
                    compare(hot, baseline[name])
                    assert cached_tokens(engine, hot) >= 704, hot
                    hot_metrics = snapshot(f"hot-{name}")
                    query_counter = "orbitkv_hll_total_requests"
                    assert math.isfinite(cold_metrics[query_counter])
                    assert math.isfinite(hot_metrics[query_counter])
                    assert cold_metrics[query_counter] >= 0
                    assert hot_metrics[query_counter] == cold_metrics[query_counter]
                    assert hot_metrics.get("orbitkv_load_bytes_total", 0) == cold_metrics.get(
                        "orbitkv_load_bytes_total", 0
                    )
            evict_dram_after_ssd_writes(http_port)
            with service(command + options, url, env, tmp_path, f"{engine}-cache-restart"):
                for name in selections:
                    old = snapshot(f"before-restart-{name}")
                    warm = generate("restart", name)
                    compare(warm, baseline[name])
                    assert cached_tokens(engine, warm) >= 704, warm
                    new = snapshot(f"after-restart-{name}")
                    assert new["orbitkv_load_bytes_total"] > old.get("orbitkv_load_bytes_total", 0)
                    assert new["orbitkv_ssd_prefetch_bytes_total"] > old.get(
                        "orbitkv_ssd_prefetch_bytes_total", 0
                    )
            write_adapter(adapters["static-a"], config, 41)
            with service(command, url, env, tmp_path, f"{engine}-native-changed"):
                changed = generate("native-changed", "static-a")
            assert output_logprobs(engine, changed) != output_logprobs(engine, baseline["static-a"])
            with service(command + options, url, env, tmp_path, f"{engine}-cache-changed"):
                old = snapshot("before-changed")
                miss = generate("changed-content", "static-a")
                compare(miss, changed)
                assert cached_tokens(engine, miss) == 0, miss
                new = snapshot("after-changed")
                assert new.get("orbitkv_load_bytes_total", 0) == old.get(
                    "orbitkv_load_bytes_total", 0
                )
                assert new["orbitkv_save_bytes_total"] > old.get("orbitkv_save_bytes_total", 0)
    finally:
        after = probe_installation(sys.executable, engine, env, tmp_path, "installed-after")
        assert before["distributions"] == after["distributions"]
