"""Token-exact streaming requests and serial cold/resident/pressure phases."""

from __future__ import annotations

import json
import random
import time
from argparse import Namespace

import requests

from .metrics import cache_source, metrics


def generate(url: str, engine: str, model: str, tokens: list[int], output_len: int) -> dict:
    if engine == "vllm":
        endpoint = "/v1/completions"
        payload = {
            "model": model,
            "prompt": tokens,
            "temperature": 0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    else:
        endpoint = "/generate"
        payload = {
            "input_ids": tokens,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
            "stream": True,
        }
    started = time.perf_counter()
    first = None
    text = ""
    usage = {}
    with requests.post(url + endpoint, json=payload, stream=True, timeout=(10, 300)) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            if engine == "vllm":
                chunk = "".join(choice.get("text", "") for choice in event.get("choices", []))
                text += chunk
                usage = event.get("usage") or usage
            else:
                chunk = event.get("text", "")
                text = chunk
                usage = event.get("meta_info", usage)
            if first is None and chunk:
                first = time.perf_counter()
    ended = time.perf_counter()
    if first is None:
        raise RuntimeError("No generated text in streaming response")
    if usage.get("prompt_tokens") != len(tokens):
        raise RuntimeError(f"Input length changed: expected {len(tokens)}, got {usage}")
    if usage.get("completion_tokens") != output_len:
        raise RuntimeError(f"Output length changed: expected {output_len}, got {usage}")
    return {
        "ttft_ms": (first - started) * 1000,
        "e2e_ms": (ended - started) * 1000,
        "text": text,
        "usage": usage,
    }


def run_workload(args: Namespace, base_url: str, manager_url: str | None) -> list[dict]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    vocabulary = tokenizer.encode(
        "The river flows past green trees and quiet houses. A researcher measures memory "
        "transfer latency and checks that repeated requests produce accurate results. ",
        add_special_tokens=False,
    )
    rng = random.Random(args.seed)

    def prompt(length: int) -> list[int]:
        # Distinct first pages prevent unintended sharing across independent prefixes.
        return [rng.choice(vocabulary) for _ in range(length)]

    samples = []
    pressure_tokens = args.gpu_tokens * 3 // 4
    for _ in range(3):
        generate(base_url, args.engine, str(args.model), prompt(256), args.output_tokens)
    with (args.output / "samples.jsonl").open("w") as output:
        for length in args.lengths:
            generate(
                base_url,
                args.engine,
                str(args.model),
                prompt(length),
                args.output_tokens,
            )
            for repeat in range(args.repeats):
                tokens = prompt(length)
                texts = []
                for phase in ("cold", "hbm_hit", "after_pressure"):
                    if phase == "after_pressure":
                        for _ in range(2):
                            generate(
                                base_url,
                                args.engine,
                                str(args.model),
                                prompt(pressure_tokens),
                                1,
                            )
                    time.sleep(args.settle_seconds)
                    before = metrics(base_url)
                    manager_before = metrics(manager_url)
                    result = generate(
                        base_url,
                        args.engine,
                        str(args.model),
                        tokens,
                        args.output_tokens,
                    )
                    time.sleep(args.settle_seconds)
                    after = metrics(base_url)
                    manager_after = metrics(manager_url)
                    result.update(
                        length=length,
                        repeat=repeat,
                        phase=phase,
                        metrics_delta={
                            key: value - before.get(key, 0)
                            for key, value in after.items()
                            if value != before.get(key, 0)
                        },
                        manager_delta={
                            key: value - manager_before.get(key, 0)
                            for key, value in manager_after.items()
                            if value != manager_before.get(key, 0)
                        },
                    )
                    texts.append(result["text"])
                    result["cache_source"] = cache_source(args.engine, result)
                    result["matches_cold_output"] = result["text"] == texts[0]
                    samples.append(result)
                    output.write(json.dumps(result, allow_nan=False) + "\n")
                    output.flush()
                    print(
                        f"{args.engine}/{args.backend} len={length} repeat={repeat} {phase}: TTFT={result['ttft_ms']:.2f}ms usage={result['usage']}",
                        flush=True,
                    )
    return samples
