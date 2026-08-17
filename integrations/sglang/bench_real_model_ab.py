from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "integrations/sglang/bench_hybrid_shadow.py"


def validate_model_plan(args) -> dict:
    model_path = Path(args.model).resolve()
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    layout = json.loads(
        subprocess.check_output(
            [args.orbitkv_bin, "emit-layout", args.plan],
            text=True,
        )
    )
    classes = {item["name"]: item for item in layout["classes"]}
    layer_types = config.get("layer_types", [])
    expected_full = [
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    ]
    expected_swa = [
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type == "sliding_attention"
    ]
    if classes["full"]["layers"] != expected_full:
        raise RuntimeError("OrbitKV full layers do not match checkpoint config")
    if classes["swa"]["layers"] != expected_swa:
        raise RuntimeError("OrbitKV SWA layers do not match checkpoint config")

    page_tokens = int(layout["page_tokens"])
    window_tokens = int(config["sliding_window"])
    expected_cells = 1 + (window_tokens - 1 + page_tokens - 1) // page_tokens
    swa_address = classes["swa"]["address"]
    if swa_address != {"kind": "periodic", "period_blocks": expected_cells}:
        raise RuntimeError("OrbitKV periodic cells do not match checkpoint window")
    if classes["swa"]["retirement"] != {
        "kind": "block_end_plus",
        "offset_tokens": window_tokens - 1,
    }:
        raise RuntimeError("OrbitKV retirement program does not match checkpoint window")

    expected_bytes_per_token_per_layer = (
        2
        * int(config["num_key_value_heads"])
        * int(config["head_dim"])
        * args.kv_dtype_bytes
    )
    for class_name in ("full", "swa"):
        if (
            int(classes[class_name]["bytes_per_token_per_layer"])
            != expected_bytes_per_token_per_layer
        ):
            raise RuntimeError(
                f"OrbitKV {class_name} KV geometry does not match checkpoint config"
            )
    return {
        "schema": "orbitkv.model-plan-validation.v1",
        "config": str(model_path / "config.json"),
        "plan": str(Path(args.plan).resolve()),
        "full_layers": expected_full,
        "swa_layers": expected_swa,
        "window_tokens": window_tokens,
        "page_tokens": page_tokens,
        "minimum_swa_cells": expected_cells,
        "bytes_per_token_per_layer": expected_bytes_per_token_per_layer,
        "status": "pass",
    }


def run_once(args, mode: str, pair_index: int) -> dict:
    command = [
        args.python,
        str(BENCH),
        "--mode",
        mode,
        "--load-format",
        "auto",
        "--orbitkv-bin",
        args.orbitkv_bin,
        "--orbitkv-owner-lib",
        args.owner_library,
        "--owner-transport",
        "ffi",
        "--model",
        args.model,
        "--plan",
        args.plan,
        "--requests",
        str(args.requests),
        "--max-running-requests",
        str(args.max_running_requests),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--decode-tokens",
        str(args.decode_tokens),
        "--iterations",
        "1",
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--context-length",
        str(args.context_length),
        "--eviction-interval",
        str(args.eviction_interval),
        "--attention-backend",
        args.attention_backend,
        "--moe-runner-backend",
        args.moe_runner_backend,
        "--no-trace-allocations",
    ]
    if args.mem_fraction_static is not None:
        command.extend(["--mem-fraction-static", str(args.mem_fraction_static)])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{ROOT / 'integrations/sglang/src'}:/workspace/sglang/python"
    )
    environment["ORBITKV_SGLANG_REVISION"] = args.sglang_revision
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    records = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith("{")
    ]
    if not records:
        raise RuntimeError(
            f"no JSON result for pair={pair_index} mode={mode}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    result = records[-1]
    result["pair"] = pair_index
    result["process_wall_seconds"] = time.perf_counter() - started
    if result["checkpoint"]["load_format"] != "auto":
        raise RuntimeError("real-model A/B did not use load_format=auto")
    if result["checkpoint"]["weight_bytes"] <= 0:
        raise RuntimeError("real-model A/B did not observe checkpoint weights")
    if not result["checkpoint"]["indexed_weights_complete"]:
        raise RuntimeError("real-model A/B observed an incomplete indexed checkpoint")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python", default=str(ROOT / ".venv-sglang-h20/bin/python")
    )
    parser.add_argument(
        "--orbitkv-bin", default=str(ROOT / "target/release/orbitkv")
    )
    parser.add_argument(
        "--owner-library",
        default=str(
            ROOT / "crates/orbitkv-ffi/target/release/liborbitkv_ffi.so"
        ),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--max-total-tokens", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument("--eviction-interval", type=int, default=32)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--moe-runner-backend", default="auto")
    parser.add_argument("--kv-dtype-bytes", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--sglang-revision",
        default="095ec6c997bfdd25d3864cb0ce77a6562a934b96",
    )
    args = parser.parse_args()
    if args.pairs <= 0:
        raise ValueError("pairs must be positive")
    model_plan_validation = validate_model_plan(args)

    pairs = []
    owner_over_stock = []
    for pair_index in range(args.pairs):
        order = ("stock", "owner") if pair_index % 2 == 0 else ("owner", "stock")
        runs = {}
        for mode in order:
            print(f"pair={pair_index} mode={mode} starting", flush=True)
            result = run_once(args, mode, pair_index)
            runs[mode] = result
            print(
                f"pair={pair_index} mode={mode} "
                f"workload={result['iteration_seconds'][0]:.6f}s "
                f"digest={result['output_digest'][:12]} "
                f"capacity={result['server_memory'].get('token_capacity')}",
                flush=True,
            )

        stock = runs["stock"]
        owner = runs["owner"]
        if stock["checkpoint"] != owner["checkpoint"]:
            raise RuntimeError("Stock and OrbitKV checkpoint identities differ")
        if stock["output_digest"] != owner["output_digest"]:
            raise RuntimeError("Stock and OrbitKV output token digests differ")
        if stock["completion_tokens"] != owner["completion_tokens"]:
            raise RuntimeError("Stock and OrbitKV completion counts differ")
        ratio = owner["iteration_seconds"][0] / stock["iteration_seconds"][0]
        owner_over_stock.append(ratio)
        pairs.append(
            {
                "pair": pair_index,
                "execution_order": list(order),
                "stock": stock,
                "owner": owner,
                "owner_over_stock_ratio": ratio,
            }
        )

    report = {
        "schema": "orbitkv.real-model-sglang-ab.v1",
        "model": str(Path(args.model).resolve()),
        "plan": str(Path(args.plan).resolve()),
        "model_plan_validation": model_plan_validation,
        "pairs": pairs,
        "owner_over_stock_ratios": owner_over_stock,
        "median_owner_over_stock_ratio": statistics.median(owner_over_stock),
        "median_owner_over_stock_percent": (
            statistics.median(owner_over_stock) - 1
        )
        * 100,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
