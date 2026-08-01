#!/usr/bin/env python3
"""Compare Loom MoE movement with vLLM's production scratch path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable

import torch

from loom_kernels.torch_ops import (
    bridge_abi_version,
    moe_combine,
    moe_permute,
)


DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--tokens", default="1,8,32,128,512,2048")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--local-experts", type=int, default=0)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    token_counts = [int(value) for value in args.tokens.split(",") if value]
    if not token_counts or min(token_counts) <= 0:
        parser.error("--tokens must contain positive integers")
    if min(
        args.hidden_size,
        args.top_k,
        args.experts,
        args.warmup,
        args.iterations,
        args.repeats,
    ) <= 0:
        parser.error("shape and timing counts must be positive")
    if args.local_experts == 0:
        args.local_experts = args.experts
    if not 0 < args.local_experts <= args.experts:
        parser.error("--local-experts must be in [1, experts]")
    if args.top_k > args.experts:
        parser.error("--top-k must not exceed --experts")
    if args.hidden_size * torch.empty((), dtype=DTYPES[args.dtype]).element_size() % 16:
        parser.error("vLLM's MoE movement requires 16-byte aligned hidden rows")
    args.token_counts = token_counts
    return args


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def elapsed_eager_us(
    operation: Callable[[], Any], warmup: int, iterations: int
) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = None
    for _ in range(iterations):
        result = operation()
    end.record()
    end.synchronize()
    del result
    return float(start.elapsed_time(end) * 1000.0 / iterations)


def elapsed_graph_us(
    operation: Callable[[], Any], warmup: int, iterations: int
) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = operation()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end) * 1000.0 / iterations)
    del result
    return elapsed


def summarize(samples: list[float]) -> dict[str, Any]:
    return {
        "median_us": statistics.median(samples),
        "p90_us": percentile(samples, 0.9),
        "samples_us": samples,
    }


def benchmark_case(
    *,
    tokens: int,
    max_tokens: int,
    hidden_size: int,
    top_k: int,
    experts: int,
    local_experts: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
        MoEPermuteScratch,
        moe_permute as vllm_moe_permute,
        moe_unpermute as vllm_moe_unpermute,
    )

    torch.manual_seed(seed + tokens)
    hidden_states = torch.randn(
        tokens, hidden_size, dtype=dtype, device="cuda"
    )
    topk_ids = torch.randint(
        0, experts, (tokens, top_k), dtype=torch.int32, device="cuda"
    )
    routing_weights = torch.softmax(
        torch.randn(tokens, top_k, dtype=torch.float32, device="cuda"), dim=-1
    )
    expert_map = None
    if local_experts != experts:
        expert_map = torch.full(
            (experts,), -1, dtype=torch.int32, device="cuda"
        )
        expert_map[:local_experts] = torch.arange(
            local_experts, dtype=torch.int32, device="cuda"
        )

    scratch = MoEPermuteScratch(
        max_num_tokens=max_tokens,
        topk=top_k,
        num_experts=experts,
        num_local_experts=local_experts,
        device=torch.device("cuda"),
        hidden_size=hidden_size,
        hidden_dtype=dtype,
    )
    vllm_output = torch.empty_like(hidden_states)

    def vllm_permute() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        permuted, _, offsets, inverse, assignment_ids = vllm_moe_permute(
            hidden_states,
            None,
            topk_ids,
            experts,
            local_experts,
            expert_map,
            scratch=scratch,
        )
        return permuted, offsets, inverse, assignment_ids

    def loom_permute() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return moe_permute(
            hidden_states,
            topk_ids,
            num_experts=experts,
            num_local_experts=local_experts,
            expert_map=expert_map,
        )

    expected_permuted, expected_offsets, expected_inverse, expected_ids = (
        vllm_permute()
    )
    actual_permuted, actual_offsets, actual_inverse, actual_ids = loom_permute()
    torch.cuda.synchronize()
    if not torch.equal(actual_offsets, expected_offsets):
        raise AssertionError("Loom expert offsets differ from vLLM")
    valid_assignments = int(actual_offsets[-1].item())
    if not torch.equal(
        actual_permuted[:valid_assignments],
        expected_permuted[:valid_assignments],
    ):
        raise AssertionError("Loom local permuted activations differ from vLLM")
    if torch.count_nonzero(actual_permuted[valid_assignments:]).item() != 0:
        raise AssertionError("Loom remote permuted activation tail is not zero")
    if not torch.equal(actual_inverse.flatten(), expected_inverse):
        raise AssertionError("Loom inverse permutation differs from vLLM")
    if not torch.equal(actual_ids, expected_ids):
        raise AssertionError("Loom permuted assignment IDs differ from vLLM")

    def vllm_combine() -> torch.Tensor:
        vllm_moe_unpermute(
            vllm_output,
            expected_permuted,
            routing_weights,
            expected_inverse,
            expected_offsets,
        )
        return vllm_output

    def loom_combine() -> torch.Tensor:
        return moe_combine(
            actual_permuted,
            routing_weights,
            actual_inverse,
            actual_offsets,
        )

    expected_output = vllm_combine().clone()
    actual_output = loom_combine()
    torch.cuda.synchronize()
    tolerance = {
        torch.float32: (1.0e-6, 1.0e-6),
        torch.float16: (2.0e-3, 2.0e-3),
        torch.bfloat16: (2.0e-2, 2.0e-2),
    }[dtype]
    torch.testing.assert_close(
        actual_output, expected_output, rtol=tolerance[0], atol=tolerance[1]
    )

    def vllm_pipeline() -> torch.Tensor:
        permuted, offsets, inverse, _ = vllm_permute()
        vllm_moe_unpermute(
            vllm_output, permuted, routing_weights, inverse, offsets
        )
        return vllm_output

    def loom_pipeline() -> torch.Tensor:
        permuted, offsets, inverse, _ = loom_permute()
        return moe_combine(permuted, routing_weights, inverse, offsets)

    vllm_eager_samples: list[float] = []
    loom_eager_samples: list[float] = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            vllm_eager_samples.append(
                elapsed_eager_us(vllm_pipeline, warmup, iterations)
            )
            loom_eager_samples.append(
                elapsed_eager_us(loom_pipeline, warmup, iterations)
            )
        else:
            loom_eager_samples.append(
                elapsed_eager_us(loom_pipeline, warmup, iterations)
            )
            vllm_eager_samples.append(
                elapsed_eager_us(vllm_pipeline, warmup, iterations)
            )

    vllm_graph_samples: list[float] = []
    loom_graph_samples: list[float] = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            vllm_graph_samples.append(
                elapsed_graph_us(vllm_pipeline, warmup, iterations)
            )
            loom_graph_samples.append(
                elapsed_graph_us(loom_pipeline, warmup, iterations)
            )
        else:
            loom_graph_samples.append(
                elapsed_graph_us(loom_pipeline, warmup, iterations)
            )
            vllm_graph_samples.append(
                elapsed_graph_us(vllm_pipeline, warmup, iterations)
            )

    component_graph_samples = {
        "vllm_permute": [],
        "loom_permute": [],
        "vllm_combine": [],
        "loom_combine": [],
    }
    for repeat in range(repeats):
        providers = (
            (
                ("vllm", vllm_permute, vllm_combine),
                ("loom", loom_permute, loom_combine),
            )
            if repeat % 2 == 0
            else (
                ("loom", loom_permute, loom_combine),
                ("vllm", vllm_permute, vllm_combine),
            )
        )
        for provider, permute, combine in providers:
            component_graph_samples[f"{provider}_permute"].append(
                elapsed_graph_us(permute, warmup, iterations)
            )
            component_graph_samples[f"{provider}_combine"].append(
                elapsed_graph_us(combine, warmup, iterations)
            )

    vllm_eager = summarize(vllm_eager_samples)
    loom_eager = summarize(loom_eager_samples)
    vllm_graph = summarize(vllm_graph_samples)
    loom_graph = summarize(loom_graph_samples)
    vllm_permute_graph = summarize(component_graph_samples["vllm_permute"])
    loom_permute_graph = summarize(component_graph_samples["loom_permute"])
    vllm_combine_graph = summarize(component_graph_samples["vllm_combine"])
    loom_combine_graph = summarize(component_graph_samples["loom_combine"])
    return {
        "tokens": tokens,
        "assignments": tokens * top_k,
        "valid_local_assignments": valid_assignments,
        "eager": {
            "vllm": vllm_eager,
            "loom": loom_eager,
            "speedup": vllm_eager["median_us"] / loom_eager["median_us"],
        },
        "cuda_graph": {
            "vllm": vllm_graph,
            "loom": loom_graph,
            "speedup": vllm_graph["median_us"] / loom_graph["median_us"],
        },
        "cuda_graph_components": {
            "permute": {
                "vllm": vllm_permute_graph,
                "loom": loom_permute_graph,
                "speedup": (
                    vllm_permute_graph["median_us"]
                    / loom_permute_graph["median_us"]
                ),
            },
            "combine": {
                "vllm": vllm_combine_graph,
                "loom": loom_combine_graph,
                "speedup": (
                    vllm_combine_graph["median_us"]
                    / loom_combine_graph["median_us"]
                ),
            },
        },
        "semantics": {
            "local_permuted_activations_exact": True,
            "loom_remote_activation_tail_zero": True,
            "expert_offsets_exact": True,
            "inverse_permutation_exact": True,
            "assignment_ids_exact": True,
            "combine_close": True,
            "maximum_combine_error": float(
                (actual_output.float() - expected_output.float()).abs().max().item()
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    import vllm

    dtype = DTYPES[args.dtype]
    max_tokens = max(args.token_counts)
    results = [
        benchmark_case(
            tokens=tokens,
            max_tokens=max_tokens,
            hidden_size=args.hidden_size,
            top_k=args.top_k,
            experts=args.experts,
            local_experts=args.local_experts,
            dtype=dtype,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            seed=args.seed,
        )
        for tokens in args.token_counts
    ]
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "vllm_version": vllm.__version__,
        "bridge_abi_version": bridge_abi_version(),
        "dtype": args.dtype,
        "hidden_size": args.hidden_size,
        "top_k": args.top_k,
        "experts": args.experts,
        "local_experts": args.local_experts,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "baseline": (
            "vLLM MoEPermuteScratch plus caller-reused unpermute output, "
            "matching the Cutlass grouped-GEMM production boundary"
        ),
        "candidate": (
            "Loom allocating public moe_permute plus moe_combine; grouped GEMM "
            "is intentionally absent from both sides"
        ),
        "timing": (
            "CUDA events with alternating provider order; eager and separately "
            "captured CUDA Graph pipelines; input creation excluded"
        ),
        "semantic_boundary": (
            "vLLM-valid local activation rows and all movement metadata are "
            "compared exactly; Loom additionally guarantees a zero-filled "
            "remote tail that vLLM scratch leaves unspecified"
        ),
        "results": results,
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
