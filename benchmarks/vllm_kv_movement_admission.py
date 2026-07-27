#!/usr/bin/env python3
"""Measure whether vLLM's default scheduler physically moves paged KV blocks."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.05)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--prefix-tokens", type=int, default=1024)
    parser.add_argument("--preemption-batch-size", type=int, default=128)
    parser.add_argument("--preemption-input-tokens", type=int, default=1536)
    parser.add_argument("--preemption-output-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("gpu-memory-utilization must be between zero and one")
    positive = (
        args.max_model_len,
        args.max_num_seqs,
        args.max_num_batched_tokens,
        args.prefix_tokens,
        args.preemption_batch_size,
        args.preemption_input_tokens,
        args.preemption_output_tokens,
    )
    if min(positive) <= 0:
        parser.error("length, batch, and capacity arguments must be positive")
    if args.preemption_batch_size > args.max_num_seqs:
        parser.error("preemption-batch-size cannot exceed max-num-seqs")
    if (
        args.preemption_input_tokens + args.preemption_output_tokens
        > args.max_model_len
    ):
        parser.error(
            "preemption input plus output tokens cannot exceed max-model-len"
        )
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_environment(cache_root: Path) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(cache_root / "vllm")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "triton")

    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    if not (cuda_home / "bin" / "nvcc").is_file():
        raise RuntimeError(f"nvcc was not found under {cuda_home}")
    os.environ["CUDA_HOME"] = str(cuda_home)
    current_path = os.environ.get("PATH", "").split(os.pathsep)
    required_path = [
        str(Path(sys.executable).absolute().parent),
        str(cuda_home / "bin"),
    ]
    os.environ["PATH"] = os.pathsep.join(
        [entry for entry in required_path if entry not in current_path] + current_path
    )


def movement_totals(events: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = sorted({str(event["kind"]) for event in events})
    return {
        "calls": len(events),
        "bytes": sum(int(event["bytes"]) for event in events),
        "kinds": {
            kind: sum(1 for event in events if event["kind"] == kind)
            for kind in kinds
        },
    }


class MovementProbe:
    def __init__(self) -> None:
        self.movement_events: list[dict[str, Any]] = []
        self.preemption_events: list[dict[str, Any]] = []
        self.prefix_queries: list[dict[str, Any]] = []

    def install(self) -> None:
        import vllm._custom_ops as custom_ops
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.simple_kv_offload import copy_backend, cuda_mem_ops

        original_swap_blocks = custom_ops.swap_blocks

        @functools.wraps(original_swap_blocks)
        def swap_blocks(
            src: Any,
            dst: Any,
            block_size_in_bytes: int,
            block_mapping: Any,
        ) -> Any:
            self.movement_events.append(
                {
                    "kind": "swap_blocks",
                    "copies": int(block_mapping.shape[0]),
                    "bytes": int(block_mapping.shape[0]) * block_size_in_bytes,
                }
            )
            return original_swap_blocks(
                src,
                dst,
                block_size_in_bytes,
                block_mapping,
            )

        custom_ops.swap_blocks = swap_blocks

        original_swap_blocks_batch = custom_ops.swap_blocks_batch

        @functools.wraps(original_swap_blocks_batch)
        def swap_blocks_batch(
            src_ptrs: Any,
            dst_ptrs: Any,
            sizes: Any,
            is_src_access_order_any: bool = False,
        ) -> Any:
            self.movement_events.append(
                {
                    "kind": "swap_blocks_batch",
                    "copies": int(sizes.numel()),
                    "bytes": int(sizes.sum().item()),
                }
            )
            return original_swap_blocks_batch(
                src_ptrs,
                dst_ptrs,
                sizes,
                is_src_access_order_any,
            )

        custom_ops.swap_blocks_batch = swap_blocks_batch

        def wrap_copy_blocks(
            owner: Any,
            original: Callable[..., Any],
            kind: str,
        ) -> None:
            @functools.wraps(original)
            def copy_blocks(
                src_block_ids: list[int],
                dst_block_ids: list[int],
                params: Any,
            ) -> Any:
                copies = len(src_block_ids) * int(params.num_layers)
                bytes_moved = len(src_block_ids) * int(params.bpb.sum())
                self.movement_events.append(
                    {
                        "kind": kind,
                        "copies": copies,
                        "bytes": bytes_moved,
                    }
                )
                return original(src_block_ids, dst_block_ids, params)

            owner.copy_blocks = copy_blocks

        wrap_copy_blocks(
            cuda_mem_ops,
            cuda_mem_ops.copy_blocks,
            "simple_offload_cuda_mem_ops",
        )
        wrap_copy_blocks(
            copy_backend,
            copy_backend.copy_blocks,
            "simple_offload_copy_backend",
        )

        original_get_computed_blocks = KVCacheManager.get_computed_blocks

        @functools.wraps(original_get_computed_blocks)
        def get_computed_blocks(manager: Any, request: Any) -> Any:
            blocks, num_computed_tokens = original_get_computed_blocks(
                manager,
                request,
            )
            self.prefix_queries.append(
                {
                    "request_id": str(request.request_id),
                    "request_tokens": int(request.num_tokens),
                    "computed_tokens": int(num_computed_tokens),
                }
            )
            return blocks, num_computed_tokens

        KVCacheManager.get_computed_blocks = get_computed_blocks

        original_preempt_request = Scheduler._preempt_request

        @functools.wraps(original_preempt_request)
        def preempt_request(
            scheduler: Any,
            request: Any,
            timestamp: float,
        ) -> Any:
            self.preemption_events.append(
                {
                    "request_id": str(request.request_id),
                    "request_tokens": int(request.num_tokens),
                    "computed_tokens_before_preemption": int(
                        request.num_computed_tokens
                    ),
                    "previous_preemptions": int(request.num_preemptions),
                }
            )
            return original_preempt_request(scheduler, request, timestamp)

        Scheduler._preempt_request = preempt_request

    def snapshot(self) -> dict[str, int]:
        return {
            "movement": len(self.movement_events),
            "preemption": len(self.preemption_events),
            "prefix": len(self.prefix_queries),
        }

    def delta(self, snapshot: dict[str, int]) -> dict[str, Any]:
        movement = self.movement_events[snapshot["movement"] :]
        return {
            "movement": movement_totals(movement),
            "movement_events": movement,
            "preemptions": self.preemption_events[snapshot["preemption"] :],
            "prefix_queries": self.prefix_queries[snapshot["prefix"] :],
        }


def make_prefix_prompts(
    prefix_tokens: int,
    block_size: int,
) -> tuple[list[dict[str, list[int]]], int]:
    aligned_prefix_tokens = prefix_tokens - (prefix_tokens % block_size)
    if aligned_prefix_tokens <= 0:
        raise ValueError("prefix-tokens must include at least one full cache block")
    common = [
        3 + ((position * 13) % 40000)
        for position in range(aligned_prefix_tokens)
    ]
    suffix_a = [50003 + position for position in range(block_size)]
    suffix_b = [51003 + position for position in range(block_size)]
    return (
        [
            {"prompt_token_ids": common + suffix_a},
            {"prompt_token_ids": common + suffix_b},
        ],
        aligned_prefix_tokens,
    )


def make_preemption_prompts(
    batch_size: int,
    input_tokens: int,
) -> list[dict[str, list[int]]]:
    return [
        {
            "prompt_token_ids": [
                3 + ((batch_index * 4099 + position * 17) % 50000)
                for position in range(input_tokens)
            ]
        }
        for batch_index in range(batch_size)
    ]


def timed_generate(
    engine: Any,
    prompts: list[dict[str, list[int]]],
    sampling: Any,
) -> tuple[list[Any], float]:
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = engine.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()
    return outputs, (time.perf_counter() - started) * 1000.0


def cache_capacity(engine: Any) -> dict[str, Any]:
    cache = engine.llm_engine.vllm_config.cache_config
    values = {
        "cache_dtype": cache.cache_dtype,
        "num_gpu_blocks": cache.num_gpu_blocks,
        "block_size": cache.block_size,
        "kv_cache_size_tokens": cache.kv_cache_size_tokens,
        "kv_cache_max_concurrency": cache.kv_cache_max_concurrency,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise RuntimeError(
            "vLLM did not expose initialized cache capacity: " + ", ".join(missing)
        )
    return {
        "cache_dtype": str(values["cache_dtype"]),
        "num_gpu_blocks": int(values["num_gpu_blocks"]),
        "block_size": int(values["block_size"]),
        "kv_cache_size_tokens": int(values["kv_cache_size_tokens"]),
        "kv_cache_max_concurrency": float(values["kv_cache_max_concurrency"]),
    }


def source_provenance(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare_environment(args.cache_root.resolve())

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    probe = MovementProbe()
    probe.install()

    model_path = Path(args.model).expanduser()
    model = str(model_path.resolve()) if model_path.exists() else args.model
    engine_args: dict[str, Any] = {
        "model": model,
        "skip_tokenizer_init": True,
        "dtype": "bfloat16",
        "enable_prefix_caching": True,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "disable_log_stats": False,
        "enforce_eager": args.enforce_eager,
    }
    if args.model_revision is not None and not model_path.exists():
        engine_args["revision"] = args.model_revision
    engine = LLM(**engine_args)
    capacity = cache_capacity(engine)

    prefix_prompts, aligned_prefix_tokens = make_prefix_prompts(
        args.prefix_tokens,
        capacity["block_size"],
    )
    prefix_prompt_tokens = aligned_prefix_tokens + capacity["block_size"]
    if prefix_prompt_tokens + 1 > args.max_model_len:
        raise ValueError(
            "the aligned prefix, one suffix block, and output must fit "
            "max-model-len"
        )
    prefix_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
    )
    first_prefix_snapshot = probe.snapshot()
    first_prefix_outputs, first_prefix_ms = timed_generate(
        engine,
        [prefix_prompts[0]],
        prefix_sampling,
    )
    first_prefix = probe.delta(first_prefix_snapshot)

    second_prefix_snapshot = probe.snapshot()
    second_prefix_outputs, second_prefix_ms = timed_generate(
        engine,
        [prefix_prompts[1]],
        prefix_sampling,
    )
    second_prefix = probe.delta(second_prefix_snapshot)

    preemption_prompts = make_preemption_prompts(
        args.preemption_batch_size,
        args.preemption_input_tokens,
    )
    preemption_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.preemption_output_tokens,
        ignore_eos=True,
    )
    preemption_snapshot = probe.snapshot()
    preemption_outputs, preemption_ms = timed_generate(
        engine,
        preemption_prompts,
        preemption_sampling,
    )
    preemption = probe.delta(preemption_snapshot)

    expected_output_tokens = args.preemption_output_tokens
    actual_output_lengths = [
        len(request.outputs[0].token_ids) for request in preemption_outputs
    ]
    if any(length != expected_output_tokens for length in actual_output_lengths):
        raise RuntimeError("vLLM returned an unexpected preemption output length")

    package_root = Path(vllm.__file__).resolve().parent
    scheduler_path = package_root / "v1" / "core" / "sched" / "scheduler.py"
    kv_manager_path = package_root / "v1" / "core" / "kv_cache_manager.py"
    block_pool_path = package_root / "v1" / "core" / "block_pool.py"
    custom_ops_path = package_root / "_custom_ops.py"
    cuda_mem_ops_path = (
        package_root / "v1" / "simple_kv_offload" / "cuda_mem_ops.py"
    )
    copy_backend_path = (
        package_root / "v1" / "simple_kv_offload" / "copy_backend.py"
    )
    physical_movement_calls = (
        first_prefix["movement"]["calls"]
        + second_prefix["movement"]["calls"]
        + preemption["movement"]["calls"]
    )
    prefix_hit_tokens = max(
        (
            int(query["computed_tokens"])
            for query in second_prefix["prefix_queries"]
        ),
        default=0,
    )
    preemption_count = len(preemption["preemptions"])
    report = {
        "benchmark": "vllm_kv_movement_admission",
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "model": model,
        "model_revision": args.model_revision,
        "scope": {
            "engine_path": "vLLM V1 default local GPU KV cache",
            "kv_offload_configured": False,
            "excluded": [
                "optional CPU KV offload",
                "distributed KV transfer",
                "beam-search cache movement",
            ],
            "instrumented_calls": [
                "vllm._custom_ops.swap_blocks",
                "vllm._custom_ops.swap_blocks_batch",
                "vllm.v1.simple_kv_offload.cuda_mem_ops.copy_blocks",
                "vllm.v1.simple_kv_offload.copy_backend.copy_blocks",
                "vllm.v1.core.kv_cache_manager.KVCacheManager.get_computed_blocks",
                "vllm.v1.core.sched.scheduler.Scheduler._preempt_request",
            ],
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": importlib.metadata.version("vllm"),
            "gpu": torch.cuda.get_device_name(0),
            "v1_multiprocessing": os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"],
            "engine_type": (
                f"{type(engine.llm_engine).__module__}."
                f"{type(engine.llm_engine).__qualname__}"
            ),
        },
        "engine": {
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_prefix_caching": True,
            "enforce_eager": args.enforce_eager,
            "cache_capacity": capacity,
            "stress_max_tokens": (
                args.preemption_batch_size
                * (
                    args.preemption_input_tokens
                    + args.preemption_output_tokens
                )
            ),
            "stress_to_cache_capacity_ratio": (
                args.preemption_batch_size
                * (
                    args.preemption_input_tokens
                    + args.preemption_output_tokens
                )
                / capacity["kv_cache_size_tokens"]
            ),
        },
        "prefix_cache": {
            "aligned_shared_prefix_tokens": aligned_prefix_tokens,
            "first_request_latency_ms": first_prefix_ms,
            "second_request_latency_ms": second_prefix_ms,
            "second_request_cache_hit_tokens": prefix_hit_tokens,
            "first_request": first_prefix,
            "second_request": second_prefix,
            "output_token_ids": [
                list(first_prefix_outputs[0].outputs[0].token_ids),
                list(second_prefix_outputs[0].outputs[0].token_ids),
            ],
        },
        "preemption": {
            "batch_size": args.preemption_batch_size,
            "input_tokens": args.preemption_input_tokens,
            "output_tokens": args.preemption_output_tokens,
            "batch_latency_ms": preemption_ms,
            "count": preemption_count,
            "details": preemption,
        },
        "source_provenance": source_provenance(
            [
                scheduler_path,
                kv_manager_path,
                block_pool_path,
                custom_ops_path,
                cuda_mem_ops_path,
                copy_backend_path,
            ]
        ),
        "admission": {
            "physical_movement_calls": physical_movement_calls,
            "prefix_hit_observed": prefix_hit_tokens > 0,
            "preemption_observed": preemption_count > 0,
            "default_scheduler_block_movement_candidate": (
                physical_movement_calls > 0
            ),
            "reason": (
                "The default prefix and preemption paths exposed physical KV "
                "movement."
                if physical_movement_calls > 0
                else "The default prefix and preemption paths exposed no "
                "physical KV block movement; prefix reuse is logical and "
                "preemption recomputes."
            ),
        },
    }
    engine.llm_engine.engine_core.shutdown()
    return report


def main() -> None:
    args = parse_args()
    report = run(args)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
