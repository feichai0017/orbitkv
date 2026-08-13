#!/usr/bin/env python3
"""Matched TileLang/FlashInfer spike for Oxide's long paged-prefill gap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import flashinfer
import tilelang
import tilelang.language as T
import torch


BATCH_SIZE = 2
MAX_NUM_PAGES = 96
QUERY_HEADS = 16
KV_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 16
NNZ_QO = 96
MAX_Q_LEN = 64
GROUP_SIZE = QUERY_HEADS // KV_HEADS
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_DIM)
OUTPUT_LIMIT = 0.015625
LSE_LIMIT = 0.01
FIXTURE_SALT = 0x8001
FIXTURE_ID = "xorshift64_mod2001_bf16_i32_paged_prefill_v1"
TILELANG_COMMIT = "8001cc4ccf6149382d2019654a19f59c1d4d0482"
QO_INDPTR = (0, 32, 96)
PAGE_INDPTR = (0, 16, 80)
PAGE_INDICES = (
    15,
    3,
    27,
    9,
    31,
    1,
    35,
    5,
    39,
    7,
    43,
    11,
    47,
    13,
    51,
    17,
    19,
    21,
    23,
    25,
    29,
    33,
    37,
    41,
    45,
    49,
    53,
    55,
    57,
    59,
    61,
    63,
    65,
    67,
    69,
    71,
    73,
    75,
    77,
    79,
    81,
    83,
    85,
    87,
    89,
    91,
    93,
    95,
    0,
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28,
    30,
    32,
    34,
    36,
    38,
    40,
    42,
    44,
    46,
    48,
    50,
    52,
    54,
    56,
    58,
    60,
    62,
)
LAST_PAGE_LEN = (16, 16)
SCHEDULE = ("tilelang", "flashinfer", "flashinfer", "tilelang", "tilelang", "flashinfer")


@dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    num_stages: int
    threads: int

    @property
    def name(self) -> str:
        return f"m{self.block_m}_n{self.block_n}_s{self.num_stages}_t{self.threads}"


@dataclass
class Inputs:
    query: torch.Tensor
    key_pages: torch.Tensor
    value_pages: torch.Tensor
    qo_indptr: torch.Tensor
    page_indptr: torch.Tensor
    page_indices: torch.Tensor
    last_page_len: torch.Tensor
    expected_output: torch.Tensor
    expected_lse: torch.Tensor


CONFIGS = (
    KernelConfig(64, 32, 1, 128),
    KernelConfig(64, 32, 2, 128),
    KernelConfig(64, 64, 0, 128),
    KernelConfig(64, 64, 1, 128),
    KernelConfig(64, 64, 2, 128),
    KernelConfig(64, 64, 3, 128),
    KernelConfig(64, 64, 4, 128),
    KernelConfig(64, 128, 1, 128),
    KernelConfig(64, 128, 2, 128),
    KernelConfig(64, 128, 3, 128),
)


@tilelang.jit(
    out_idx=[],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def paged_prefill(config: KernelConfig):
    block_m = config.block_m
    block_n = config.block_n
    num_stages = config.num_stages
    threads = config.threads
    scale_log2e = SOFTMAX_SCALE * 1.4426950408889634

    @T.prim_func
    def main(
        query: T.Tensor([NNZ_QO, QUERY_HEADS, HEAD_DIM], T.bfloat16),
        key_pages: T.Tensor(
            [MAX_NUM_PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM], T.bfloat16
        ),
        value_pages: T.Tensor(
            [MAX_NUM_PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM], T.bfloat16
        ),
        qo_indptr: T.Tensor([BATCH_SIZE + 1], T.int32),
        page_indptr: T.Tensor([BATCH_SIZE + 1], T.int32),
        page_indices: T.Tensor([len(PAGE_INDICES)], T.int32),
        last_page_len: T.Tensor([BATCH_SIZE], T.int32),
        output: T.Tensor([NNZ_QO, QUERY_HEADS, HEAD_DIM], T.bfloat16),
        lse: T.Tensor([NNZ_QO, QUERY_HEADS], T.float32),
    ):
        with T.Kernel(
            T.ceildiv(MAX_Q_LEN, block_m),
            QUERY_HEADS,
            BATCH_SIZE,
            threads=threads,
        ) as (query_block, query_head, request):
            query_shared = T.alloc_shared([block_m, HEAD_DIM], T.bfloat16)
            key_shared = T.alloc_shared([block_n, HEAD_DIM], T.bfloat16)
            value_shared = T.alloc_shared([block_n, HEAD_DIM], T.bfloat16)
            output_shared = T.alloc_shared([block_m, HEAD_DIM], T.bfloat16)
            score = T.alloc_fragment([block_m, block_n], T.float32)
            probability = T.alloc_fragment([block_m, block_n], T.bfloat16)
            accumulator = T.alloc_fragment([block_m, HEAD_DIM], T.float32)
            row_max = T.alloc_fragment([block_m], T.float32)
            previous_row_max = T.alloc_fragment([block_m], T.float32)
            row_scale = T.alloc_fragment([block_m], T.float32)
            row_sum = T.alloc_fragment([block_m], T.float32)
            normalizer = T.alloc_fragment([block_m], T.float32)

            query_start = qo_indptr[request]
            query_len = qo_indptr[request + 1] - query_start
            page_start = page_indptr[request]
            page_count = page_indptr[request + 1] - page_start
            kv_len = (page_count - 1) * PAGE_SIZE + last_page_len[request]
            kv_head = query_head // GROUP_SIZE

            T.copy(
                query[
                    query_start
                    + query_block * block_m : query_start
                    + (query_block + 1) * block_m,
                    query_head,
                    :,
                ],
                query_shared,
            )
            T.fill(accumulator, 0)
            T.fill(normalizer, 0)
            T.fill(row_max, -T.infinity(T.float32))

            causal_offset = kv_len - query_len
            visible_kv_end = causal_offset + (query_block + 1) * block_m
            kv_blocks = T.min(
                T.ceildiv(visible_kv_end, block_n), T.ceildiv(kv_len, block_n)
            )

            for kv_block in T.Pipelined(kv_blocks, num_stages=num_stages):
                for row, component in T.Parallel(block_n, HEAD_DIM):
                    logical_token = kv_block * block_n + row
                    physical_page = page_indices[
                        page_start + logical_token // PAGE_SIZE
                    ]
                    key_shared[row, component] = key_pages[
                        physical_page,
                        logical_token % PAGE_SIZE,
                        kv_head,
                        component,
                    ]
                    value_shared[row, component] = value_pages[
                        physical_page,
                        logical_token % PAGE_SIZE,
                        kv_head,
                        component,
                    ]

                for row, column in T.Parallel(block_m, block_n):
                    query_index = query_block * block_m + row
                    kv_index = kv_block * block_n + column
                    score[row, column] = T.if_then_else(
                        query_index >= query_len
                        or kv_index >= kv_len
                        or kv_index > causal_offset + query_index,
                        -1e9,
                        0,
                    )

                T.gemm(
                    query_shared,
                    key_shared,
                    score,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                T.copy(row_max, previous_row_max)
                T.fill(row_max, -T.infinity(T.float32))
                T.reduce_max(score, row_max, dim=1, clear=False)
                for row in T.Parallel(block_m):
                    row_max[row] = T.max(row_max[row], previous_row_max[row])
                    row_scale[row] = T.exp2(
                        previous_row_max[row] * scale_log2e
                        - row_max[row] * scale_log2e
                    )
                for row, column in T.Parallel(block_m, block_n):
                    score[row, column] = T.exp2(
                        score[row, column] * scale_log2e
                        - row_max[row] * scale_log2e
                    )
                T.reduce_sum(score, row_sum, dim=1)
                for row in T.Parallel(block_m):
                    normalizer[row] = (
                        normalizer[row] * row_scale[row] + row_sum[row]
                    )
                T.copy(score, probability)
                for row, component in T.Parallel(block_m, HEAD_DIM):
                    accumulator[row, component] *= row_scale[row]
                T.gemm(
                    probability,
                    value_shared,
                    accumulator,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            for row, component in T.Parallel(block_m, HEAD_DIM):
                accumulator[row, component] /= normalizer[row]
            T.copy(accumulator, output_shared)
            for row in T.Parallel(block_m):
                normalizer[row] = (
                    T.log2(normalizer[row]) + row_max[row] * scale_log2e
                )
            for row, component in T.Parallel(block_m, HEAD_DIM):
                query_index = query_block * block_m + row
                if query_index < query_len:
                    output[
                        query_start + query_index, query_head, component
                    ] = output_shared[row, component]
            for row in T.Parallel(block_m):
                query_index = query_block * block_m + row
                if query_index < query_len:
                    lse[query_start + query_index, query_head] = normalizer[row]

    return main


def deterministic_bf16(length: int, salt: int) -> torch.Tensor:
    state = 0x9E3779B97F4A7C15 ^ salt
    values: list[float] = []
    for _ in range(length):
        state ^= (state << 13) & ((1 << 64) - 1)
        state ^= state >> 7
        state ^= (state << 17) & ((1 << 64) - 1)
        state &= (1 << 64) - 1
        values.append((state % 2001 - 1000) / 2048.0)
    return torch.tensor(values, dtype=torch.float32).to(torch.bfloat16)


def reference(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(query)
    lse = torch.empty((NNZ_QO, QUERY_HEADS), dtype=torch.float32)
    for request in range(BATCH_SIZE):
        query_start = QO_INDPTR[request]
        query_end = QO_INDPTR[request + 1]
        query_len = query_end - query_start
        page_start = PAGE_INDPTR[request]
        page_end = PAGE_INDPTR[request + 1]
        pages = PAGE_INDICES[page_start:page_end]
        key = key_pages[list(pages)].reshape(-1, KV_HEADS, HEAD_DIM)
        value = value_pages[list(pages)].reshape(-1, KV_HEADS, HEAD_DIM)
        kv_len = (len(pages) - 1) * PAGE_SIZE + LAST_PAGE_LEN[request]
        key = key[:kv_len].float()
        value = value[:kv_len].float()
        for query_index in range(query_len):
            visible = kv_len - query_len + query_index + 1
            for query_head in range(QUERY_HEADS):
                kv_head = query_head // GROUP_SIZE
                scores = (
                    query[query_start + query_index, query_head].float()
                    @ key[:visible, kv_head].T
                ) * SOFTMAX_SCALE
                probabilities = torch.softmax(scores, dim=-1)
                result = probabilities @ value[:visible, kv_head]
                output[query_start + query_index, query_head] = result.to(
                    torch.bfloat16
                )
                lse[query_start + query_index, query_head] = torch.logsumexp(
                    scores, dim=-1
                ) * math.log2(math.e)
    return output, lse


def make_inputs() -> Inputs:
    query_host = deterministic_bf16(NNZ_QO * QUERY_HEADS * HEAD_DIM, FIXTURE_SALT)
    key_host = deterministic_bf16(
        MAX_NUM_PAGES * PAGE_SIZE * KV_HEADS * HEAD_DIM,
        FIXTURE_SALT ^ 0x4B455900,
    )
    value_host = deterministic_bf16(
        MAX_NUM_PAGES * PAGE_SIZE * KV_HEADS * HEAD_DIM,
        FIXTURE_SALT ^ 0x56414C554500,
    )
    query_host = query_host.reshape(NNZ_QO, QUERY_HEADS, HEAD_DIM)
    key_host = key_host.reshape(MAX_NUM_PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM)
    value_host = value_host.reshape(
        MAX_NUM_PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM
    )
    expected_output, expected_lse = reference(query_host, key_host, value_host)
    return Inputs(
        query=query_host.cuda(),
        key_pages=key_host.cuda(),
        value_pages=value_host.cuda(),
        qo_indptr=torch.tensor(QO_INDPTR, dtype=torch.int32, device="cuda"),
        page_indptr=torch.tensor(PAGE_INDPTR, dtype=torch.int32, device="cuda"),
        page_indices=torch.tensor(
            PAGE_INDICES, dtype=torch.int32, device="cuda"
        ),
        last_page_len=torch.tensor(
            LAST_PAGE_LEN, dtype=torch.int32, device="cuda"
        ),
        expected_output=expected_output.cuda(),
        expected_lse=expected_lse.cuda(),
    )


def errors(
    output: torch.Tensor,
    lse: torch.Tensor,
    inputs: Inputs,
) -> dict[str, float]:
    output_error = (output.float() - inputs.expected_output.float()).abs()
    lse_error = (lse - inputs.expected_lse).abs()
    result = {
        "output_max_abs": output_error.max().item(),
        "output_mean_abs": output_error.mean().item(),
        "lse_max_abs": lse_error.max().item(),
        "lse_mean_abs": lse_error.mean().item(),
    }
    if result["output_max_abs"] > OUTPUT_LIMIT:
        raise RuntimeError(f"output error exceeded limit: {result}")
    if result["lse_max_abs"] > LSE_LIMIT:
        raise RuntimeError(f"LSE error exceeded limit: {result}")
    return result


def measure(
    run: Callable[[], None],
    warmups: int,
    launches: int,
    samples: int,
) -> list[float]:
    for _ in range(warmups):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(samples):
        start.record()
        for _ in range(launches):
            run()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / launches)
    return values


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def make_flashinfer(inputs: Inputs) -> tuple[Callable[[], None], torch.Tensor, torch.Tensor]:
    output = torch.empty_like(inputs.query)
    lse = torch.empty((NNZ_QO, QUERY_HEADS), dtype=torch.float32, device="cuda")
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=False,
        backend="fa2",
    )
    wrapper.plan(
        inputs.qo_indptr,
        inputs.page_indptr,
        inputs.page_indices,
        inputs.last_page_len,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
        pos_encoding_mode="NONE",
        window_left=-1,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        sm_scale=SOFTMAX_SCALE,
        disable_split_kv=False,
    )

    def run() -> None:
        wrapper.run(
            inputs.query,
            (inputs.key_pages, inputs.value_pages),
            out=output,
            lse=lse,
            return_lse=True,
            enable_pdl=False,
        )

    return run, output, lse


def compile_tilelang(
    config: KernelConfig, inputs: Inputs
) -> tuple[Callable[[], None], torch.Tensor, torch.Tensor, str, float]:
    started = time.perf_counter()
    kernel = paged_prefill(config)
    compile_seconds = time.perf_counter() - started
    output = torch.empty_like(inputs.query)
    lse = torch.empty((NNZ_QO, QUERY_HEADS), dtype=torch.float32, device="cuda")

    def run() -> None:
        kernel(
            inputs.query,
            inputs.key_pages,
            inputs.value_pages,
            inputs.qo_indptr,
            inputs.page_indptr,
            inputs.page_indices,
            inputs.last_page_len,
            output,
            lse,
        )

    run()
    torch.cuda.synchronize()
    source_hash = hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest()
    return run, output, lse, source_hash, compile_seconds


def source_identity() -> dict[str, str | bool | None]:
    repository = Path(__file__).resolve().parents[2]
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": (
            commit_result.stdout.strip()
            if commit_result.returncode == 0
            else None
        ),
        "worktree_clean": (
            status_result.returncode == 0 and not status_result.stdout.strip()
        ),
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tune-warmups", type=int, default=10)
    parser.add_argument("--tune-launches", type=int, default=20)
    parser.add_argument("--tune-samples", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--launches", type=int, default=100)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    if min(
        args.tune_warmups,
        args.tune_launches,
        args.tune_samples,
        args.warmups,
        args.launches,
        args.samples,
    ) <= 0:
        raise ValueError("all measurement counts must be positive")

    torch.manual_seed(0)
    inputs = make_inputs()
    flashinfer_run, flashinfer_output, flashinfer_lse = make_flashinfer(inputs)
    flashinfer_run()
    torch.cuda.synchronize()
    flashinfer_correctness = errors(flashinfer_output, flashinfer_lse, inputs)

    tuning = []
    compiled = {}
    for config in CONFIGS:
        print(f"compiling TileLang {config.name}", flush=True)
        try:
            run, output, lse, source_hash, compile_seconds = compile_tilelang(
                config, inputs
            )
            correctness = errors(output, lse, inputs)
            samples = measure(
                run,
                args.tune_warmups,
                args.tune_launches,
                args.tune_samples,
            )
            tuning.append(
                {
                    "config": asdict(config),
                    "compile_seconds": compile_seconds,
                    "source_sha256": source_hash,
                    "correctness": correctness,
                    "samples_microseconds": samples,
                    "distribution_microseconds": distribution(samples),
                }
            )
            compiled[config.name] = (run, output, lse)
        except Exception as error:
            tuning.append({"config": asdict(config), "error": repr(error)})
            print(f"candidate failed: {config.name}: {error!r}", flush=True)

    successful = [entry for entry in tuning if "error" not in entry]
    if not successful:
        raise RuntimeError("every TileLang configuration failed")
    best = min(
        successful,
        key=lambda entry: entry["distribution_microseconds"]["p50"],
    )
    best_config = KernelConfig(**best["config"])
    tilelang_run, tilelang_output, tilelang_lse = compiled[best_config.name]
    print(f"selected TileLang {best_config.name}", flush=True)

    providers = {
        "tilelang": (tilelang_run, tilelang_output, tilelang_lse),
        "flashinfer": (flashinfer_run, flashinfer_output, flashinfer_lse),
    }
    blocks = []
    for block_index, provider in enumerate(SCHEDULE):
        print(f"running block {block_index + 1}/{len(SCHEDULE)} {provider}", flush=True)
        run, output, lse = providers[provider]
        samples = measure(run, args.warmups, args.launches, args.samples)
        blocks.append(
            {
                "block_index": block_index,
                "provider": provider,
                "samples_microseconds": samples,
                "distribution_microseconds": distribution(samples),
                "correctness": errors(output, lse, inputs),
            }
        )

    summaries = []
    for provider in providers:
        pooled = [
            value
            for block in blocks
            if block["provider"] == provider
            for value in block["samples_microseconds"]
        ]
        summaries.append(
            {
                "provider": provider,
                "samples_microseconds": pooled,
                "distribution_microseconds": distribution(pooled),
            }
        )
    tilelang_p50 = next(
        item["distribution_microseconds"]["p50"]
        for item in summaries
        if item["provider"] == "tilelang"
    )
    flashinfer_p50 = next(
        item["distribution_microseconds"]["p50"]
        for item in summaries
        if item["provider"] == "flashinfer"
    )

    record = {
        "schema": "oxide.tilelang-paged-prefill-spike.v1",
        "unix_time_seconds": int(time.time()),
        "source": source_identity(),
        "fixture_id": FIXTURE_ID,
        "shape": {
            "batch_size": BATCH_SIZE,
            "query_lengths": [32, 64],
            "kv_lengths": [256, 1024],
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "page_size": PAGE_SIZE,
            "layout": "NHD",
            "dtype": "bf16",
            "causal": "bottom_right",
        },
        "software": {
            "tilelang_version": tilelang.__version__,
            "tilelang_source_reference": {
                "tag": "v0.1.13",
                "commit": TILELANG_COMMIT,
                "qualifies_installed_wheel_provenance": False,
            },
            "flashinfer_version": flashinfer.__version__,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        },
        "hardware": {
            "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
        },
        "protocol": {
            "same_process": True,
            "same_tensors": True,
            "same_stream": True,
            "preallocated_outputs": True,
            "compilation_timed": False,
            "planning_timed": False,
            "page_materialization_timed": False,
            "schedule": list(SCHEDULE),
            "warmups_per_block": args.warmups,
            "launches_per_sample": args.launches,
            "samples_per_block": args.samples,
            "measurement": "eager_stream_batch_cuda_event",
        },
        "flashinfer_correctness": flashinfer_correctness,
        "tuning": tuning,
        "selected_config": best["config"],
        "selected_source_sha256": best["source_sha256"],
        "blocks": blocks,
        "summaries": summaries,
        "comparison": {
            "tilelang_p50_microseconds": tilelang_p50,
            "flashinfer_p50_microseconds": flashinfer_p50,
            "tilelang_over_flashinfer": tilelang_p50 / flashinfer_p50,
            "tilelang_minus_flashinfer_microseconds": tilelang_p50
            - flashinfer_p50,
        },
        "excluded_claims": [
            "This spike does not include Oxide host validation or completion overhead.",
            "This one fixed shape does not establish model or serving performance.",
            "The selected TileLang configuration is not qualified for production.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
