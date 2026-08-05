#!/usr/bin/env python3
"""Matched FlashInfer baseline for Loom Infer's admitted BF16 contracts."""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Callable

import torch

import flashinfer
from flashinfer.decode import get_single_decode_module
from flashinfer.gemm.gemm_base import (
    DEFAULT_WORKSPACE_SIZE,
    get_mm_bf16_cublaslt_module,
)
from flashinfer.utils import SINGLE_KERNEL_TMP_SIZE


PROVIDER_COMMIT = "5f3d1b3fc6e1ed8a79429986b3637802f1bd2b57"
MEASUREMENT = "eager_stream_batch_cuda_event"


def env_positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


WARMUP = env_positive_int("LOOM_BENCH_WARMUP", 20)
LAUNCHES = env_positive_int("LOOM_BENCH_LAUNCHES", 100)
SAMPLES = env_positive_int("LOOM_BENCH_SAMPLES", 30)
RUN_LABEL = os.environ.get("LOOM_BENCH_RUN_LABEL", "unlabeled")


def benchmark(fn: Callable[[], None]) -> list[float]:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples_us: list[float] = []
    for _ in range(SAMPLES):
        start.record()
        for _ in range(LAUNCHES):
            fn()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0 / LAUNCHES)
    return samples_us


def write_record(
    operator: str,
    case: str,
    layout: str,
    shape: dict[str, int],
    samples_us: list[float],
) -> None:
    print(
        json.dumps(
            {
                "schema_version": 1,
                "provider": "flashinfer",
                "provider_version": flashinfer.__version__,
                "provider_commit": PROVIDER_COMMIT,
                "run_label": RUN_LABEL,
                "measurement": MEASUREMENT,
                "operator": operator,
                "case": case,
                "dtype": "bf16",
                "layout": layout,
                "shape": shape,
                "warmup_launches": WARMUP,
                "launches_per_sample": LAUNCHES,
                "samples_us": samples_us,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def benchmark_rmsnorm(rows: int, hidden_size: int) -> None:
    input_tensor = torch.randn(
        (rows, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn((hidden_size,), device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(input_tensor)

    def run() -> None:
        flashinfer.rmsnorm(
            input_tensor,
            weight,
            eps=1.0e-5,
            out=output,
            enable_pdl=False,
        )

    samples_us = benchmark(run)
    write_record(
        "rms_norm",
        f"bf16_r{rows}_h{hidden_size}",
        "contiguous_rows_hidden",
        {"rows": rows, "hidden_size": hidden_size},
        samples_us,
    )


def benchmark_gemm() -> None:
    m, n, k = 1, 4096, 4096
    activation = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    # Setup transposes once. The timed API sees the FlashInfer contract's
    # column-major [K,N] weight view without a timed tensor copy.
    weight_storage = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    weight = weight_storage.transpose(0, 1)
    output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(
        DEFAULT_WORKSPACE_SIZE, device="cuda", dtype=torch.uint8
    )
    runner = get_mm_bf16_cublaslt_module().cublaslt_bf16_gemm_runner()
    inputs = [activation, weight, None, False, output, workspace]
    # Freeze the same first heuristic tactic policy Loom currently uses.
    runner.forward(inputs, tactic=0, do_preparation=True)
    torch.cuda.synchronize()

    def run() -> None:
        runner.forward(inputs, tactic=0)

    samples_us = benchmark(run)
    write_record(
        "gemm",
        "bf16_m1_n4096_k4096_cublaslt",
        "A_row_major_W_row_major_transposed",
        {"m": m, "n": n, "k": k},
        samples_us,
    )


def benchmark_decode(
    case: str, kv_len: int, query_heads: int, kv_heads: int
) -> None:
    head_dim = 128
    query = torch.randn(
        (query_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    key = torch.randn(
        (kv_len, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    lse = torch.empty((query_heads,), device="cuda", dtype=torch.float32)
    tmp = torch.empty(SINGLE_KERNEL_TMP_SIZE, device="cuda", dtype=torch.uint8)
    module = get_single_decode_module(
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        head_dim,
        head_dim,
        0,  # PosEncodingMode.NONE
        False,  # sliding window
        False,  # logits soft cap
    )
    sm_scale = 1.0 / math.sqrt(head_dim)

    def run() -> None:
        module.run(
            query,
            key,
            value,
            tmp,
            output,
            lse,
            None,  # alibi slopes
            0,  # TensorLayout.NHD
            -1,  # full window
            0.0,  # logits soft cap
            sm_scale,
            1.0,  # rope scale
            1.0e4,  # rope theta
        )

    samples_us = benchmark(run)
    write_record(
        "single_decode",
        case,
        "NHD_D128",
        {
            "kv_len": kv_len,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
        },
        samples_us,
    )


def main() -> None:
    requested = os.environ.get(
        "LOOM_BENCH_OPERATORS", "rms_norm,gemm,single_decode"
    ).split(",")
    if "rms_norm" in requested:
        for rows, hidden_size in ((1, 4096), (8, 4096), (64, 4096), (16, 8192)):
            try:
                benchmark_rmsnorm(rows, hidden_size)
            except Exception as error:
                print(
                    f"flashinfer rms_norm unavailable: {error}",
                    file=sys.stderr,
                )
                break
    if "gemm" in requested:
        benchmark_gemm()
    if "single_decode" in requested:
        for args in (
            ("bf16_mha_l1_qh8_kvh8_d128", 1, 8, 8),
            ("bf16_mqa_l33_qh8_kvh1_d128", 33, 8, 1),
            ("bf16_gqa_l127_qh16_kvh4_d128", 127, 16, 4),
            ("bf16_gqa_l4096_qh32_kvh4_d128", 4096, 32, 4),
        ):
            benchmark_decode(*args)


if __name__ == "__main__":
    main()
