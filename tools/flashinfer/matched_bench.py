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
FIXTURE_ID = "xorshift64_mod2001_bf16_v1"
FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
MASK64 = (1 << 64) - 1


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


def deterministic_bf16(length: int, salt: int) -> torch.Tensor:
    state = 0x9E3779B97F4A7C15 ^ salt
    values: list[float] = []
    for _ in range(length):
        state ^= (state << 13) & MASK64
        state ^= state >> 7
        state ^= (state << 17) & MASK64
        state &= MASK64
        signed = state % 2001 - 1000
        values.append(signed / 2048.0)
    return torch.tensor(values, dtype=torch.float32).to(torch.bfloat16)


def digest_bf16(values: torch.Tensor) -> str:
    bits = values.contiguous().view(torch.int16).view(-1).tolist()
    digest = FNV_OFFSET_BASIS
    for value in bits:
        digest ^= value & 0xFFFF
        digest = (digest * FNV_PRIME) & MASK64
    return f"{digest:016x}"


def write_record(
    operator: str,
    case: str,
    layout: str,
    execution: dict[str, object],
    kernels_per_call: int,
    shape: dict[str, int],
    fixture_digests: dict[str, str],
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
                "execution": execution,
                "kernels_per_call": kernels_per_call,
                "shape": shape,
                "fixture_id": FIXTURE_ID,
                "fixture_digests": fixture_digests,
                "warmup_launches": WARMUP,
                "launches_per_sample": LAUNCHES,
                "samples_us": samples_us,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def benchmark_rmsnorm(rows: int, hidden_size: int) -> None:
    input_host = deterministic_bf16(rows * hidden_size, 0x524D534E)
    weight_host = deterministic_bf16(hidden_size, 0x57454947)
    input_tensor = input_host.reshape(rows, hidden_size).to("cuda")
    weight = weight_host.to("cuda")
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
        {"algorithm": "flashinfer_rmsnorm", "enable_pdl": False},
        1,
        {"rows": rows, "hidden_size": hidden_size},
        {"input": digest_bf16(input_host), "weight": digest_bf16(weight_host)},
        samples_us,
    )


def benchmark_gemm() -> None:
    m, n, k = 1, 4096, 4096
    activation_host = deterministic_bf16(m * k, 0x41435449)
    activation = activation_host.reshape(m, k).to("cuda")
    # Setup transposes once. The timed API sees the FlashInfer contract's
    # column-major [K,N] weight view without a timed tensor copy.
    weight_host = deterministic_bf16(n * k, 0x47454D4D)
    weight_storage = weight_host.reshape(n, k).to("cuda")
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
        {"algorithm": "cublaslt", "tactic": 0},
        1,
        {"m": m, "n": n, "k": k},
        {
            "activation": digest_bf16(activation_host),
            "weight_storage": digest_bf16(weight_host),
        },
        samples_us,
    )


def benchmark_decode(
    case: str, kv_len: int, query_heads: int, kv_heads: int
) -> None:
    head_dim = 128
    query_host = deterministic_bf16(query_heads * head_dim, 0x51554552)
    key_host = deterministic_bf16(kv_len * kv_heads * head_dim, 0x4B455900)
    value_host = deterministic_bf16(kv_len * kv_heads * head_dim, 0x56414C55)
    query = query_host.reshape(query_heads, head_dim).to("cuda")
    key = key_host.reshape(kv_len, kv_heads, head_dim).to("cuda")
    value = value_host.reshape(kv_len, kv_heads, head_dim).to("cuda")
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
        {"algorithm": "flashinfer_single_decode_module"},
        1,
        {
            "kv_len": kv_len,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
        },
        {
            "query": digest_bf16(query_host),
            "key": digest_bf16(key_host),
            "value": digest_bf16(value_host),
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
