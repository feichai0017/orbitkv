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
from flashinfer.page import _append_paged_kv_cache_kernel, append_paged_kv_cache
from flashinfer.rope import _apply_rope_pos_ids
from flashinfer.trace.templates.rope import _apply_rope_pos_ids_reference
from flashinfer.utils import SINGLE_KERNEL_TMP_SIZE


PROVIDER_COMMIT = "5f3d1b3fc6e1ed8a79429986b3637802f1bd2b57"
MEASUREMENT = "eager_stream_batch_cuda_event"
FIXTURE_ID = "xorshift64_mod2001_bf16_v1"
PAGED_FIXTURE_ID = "xorshift64_mod2001_bf16_i32_page_table_v1"
RAGGED_FIXTURE_ID = "xorshift64_mod2001_bf16_i32_ragged_indptr_v1"
ROPE_FIXTURE_ID = "xorshift64_mod2001_bf16_i32_rope_pos_ids_v1"
ROPE_APPEND_FIXTURE_ID = "xorshift64_mod2001_bf16_i32_rope_paged_append_v1"
ROPE_APPEND_TOKENS_FIXTURE_ID = (
    "xorshift64_mod2001_bf16_i32_rope_paged_append_tokens_v1"
)
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


def digest_i32(values: torch.Tensor) -> str:
    digest = FNV_OFFSET_BASIS
    for value in values.contiguous().view(-1).tolist():
        digest ^= value & 0xFFFFFFFF
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
    fixture_id: str = FIXTURE_ID,
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
                "fixture_id": fixture_id,
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


def benchmark_paged_decode(
    case: str,
    batch_size: int,
    max_num_pages: int,
    query_heads: int,
    kv_heads: int,
    page_indptr_values: tuple[int, ...],
    page_indices_values: tuple[int, ...],
    last_page_len_values: tuple[int, ...],
    salt: int,
) -> None:
    head_dim = 128
    page_size = 16
    query_host = deterministic_bf16(
        batch_size * query_heads * head_dim, salt
    )
    key_host = deterministic_bf16(
        max_num_pages * page_size * kv_heads * head_dim,
        salt ^ 0x4B455900,
    )
    value_host = deterministic_bf16(
        max_num_pages * page_size * kv_heads * head_dim,
        salt ^ 0x56414C554500,
    )
    page_indptr_host = torch.tensor(page_indptr_values, dtype=torch.int32)
    page_indices_host = torch.tensor(page_indices_values, dtype=torch.int32)
    last_page_len_host = torch.tensor(last_page_len_values, dtype=torch.int32)
    query = query_host.reshape(batch_size, query_heads, head_dim).to("cuda")
    key_pages = key_host.reshape(
        max_num_pages, page_size, kv_heads, head_dim
    ).to("cuda")
    value_pages = value_host.reshape(
        max_num_pages, page_size, kv_heads, head_dim
    ).to("cuda")
    page_indptr = page_indptr_host.to("cuda")
    page_indices = page_indices_host.to("cuda")
    last_page_len = last_page_len_host.to("cuda")
    output = torch.empty_like(query)
    lse = torch.empty(
        (batch_size, query_heads), device="cuda", dtype=torch.float32
    )
    workspace = torch.zeros(
        128 * 1024 * 1024, device="cuda", dtype=torch.uint8
    )
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=False,
        use_tensor_cores=False,
        backend="fa2",
    )
    wrapper.plan(
        page_indptr,
        page_indices,
        last_page_len,
        query_heads,
        kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        window_left=-1,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        sm_scale=1.0 / math.sqrt(head_dim),
        disable_split_kv=False,
    )
    torch.cuda.synchronize()

    def run() -> None:
        wrapper.run(
            query,
            (key_pages, value_pages),
            out=output,
            lse=lse,
            return_lse=True,
            enable_pdl=False,
        )

    samples_us = benchmark(run)
    request_kv_lens = [
        (page_indptr_values[index + 1] - page_indptr_values[index] - 1)
        * page_size
        + last_page_len_values[index]
        for index in range(batch_size)
    ]
    write_record(
        "paged_batch_decode",
        case,
        "NHD_D128_page16",
        {
            "algorithm": "flashinfer_batch_decode_wrapper",
            "backend": "fa2",
            "use_tensor_cores": False,
            "disable_split_kv": False,
            "page_table_location": "device",
        },
        1,
        {
            "batch_size": batch_size,
            "max_num_pages": max_num_pages,
            "referenced_pages": len(page_indices_values),
            "request_kv_lens": request_kv_lens,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
        },
        {
            "query": digest_bf16(query_host),
            "key_pages": digest_bf16(key_host),
            "value_pages": digest_bf16(value_host),
            "page_indptr": digest_i32(page_indptr_host),
            "page_indices": digest_i32(page_indices_host),
            "last_page_len": digest_i32(last_page_len_host),
        },
        samples_us,
        PAGED_FIXTURE_ID,
    )


def benchmark_ragged_prefill(
    case: str,
    batch_size: int,
    query_heads: int,
    kv_heads: int,
    qo_indptr_values: tuple[int, ...],
    kv_indptr_values: tuple[int, ...],
    salt: int,
) -> None:
    head_dim = 128
    nnz_qo = qo_indptr_values[-1]
    nnz_kv = kv_indptr_values[-1]
    query_host = deterministic_bf16(nnz_qo * query_heads * head_dim, salt)
    key_host = deterministic_bf16(
        nnz_kv * kv_heads * head_dim, salt ^ 0x4B455900
    )
    value_host = deterministic_bf16(
        nnz_kv * kv_heads * head_dim, salt ^ 0x56414C554500
    )
    qo_indptr_host = torch.tensor(qo_indptr_values, dtype=torch.int32)
    kv_indptr_host = torch.tensor(kv_indptr_values, dtype=torch.int32)
    query = query_host.reshape(nnz_qo, query_heads, head_dim).to("cuda")
    key = key_host.reshape(nnz_kv, kv_heads, head_dim).to("cuda")
    value = value_host.reshape(nnz_kv, kv_heads, head_dim).to("cuda")
    qo_indptr = qo_indptr_host.to("cuda")
    kv_indptr = kv_indptr_host.to("cuda")
    output = torch.empty_like(query)
    lse = torch.empty(
        (nnz_qo, query_heads), device="cuda", dtype=torch.float32
    )
    workspace = torch.zeros(
        128 * 1024 * 1024, device="cuda", dtype=torch.uint8
    )
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=False,
        backend="fa2",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        query_heads,
        kv_heads,
        head_dim,
        causal=True,
        pos_encoding_mode="NONE",
        window_left=-1,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        sm_scale=1.0 / math.sqrt(head_dim),
        disable_split_kv=False,
    )
    torch.cuda.synchronize()

    def run() -> None:
        wrapper.run(
            query,
            key,
            value,
            out=output,
            lse=lse,
            return_lse=True,
            enable_pdl=False,
        )

    samples_us = benchmark(run)
    request_qo_lens = [
        qo_indptr_values[index + 1] - qo_indptr_values[index]
        for index in range(batch_size)
    ]
    request_kv_lens = [
        kv_indptr_values[index + 1] - kv_indptr_values[index]
        for index in range(batch_size)
    ]
    write_record(
        "ragged_prefill",
        case,
        "NHD_D128_ragged",
        {
            "algorithm": "flashinfer_batch_ragged_prefill_wrapper",
            "backend": "fa2",
            "causal": "bottom_right",
            "disable_split_kv": False,
            "indptr_location": "device",
        },
        1,
        {
            "batch_size": batch_size,
            "nnz_qo": nnz_qo,
            "nnz_kv": nnz_kv,
            "request_qo_lens": request_qo_lens,
            "request_kv_lens": request_kv_lens,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
        },
        {
            "query": digest_bf16(query_host),
            "key": digest_bf16(key_host),
            "value": digest_bf16(value_host),
            "qo_indptr": digest_i32(qo_indptr_host),
            "kv_indptr": digest_i32(kv_indptr_host),
        },
        samples_us,
        RAGGED_FIXTURE_ID,
    )


def benchmark_rope() -> None:
    tokens = 96
    query_heads = 16
    key_heads = 4
    head_dim = 128
    query_host = deterministic_bf16(
        tokens * query_heads * head_dim, 0x524F5045
    )
    key_host = deterministic_bf16(tokens * key_heads * head_dim, 0x4B455900)
    position_ids_host = torch.tensor(
        [*range(224, 256), *range(960, 1024)], dtype=torch.int32
    )
    query = query_host.reshape(tokens, query_heads, head_dim).to("cuda")
    key = key_host.reshape(tokens, key_heads, head_dim).to("cuda")
    position_ids = position_ids_host.to("cuda")
    query_output = torch.empty_like(query)
    key_output = torch.empty_like(key)

    def run() -> None:
        _apply_rope_pos_ids(
            query,
            key,
            query_output,
            key_output,
            position_ids,
            head_dim,
            False,
            1.0,
            10000.0,
        )

    # Resolve and compile the fixed module before entering the timed region.
    flashinfer.apply_rope_pos_ids(
        query,
        key,
        position_ids,
        rotary_dim=head_dim,
        interleave=False,
        rope_scale=1.0,
        rope_theta=10000.0,
    )
    torch.cuda.synchronize()
    samples_us = benchmark(run)
    expected_query, expected_key = _apply_rope_pos_ids_reference(
        query,
        key,
        position_ids,
        rotary_dim=head_dim,
        interleave=False,
        rope_scale=1.0,
        rope_theta=10000.0,
    )
    expected_query = expected_query.to(torch.bfloat16)
    expected_key = expected_key.to(torch.bfloat16)
    torch.cuda.synchronize()
    query_max_abs = (
        (query_output.float() - expected_query.float()).abs().max().item()
    )
    key_max_abs = (key_output.float() - expected_key.float()).abs().max().item()
    query_bit_mismatches = (
        query_output.view(torch.int16) != expected_query.view(torch.int16)
    ).sum().item()
    key_bit_mismatches = (
        key_output.view(torch.int16) != expected_key.view(torch.int16)
    ).sum().item()
    if query_max_abs > 0.015625 or key_max_abs > 0.015625:
        raise RuntimeError(
            "FlashInfer RoPE output exceeded the BF16 correctness limit: "
            f"query={query_max_abs}, key={key_max_abs}"
        )
    write_record(
        "rope",
        "bf16_rope_pos_ids_t96_qh16_kh4_d128_neox",
        "NHD_D128_neox_split_half",
        {
            "algorithm": "flashinfer_apply_rope_pos_ids",
            "position_mode": "explicit_i32",
            "rotary_dim": 128,
            "rope_scale": 1.0,
            "rope_theta": 10000.0,
            "correctness": {
                "reference": "FlashInfer v0.6.16.post1 trace reference",
                "query_max_abs": query_max_abs,
                "query_bit_mismatches": query_bit_mismatches,
                "query_digest": digest_bf16(query_output),
                "query_reference_digest": digest_bf16(expected_query),
                "key_max_abs": key_max_abs,
                "key_bit_mismatches": key_bit_mismatches,
                "key_digest": digest_bf16(key_output),
                "key_reference_digest": digest_bf16(expected_key),
            },
        },
        1,
        {
            "tokens": tokens,
            "query_heads": query_heads,
            "key_heads": key_heads,
            "head_dim": head_dim,
            "rotary_dim": head_dim,
            "position_ranges": [[224, 256], [960, 1024]],
        },
        {
            "query": digest_bf16(query_host),
            "key": digest_bf16(key_host),
            "position_ids": digest_i32(position_ids_host),
        },
        samples_us,
        ROPE_FIXTURE_ID,
    )


def benchmark_rope_paged_append() -> None:
    batch_size = 4
    max_num_pages = 8
    query_heads = 16
    key_heads = 4
    head_dim = 128
    page_size = 16
    query_host = deterministic_bf16(
        batch_size * query_heads * head_dim, 0x51504147
    )
    key_host = deterministic_bf16(batch_size * key_heads * head_dim, 0x4B504147)
    value_host = deterministic_bf16(
        batch_size * key_heads * head_dim, 0x56504147
    )
    key_pages_host = deterministic_bf16(
        max_num_pages * page_size * key_heads * head_dim, 0x4B434143
    )
    value_pages_host = deterministic_bf16(
        max_num_pages * page_size * key_heads * head_dim, 0x56434143
    )
    page_indptr_host = torch.tensor([0, 1, 3, 5, 8], dtype=torch.int32)
    page_indices_host = torch.tensor(
        [7, 2, 6, 5, 1, 7, 0, 4], dtype=torch.int32
    )
    last_page_len_host = torch.tensor([3, 16, 1, 9], dtype=torch.int32)
    batch_indices_host = torch.arange(batch_size, dtype=torch.int32)
    positions_host = torch.tensor([2, 31, 16, 40], dtype=torch.int32)

    query = query_host.reshape(batch_size, query_heads, head_dim).to("cuda")
    key = key_host.reshape(batch_size, key_heads, head_dim).to("cuda")
    value = value_host.reshape(batch_size, key_heads, head_dim).to("cuda")
    query_output = torch.empty_like(query)
    rotated_key = torch.empty_like(key)
    key_pages = key_pages_host.reshape(
        max_num_pages, page_size, key_heads, head_dim
    ).to("cuda")
    value_pages = value_pages_host.reshape(
        max_num_pages, page_size, key_heads, head_dim
    ).to("cuda")
    page_indptr = page_indptr_host.to("cuda")
    page_indices = page_indices_host.to("cuda")
    last_page_len = last_page_len_host.to("cuda")
    batch_indices = batch_indices_host.to("cuda")
    positions = positions_host.to("cuda")

    def run() -> None:
        _apply_rope_pos_ids(
            query,
            key,
            query_output,
            rotated_key,
            positions,
            head_dim,
            False,
            1.0,
            10000.0,
        )
        _append_paged_kv_cache_kernel(
            rotated_key,
            value,
            batch_indices,
            positions,
            key_pages,
            value_pages,
            page_indices,
            page_indptr,
            last_page_len,
            0,
        )

    # Resolve both fixed modules before entering the timed region.
    flashinfer.apply_rope_pos_ids(
        query,
        key,
        positions,
        rotary_dim=head_dim,
        interleave=False,
        rope_scale=1.0,
        rope_theta=10000.0,
    )
    append_paged_kv_cache(
        key,
        value,
        batch_indices,
        positions,
        (key_pages, value_pages),
        page_indices,
        page_indptr,
        last_page_len,
        kv_layout="NHD",
    )
    # Restore the matched initial cache after module warmup.
    key_pages.copy_(
        key_pages_host.reshape(max_num_pages, page_size, key_heads, head_dim)
    )
    value_pages.copy_(
        value_pages_host.reshape(max_num_pages, page_size, key_heads, head_dim)
    )
    torch.cuda.synchronize()
    samples_us = benchmark(run)
    expected_query, expected_key = _apply_rope_pos_ids_reference(
        query,
        key,
        positions,
        rotary_dim=head_dim,
        interleave=False,
        rope_scale=1.0,
        rope_theta=10000.0,
    )
    expected_query = expected_query.to(torch.bfloat16)
    expected_key = expected_key.to(torch.bfloat16)
    expected_key_pages = key_pages_host.reshape(
        max_num_pages, page_size, key_heads, head_dim
    ).to("cuda")
    expected_value_pages = value_pages_host.reshape(
        max_num_pages, page_size, key_heads, head_dim
    ).to("cuda")
    for request in range(batch_size):
        position = positions_host[request].item()
        page_slot = position // page_size
        page_offset = position % page_size
        physical_page = page_indices_host[
            page_indptr_host[request].item() + page_slot
        ].item()
        expected_key_pages[physical_page, page_offset].copy_(expected_key[request])
        expected_value_pages[physical_page, page_offset].copy_(value[request])
    torch.cuda.synchronize()
    query_max_abs = (
        (query_output.float() - expected_query.float()).abs().max().item()
    )
    key_pages_max_abs = (
        (key_pages.float() - expected_key_pages.float()).abs().max().item()
    )
    value_pages_max_abs = (
        (value_pages.float() - expected_value_pages.float()).abs().max().item()
    )
    query_bit_mismatches = (
        query_output.view(torch.int16) != expected_query.view(torch.int16)
    ).sum().item()
    key_pages_bit_mismatches = (
        key_pages.view(torch.int16) != expected_key_pages.view(torch.int16)
    ).sum().item()
    value_pages_bit_mismatches = (
        value_pages.view(torch.int16) != expected_value_pages.view(torch.int16)
    ).sum().item()
    if (
        query_max_abs > 0.015625
        or key_pages_max_abs > 0.015625
        or value_pages_max_abs != 0.0
    ):
        raise RuntimeError(
            "FlashInfer RoPE paged append exceeded the BF16 correctness limits: "
            f"query={query_max_abs}, key_pages={key_pages_max_abs}, "
            f"value_pages={value_pages_max_abs}"
        )
    torch.cuda.synchronize()
    write_record(
        "rope_paged_kv_append",
        "bf16_rope_paged_append_b4_qh16_kh4_d128_p16",
        "NHD_D128_neox_split_half_page16",
        {
            "algorithm": "flashinfer_rope_then_paged_append",
            "kernels": 2,
            "positions": [2, 31, 16, 40],
            "physical_slots": [[7, 2], [6, 15], [1, 0], [4, 8]],
            "correctness": {
                "reference": "FlashInfer v0.6.16.post1 trace RoPE reference plus explicit page writes",
                "query_max_abs": query_max_abs,
                "query_bit_mismatches": query_bit_mismatches,
                "query_digest": digest_bf16(query_output),
                "query_reference_digest": digest_bf16(expected_query),
                "key_pages_max_abs": key_pages_max_abs,
                "key_pages_bit_mismatches": key_pages_bit_mismatches,
                "key_pages_digest": digest_bf16(key_pages),
                "key_pages_reference_digest": digest_bf16(expected_key_pages),
                "value_pages_max_abs": value_pages_max_abs,
                "value_pages_bit_mismatches": value_pages_bit_mismatches,
                "value_pages_digest": digest_bf16(value_pages),
                "value_pages_reference_digest": digest_bf16(expected_value_pages),
            },
        },
        2,
        {
            "batch_size": batch_size,
            "max_num_pages": max_num_pages,
            "query_heads": query_heads,
            "key_heads": key_heads,
            "head_dim": head_dim,
            "page_size": page_size,
        },
        {
            "query": digest_bf16(query_host),
            "key": digest_bf16(key_host),
            "value": digest_bf16(value_host),
            "key_pages_initial": digest_bf16(key_pages_host),
            "value_pages_initial": digest_bf16(value_pages_host),
            "page_indptr": digest_i32(page_indptr_host),
            "page_indices": digest_i32(page_indices_host),
            "last_page_len": digest_i32(last_page_len_host),
        },
        samples_us,
        ROPE_APPEND_FIXTURE_ID,
    )


def benchmark_rope_paged_append_tokens() -> None:
    tokens = 6
    batch_size = 3
    max_num_pages = 8
    query_heads = 16
    key_heads = 4
    head_dim = 128
    page_size = 16
    query_host = deterministic_bf16(tokens * query_heads * head_dim, 0x54514147)
    key_host = deterministic_bf16(tokens * key_heads * head_dim, 0x544B4147)
    value_host = deterministic_bf16(tokens * key_heads * head_dim, 0x54564147)
    key_pages_host = deterministic_bf16(
        max_num_pages * page_size * key_heads * head_dim, 0x544B4343
    )
    value_pages_host = deterministic_bf16(
        max_num_pages * page_size * key_heads * head_dim, 0x54564343
    )
    batch_indices_host = torch.tensor([2, 0, 1, 0, 2, 1], dtype=torch.int32)
    positions_host = torch.tensor([5, 17, 20, 16, 4, 19], dtype=torch.int32)
    page_indptr_host = torch.tensor([0, 2, 4, 5], dtype=torch.int32)
    page_indices_host = torch.tensor([7, 3, 2, 6, 3], dtype=torch.int32)
    last_page_len_host = torch.tensor([2, 5, 6], dtype=torch.int32)

    query = query_host.reshape(tokens, query_heads, head_dim).to("cuda")
    key = key_host.reshape(tokens, key_heads, head_dim).to("cuda")
    value = value_host.reshape(tokens, key_heads, head_dim).to("cuda")
    query_output = torch.empty_like(query)
    rotated_key = torch.empty_like(key)
    key_pages = key_pages_host.reshape(
        max_num_pages, page_size, key_heads, head_dim
    ).to("cuda")
    value_pages = value_pages_host.reshape(
        max_num_pages, page_size, key_heads, head_dim
    ).to("cuda")
    batch_indices = batch_indices_host.to("cuda")
    positions = positions_host.to("cuda")
    page_indptr = page_indptr_host.to("cuda")
    page_indices = page_indices_host.to("cuda")
    last_page_len = last_page_len_host.to("cuda")

    def run() -> None:
        _apply_rope_pos_ids(
            query,
            key,
            query_output,
            rotated_key,
            positions,
            head_dim,
            False,
            1.0,
            10000.0,
        )
        _append_paged_kv_cache_kernel(
            rotated_key,
            value,
            batch_indices,
            positions,
            key_pages,
            value_pages,
            page_indices,
            page_indptr,
            last_page_len,
            0,
        )

    flashinfer.apply_rope_pos_ids(
        query,
        key,
        positions,
        rotary_dim=head_dim,
        interleave=False,
        rope_scale=1.0,
        rope_theta=10000.0,
    )
    append_paged_kv_cache(
        key,
        value,
        batch_indices,
        positions,
        (key_pages, value_pages),
        page_indices,
        page_indptr,
        last_page_len,
        kv_layout="NHD",
    )
    key_pages.copy_(
        key_pages_host.reshape(max_num_pages, page_size, key_heads, head_dim)
    )
    value_pages.copy_(
        value_pages_host.reshape(max_num_pages, page_size, key_heads, head_dim)
    )
    torch.cuda.synchronize()
    samples_us = benchmark(run)
    expected_query, expected_key = _apply_rope_pos_ids_reference(
        query,
        key,
        positions,
        rotary_dim=head_dim,
        interleave=False,
        rope_scale=1.0,
        rope_theta=10000.0,
    )
    expected_query = expected_query.to(torch.bfloat16)
    expected_key = expected_key.to(torch.bfloat16)
    expected_key_pages = key_pages_host.reshape(
        max_num_pages, page_size, key_heads, head_dim
    ).to("cuda")
    expected_value_pages = value_pages_host.reshape(
        max_num_pages, page_size, key_heads, head_dim
    ).to("cuda")
    physical_slots: list[list[int]] = []
    for token in range(tokens):
        request = batch_indices_host[token].item()
        position = positions_host[token].item()
        page_slot = position // page_size
        page_offset = position % page_size
        physical_page = page_indices_host[
            page_indptr_host[request].item() + page_slot
        ].item()
        physical_slots.append([physical_page, page_offset])
        expected_key_pages[physical_page, page_offset].copy_(expected_key[token])
        expected_value_pages[physical_page, page_offset].copy_(value[token])
    torch.cuda.synchronize()
    query_max_abs = (
        (query_output.float() - expected_query.float()).abs().max().item()
    )
    key_pages_max_abs = (
        (key_pages.float() - expected_key_pages.float()).abs().max().item()
    )
    value_pages_max_abs = (
        (value_pages.float() - expected_value_pages.float()).abs().max().item()
    )
    query_bit_mismatches = (
        query_output.view(torch.int16) != expected_query.view(torch.int16)
    ).sum().item()
    key_pages_bit_mismatches = (
        key_pages.view(torch.int16) != expected_key_pages.view(torch.int16)
    ).sum().item()
    value_pages_bit_mismatches = (
        value_pages.view(torch.int16) != expected_value_pages.view(torch.int16)
    ).sum().item()
    if (
        query_max_abs > 0.015625
        or key_pages_max_abs > 0.015625
        or value_pages_max_abs != 0.0
    ):
        raise RuntimeError(
            "FlashInfer explicit RoPE paged append exceeded correctness limits: "
            f"query={query_max_abs}, key_pages={key_pages_max_abs}, "
            f"value_pages={value_pages_max_abs}"
        )
    torch.cuda.synchronize()
    write_record(
        "rope_paged_kv_append_tokens",
        "bf16_rope_paged_append_t6_b3_qh16_kh4_d128_p16",
        "NHD_D128_neox_split_half_page16",
        {
            "algorithm": "flashinfer_rope_then_paged_append_explicit_tokens",
            "kernels": 2,
            "batch_indices": [2, 0, 1, 0, 2, 1],
            "positions": [5, 17, 20, 16, 4, 19],
            "physical_slots": physical_slots,
            "correctness": {
                "reference": "FlashInfer v0.6.16.post1 trace RoPE reference plus explicit page writes",
                "query_max_abs": query_max_abs,
                "query_bit_mismatches": query_bit_mismatches,
                "query_digest": digest_bf16(query_output),
                "query_reference_digest": digest_bf16(expected_query),
                "key_pages_max_abs": key_pages_max_abs,
                "key_pages_bit_mismatches": key_pages_bit_mismatches,
                "key_pages_digest": digest_bf16(key_pages),
                "key_pages_reference_digest": digest_bf16(expected_key_pages),
                "value_pages_max_abs": value_pages_max_abs,
                "value_pages_bit_mismatches": value_pages_bit_mismatches,
                "value_pages_digest": digest_bf16(value_pages),
                "value_pages_reference_digest": digest_bf16(expected_value_pages),
            },
        },
        2,
        {
            "tokens": tokens,
            "batch_size": batch_size,
            "max_num_pages": max_num_pages,
            "query_heads": query_heads,
            "key_heads": key_heads,
            "head_dim": head_dim,
            "page_size": page_size,
        },
        {
            "query": digest_bf16(query_host),
            "key": digest_bf16(key_host),
            "value": digest_bf16(value_host),
            "key_pages_initial": digest_bf16(key_pages_host),
            "value_pages_initial": digest_bf16(value_pages_host),
            "batch_indices": digest_i32(batch_indices_host),
            "positions": digest_i32(positions_host),
            "page_indptr": digest_i32(page_indptr_host),
            "page_indices": digest_i32(page_indices_host),
            "last_page_len": digest_i32(last_page_len_host),
        },
        samples_us,
        ROPE_APPEND_TOKENS_FIXTURE_ID,
    )


def main() -> None:
    requested = os.environ.get(
        "LOOM_BENCH_OPERATORS",
        "rms_norm,gemm,single_decode,paged_batch_decode,ragged_prefill,rope,rope_paged_kv_append,rope_paged_kv_append_tokens",
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
    if "paged_batch_decode" in requested:
        for args in (
            (
                "bf16_paged_mha_b1_l1_qh8_kvh8_d128_p16",
                1,
                2,
                8,
                8,
                (0, 1),
                (1,),
                (1,),
                0x1001,
            ),
            (
                "bf16_paged_mqa_b3_l16_23_48_qh8_kvh1_d128_p16",
                3,
                7,
                8,
                1,
                (0, 1, 3, 6),
                (4, 6, 1, 5, 0, 3),
                (16, 7, 16),
                0x2001,
            ),
            (
                "bf16_paged_gqa4_b4_l3_32_17_41_qh16_kvh4_d128_p16",
                4,
                8,
                16,
                4,
                (0, 1, 3, 5, 8),
                (7, 2, 6, 5, 1, 7, 0, 4),
                (3, 16, 1, 9),
                0x4001,
            ),
        ):
            benchmark_paged_decode(*args)
    if "ragged_prefill" in requested:
        for args in (
            (
                "bf16_ragged_mha_b1_q16_kv16_qh8_kvh8_d128",
                1,
                8,
                8,
                (0, 16),
                (0, 16),
                0x1001,
            ),
            (
                "bf16_ragged_mqa_b3_q1_4_16_kv128_256_512_qh8_kvh1_d128",
                3,
                8,
                1,
                (0, 1, 5, 21),
                (0, 128, 384, 896),
                0x2001,
            ),
            (
                "bf16_ragged_gqa4_b2_q32_64_kv256_1024_qh16_kvh4_d128",
                2,
                16,
                4,
                (0, 32, 96),
                (0, 256, 1280),
                0x4001,
            ),
        ):
            benchmark_ragged_prefill(*args)
    if "rope" in requested:
        benchmark_rope()
    if "rope_paged_kv_append" in requested:
        benchmark_rope_paged_append()
    if "rope_paged_kv_append_tokens" in requested:
        benchmark_rope_paged_append_tokens()


if __name__ == "__main__":
    main()
