#!/usr/bin/env python3
"""Matched CUDA Graph replay baseline for explicit RoPE paged KV append."""

from __future__ import annotations

import json
import os

import torch

import flashinfer
from flashinfer.page import _append_paged_kv_cache_kernel, append_paged_kv_cache
from flashinfer.rope import _apply_rope_pos_ids

from matched_bench import (
    PROVIDER_COMMIT,
    ROPE_APPEND_TOKENS_FIXTURE_ID,
    deterministic_bf16,
    digest_bf16,
    digest_i32,
    page_refcounts_i32,
    validate_flashinfer_installation,
)


MEASUREMENT = "fixed_address_cuda_graph_single_replay_event"
CASE = "bf16_rope_paged_append_t6_b3_qh16_kh4_d128_p16"


def env_positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def main() -> None:
    validate_flashinfer_installation()
    warmup = env_positive_int("OXIDE_BENCH_WARMUP", 200)
    launches = env_positive_int("OXIDE_BENCH_LAUNCHES", 1)
    samples = env_positive_int("OXIDE_BENCH_SAMPLES", 100)
    if launches != 1:
        raise ValueError("Graph benchmark requires OXIDE_BENCH_LAUNCHES=1")
    run_label = os.environ.get("OXIDE_BENCH_RUN_LABEL", "unlabeled")

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
    page_indices_host = torch.tensor([7, 3, 2, 6, 5], dtype=torch.int32)
    last_page_len_host = torch.tensor([2, 5, 6], dtype=torch.int32)
    page_refcounts_host = page_refcounts_i32(max_num_pages, page_indices_host)

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
    run()
    torch.cuda.synchronize()
    reference_query = query_output.clone()
    reference_key_pages = key_pages.clone()
    reference_value_pages = value_pages.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    completion = torch.cuda.Event(enable_timing=False)
    end = torch.cuda.Event(enable_timing=True)
    samples_us: list[float] = []
    for _ in range(samples):
        start.record()
        graph.replay()
        completion.record()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)

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

    # A replay must overwrite every logical result. Poison only the append
    # targets so untouched cache entries retain their fixture values.
    query_output.fill_(torch.nan)
    for physical_page, page_offset in physical_slots:
        key_pages[physical_page, page_offset].fill_(torch.nan)
        value_pages[physical_page, page_offset].fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    query_max_abs = (
        (query_output.float() - reference_query.float()).abs().max().item()
    )
    key_pages_max_abs = (
        (key_pages.float() - reference_key_pages.float()).abs().max().item()
    )
    value_pages_max_abs = (
        (value_pages.float() - reference_value_pages.float()).abs().max().item()
    )
    query_bit_mismatches = (
        query_output.view(torch.int16) != reference_query.view(torch.int16)
    ).sum().item()
    key_pages_bit_mismatches = (
        key_pages.view(torch.int16) != reference_key_pages.view(torch.int16)
    ).sum().item()
    value_pages_bit_mismatches = (
        value_pages.view(torch.int16) != reference_value_pages.view(torch.int16)
    ).sum().item()
    if (
        query_bit_mismatches != 0
        or key_pages_bit_mismatches != 0
        or value_pages_bit_mismatches != 0
    ):
        raise RuntimeError(
            "FlashInfer Graph replay changed output versus eager reference: "
            f"query={query_bit_mismatches}, key_pages={key_pages_bit_mismatches}, "
            f"value_pages={value_pages_bit_mismatches}"
        )

    print(
        json.dumps(
            {
                "schema_version": 1,
                "provider": "flashinfer",
                "provider_version": flashinfer.__version__,
                "provider_commit": PROVIDER_COMMIT,
                "run_label": run_label,
                "measurement": MEASUREMENT,
                "operator": "rope_paged_kv_append_tokens",
                "case": CASE,
                "dtype": "bf16",
                "layout": "NHD_D128_neox_split_half_page16",
                "execution": {
                    "algorithm": "flashinfer_rope_then_paged_append_explicit_tokens",
                    "graph": "torch_cuda_graph_fixed_address",
                    "completion_event_inside_timed_interval": True,
                    "kernel_count": {"status": "unverified"},
                    "batch_indices": [2, 0, 1, 0, 2, 1],
                    "positions": [5, 17, 20, 16, 4, 19],
                    "physical_slots": physical_slots,
                    "correctness": {
                        "reference": "eager composition after poisoned-output replay",
                        "query_max_abs": query_max_abs,
                        "query_bit_mismatches": query_bit_mismatches,
                        "query_digest": digest_bf16(query_output),
                        "key_pages_max_abs": key_pages_max_abs,
                        "key_pages_bit_mismatches": key_pages_bit_mismatches,
                        "key_pages_digest": digest_bf16(key_pages),
                        "value_pages_max_abs": value_pages_max_abs,
                        "value_pages_bit_mismatches": value_pages_bit_mismatches,
                        "value_pages_digest": digest_bf16(value_pages),
                    },
                },
                "shape": {
                    "tokens": tokens,
                    "batch_size": batch_size,
                    "max_num_pages": max_num_pages,
                    "query_heads": query_heads,
                    "key_heads": key_heads,
                    "head_dim": head_dim,
                    "page_size": page_size,
                },
                "fixture_id": ROPE_APPEND_TOKENS_FIXTURE_ID,
                "fixture_digests": {
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
                    "page_refcounts": digest_i32(page_refcounts_host),
                },
                "warmup_launches": warmup,
                "launches_per_sample": 1,
                "samples_us": samples_us,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
