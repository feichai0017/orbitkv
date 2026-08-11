#!/usr/bin/env python3
"""Matched CUDA Graph replay baseline for paged FlashInfer prefill."""

from __future__ import annotations

import json
import math
import os

import torch

import flashinfer
from matched_bench import (
    PAGED_PREFILL_FIXTURE_ID,
    PROVIDER_COMMIT,
    check_attention_output,
    deterministic_bf16,
    digest_bf16,
    digest_i32,
    paged_attention_reference,
    validate_flashinfer_installation,
)


MEASUREMENT = "fixed_address_cuda_graph_single_replay_event"
CASE = "bf16_paged_prefill_gqa4_b2_q4_2_kv23_18_qh16_kvh4_d128_p16"


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

    batch_size = 2
    nnz_qo = 6
    max_num_pages = 6
    query_heads = 16
    kv_heads = 4
    head_dim = 128
    page_size = 16
    qo_indptr_values = (0, 4, 6)
    page_indptr_values = (0, 2, 4)
    page_indices_values = (5, 1, 5, 3)
    last_page_len_values = (7, 2)
    salt = 0x4001

    query_host = deterministic_bf16(nnz_qo * query_heads * head_dim, salt)
    key_host = deterministic_bf16(
        max_num_pages * page_size * kv_heads * head_dim,
        salt ^ 0x4B455900,
    )
    value_host = deterministic_bf16(
        max_num_pages * page_size * kv_heads * head_dim,
        salt ^ 0x56414C554500,
    )
    qo_indptr_host = torch.tensor(qo_indptr_values, dtype=torch.int32)
    page_indptr_host = torch.tensor(page_indptr_values, dtype=torch.int32)
    page_indices_host = torch.tensor(page_indices_values, dtype=torch.int32)
    last_page_len_host = torch.tensor(last_page_len_values, dtype=torch.int32)

    query = query_host.reshape(nnz_qo, query_heads, head_dim).to("cuda")
    key_pages = key_host.reshape(
        max_num_pages, page_size, kv_heads, head_dim
    ).to("cuda")
    value_pages = value_host.reshape(
        max_num_pages, page_size, kv_heads, head_dim
    ).to("cuda")
    qo_indptr = qo_indptr_host.to("cuda")
    page_indptr = page_indptr_host.to("cuda")
    page_indices = page_indices_host.to("cuda")
    last_page_len = last_page_len_host.to("cuda")
    qo_indptr_buf = torch.empty_like(qo_indptr)
    page_indptr_buf = torch.empty_like(page_indptr)
    page_indices_buf = torch.empty_like(page_indices)
    last_page_len_buf = torch.empty_like(last_page_len)
    output = torch.empty_like(query)
    lse = torch.empty(
        (nnz_qo, query_heads), device="cuda", dtype=torch.float32
    )
    workspace = torch.zeros(
        128 * 1024 * 1024, device="cuda", dtype=torch.uint8
    )
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=qo_indptr_buf,
        paged_kv_indptr_buf=page_indptr_buf,
        paged_kv_indices_buf=page_indices_buf,
        paged_kv_last_page_len_buf=last_page_len_buf,
        backend="fa2",
    )
    wrapper.plan(
        qo_indptr,
        page_indptr,
        page_indices,
        last_page_len,
        query_heads,
        kv_heads,
        head_dim,
        page_size,
        causal=True,
        pos_encoding_mode="NONE",
        window_left=-1,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        sm_scale=1.0 / math.sqrt(head_dim),
        disable_split_kv=False,
    )

    def run() -> None:
        wrapper.run(
            query,
            (key_pages, value_pages),
            out=output,
            lse=lse,
            return_lse=True,
            enable_pdl=False,
        )

    torch.cuda.synchronize()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    expected_output, expected_lse = paged_attention_reference(
        query_host.reshape(nnz_qo, query_heads, head_dim),
        key_host.reshape(max_num_pages, page_size, kv_heads, head_dim),
        value_host.reshape(max_num_pages, page_size, kv_heads, head_dim),
        page_indptr_values,
        page_indices_values,
        last_page_len_values,
        qo_indptr_values,
        causal=True,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize()
    output.fill_(math.nan)
    lse.fill_(math.nan)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    correctness = check_attention_output(
        "FlashInfer paged-prefill Graph replay",
        output,
        lse,
        expected_output,
        expected_lse,
    )
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

    print(
        json.dumps(
            {
                "schema_version": 1,
                "provider": "flashinfer",
                "provider_version": flashinfer.__version__,
                "provider_commit": PROVIDER_COMMIT,
                "run_label": run_label,
                "measurement": MEASUREMENT,
                "operator": "paged_prefill",
                "case": CASE,
                "dtype": "bf16",
                "layout": "NHD_D128_page16",
                "execution": {
                    "algorithm": "flashinfer_batch_paged_prefill_wrapper",
                    "backend": "fa2",
                    "causal": "bottom_right",
                    "disable_split_kv": False,
                    "graph": "torch_cuda_graph_fixed_address",
                    "wrapper_cuda_graph_mode": True,
                    "completion_event_inside_timed_interval": True,
                    "replay_output_poisoned": True,
                    "kernel_count": {"status": "unverified"},
                    "correctness": correctness,
                },
                "shape": {
                    "batch_size": batch_size,
                    "nnz_qo": nnz_qo,
                    "max_num_pages": max_num_pages,
                    "referenced_pages": len(page_indices_values),
                    "request_qo_lens": [4, 2],
                    "request_kv_lens": [23, 18],
                    "query_heads": query_heads,
                    "kv_heads": kv_heads,
                    "head_dim": head_dim,
                    "page_size": page_size,
                },
                "fixture_id": PAGED_PREFILL_FIXTURE_ID,
                "fixture_digests": {
                    "query": digest_bf16(query_host),
                    "key_pages": digest_bf16(key_host),
                    "value_pages": digest_bf16(value_host),
                    "qo_indptr": digest_i32(qo_indptr_host),
                    "page_indptr": digest_i32(page_indptr_host),
                    "page_indices": digest_i32(page_indices_host),
                    "last_page_len": digest_i32(last_page_len_host),
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
