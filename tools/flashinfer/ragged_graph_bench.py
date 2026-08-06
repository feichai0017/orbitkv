#!/usr/bin/env python3
"""Matched CUDA Graph replay baseline for ragged FlashInfer prefill."""

from __future__ import annotations

import json
import math
import os

import torch

import flashinfer
from matched_bench import (
    PROVIDER_COMMIT,
    RAGGED_FIXTURE_ID,
    deterministic_bf16,
    digest_bf16,
    digest_i32,
)


MEASUREMENT = "fixed_address_cuda_graph_single_replay_event"
CASE = "bf16_ragged_gqa4_b2_q32_64_kv256_1024_qh16_kvh4_d128"
FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
MASK64 = (1 << 64) - 1


def env_positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def digest_f32(values: torch.Tensor) -> str:
    digest = FNV_OFFSET_BASIS
    for value in values.contiguous().view(torch.int32).view(-1).tolist():
        digest ^= value & 0xFFFFFFFF
        digest = (digest * FNV_PRIME) & MASK64
    return f"{digest:016x}"


def main() -> None:
    warmup = env_positive_int("LOOM_BENCH_WARMUP", 200)
    launches = env_positive_int("LOOM_BENCH_LAUNCHES", 1)
    samples = env_positive_int("LOOM_BENCH_SAMPLES", 100)
    if launches != 1:
        raise ValueError("Graph benchmark requires LOOM_BENCH_LAUNCHES=1")
    run_label = os.environ.get("LOOM_BENCH_RUN_LABEL", "unlabeled")

    batch_size = 2
    query_heads = 16
    kv_heads = 4
    head_dim = 128
    qo_indptr_values = (0, 32, 96)
    kv_indptr_values = (0, 256, 1280)
    nnz_qo = qo_indptr_values[-1]
    nnz_kv = kv_indptr_values[-1]
    salt = 0x4001

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
    qo_indptr_buf = torch.empty_like(qo_indptr)
    kv_indptr_buf = torch.empty_like(kv_indptr)
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
        use_cuda_graph=True,
        qo_indptr_buf=qo_indptr_buf,
        kv_indptr_buf=kv_indptr_buf,
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

    torch.cuda.synchronize()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    run()
    torch.cuda.synchronize()
    reference_output = output.clone()
    reference_lse = lse.clone()

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

    torch.cuda.synchronize()
    output_max_abs = (
        (output.float() - reference_output.float()).abs().max().item()
    )
    output_bit_mismatches = (
        output.view(torch.int16) != reference_output.view(torch.int16)
    ).sum().item()
    lse_max_abs = (lse - reference_lse).abs().max().item()
    lse_bit_mismatches = (
        lse.view(torch.int32) != reference_lse.view(torch.int32)
    ).sum().item()
    if output_bit_mismatches != 0 or lse_bit_mismatches != 0:
        raise RuntimeError(
            "FlashInfer graph replay changed output versus its eager reference: "
            f"output_mismatches={output_bit_mismatches}, "
            f"lse_mismatches={lse_bit_mismatches}"
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
                "operator": "ragged_prefill",
                "case": CASE,
                "dtype": "bf16",
                "layout": "NHD_D128_ragged",
                "execution": {
                    "algorithm": "flashinfer_batch_ragged_prefill_wrapper",
                    "backend": "fa2",
                    "causal": "bottom_right",
                    "disable_split_kv": False,
                    "graph": "torch_cuda_graph_fixed_address",
                    "wrapper_cuda_graph_mode": True,
                    "completion_event_inside_timed_interval": True,
                    "correctness": {
                        "reference": "same fixed wrapper eager run",
                        "output_max_abs": output_max_abs,
                        "output_bit_mismatches": output_bit_mismatches,
                        "output_digest": digest_bf16(output),
                        "lse_max_abs": lse_max_abs,
                        "lse_bit_mismatches": lse_bit_mismatches,
                        "lse_digest": digest_f32(lse),
                    },
                },
                "kernels_per_call": 1,
                "shape": {
                    "batch_size": batch_size,
                    "nnz_qo": nnz_qo,
                    "nnz_kv": nnz_kv,
                    "request_qo_lens": [32, 64],
                    "request_kv_lens": [256, 1024],
                    "query_heads": query_heads,
                    "kv_heads": kv_heads,
                    "head_dim": head_dim,
                },
                "fixture_id": RAGGED_FIXTURE_ID,
                "fixture_digests": {
                    "query": digest_bf16(query_host),
                    "key": digest_bf16(key_host),
                    "value": digest_bf16(value_host),
                    "qo_indptr": digest_i32(qo_indptr_host),
                    "kv_indptr": digest_i32(kv_indptr_host),
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
