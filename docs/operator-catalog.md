# Operator catalog

This catalog defines the target Loom Infer surface. It does not claim
FlashInfer or FlashAttention parity. The [FlashInfer parity matrix](flashinfer-parity.md)
tracks the pinned upstream comparison.

## States

| State | Meaning |
| --- | --- |
| `contract` | A public Rust specification and CPU reference exist |
| `in progress` | A permanent Rust device provider is under implementation |
| `device correct` | The permanent provider passed its declared device correctness gate |
| `planned` | The operator is in the roadmap |

## Current work

| Operator | Provider | State | Current boundary |
| --- | --- | --- | --- |
| RMSNorm F32 | Rust / cuda-oxide | device correct | Owned bindings pass H20 correctness and sanitizer gates |
| RMSNorm FP16 and BF16 | Rust / cuda-oxide | device correct | Scalar and packed paths pass H20 correctness and sanitizer gates |
| BF16 dense GEMM | cuBLASLt | device correct | Fixed `D=A×Wᵀ` plan and Graph chain pass H20 correctness and sanitizer gates |
| BF16 single decode | Rust / cuda-oxide | device correct | NHD D128 direct and split-K MHA/MQA/GQA paths pass H20 correctness and sanitizer gates |
| BF16 paged batch decode | Rust contract and CPU reference | contract | NHD D128, page size 16, MHA/MQA/GQA, validated `i32` page tables, full window |

The matched parallel-merge H20 result is shape-specific. The complete split-K
path lowers Loom median latency by 5.39x at GQA KV length 127 and 38.19x at KV
length 4096 relative to the recorded direct baseline. FlashInfer remains 1.17x
and 2.09x lower-latency at those shapes. See the
[performance record](results/h20-flashinfer-v0.6.16.post1-parallel-merge-eager-performance-20260805.json)
for raw samples, execution metadata, order variance, and excluded claims.

## Target surface

| Family | Operators | Provider | State |
| --- | --- | --- | --- |
| Normalization | RMSNorm, Add+RMSNorm, quantized output | Rust / cuda-oxide | planned |
| Attention | Single decode, ragged prefill, paged decode, split-K, state merge | Rust / cuda-oxide | in progress |
| KV cache | RoPE append, gather, scatter, compaction, quantized storage | Rust / cuda-oxide | planned |
| Decode tail | Logits processing, penalties, top-k, top-p, Min-P, logprobs, sampling | Rust / cuda-oxide | planned |
| Speculation | Greedy, stochastic, and tree verification | Rust / cuda-oxide | planned |
| MoE | Routing support, permutation, grouped-GEMM input, combine | Rust plus vendor GEMM | planned |
| Quantization | Scale, pack, unpack, dequantize, layout conversion | Rust / cuda-oxide | planned |
| Matrix work | Dense, quantized, and grouped GEMM | Qualified vendor libraries | planned |
| Communication | Tensor-parallel and expert-parallel collectives | Qualified vendor libraries | planned |

## Attention baseline

Attention implementations compare against pinned FlashAttention, FlashInfer,
and engine providers when contracts match. The comparison must keep shapes,
dtypes, layouts, masks, workspaces, streams, and graph mode equal.

The current single-decode correctness record remains a Loom GPU versus
CPU-oracle gate. Performance is tracked separately because eager-provider,
kernel, Graph, engine, and serving measurements have different timed regions.

The paged batch-decode contract matches FlashInfer's `indptr`, physical page
index, and last-page-length semantics for one query token per request. Its
current tests cover exact spans, malformed metadata, logical-to-physical token
mapping, and numerical equivalence to contiguous decode. It has no CUDA,
Graph, H20 correctness, or performance claim yet.

## Admission

Before implementation, each operator records:

- the model or engine call site.
- shapes, dtypes, layouts, alignment, and aliasing.
- the numerical or quality limit.
- the named baseline and target hardware.
- the target metric and stop condition.

The provider then passes contract, device, performance, graph, engine, and
serving gates independently. Unsupported contracts return an error.
