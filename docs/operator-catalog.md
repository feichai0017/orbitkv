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
| BF16 single decode | Rust / cuda-oxide | device correct | NHD D128 MHA/MQA/GQA slice passes H20 correctness and sanitizer gates |

No permanent GPU provider has a published performance result yet. The
[attention record](results/h20-bf16-single-decode-correctness-20260803.json)
contains the latest accepted and excluded claims.

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

The current single-decode record is a Loom GPU versus CPU-oracle correctness
gate. It contains no matched FlashInfer performance result.

## Admission

Before implementation, each operator records:

- the model or engine call site.
- shapes, dtypes, layouts, alignment, and aliasing.
- the numerical or quality limit.
- the named baseline and target hardware.
- the target metric and stop condition.

The provider then passes contract, device, performance, graph, engine, and
serving gates independently. Unsupported contracts return an error.
