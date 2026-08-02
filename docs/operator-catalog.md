# Operator catalog

This catalog defines the target Loom Infer surface. It does not claim
FlashInfer or FlashAttention parity.

## States

| State | Meaning |
| --- | --- |
| `contract` | A public Rust specification and CPU reference exist |
| `in progress` | A permanent Rust device provider is under implementation |
| `device correct` | The permanent provider passed its declared device correctness gate |
| `planned` | The operator is in the roadmap |
| `vendor` | Loom plans and calls a qualified vendor implementation |

## Current work

| Operator | State | Current boundary |
| --- | --- | --- |
| RMSNorm F32 | device correct | Four shapes, checked bindings, reusable scopes, two-launch chaining, and partial-scope rejection pass on H20 |
| RMSNorm FP16 and BF16 | device correct | Scalar and packed paths, eight shapes per dtype, three short-buffer checks, signed zero, and two-launch chaining pass on H20 |

No permanent GPU provider has a published performance result yet. The
[low-precision correctness record](results/h20-rms-norm-low-precision-20260802.json)
contains the accepted and excluded claims.

## Target surface

| Family | Operators | State |
| --- | --- | --- |
| Normalization | RMSNorm, Add+RMSNorm, quantized output | planned |
| Attention | Ragged prefill, paged decode, split-K, state merge | planned |
| KV cache | RoPE append, gather, scatter, compaction, quantized storage | planned |
| Decode tail | Logits processing, penalties, top-k, top-p, Min-P, logprobs, sampling | planned |
| Speculation | Greedy, stochastic, and tree verification | planned |
| MoE | Routing support, permutation, grouped-GEMM input, combine | planned |
| Quantization | Scale, pack, unpack, dequantize, layout conversion | planned |
| Matrix work | Dense, quantized, and grouped GEMM | vendor |
| Communication | Tensor-parallel and expert-parallel collectives | vendor |

## Attention baseline

Attention implementations compare against pinned FlashAttention, FlashInfer,
and engine providers when contracts match. The comparison must keep shapes,
dtypes, layouts, masks, workspaces, streams, and graph mode equal.

## Admission

Before implementation, each operator records:

- the model or engine call site.
- shapes, dtypes, layouts, alignment, and aliasing.
- the numerical or quality limit.
- the named baseline and target hardware.
- the target metric and stop condition.

The provider then passes contract, device, performance, graph, engine, and
serving gates independently. Unsupported contracts return an error.
