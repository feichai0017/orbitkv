# Operator catalog

This catalog defines the target surface for Loom Infer's high-performance Rust
and cuda-oxide operator layer. It does not claim complete FlashInfer or
FlashAttention parity. The [FlashInfer parity matrix](flashinfer-parity.md)
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
| BF16 paged batch decode | Rust / cuda-oxide | device correct | NHD D128 page-size-16 MHA/MQA/GQA passes H20 correctness and sanitizer gates |
| BF16 paged causal prefill | Rust / cuda-oxide | device correct | Ragged-query NHD D128 page-size-16 direct MHA/MQA/GQA passes H20 correctness, sanitizer, and fixed-address Graph gates |
| BF16 ragged causal prefill | Rust / cuda-oxide | device correct | NHD D128 direct, eight/sixteen-warp, and tiled eight-partition bottom-right causal MHA/MQA/GQA passes H20 correctness, sanitizer, and fixed-address Graph gates |
| BF16 standard RoPE | Rust / cuda-oxide | device correct | NHD D128 NeoX split-half with explicit I32 positions passes H20 correctness, sanitizer, and matched eager gates |
| BF16 fused RoPE paged KV append | Rust / cuda-oxide | device correct | Explicit 1..=64 token, NHD D128, page-size-16 NeoX paths pass H20 correctness, sanitizer, matched eager, and fixed-address Graph gates |

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
| Attention | Single decode, ragged/paged prefill, paged decode, split-K, state merge | Rust / cuda-oxide | in progress |
| KV cache | RoPE append, gather, scatter, compaction, quantized storage | Rust / cuda-oxide | in progress |
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
CPU tests cover malformed metadata, logical-to-physical token mapping, and
numerical equivalence to contiguous decode. The permanent CUDA provider uses
direct MHA and token-parallel MQA/GQA paths. It passes H20 correctness,
exact-span preflight, an invalid-page device guard, and four Compute Sanitizer
tools.

The current matched paged result retains 600 raw eager samples and both
provider orders. Eight-warp block-local token parallelism lowers Loom MQA and
GQA latency by 3.78x and 3.32x relative to the immutable direct record. Loom
is now 4.41x lower-latency for batch-1 MHA and 2.35x lower-latency for
mixed-length batch-3 MQA than FlashInfer. The batch-4 GQA ranking remains
excluded because FlashInfer's order delta is 60.62%. Graph, engine, and serving
gates remain open.

The first paged-prefill contract combines ragged query `indptr` with
FlashInfer-compatible page `indptr`, physical indices, and last-page lengths.
Its direct one-warp-per-query-row/head provider passes MHA, MQA, and GQA H20
correctness, mixed query/KV lengths, physical-page reordering and reuse,
short-metadata and duplicate-binding preflight, an invalid-page device guard,
and all four Compute Sanitizer tools.

The matched eager result records Loom at `4.816` and `12.171` microseconds for
short MHA and GQA4 versus FlashInfer at `15.718` and `15.677` microseconds.
FlashInfer records `15.630` microseconds on mixed MQA versus Loom at `16.902`
microseconds. All provider-order deltas are below `2.1%`. Token-parallel
optimization and Graph performance remain open. The direct GQA4 command passes
fixed-address Graph correctness after two replays and external owner teardown.

The first fused KV mutation contract rotates one Q/K token per request at
`request_kv_len - 1`, writes Q to caller-owned output, and appends rotated K
plus unmodified V into the final physical NHD slot. It rejects duplicate final
physical slots and performs a device-side full page-table range guard before
any write. The admitted batch-4 Q16/K4 D128 case is bit-exact with the CPU
reference and passes all four Compute Sanitizer tools. Its one-kernel eager
combined median is `3.989` microseconds versus `11.735` microseconds for
FlashInfer's two-kernel standard RoPE plus paged append composition, or
`2.942x` lower latency under the fixed-affinity matched metric. That immutable
first record is limited to one token per request.

The explicit extension accepts 1 through 64 tokens with caller-supplied
`batch_indices` and `positions`. The admitted performance case shuffles six
tokens covering each request's final two positions, including page-boundary
crossings and shared physical pages at different offsets. A two-warp
device-side guard validates the full page table, token mappings, and physical
slot uniqueness before any output write. The 64-token limit and four invalid
metadata classes pass H20 and sanitizer gates. Under the fixed-affinity eager
metric, Loom's one-kernel combined median is `5.510` microseconds versus
`11.732` microseconds for FlashInfer's two-kernel composition, or `2.129x`
lower latency. Larger token batches, other layouts/dtypes, engine, and serving
gates remain open.

The six-token command also captures into one fixed-address Graph node and
replays after external provider, plan, and read-buffer owners are dropped.
The matched single-replay medians are `8.288` microseconds for Loom and
`13.728` microseconds for FlashInfer's two-node graph, making Loom `1.656x`
lower-latency. Both provider-order deltas are below five percent and all 400
samples are retained. Graph updates, engine integration, and serving gates
remain open.

The ragged prefill contract uses contiguous NHD query/KV storage with separate
`i32` query and KV `indptr` arrays. Its causal mask is bottom-right aligned:
`kv_index <= kv_len - qo_len + query_index`. Short average-KV plans use the
direct warp; long single-KV-head MQA uses sixteen-warp token partitioning;
admitted long GQA4 plans use fused tensor-core QK/online-softmax/PV over eight
KV partitions plus an F32 merge. Other declared long plans use eight warps.
All paths pass H20 correctness, exact-span preflight, a missing-workspace
preflight gate, a nonmonotonic-metadata device guard, and all four Compute
Sanitizer tools.

The matched eager result lowers Loom long-GQA latency to `48.232`
microseconds. Unrolled 16-byte `cp.async` K/V staging is `1.148x` faster than
the previous tiled split-K result and the complete path is `7.729x` faster than
direct. FlashInfer remains `2.206x` lower-latency on stable long GQA.
Short-MHA and mixed-MQA rankings are excluded because FlashInfer's
provider-order deltas are `10.643%` and `14.097%`. The tiled
partial-plus-merge plan passes fixed-address Graph correctness after two
replays and external owner teardown, with four Compute Sanitizer tools
reporting no errors or leaks. The matched single-replay Graph medians are
`50.480` microseconds for Loom and `32.640` microseconds for FlashInfer;
FlashInfer is `1.547x` lower-latency. Engine and serving gates remain open.

## Admission

Before implementation, each operator records:

- the model or engine call site.
- shapes, dtypes, layouts, alignment, and aliasing.
- the numerical or quality limit.
- the named baseline and target hardware.
- the target metric and stop condition.

The provider then passes contract, device, performance, graph, engine, and
serving gates independently. Unsupported contracts return an error.
