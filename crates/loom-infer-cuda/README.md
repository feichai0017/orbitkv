# loom-infer-cuda

`loom-infer-cuda` contains Loom Infer's Rust CUDA host and device code.
`cuda-oxide` compiles each Loom-owned kernel. Audited Rust FFI calls qualified
vendor libraries. The crate does not use CUDA C++, Python, or framework
bindings.

CUDA names the target platform and execution contract. cuda-oxide is the Rust
toolchain used for Loom-owned kernels; it does not replace CUDA contexts,
streams, events, Graphs, or vendor libraries.

The default feature set keeps CPU-only workspace checks platform independent.
Enable `cuda` only inside the pinned CUDA build environment.

Hardware runners live in the sibling `loom-infer-validation` crate. Use
`make cuda-test` and `make h20` from the repository root.

## Module layout

```text
src/
|-- attention/{mod.rs,decode.rs,prefill.rs}
|-- command/{mod.rs,binding.rs,resolve.rs,submission.rs,completion.rs}
|-- gemm/mod.rs
|-- graph/mod.rs
|-- rms_norm/mod.rs
|-- rope/mod.rs
|-- driver.rs
`-- lib.rs
```

Public operator modules are stable directory facades. Attention's decode and
prefill domains keep one file per complete cuda-oxide artifact bundle because
the macro must discover each bundle in one token tree. Future MLA and KV
domains become sibling private modules without changing
`loom_infer_cuda::attention::*`.

## Current providers

The Rust providers implement contiguous RMSNorm, BF16 single-request decode,
BF16 paged batch decode, BF16 ragged and paged causal prefill attention,
standard RoPE, and fused RoPE plus paged KV append for one-token/request and
explicit 1-through-64-token contracts. The vendor provider freezes one
contiguous BF16 cuBLASLt GEMM algorithm during planning. All use typed bindings
and one completion event.

The current owned-binding revision passed its H20 correctness gates. The fixed
RMSNorm-to-GEMM Graph also passed replay and Compute Sanitizer gates. The
single-decode attention slice passed its H20 correctness and sanitizer gates.
Its split-K partial and F32 state-merge kernels pass H20 correctness and
sanitizer gates with caller-owned workspace and two-command preflight. The
eight-warp block-local merge lowers its isolated KV4096 duration from `20.192`
to `5.056` microseconds. The matched result records 5.39x and 38.19x total Loom
speedups at GQA KV lengths 127 and 4096 relative to the direct baseline;
FlashInfer remains 1.17x and 2.09x lower-latency. Hardware-counter metrics and
Graph performance gates remain open.

Paged MHA keeps a direct warp, while MQA/GQA use eight-warp block-local token
parallelism. The current matched result puts Loom 4.41x lower-latency for MHA
and 2.35x lower-latency for MQA than the pinned FlashInfer path; GQA remains
excluded from stable ranking because the baseline is provider-order sensitive.

The fused RoPE append path uses one 64-thread CTA per request/head state to
rotate Q/K and write rotated K plus unmodified V into the final physical NHD
slot. Full Q and K/V page pools pass the H20 CPU oracle bit-exactly, and all
four Compute Sanitizer tools report no errors. Its admitted fixed-affinity
eager median is `3.989` microseconds versus FlashInfer's `11.735` microsecond
two-kernel composition.

The explicit extension accepts 1 through 64 tokens with caller-supplied batch
indices and positions. Two validation warps establish page-table and
physical-slot safety before Q/K/V writes. The 64-token boundary and four
invalid-metadata guards pass H20 and all four Compute Sanitizer tools. Its
admitted six-token fixed-affinity eager median is `5.510` microseconds versus
FlashInfer's `11.732` microsecond two-kernel composition.

The same command captures into one fixed-address Graph node and replays after
external provider, plan, and read-buffer owners are dropped. Its matched
single-replay median is `8.288` microseconds versus `13.728` microseconds for
FlashInfer's two-node graph.

Paged prefill combines ragged query `indptr` with page-size-16 NHD KV metadata.
The first direct one-warp-per-query-row/head provider passes MHA, MQA, and GQA
H20 correctness, mixed-length, physical-page reuse, metadata preflight, and all
four Compute Sanitizer gates. Performance, Graph, and optimized long-context
paths remain open.

Ragged prefill keeps short requests on one direct warp per query-row/head,
uses sixteen-warp token partitioning for long single-KV-head MQA, and keeps
eight-warp partitioning for other declared long requests. MHA/MQA/GQA pass the
H20 correctness and sanitizer gates. The admitted long-GQA path uses fused
tensor-core tiling, eight KV partitions, and unrolled 16-byte `cp.async`
staging. It is 7.729x faster than the direct baseline; FlashInfer remains
2.206x lower-latency on the stable long-GQA case. The fixed-address
partial-plus-merge Graph also passes correctness and matched replay gates.
Broader query tiling and engine integration remain open.
