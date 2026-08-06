# loom-infer-cuda

`loom-infer-cuda` contains Loom Infer's Rust CUDA host and device code.
`cuda-oxide` compiles each Loom-owned kernel. Audited Rust FFI calls qualified
vendor libraries. The crate does not use CUDA C++, Python, or framework
bindings.

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
BF16 paged batch decode, and BF16 ragged causal prefill attention. The vendor
provider freezes one contiguous BF16 cuBLASLt GEMM algorithm during planning.
All use typed bindings and one completion event.

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

Ragged prefill keeps short requests on one direct warp per query-row/head and
uses eight-warp token partitioning with a block-local F32 merge for longer KV
ranges. MHA/MQA/GQA pass the H20 correctness and sanitizer gates. Matched eager
measurements improve mixed MQA and long GQA by 5.779x and 1.689x versus direct
Loom; FlashInfer remains 10.114x lower-latency on stable long GQA. Graph replay
and engine integration remain separate gates.
