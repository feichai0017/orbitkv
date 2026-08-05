# loom-infer-cuda

`loom-infer-cuda` contains Loom Infer's Rust CUDA host and device code.
`cuda-oxide` compiles each Loom-owned kernel. Audited Rust FFI calls qualified
vendor libraries. The crate does not use CUDA C++, Python, or framework
bindings.

The default feature set keeps CPU-only workspace checks platform independent.
Enable `cuda` only inside the pinned CUDA build environment.

Hardware runners live in the sibling `loom-infer-validation` crate. Use
`make cuda-test` and `make h20` from the repository root.

## Current providers

The Rust providers implement contiguous RMSNorm and BF16 single-request decode
attention. The vendor provider freezes one contiguous BF16 cuBLASLt GEMM
algorithm during planning. All use typed bindings and one completion event.

The current owned-binding revision passed its H20 correctness gates. The fixed
RMSNorm-to-GEMM Graph also passed replay and Compute Sanitizer gates. The
single-decode attention slice passed its H20 correctness and sanitizer gates.
Fixed Rust-kernel argument packs and performance gates remain open.
