# loom-infer-cuda

`loom-infer-cuda` contains Loom Infer's Rust CUDA host and device code.
`cuda-oxide` compiles each Loom-owned kernel. Audited Rust FFI calls qualified
vendor libraries. The crate does not use CUDA C++, Python, or framework
bindings.

The default feature set keeps CPU-only workspace checks platform independent.
Enable `cuda` only inside the pinned CUDA build environment.

```bash
cargo oxide run rms_norm_h20 --bin rms_norm_h20 --features cuda --arch sm_90
cargo oxide run bf16_gemm_h20 --bin bf16_gemm_h20 --features cuda --arch sm_90
cargo oxide run single_decode_h20 --bin single_decode_h20 --features cuda --arch sm_90
```

## Current providers

The Rust providers implement contiguous RMSNorm and BF16 single-request decode
attention. The vendor provider freezes one contiguous BF16 cuBLASLt GEMM
algorithm during planning. All use typed bindings and one completion event.

The current owned-binding revision passed its H20 correctness gates. The fixed
RMSNorm-to-GEMM Graph also passed replay and Compute Sanitizer gates. The
single-decode attention slice passed its H20 correctness and sanitizer gates.
Fixed Rust-kernel argument packs and performance gates remain open.
