# loom-infer-cuda

`loom-infer-cuda` contains Loom Infer's Rust CUDA host and device code.
`cuda-oxide` compiles each Loom-owned kernel. The crate does not use CUDA C++,
`nvcc`, Python, or framework bindings.

The default feature set keeps CPU-only workspace checks platform independent.
Enable `cuda` only inside the pinned CUDA build environment.

```bash
cargo oxide run rms_norm_h20 --bin rms_norm_h20 --features cuda --arch sm_90
```

## Current provider

The current provider implements contiguous F32, FP16, and BF16 RMSNorm with
typed heterogeneous bindings and one completion event. Scalar and packed paths
pass H20 correctness gates. Fixed argument packs, graph, sanitizer, and
performance gates remain open.
