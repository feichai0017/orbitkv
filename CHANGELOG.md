# Changelog

## Unreleased

### Added

- Added a contiguous BF16 GEMM contract and CPU reference with F32 accumulation.
- Added one fixed cuBLASLt plan with checked layouts, spans, alignment,
  workspace, context, and caller-owned stream.
- Added H20 correctness coverage for standalone GEMM, row-major transpose,
  capacity and buffer rejection, plan reuse, and an RMSNorm-to-GEMM chain.

### Changed

- Generalized command scopes to retain Rust kernel functions and external
  provider plans under one completion fence.
- Renamed launch-specific command capacity APIs to provider-neutral command
  terminology.
- Separated external provider submission errors from CUDA driver errors.

## 1.0.0-alpha.1

### Breaking

- Reduced the workspace to `loom-infer` and `loom-infer-cuda`.
- Removed the Python package, CUDA C++ kernels, C bridge, raw FFI crate,
  compatibility paths, benchmark scripts, spikes, and historical result set.
- Removed all operator APIs except the RMSNorm slice with a permanent Rust
  device implementation.
- Made cuda-oxide the only custom GPU-kernel toolchain.

### Added

- Added a checked `RmsNormSpec` and F32, F16, and BF16 CPU references.
- Added the permanent F32 RMSNorm cuda-oxide module, provider, immutable launch
  plan, and H20 correctness program.
- Added a two-crate architecture, operator catalog, staged roadmap, and strict
  evidence contract.
