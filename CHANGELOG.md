# Changelog

## Unreleased

### Added

- Defined the product target as a high-performance, FlashInfer-class operator
  layer for Rust inference engines, with cuda-oxide compiling Loom-owned Rust
  kernels for the CUDA platform.
- Added a contiguous BF16 GEMM contract and CPU reference with F32 accumulation.
- Added one fixed cuBLASLt plan with checked layouts, spans, alignment,
  workspace, context, and caller-owned stream.
- Added H20 correctness coverage for standalone GEMM, row-major transpose,
  capacity and buffer rejection, plan reuse, and an RMSNorm-to-GEMM chain.
- Added a BF16 NHD D128 single-decode contract, CPU reference, and Rust CUDA
  provider for MHA, MQA, and GQA.
- Added H20 attention correctness, exact-span rejection, duplicate-binding checks,
  Compute Sanitizer, and SM90 artifact evidence.
- Added BF16 paged batch decode, ragged causal prefill, standard RoPE, and fused
  RoPE plus paged-KV append contracts, CUDA providers, validation, and evidence.
- Added fixed-address CUDA Graph execution for admitted command chains.
- Added a GPU-less CUDA host CI gate for the `cuda` feature and cuBLASLt
  surface.

### Changed

- Simplified the project README and documentation site around the Rust,
  cuda-oxide, CUDA, and FlashInfer-class product boundary.
- Updated the website dependency lock to use the patched `js-yaml` release.
- Renamed the shared decode provider to `DecodeProvider` so decode and prefill
  provider domains use symmetric names.
- Generalized command scopes to retain Rust kernel functions and external
  provider plans under one completion fence.
- Renamed launch-specific command capacity APIs to provider-neutral command
  terminology.
- Separated external provider submission errors from CUDA driver errors.
- Added checked three-read, two-write resolution to the shared binding path.

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
