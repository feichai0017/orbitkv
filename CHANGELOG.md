# Changelog

## Unreleased

### Breaking

- Reduced the workspace to `loom-infer` and `loom-infer-cuda`.
- Removed the Python package, CUDA C++ kernels, C bridge, raw FFI crate,
  compatibility paths, benchmark scripts, spikes, and historical result set.
- Removed all operator APIs except the RMSNorm slice that has a Rust device
  implementation in progress.
- Made cuda-oxide the only custom GPU-kernel toolchain.

### Added

- Added a checked `RmsNormSpec` and F32, F16, and BF16 CPU references.
- Added the permanent F32 RMSNorm cuda-oxide module, provider, immutable launch
  plan, and H20 correctness program.
- Added a two-crate architecture, operator catalog, staged roadmap, and strict
  evidence contract.

## 1.0.0-alpha.1

- Reserved for the first published Rust-native Loom Infer release.
