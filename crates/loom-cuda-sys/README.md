# loom-cuda-sys

Raw Rust declarations for the Loom Kernels C ABI, plus the packaged
handwritten CUDA sources and build plumbing.

The default build exposes constants and availability metadata without needing
an NVIDIA toolkit. Enabling `cuda` compiles the bundled `.cu` files with
`nvcc` and links the CUDA runtime:

```toml
[dependencies]
loom-cuda-sys = { version = "1.0.0-alpha.1", features = ["cuda"] }
```

Set `CUDA_HOME` (or `CUDA_PATH`) and optionally set `LOOM_CUDA_ARCHS` to a
comma-separated list such as `80,89,90`. Most callers should use the checked
safe API in [`loom-cuda`](https://crates.io/crates/loom-cuda) instead of this
raw crate.

See the [project documentation](https://feichai0017.github.io/loom-kernels/)
for supported operators and tested hardware boundaries.

Licensed under MIT.
