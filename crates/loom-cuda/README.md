# loom-cuda

Safe Rust execution for Loom Kernels' handwritten CUDA operators.

The crate owns CUDA streams, allocations, and events; validates every public
operator contract before launch; and reports unsupported inputs explicitly.
CUDA is opt-in so ordinary documentation and CPU-only dependency builds do not
require an NVIDIA toolkit.

```toml
[dependencies]
loom-cuda = { version = "1.0.0-alpha.1", features = ["cuda"] }
loom-kernels = "1.0.0-alpha.1"
```

```bash
CUDA_HOME=/usr/local/cuda LOOM_CUDA_ARCHS=90 \
  cargo run -p loom-cuda --features cuda --release \
  --example rust_cuda_smoke
```

The alpha API supports normalization and quantization, SwiGLU, RoPE plus
paged-KV writes, decode-tail sampling/logprob operations, Min-P, and paged
MQA/GQA decode attention. See the
[project documentation](https://feichai0017.github.io/loom-kernels/) for exact
shape gates and H20 evidence.

Licensed under MIT.
