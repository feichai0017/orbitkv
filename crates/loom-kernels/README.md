# loom-kernels

Backend-independent Rust contracts and deterministic CPU reference
implementations for inference operators.

This crate deliberately contains no CUDA, FFI, framework, or device-runtime
dependency. Use it to validate shapes and dtypes, generate oracle outputs, and
query backend capabilities. Pair it with
[`loom-cuda`](https://crates.io/crates/loom-cuda) for GPU execution.

```toml
[dependencies]
loom-kernels = "1.0.0-alpha.1"
```

The alpha surface currently covers normalization and quantization, SwiGLU,
RoPE plus paged-KV writes, decode-tail sampling/logprob operations, Min-P, and
paged MQA/GQA decode attention.

See the [project documentation](https://feichai0017.github.io/loom-kernels/)
for contracts, integration boundaries, and H20-qualified evidence.

Licensed under MIT.
