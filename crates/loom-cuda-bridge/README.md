# loom-cuda-bridge

Checked C entrypoints into Loom Kernels' safe Rust CUDA runtime.

This crate is the narrow boundary used by framework adapters that already own
CUDA tensors and a current stream. The adapter passes raw pointers, element
counts, and the stream handle once; Rust constructs non-owning typed views,
validates the operator contract, and launches asynchronously without copying,
allocating device memory, synchronizing, or taking ownership.

The first admitted bridge paths are fused Add+RMSNorm and RMSNorm followed by
dynamic per-token FP8 quantization. Other C ABI operators remain in
`loom-cuda-sys` until their framework paths are migrated and validated
independently.

```bash
CUDA_HOME=/usr/local/cuda LOOM_CUDA_ARCHS=90 \
  cargo build -p loom-cuda-bridge --features cuda --release
```

Raw entrypoints are inherently unsafe for their C/C++ caller: pointers must
refer to correctly typed allocations on the active CUDA context, remain alive
until stream work completes, and obey the documented aliasing contract.

Licensed under MIT.
