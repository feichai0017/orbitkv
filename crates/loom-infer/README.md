# loom-infer

`loom-infer` defines backend-independent operator contracts and CPU reference
implementations. The crate has no CUDA, FFI, runtime, or framework dependency.

The current vertical slice contains RMSNorm for F32, FP16, and BF16. Loom Infer
adds another contract only when its Rust device implementation starts.
