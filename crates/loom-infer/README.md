# loom-infer

`loom-infer` defines backend-independent operator contracts and CPU reference
implementations. The crate has no CUDA, FFI, runtime, or framework dependency.

The current contracts cover RMSNorm for F32, FP16, and BF16, plus contiguous
BF16 GEMM with F32 accumulation. Loom Infer adds another contract only when a
permanent device provider starts.
