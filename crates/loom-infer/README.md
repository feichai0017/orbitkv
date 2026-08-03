# loom-infer

`loom-infer` defines backend-independent operator contracts and CPU reference
implementations. The crate has no CUDA, FFI, runtime, or framework dependency.

The current contracts cover RMSNorm for F32, FP16, and BF16, contiguous BF16
GEMM, and BF16 single-request decode attention. The attention contract fixes
NHD caches, head dimension 128, F32 accumulation, BF16 output, and F32 log2-LSE.
