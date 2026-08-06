# loom-infer

`loom-infer` defines backend-independent operator contracts and CPU reference
implementations. The crate has no CUDA, FFI, runtime, or framework dependency.

The current contracts cover RMSNorm for F32, FP16, and BF16, contiguous BF16
GEMM, BF16 single-request decode attention, and BF16 paged batch decode. The
paged contract fixes NHD pages, head dimension 128, page size 16, validated
FlashInfer-compatible page tables, F32 accumulation, BF16 output, and F32
log2-LSE. A separate `loom-infer-cuda` provider implements the admitted H20
device path.

The public `attention` module is a facade. Contiguous and split-K behavior
lives in `attention/single_decode.rs`; paged contracts and page-table behavior
live in `attention/paged_decode.rs`. These private modules do not change the
public `loom_infer::attention::*` paths.
