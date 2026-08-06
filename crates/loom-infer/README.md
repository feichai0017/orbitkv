# loom-infer

`loom-infer` defines backend-independent operator contracts and CPU reference
implementations. The crate has no CUDA, FFI, runtime, or framework dependency.

The current contracts cover RMSNorm for F32, FP16, and BF16, contiguous BF16
GEMM, BF16 single-request decode, BF16 paged batch decode, and BF16 ragged
causal prefill. A separate `loom-infer-cuda` crate implements the admitted H20
device paths.

## Module layout

Operator families use directory modules consistently:

```text
src/
|-- attention/
|   |-- mod.rs
|   |-- single_decode/{mod.rs,tests.rs}
|   |-- paged_decode/{mod.rs,tests.rs}
|   `-- ragged_prefill/{mod.rs,tests.rs}
|-- gemm/{mod.rs,tests.rs}
|-- rms_norm/{mod.rs,tests.rs}
|-- dtype.rs
|-- error.rs
`-- lib.rs
```

Each family `mod.rs` is a stable public facade. Private domain directories own
their contract, CPU reference, metadata views, and tests. This keeps
`loom_infer::attention::*` stable without mixing facade files and same-named
directories at one source level.
