# OrbitKV executor

`executor/` joins OrbitKV's state compiler with the forked Luminal model
compiler. It does not own a second page allocator.

The Rust composition crate lowers an OrbitKV `RuntimeManifest` into immutable
attention-class geometry, produces FlashInfer CSR metadata from authoritative
request page views, translates prepared write/COW actions into physical token
slots, and lowers manager-authored token relocation into per-layer K/V byte
ranges. Relocation success evidence is exposed only after the Luminal CUDA
stream event completes. The complete Luminal fork lives in `luminal/` as a Git
submodule tracking `feichai0017/orbitkv-luminal`; its `upstream` remote is
`luminal-ai/luminal`.

The fork adds an external paged-attention entry point that accepts page size,
page indices, query/KV indptrs, and last-page lengths directly. This is a
source-level integration boundary. Accelerator correctness and throughput are
not qualified by host compilation alone. The current pin has real-device
correctness coverage for externally planned block pages, packed relocation,
and a minimal released-checkpoint prefill/decode path; throughput remains
unqualified.

The exact fork delta and upstream update procedure are documented in
[`docs/executor-upstream.md`](../docs/executor-upstream.md).
