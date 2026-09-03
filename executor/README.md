# OrbitKV executor

`executor/` joins OrbitKV's state compiler with the forked Luminal model
compiler. It does not own a second page allocator.

The Rust composition crate lowers an OrbitKV `RuntimeManifest` into immutable
attention-class geometry, produces FlashInfer CSR metadata from authoritative
request page views, and translates prepared write/COW actions into physical
token slots. The complete Luminal fork lives in `luminal/` as a Git submodule
tracking `feichai0017/orbitkv-luminal`; its `upstream` remote is
`luminal-ai/luminal`.

The fork adds an external paged-attention entry point that accepts page size,
page indices, query/KV indptrs, and last-page lengths directly. This is a
source-level integration boundary. Accelerator correctness and throughput are
not qualified by host compilation alone.
