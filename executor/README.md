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
and a released-checkpoint prefill/decode path. `CompiledDecoder` builds one
symbolic graph, performs one real search over separate decode and prefill
buckets, reserves stable-capacity dynamic inputs, and retains one persistent
K/V arena across every dispatch. The selected plan may update that arena in
place or pay a graph-visible device copy back into it; the current smoke
observed the latter. Prefill and repeated decode now use one runtime; there is
no cross-runtime `transfer_cache` path. Throughput remains unqualified.
Greedy argmax is compiled into the same graph; the default runtime API reads
only token IDs. Full-logit transfer remains available through an explicit
diagnostic API for correctness comparison. `capture_decode` performs one
ordinary warmup and records the prepared decode work as a caller-owned outer
CUDA Graph; `replay_decode` updates stable input allocations and launches it.
The initial contract freezes query/batch/context shape and the CSR indptr
arrays. Page identities and last-page lengths may change without recapture.
The Luminal capture primitive has an explicit CUDA regression, but neither that
test nor the complete released-checkpoint replay was executed in the current
qualification environment, so no device-correctness or latency claim is made.

The exact fork delta and upstream update procedure are documented in
[`docs/executor-upstream.md`](../docs/executor-upstream.md).
