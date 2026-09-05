# OrbitKV

OrbitKV is a Rust attention-state compiler and KV block manager for a native
inference engine. It compiles attention visibility into physical layouts and
lifetime rules, owns every KV page generation, and reuses storage only after
both semantic death and device completion are proven.

The product has three layers:

| Layer | Responsibility |
| --- | --- |
| `core/` | Attention-state compilation, request and snapshot identity, Prefix/COW, token disposition, physical-page ownership, retirement, acknowledgement, and safe reuse |
| `executor/` | OrbitKV plan lowering plus the forked Luminal graph compiler and device executor |
| `server/` | Rust API and scheduling boundary; optionally reuses vLLM's Rust OpenAI/tokenizer/chat frontend through a narrow protocol adapter |

OrbitKV is the only KV authority. The executor consumes manager-authored pages
and the server cannot name a physical page. There is no compatibility layer, C
boundary, Python runtime, or second allocator in the active product.

## Repository layout

```text
orbitkv/
├── core/
│   ├── src/                 compiler, KV manager, RuntimeSession
│   ├── examples/            generic attention-state inputs
│   └── fixtures/            generic compiler fixtures
├── executor/
│   ├── src/                 native execution-plan lowering
│   └── luminal/             pinned Luminal fork (Git submodule)
├── server/                  local async Engine and request contracts
├── docs/                    architecture and qualification boundary
├── tools/                   repository invariants
├── website/                 project documentation site
└── results/                 append-only historical evidence
```

The Luminal fork is pinned by the parent repository. Its paged-attention path
accepts externally managed page indices, query/KV indptrs, last-page lengths,
and page geometry. It may cache those inputs for execution, but it does not
allocate or recycle KV pages.

The native decoder compiles one symbolic graph into decode and prefill buckets.
Both phases share the same runtime, preallocated dynamic inputs, and persistent
K/V arena; requests update only tokens, positions, write slots, CSR metadata,
and dynamic dimensions before dispatch. Greedy sampling is part of the graph,
so the default execution path returns one token ID per query row instead of
copying vocabulary-sized logits to the host. A fixed-signature decode can also
be captured as one outer CUDA Graph after warmup. Replay updates the same input
allocations before launching the graph; a change to query/batch/context shape
or CSR indptr geometry is rejected and requires a new capture. The generic
CUDA capture path and a released-checkpoint prefill/capture/replay lifecycle
both pass on H20. The first flattened outer graph was 24.9% slower than eager;
composing Luminal's selected executables as child graphs reversed that result,
reducing matched fixed-step decode wall time by 8.3% over 20 iterations and
5.8% over a 100-iteration confirmation. This is a narrow batch-one result, not
a throughput or general model-speed claim.

## Compilation and execution

```text
attention visibility / retention semantics
        |
        v
RuntimeManifest
        |
        +--> RuntimeSession: identities, pages, COW, retirement, reuse
        |
        v
ExecutorPlan: attention classes and physical metadata
        |
        v
Luminal graph + kernels
        |
        v
completion evidence -> RuntimeSession publication and acknowledgement
```

Ring layouts, append-only layouts, resettable arenas, and page-level COW are
compiled physical choices, not separate product modes. Token-level management
is always present: the manager tracks logical token placement and disposition,
then relocates only when the state semantics and cost policy permit it.

Current compiled token lifetimes include:

- Full attention: append-only placement, optional shared Prefix, and COW.
- Sliding attention: periodic placement and retirement after the visibility
  window passes.
- Full + Sliding: class-separated placement and independent retirement.
- Exact Chunked attention: resettable epoch arenas.
- Full latent KV: component-aware token storage in the core; executor support is
  still pending.
- Recurrent and convolution state: generation-checked checkpoints in the core;
  unified executor transactions are still pending.

## Server boundary

`orbitkv-server` defines an async, in-process `Engine` interface with explicit
execution and cancellation. Its optional `vllm-frontend` feature launches the
pinned vLLM Rust OpenAI HTTP/tokenizer/chat/SSE stack and translates only
tokenized Add/Abort traffic to the local engine. PegaInfer informed this bridge
shape, but its scheduler, KV cache, model runtime, and CUDA ownership are not
part of OrbitKV. The current adapter is greedy text-only and rejects unsupported
semantics rather than silently dropping them.

## Build and test

```bash
git submodule update --init --recursive
cargo test --locked --workspace --all-targets
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
python tools/verify_active_source.py
```

Compile a canonical runtime manifest:

```bash
cargo run --locked -p orbitkv --bin orbitkv -- \
  compile-runtime-manifest core/examples/hybrid-attention-state-plan.json
```

## Evidence boundary

Core lifecycle and executor lowering are host-verified. The current pinned
executor also has two same-source real-device correctness closures: a released
full-attention checkpoint completes prefill and one decode step with
OrbitKV-owned pages, and a token-relocation transaction copies K/V on one CUDA
stream, waits on an event, publishes the packed view, and serves a following
paged decode. These are correctness qualifications, not matched performance
measurements. OrbitKV therefore still makes no speedup, capacity,
memory-saving, production-readiness, or complete-replacement claim.

`results/**` is append-only provenance from earlier experiments. Those records
retain their exact model, hardware, source, and outcome, but they do not qualify
the current architecture. See the [Capability Matrix](docs/capability-matrix.md)
and [Results Index](results/README.md).

## Documentation

- [Architecture](docs/architecture.md)
- [Capability Matrix](docs/capability-matrix.md)
- [Executor fork and upstream policy](docs/executor-upstream.md)
- [RuntimeSession](docs/runtime-session.md)
- [State lifetime and reclamation](docs/state-lifecycle.md)
- [Roadmap](docs/roadmap.md)
