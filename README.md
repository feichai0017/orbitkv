# OrbitKV

OrbitKV is a Rust attention-state compiler and KV block manager for a native
inference engine. It compiles attention visibility into physical layouts and
lifetime rules, owns every KV page generation, and reuses storage only after
both semantic death and device completion are proven.

The product has four owned crates:

| Layer | Responsibility |
| --- | --- |
| `crates/orbitkv/` | Attention-state compilation, request and snapshot identity, Prefix/COW, physical-page ownership, retirement, acknowledgement, and safe reuse |
| `crates/orbitkv-executor/` | OrbitKV plan lowering, Luminal graph/device execution, and external byte transports |
| `crates/orbitkv-server/` | Rust API and scheduling boundary; optionally reuses vLLM's Rust OpenAI/tokenizer/chat frontend through a narrow protocol adapter |
| `crates/orbitkv-engine/` | Single-process composition root that joins logical requests, one `RuntimeSession`, and one compiled Luminal decoder without creating a second KV authority |

OrbitKV is the only KV authority. The executor consumes manager-authored pages
and the server cannot name a physical page. There is no compatibility layer, C
boundary, Python runtime, or second allocator in the active product.

## Repository layout

```text
orbitkv/
├── crates/
│   ├── orbitkv/             compiler, KV manager, RuntimeSession
│   ├── orbitkv-executor/    plan lowering and device execution
│   ├── orbitkv-engine/      in-process composition root
│   └── orbitkv-server/      local Engine and protocol contracts
├── third_party/
│   └── luminal/             pinned compiler/executor fork
├── docs/                    architecture and qualification boundary
├── tools/                   repository invariants
├── website/                 project documentation site
└── results/                 compact current evidence only
```

The Luminal fork is pinned by the parent repository. Its paged-attention path
accepts externally managed page indices, query/KV indptrs, last-page lengths,
and page geometry. It may cache those inputs for execution, but it does not
allocate or recycle KV pages. The executor depends on the visible submodule by
local path, so the reviewed fork and the code linked into the product cannot
silently diverge.

The same boundary carries compiler-visible persistent-state facts. The reusable
`orbitkv` crate derives storage components, retention and retirement geometry,
and address programs from a validated manifest. `orbitkv-executor` binds those
facts to stable arenas and injects them into Luminal's e-graph; each
paged-attention custom op identifies its owning state class. Selected artifacts
include the facts digest in their identity. Today these facts enforce ownership
and artifact compatibility; no current rewrite uses them to choose among
multiple attention backends or KV layouts.

The native decoder compiles one symbolic graph into decode and prefill buckets.
Both phases share the same runtime, preallocated dynamic inputs, and one
persistent K/V arena per attention class; requests update only tokens, positions,
write slots, CSR metadata, and dynamic dimensions before dispatch. Greedy
sampling is part of the graph,
so the default execution path returns one token ID per query row instead of
copying vocabulary-sized logits to the host. A fixed-signature decode can also
be captured as one outer CUDA Graph after warmup. Replay updates the same input
allocations before launching the graph; a change to query/batch/context shape
or CSR indptr geometry is rejected and requires a new capture. The generic
CUDA capture path and a released-checkpoint prefill/capture/replay lifecycle
both pass on H20. In its recorded source closure, the first flattened outer
graph was 24.9% slower than eager; composing Luminal's selected executables as
child graphs reversed that result, reducing matched fixed-step decode wall time
by 8.3% over 20 iterations and 5.8% over a 100-iteration confirmation. This is
a historical, narrow batch-one result, not a current-tree throughput or general
model-speed claim.

The selected decode/prefill schedule can be persisted with
`--decoder-artifact`. A missing path is atomically created after search; an
existing artifact is loaded without re-running search. The artifact contains no
weights, pointers, or KV data and is bound to the manifest, decoder and weight
geometry, arena shape, and compile buckets. Custom-op schedules are re-extracted
against the current paged-attention operators and LLIR-fingerprint checked; any
mismatch fails closed instead of silently searching a replacement.

The core also has a backend-neutral external-export transaction for immutable
request snapshots. It pins exact local page generations, emits logical-page
copy records, requires per-page durable receipts and a monotonic completion
frontier, then publishes an external replica catalog entry. This is the intended
integration boundary for transports such as Mooncake or NIXL. The symmetric
request-private restore transaction allocates fresh OrbitKV-owned pages,
validates external copy receipts, and publishes through the native append
lifecycle. An object-safe async transport contract and a host-memory reference
adapter now execute the plans against real byte buffers, including compact
partial tails, checksums, deletion, and deterministic unobserved/ambiguous
faults. Mooncake/NIXL adapters and external-system qualification remain open;
see [external KV tiers](docs/external-kv.md).

## Compilation and execution

```text
attention visibility / retention semantics
        |
        v
RuntimeManifest
        |
        +--> RuntimeSession: identities, pages, COW, retirement, reuse
        +--> ExecutorPlan: attention classes and physical metadata
                         |
                         v
BatchIntent --> ModelEngine coordinator --> Luminal graph + kernels
                         |
                         +--> ExternalKvTransport (tier movement)
                         |
                         v
              completion evidence --> RuntimeSession publication + ACK
```

Periodic layouts, append-only layouts, resettable arenas, and page-level COW are
compiled physical choices, not separate product modes. Token-level relocation
and compaction are intentionally absent: the manager compiles token visibility
into page placement and retirement, then reuses only whole page generations
after semantic and execution completion.
For attribution testing, an explicit request-lifetime residence baseline keeps
the same attention visibility but delays physical reclamation; normal
construction always uses the compiled policy.

Current compiled token lifetimes include:

- Full attention: append-only placement, optional shared Prefix, and COW.
- Sliding attention: periodic placement and retirement after the visibility
  window passes.
- Full + Sliding: class-separated placement and independent retirement.
- Exact Chunked attention: resettable epoch arenas.
- Full latent KV: component-aware token storage in the core; executor support is
  still pending.
- Recurrent and convolution state: generation-checked checkpoints in the core;
  executor plans and compiler facts now preserve their geometry, while unified
  device transactions are still pending.

The primary model target is a 27B block-FP8 hybrid decoder with a 3:1 Gated
DeltaNet/Full-attention schedule. Its real configuration already compiles to 16
Full token-KV layers and 48 recurrent/convolution layers. The executor parses
the nested text configuration, partial rotary geometry, and block-FP8 contract,
but deliberately rejects execution until GDN state transactions and FP8 model
loading are implemented. A smaller BF16 checkpoint with the same architecture
is the bring-up target; existing dense checkpoints remain regression witnesses.
The latest open DeepSeek V4 family is the second architecture target.

## Server boundary

`orbitkv-server` defines an async, in-process `Engine` interface with explicit
execution and cancellation. Its optional `vllm-frontend` feature launches the
pinned vLLM Rust OpenAI HTTP/tokenizer/chat/SSE stack and translates only
tokenized Add/Abort traffic to the local engine. PegaInfer informed this bridge
shape, but its scheduler, KV cache, model runtime, and CUDA ownership are not
part of OrbitKV. The current adapter is greedy text-only and rejects unsupported
semantics rather than silently dropping them.

`orbitkv-engine` provides the concrete model-backed implementation of that
interface. Its dedicated execution thread owns a bounded admission queue and an
active-request set. Each iteration builds one token-budgeted, decode-first batch,
then executes one atomic `RuntimeSession` transaction and one Luminal dispatch.
Slow output consumers are skipped until their bounded event buffer has room;
stop, cancellation, and disconnect still converge through release and exact
reclamation acknowledgement. Submissions remain one fresh request each, while
the engine combines them internally.

With the optional `server` feature, `orbitkv-serve` composes this engine with
the pinned vLLM Rust OpenAI frontend in the same process. It reuses vLLM's
request schemas, tokenizer/chat backends, SSE framing, request identity, and
stream-drop auto-abort instead of reimplementing them. The narrow local IPC is
an adapter required by `EngineCoreClient`; it does not start another inference
server or grant vLLM KV ownership. Model weights/config and frontend tokenizer
assets may be supplied from separate directories without model-specific code.

## Build and test

```bash
git submodule update --init --recursive
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
cargo fmt --all -- --check
python tools/verify_active_source.py
```

Compile a canonical runtime manifest:

```bash
cargo run --locked -p orbitkv --bin orbitkv -- \
  compile-runtime-manifest crates/orbitkv/examples/hybrid-attention-state-plan.json
```

## Evidence boundary

Core lifecycle and executor lowering are host-verified. On H20, the current
source also executes a released 18-layer Full+Sliding checkpoint with its native
3 Full / 15 Sliding schedule and 512-token window. A 512-token prefill plus 33
decode steps matches a separately generated Transformers greedy-token sequence,
crosses the window, reclaims and reuses a page generation, executes a second
request from reused storage, and drains every arena after release. The direct
explicit-CSR prefill kernel also matches its independent BF16 reference within
`1.93e-4`. See `results/released-hybrid-lifecycle-20260906`.

The concrete `ModelEngine` composition root also passes released-checkpoint H20
closures on the same hybrid model. In addition to length, stop, cancellation,
and drain coverage, two concurrent 512-token prompts are combined into B=2
prefill and decode dispatches and both match the independent eight-token
reference. A separate run inserts a new prefill while another request is already
decoding and observes a mixed-phase dispatch. These qualify continuous-batching
correctness. A final real-device HTTP closure covers non-streaming completions,
ordered SSE token IDs plus `[DONE]`, two concurrent requests, dropped-stream
auto-abort, graceful shutdown, and final manager drain. A subsequent fixed-load
HTTP run completed 16 requests at each of C1/C2/C4/C8, with all 256 requested
output tokens and no per-request errors. Throughput scaled from 184.24 to 518.88
output token/s while TTFT and TPOT increased, so this is a narrow load closure
and measured tradeoff, not a performance-advantage claim.

Run the server with explicit per-class page budgets:

```bash
cargo run --release --locked -p orbitkv-engine --features server --bin orbitkv-serve -- \
  --model /models/checkpoint \
  --decoder-artifact /artifacts/decoder.json \
  --page-counts 128,66 \
  --max-model-tokens 1024 \
  --max-prefill-tokens 512 \
  --max-batch-tokens 1024 \
  --max-active-requests 2
```

Use `--frontend-model` only when tokenizer/chat assets and compatible model
metadata are stored separately from the weight directory.

A separate fixed-signature child-graph experiment records a narrow matched
dispatch improvement for its exact source closure. A release-mode same-executor
test on the released hybrid
checkpoint now gives the first narrow compiler-benefit closure: over ten paired
alternating runs, compiled residence reduced live payload by 27.8%, extended a
fixed page budget from boundary 528 to 560, and reduced median single-request
test-path time by 0.77% while preserving all 256 generated tokens. This remains
a batch-one engine microbenchmark, not an end-to-end serving-throughput result.
There is still no production-serving, broad model-family, or complete SGLang
replacement claim.

A four-epoch product comparison with tuned stock SGLang v0.5.17 is now also
recorded. The first fixed decoder artifact reached 53.4% of SGLang's C2 output
throughput. Persistent K/V updates are now compiler constraints: every selected
bucket must alias all 36 K/V tensors back to the stable arena or compilation and
artifact loading fail closed. A deeper constrained search improved the same
engine's median throughput by 14.5%, TTFT by 32.2%, TPOT by 10.2%, and E2E by
12.8% over four alternating epochs. Against stock SGLang, the improved artifact
reached 59.8% throughput with 1.59x TPOT and 4.79x TTFT. OrbitKV's configured
K/V tensor payload remained 40.4% smaller because SGLang disabled hybrid
Sliding-Window memory for this Gemma3 path. This is a reference-gated,
measured compiler-selection improvement and a smaller—but still clear—serving
performance deficit, not a win claim. The new artifact passes the existing independent B2 reference-token
probe, while random-trace text differs across schedules and engines.

`results/**` contains only compact reviewed evidence directly relevant to the
current architecture. Removed historical archives remain recoverable from Git
history. See the [Capability Matrix](docs/capability-matrix.md) and
[Results Index](results/README.md).

## Documentation

- [Architecture](docs/architecture.md)
- [Components and external projects](docs/components.md)
- [Implementation status](docs/implementation-status.md)
- [Capability Matrix](docs/capability-matrix.md)
- [Executor fork and upstream policy](docs/executor-upstream.md)
- [Matched serving benchmarks](docs/benchmarking.md)
- [RuntimeSession](docs/runtime-session.md)
- [State lifetime and reclamation](docs/state-lifecycle.md)
- [Roadmap](docs/roadmap.md)
