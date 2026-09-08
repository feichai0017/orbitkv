# Architecture

OrbitKV is one native inference stack with four owned crates.

```text
HTTP / SSE / WebSocket
        |
        v
crates/orbitkv-server/  HTTP/tokenization + local scheduling/sampling boundary
        | BatchIntent (logical request state only)
        v
crates/orbitkv-engine/  single-process request/lifecycle/execution coordinator
        |
        +--> crates/orbitkv/ compile visibility and own the KV lifecycle
        |       | prepared pages, copies, views, retirement rules
        |       v
        +--> crates/orbitkv-executor/ Luminal graph and device execution
                | local or external completion evidence
                v
        crates/orbitkv/ publish, retire, acknowledge, reuse
```

## Authority

The `orbitkv` crate is the sole owner of request and snapshot identities, page allocation,
page generations, Prefix/COW, semantic liveness, retirement, and reuse.

The `orbitkv-executor` crate owns tensors, compiled graphs, kernel selection, streams, events,
and sampling execution. It consumes immutable physical metadata selected by
OrbitKV. Cached metadata is an execution artifact, never a second page table.

The `orbitkv-server` crate owns client protocols, admission queues, batching, cancellation, and
backpressure. Its public types contain request IDs, tokens, logical boundaries,
sampling intent, and output events; they contain no page or device-buffer
identity. The optional vLLM frontend reuses OpenAI HTTP, tokenizer, chat
template, and streaming code through a narrow Add/Abort transport. It does not
import a second scheduler, KV allocator, or device runtime.

The `orbitkv-engine` crate is the composition root. It implements the server's logical `Engine`
trait while holding `RuntimeSession`, `ExecutorPlan`, stable device arenas, and
`CompiledDecoder` behind one dedicated execution thread. It is allowed to join
the other three layers, but it cannot mint pages or bypass manager transactions.
The scheduler has bounded admission and per-request output queues, keeps a
bounded active set, and forms decode-first token-budgeted dispatches. Each public
submission is one fresh prompt; continuation is internal scheduler state.

External storage providers own byte movement and storage resources only. The
core pins generation-checked source pages, validates exact durable receipts,
and catalogs immutable replicas. Restore allocates fresh local generations and
uses the native append/submission/publication transaction. External metadata
cannot create, retire, or reuse an OrbitKV page. See
[external-kv.md](external-kv.md).

`executor::transport::ExternalKvTransport` is the single object-safe async data
plane seam. It accepts only executor-lowered tensor spans and returns checksums
computed from moved bytes plus confirmed completion evidence. Failures are
classified as `Unobserved` or `Ambiguous`, mapping respectively to safe abort or
fail-stop quarantine in `RuntimeSession`. The included host-memory adapter is a
reference implementation and fault oracle, not a performance backend.

## Native data flow

1. The engine loads model configuration and the compiler turns its attention
   semantics into a fingerprinted
   `RuntimeManifest`.
2. The executor derives an `ExecutorPlan` directly from that manifest.
3. The model executor compiles one symbolic decoder graph into decode and
   prefill buckets and retains one stable K/V arena per attention class. Dynamic input buffers are
   allocated to their configured capacities before search. OrbitKV registers
   every persistent K/V update as a required output-to-input alias; Luminal may
   search freely only among schedules that preserve that state contract.
4. `RuntimeSession` prepares append, Prefix/COW, external transfer, or release
   work.
5. The executor updates bounded dynamic inputs, dispatches the matching Luminal
   bucket, runs on the owning stream without recompiling the model, and samples
   greedy token IDs on device. A warmed fixed-signature decode may instead
   replay one outer CUDA Graph; shape or CSR-indptr changes fail closed and
   require recapture.
6. Completion evidence advances the Execution Frontier.
7. RuntimeSession publishes new request heads, retires unreachable generations,
   validates cleanup acknowledgement, and only then permits reuse.
8. Normal length, stop, cancellation, or disconnected-output termination all
   converge on request release and final drain. Ambiguous device execution is
   quarantined and the worker stops instead of fabricating completion.

## Physical-residence ablation

`CanonicalKvManager::new` executes the compiler-authored physical plan by
default. `new_with_residence` additionally accepts a backend-neutral
`PhysicalResidencePolicy` for controlled experiments:

- `Compiled` uses generated address and retirement programs.
- `RequestLifetime` keeps the same attention visibility and Luminal kernels,
  but uses append-only addresses and retains physical pages until request
  release.

The conservative policy is an attribution baseline, not a serving mode. It
fails closed for resettable Chunked layouts, Prefix publication, and external
export. Prepared attention views omit old physically resident
pages that are unnecessary for the current query range, so the executor sees
the same CSR geometry while underlying page identities may differ.

The execution control path is in-process Rust. The optional vLLM HTTP frontend
uses a process-local IPC protocol adapter because that crate is coupled to its
`EngineCoreClient`; no Python process or upstream inference HTTP hop is present.
The `orbitkv-serve` composition binary starts that frontend and `ModelEngine`
in the same process. vLLM owns OpenAI schemas, tokenization, chat rendering, SSE,
request IDs, and stream-drop auto-abort. OrbitKV still owns every KV page and
lifecycle transition. The frontend handshake reports logical scheduler blocks,
never the sum of heterogeneous physical class arenas.

## Source boundaries

```text
crates/orbitkv/
  src/                    compiler, manager, RuntimeSession, checkpoint pool
crates/orbitkv-engine/
  src/                    single-process model coordinator and failure policy
  src/bin/serve.rs        typed single-process OpenAI server entry point
  tests/                  released-model stream/stop/cancel/drain closure
crates/orbitkv-executor/
  src/
    model.rs              graph/runtime orchestration
    model/config.rs       structural decoder semantics from model config
    model/weights.rs      fail-closed checkpoint tensor contract
    model/block.rs        generic dense transformer block math
    model/runtime_input.rs bounded multi-class step validation
    cuda.rs               direct paged-attention and stream/event boundary
    transport.rs          external byte-movement contract
  tests/                  real-byte reference transport closures
crates/orbitkv-server/
  src/                    local Engine, semantic requests, optional HTTP adapter
docs/                     current product contracts
third_party/luminal/      complete pinned compiler/executor fork
results/                  compact reviewed current evidence, never active source
```

The Luminal submodule preserves its upstream history. `crates/orbitkv-executor/Cargo.toml`
depends directly on its local crates, so the reviewed fork source and the code
compiled into the product are the same tree. OrbitKV-specific changes are made
in the fork and pinned by the parent submodule pointer, rather than copied into
model- or hardware-specific directories. See
[executor-upstream.md](executor-upstream.md) for the update procedure.

The root Cargo workspace contains exactly the four owned `crates/orbitkv-*`
packages. `third_party/luminal` is a path dependency but is explicitly excluded
from workspace membership, so upstream crates remain visibly third-party and
can be synchronized and qualified independently.

## Enforced dependency boundaries

The crate graph is intentionally one-way. `orbitkv` has no executor, Luminal,
server, async-runtime, or HTTP dependency. `orbitkv-executor` depends inward on
`orbitkv` and the embedded compiler crates, but never on `orbitkv-server`.
`orbitkv-server` owns only logical protocol contracts and depends on neither
`orbitkv` nor `orbitkv-executor`. `orbitkv-engine` is the only outer crate
allowed to depend on all three and implements the server trait without moving
page types into the server. External KV transports live in `orbitkv-executor`
because they operate on lowered tensor spans, while replica identity, pins,
publication, and deletion authority stay in `orbitkv`.

The compiler boundary shares semantic lifetime, physical arena, and
persistent-state constraints in the OrbitKV-to-Luminal direction. A candidate
that violates a required state alias is rejected before profiling, and an
artifact containing such a candidate is rejected during load. Backend-specific
egglog and CUDA types stay inside the fork and executor; Luminal never receives
page-allocation, publication, or lifecycle authority.

Concretely, `RuntimeManifest::state_layout_facts` emits a backend-neutral view
of every state class. `orbitkv-executor` joins token classes with stable arena
registrations, emits deterministic e-graph facts, and binds each paged-attention
custom op to its manager class. The fact digest participates in decoder artifact
identity, so an artifact cannot silently survive a changed state/search
contract. These facts currently constrain identity and provide a rewrite input.
The direct OrbitKV paged-attention node is a FlashInfer custom op, so the current
search can optimize the surrounding decoder graph and schedule but does not yet
choose among multiple attention implementations or jointly derive a KV layout.

`tools/verify_active_source.py` enforces these forbidden dependency edges,
rejects physical KV ownership types in server source, requires all product
Luminal dependencies to resolve through the visible submodule, rejects removed
compatibility paths and model/hardware/version-specific active filenames, and
bounds source-file size. The gate cannot prove every semantic ownership rule,
so transaction and failure-atomicity tests remain the executable authority.

## Qualification boundary

The current tree proves compiler, manager, lifecycle, and executor-metadata
contracts on the host. Real-device tests additionally cover external block-page
attention, explicit-CSR causal prefill, and released Full and Full+Sliding
checkpoints. The released hybrid closure runs its native 3 Full / 15 Sliding
layer schedule,
crosses the 512-token window with 512 prefill plus 33 decode steps, matches a
separate Transformers greedy-token reference, reuses an acknowledged retired
generation, executes a second request from recycled storage, and drains all
state after token-boundary cancellation.

The concrete `ModelEngine` is separately exercised on that released checkpoint.
It keeps one compiled decoder alive across length, stop, and cancel requests;
executes two 512-token requests through B=2 prefill/decode; admits a new prefill
while another request is decoding; and proves complete manager drain. The
model-backed HTTP path passes non-streaming, SSE, concurrent-request,
dropped-stream cancellation, graceful shutdown, and final-drain checks. Serving
performance qualification remains outside this closure.

Model-step completion still relies on the embedding runtime's completion
assertion. Generic stable-input outer-graph replay and a released-checkpoint
capture/replay lifecycle pass on H20. The current parent graph preserves
Luminal's searched executables
as child graphs and orders persistent-state D2D epilogues after them. In their
recorded source closure, narrow fixed-step matched tests improved decode wall
time by 5.8-8.3%; that result is not a current-tree claim and broader
performance remains unqualified. A released-hybrid same-executor ablation now
also proves a narrow lifecycle benefit: ten paired release-mode runs reduce live
payload by 27.8%, increase the fixed-budget sequence boundary by 32 tokens, and
show a positive paired total-time confidence interval with identical output. The
tree does not yet prove continuous-batching throughput, multi-user capacity, or
long-running behavior.
