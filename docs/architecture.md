# Architecture

OrbitKV is one native inference stack with seven owned crates.

The target compiler architecture and its implementation gates are described in
[Joint compilation](joint-compilation.md). The first acceptance model is
Qwen3.8 27B block-FP8; product graph construction and optimization remain
structural rather than checkpoint-specific.
The implemented compiler/backend boundary, search objective and fusion limits
are described in [OrbitKV compiler design](compiler.md).

```text
HTTP / SSE / WebSocket
        |
        v
crates/orbitkv-engine/
  frontend             HTTP/tokenization and transport adapter
        | protocol::Engine / BatchIntent (logical request state only)
        v
  model_engine         single-process request/lifecycle/execution coordinator
        |
        +--> crates/orbitkv/ compile visibility and own the KV lifecycle
        |       | prepared pages, copies, views, retirement rules
        |       v
        +--> crates/orbitkv-executor/ OrbitKV compiler graph and device execution
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

The `orbitkv-engine` crate owns the logical protocol, optional HTTP frontend,
and request coordinator. Its `protocol` module defines request IDs, tokens,
logical boundaries, sampling intent, and output events; those types contain no
page or device-buffer identity. The `frontend` module depends on that logical
contract and remains independent of the concrete coordinator. The optional vLLM frontend reuses OpenAI HTTP, tokenizer, chat
template, and streaming code through a narrow Add/Abort transport. It does not
import a second scheduler, KV allocator, or device runtime.

Its `model_engine` module implements the logical `Engine` trait while holding `RuntimeSession`, `ExecutorPlan`, stable device arenas, and
`CompiledDecoder` behind one dedicated execution thread. It joins the core and executor, but it cannot mint pages or bypass manager
transactions.
The scheduler has bounded admission and per-request output queues, keeps a
bounded active set, and forms decode-first token-budgeted dispatches. Each public
submission is one fresh prompt; continuation is internal scheduler state.

`ModelEngine::shutdown` closes admission, cancels outstanding work and joins the
execution worker. It returns the final scheduler, KV, fixed-state and graph
statistics only after the worker has retired its decoder. Pending ownership or
quarantined state makes shutdown fail; execution errors and worker panics reach
the caller. The HTTP executable performs this blocking join outside the async
frontend tasks and writes `ORBITKV_ENGINE_SHUTDOWN` as a structured diagnostic.

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
   every persistent K/V update as a required output-to-input alias; OrbitKV compiler may
   search freely only among schedules that preserve that state contract.
4. `RuntimeSession` prepares append, Prefix/COW, external transfer, or release
   work.
5. The executor updates bounded dynamic inputs, dispatches the matching OrbitKV compiler
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

Materialized bucket capacity is an explicit deployment policy, separate from
the selected artifact. Provider metadata stays owned by its captured plan;
shared scratch is dependency-ordered, and replacing plans reserves both old and
new allocations at peak. The runtime evicts least recently used buckets before
constructing replacements. See [graph residency](graph-residency.md) for the
engine/CLI configuration and qualification scope.

## Joint compilation boundary

The executor coordinates state and computation without moving their ownership
boundaries. OrbitKV derives backend-neutral state facts from a validated
manifest; the executor binds those facts to arena contracts and lowers them into
OrbitKV compiler's compiler vocabulary. OrbitKV compiler matches and selects implementations in
egglog, while the CUDA runtime owns candidate preparation, resource validation,
profiling, and loading.

The intended implementation space has three levels: complete optimized
providers, generated algorithm regions and operator fusion, and later bounded
persistent schedules or megakernels. A selected program can contain all three.
FlashInfer, FlashAttention-3 and DeepGEMM are current providers; FlashMLA requires a future MLA
contract and is not an interchangeable provider for arbitrary softmax/GQA
attention. Triton and TileLang are possible code-generation providers, not
current integrations.

Current state facts describe one selected physical realization. Full joint
layout search still requires explicit physical-choice inputs, deterministic
OrbitKV validation, and matching manifest/arena/artifact identities. Runtime
facts such as current sharing and page contiguity must be guarded rather than
inferred from one profiling fixture. OrbitKV compiler may optimize graph-local buffer
lifetimes; OrbitKV retains page generations, cross-request references,
publication, retirement, and reuse authority.

Persistent-kernel execution is a planned candidate form. Its synchronization,
resident resources, state effects, and completion protocol must be represented
before deployment. Existing host-launched providers are execution boundaries;
combining their binaries does not generate a fused device kernel.

## Physical-residence ablation

`CanonicalKvManager::new` executes the compiler-authored physical plan by
default. `new_with_residence` additionally accepts a backend-neutral
`PhysicalResidencePolicy` for controlled experiments:

- `Compiled` uses generated address and retirement programs.
- `RequestLifetime` keeps the same attention visibility and OrbitKV compiler kernels,
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
  src/protocol.rs         logical Engine, request, and event contracts
  src/frontend.rs         optional HTTP/tokenizer/protocol adapter
  src/model_engine.rs     scheduling, lifecycle coordination, failure policy
  src/bin/serve/main.rs    typed single-process OpenAI server entry point
  tests/                  released-model stream/stop/cancel/drain closure
crates/orbitkv-executor/
  src/
    model.rs              graph/runtime orchestration
    model/artifact.rs     schedule serialization and structural identity
    model/compiler.rs     graph preparation and persistent-device binding
    model/config.rs       normalized decoder semantics
    model/import.rs       explicit checkpoint architecture import
    model/import/input.rs shared serialization and numeric validation
    model/weights.rs      fail-closed checkpoint tensor contract
    model/block.rs        generic dense transformer block math
    model/topology.rs     joint token-KV and fixed-state layer ownership
    model/recurrent_layer.rs checkpoint-backed gated-delta graph composition
    model/runtime_input.rs bounded multi-class step validation
    recurrent/            recurrent semantics and arena views
    convolution.rs        minimal-history causal convolution semantics
    state_graph.rs        shared fixed-state arena addressing and bindings
    cuda.rs               provider-neutral paged-attention and stream/event boundary
    transport.rs          external byte-movement contract
  tests/                  real-byte reference transport closures
docs/                     current product contracts
crates/orbitkv-compiler/   symbolic graphs, equivalence compilation, search utilities
crates/orbitkv-ops/        portable operation contracts and graph builders
crates/orbitkv-cuda/       CUDA providers, compilation, profiling and execution
crates/orbitkv-tracing/    compiler/runtime diagnostics
results/                  compact reviewed evidence, never active source
```

The [code and test layout](code-layout.md) applies to all seven owned crates.
They share the root workspace and lockfile. Default members include every host
crate; CUDA device tests are explicit. The [compiler maintenance policy](compiler-maintenance.md)
records source ancestry and licenses. Compiler and state-contract updates are
reviewed and qualified together in this repository.

## Enforced dependency boundaries

The crate graph is intentionally one-way. `orbitkv` has no executor, OrbitKV compiler,
server, async-runtime, or HTTP dependency. `orbitkv-executor` depends inward on
`orbitkv` and the embedded compiler crates, but never on `orbitkv-engine`.
`orbitkv-engine` joins core and executor and owns the optional frontend. Inside
that crate, `protocol` and `frontend` remain logical boundaries: they cannot
import core/executor/OrbitKV compiler implementations or name page, arena, and decoder
implementation types. `tools/verify_active_source.py` checks this source boundary
in addition to the Cargo dependency graph. The frontend feature compiles and
tests without enabling CUDA. External KV transports live in `orbitkv-executor`
because they operate on lowered tensor spans, while replica identity, pins,
publication, and deletion authority stay in `orbitkv`.

The compiler boundary shares semantic lifetime, physical arena, and
persistent-state constraints in the OrbitKV-to-OrbitKV compiler direction. A candidate
that violates a required state alias is rejected before profiling, and an
artifact containing such a candidate is rejected during load. Backend-specific
egglog and CUDA types stay inside the compiler/backend and executor; OrbitKV compiler never receives
page-allocation, publication, or lifecycle authority.

Concretely, `RuntimeManifest::state_layout_facts` emits a backend-neutral view
of every state class. `orbitkv-executor` joins token classes with stable arena
registrations, emits deterministic e-graph facts, binds each paged-attention
custom op to its manager class, and preserves recurrent/convolution class
identity, layer coverage, byte geometry, and checkpoint slots. The fact digest
participates in decoder artifact identity, so an artifact cannot silently
survive a changed state/search contract. Fixed-state lifecycle joins token KV in
one RuntimeSession transaction. The executor owns stable per-class CUDA
allocations, lowers session-authored generation leases to byte ranges, and
shares those allocations with OrbitKV compiler through typed state bindings without
transferring lifecycle authority. A fixed graph sees the complete arena plus a
small dynamic destination-slot tensor; manifest layer order determines the
per-layer byte region. Search uses a private scratch allocation, so profiling
cannot mutate live manager state. Success evidence is withheld until a OrbitKV compiler
execution receipt tied to the exact shared allocation and runtime registration
has synchronized its CUDA event. The ignored-by-default real-device
qualification gate passes on H20 and executes two recurrent transitions and
final drain. The production decoder now
owns the same fixed-state device arenas, and the engine joins their event-backed
receipts to token-KV evidence before submitting the transaction.

The fixed-state graph substrate is dtype- and shape-parameterized. Recurrent
f32 matrices and BF16 causal-convolution histories share stable arena
addressing, dynamic slot metadata, manifest layer ordering, and completion
receipts. Binding policy is explicit: packed recurrent and convolution updates
both require an in-place state-commit schedule selected through egglog.

The recurrent computation boundary is semantic rather than model-specific. A
normalized gated-delta transition is expressed as pure OrbitKV compiler HLIR over
`query`, `key`, `value`, `log_decay`, `update_gate`, and previous state, yielding
both token values and next state. An independent Rust sequence oracle defines
f32 accumulation and proves that chunked continuation from a returned state is
equivalent to one-shot execution. The semantic HLIR and packed CUDA path support
different key and value-head counts through an exact grouped-head mapping. A
second graph composes checkpoint-shaped input projections, minimal `K-1`
causal-convolution history, delta gates, recurrence, gated RMS normalization,
and output projection. The OrbitKV compiler recognizes the exact rank-four
`state * decay + key * delta` subgraph and adds an in-place CUDA state-update
candidate to the same e-class. Its static resource pass proves ordered reads of
the old state precede mutation and rejects competing reads. Binding selected
source/destination slots is now represented by generic Gather/Scatter graph
views over the stable arena, with source-to-destination initialization outside
the graph and dynamic destination ids inside it. A joint topology compiler
proves each layer is owned by exactly one token-KV class or by the matching
recurrent-plus-convolution pair. The production graph dispatches from this
topology, so fixed-state layers do not fabricate token-KV tensors or attention
weights. Stateful artifacts now contain decode and packed-prefill buckets.
Production-rendered static and symbolic CUDA sources pass NVRTC compilation,
and a ragged two-request H20 test matches independent convolution and recurrent
references. Full released-model parity and fused projection/readout kernels
remain open.
The OrbitKV graph emits a provider-neutral paged-attention semantic op. Egglog
contributes explicit FlashInfer CUDA-core/tensor-core algorithms and optional
FlashAttention-3 candidates. Each adapter declares its complete target and ABI
scope. FA3 retains K/V payload allocations and converts CSR page metadata on the
GPU; its scheduler and attention launches join the same measured program. The
handwritten native attention provider has been removed. Joint KV-layout search
remains open; see [attention providers](attention-providers.md).

Every selectable LLIR must also preserve non-attention layout semantics. During
the 27B 16-candidate closure, exact LLIR comparison exposed a fused RMSNorm
candidate that flattened a 3-D Q/K slice even though its physical row pitch came
from a wider projection. The compiler now offers that kernel only for 2-D dense
rows proven by egglog stride facts; 3-D views stay decomposed until the graph IR
can carry an explicit base-layout proof. Decoder artifact schema 5 invalidates
the older unsafe search space and additionally requires every selected token-KV
update to alias the manager-owned arena, eliminating full-cache copy-back.

Block-scaled linear execution follows the same compiler boundary rather than a
model-side backend switch. The decoder emits a provider-neutral semantic node
whose inputs are BF16 activations, FP8 E4M3 weights, and 128x128 inverse scales.
Its independent two-kernel CUDA implementation is a correctness oracle and is
not a deployment-eligible fallback.
Egglog unions four legal DeepGEMM tile schedules into that e-class;
candidate preparation JIT-compiles the resolved provider source before timing, and
OrbitKV compiler's normal device profiler chooses the implementation per dynamic bucket.
The selected LLIR contains a digest of provider/dependency and wrapper contents
plus the tile variant. Strict replay checks the recorded provider identity
against current sources before loading that implementation.
DeepGEMM receives only tensor pointers and a stream and has no state-management
authority.

An opt-in egglog alternative exposes activation quantization as a graph-owned
packed byte buffer consumed by prequantized DeepGEMM nodes. Identical activation,
geometry, provider and ABI identities share the producer through the e-graph.
The original combined providers remain legal choices. This reuses preparation
across projections; it still launches a quantizer and GEMM kernels. It is not a
fused megakernel. The shared quantizer source and packed ABI enter provider
identity and resource validation.

`DecoderTuningProfile` separates workload representatives and search budgets
from executable capacity. The executor proposes feasible joint batch/query/page
buckets and supplies synthetic query/page CSR metadata, unique private-page
write slots, positions, and fixed-state slots before candidate preparation.
CUDA compares the configured number of finalists on its deployment graph path.
The profile is artifact-bound. Ordinary execution continues to use
OrbitKV-authored metadata. Shared-Prefix physical-layout profiling and joint
layout search remain open; see [FP8 region tuning](fp8-region-tuning.md).

`tools/verify_active_source.py` enforces these forbidden dependency edges,
rejects physical KV ownership types in server source, requires all product
compiler dependencies to resolve through the root workspace, rejects removed
compatibility paths and model/hardware/version-specific active filenames, and
bounds source-file size. The gate cannot prove every semantic ownership rule,
so transaction and failure-atomicity tests remain the executable authority.

## Qualification boundary

The compiler/provider modules, budget ownership, and structured candidate trace
are mapped in [Compiler boundaries and extension points](compiler-boundaries.md).
Compile/load orchestration is in `model/compiler.rs`; workload policy, feasible
bucket construction and profiling fixtures are separate modules. Numerical ABI
definitions, scratch ownership and DeepGEMM tile ordering have separate owners.

[Generated module artifacts](module-artifacts.md) sit at the CUDA backend
boundary. OrbitKV embeds the selected images with its schedule identity, while
OrbitKV compiler owns serialization, target/compiler checks and strict source lookup.
The runtime retains that policy through bucket materialization and execution;
weight loading, provider planning and live CUDA resources keep their own owners.

The CUDA runtime's [weight loader](weight-loading.md) owns mapped checkpoint
bytes, explicit dtype conversion and device upload. The executor owns the
earlier model-wide tensor/shape contract. Storage-compatible bytes are borrowed
from the mapping; conversions keep one typed allocation. A fallible load returns
actual tensor/byte counts, drains each shard before releasing its mapping and
propagates failures to decoder initialization.

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
OrbitKV compiler's searched executables
as child graphs and orders persistent-state D2D epilogues after them. In their
recorded source closure, narrow fixed-step matched tests improved decode wall
time by 5.8-8.3%; that result is not a current-tree claim and broader
performance remains unqualified. A released-hybrid same-executor ablation now
also proves a narrow lifecycle benefit: ten paired release-mode runs reduce live
payload by 27.8%, increase the fixed-budget sequence boundary by 32 tokens, and
show a positive paired total-time confidence interval with identical output. The
tree does not yet prove continuous-batching throughput, multi-user capacity, or
long-running behavior.
