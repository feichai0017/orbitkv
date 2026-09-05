# Architecture

OrbitKV is one native inference stack with three ownership layers.

```text
HTTP / SSE / WebSocket
        |
        v
server/                 HTTP/tokenization + local scheduling/sampling boundary
        | BatchIntent (logical request state only)
        v
core/                   compile visibility and own the KV lifecycle
        | prepared pages, copies, views, retirement rules
        v
executor/               Luminal graph compilation and device execution
        | local or external completion evidence
        v
core/                   publish, retire, acknowledge, reuse
```

## Authority

`core/` is the sole owner of request and snapshot identities, page allocation,
page generations, Prefix/COW, semantic liveness, retirement, and reuse.

`executor/` owns tensors, compiled graphs, kernel selection, streams, events,
and sampling execution. It consumes immutable physical metadata selected by
OrbitKV. Cached metadata is an execution artifact, never a second page table.

`server/` owns client protocols, admission queues, batching, cancellation, and
backpressure. Its public types contain request IDs, tokens, logical boundaries,
sampling intent, and output events; they contain no page or device-buffer
identity. The optional vLLM frontend reuses OpenAI HTTP, tokenizer, chat
template, and streaming code through a narrow Add/Abort transport. It does not
import a second scheduler, KV allocator, or device runtime.

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

1. The compiler turns model attention semantics into a fingerprinted
   `RuntimeManifest`.
2. The executor derives an `ExecutorPlan` directly from that manifest.
3. The model executor compiles one symbolic decoder graph into decode and
   prefill buckets and retains one stable K/V arena per attention class. Dynamic input buffers are
   allocated to their configured capacities before search.
4. `RuntimeSession` prepares append, Prefix/COW, relocation, or release work.
5. The executor updates bounded dynamic inputs, dispatches the matching Luminal
   bucket, runs on the owning stream without recompiling the model, and samples
   greedy token IDs on device. A warmed fixed-signature decode may instead
   replay one outer CUDA Graph; shape or CSR-indptr changes fail closed and
   require recapture.
6. Completion evidence advances the Execution Frontier.
7. RuntimeSession publishes new request heads, retires unreachable generations,
   validates cleanup acknowledgement, and only then permits reuse.

## Physical-residence ablation

`CanonicalKvManager::new` executes the compiler-authored physical plan by
default. `new_with_residence` additionally accepts a backend-neutral
`PhysicalResidencePolicy` for controlled experiments:

- `Compiled` uses generated address and retirement programs.
- `RequestLifetime` keeps the same attention visibility and Luminal kernels,
  but uses append-only addresses and retains physical pages until request
  release.

The conservative policy is an attribution baseline, not a serving mode. It
fails closed for resettable Chunked layouts, relocation, Prefix publication,
and external export. Prepared attention views omit old physically resident
pages that are unnecessary for the current query range, so the executor sees
the same CSR geometry while underlying page identities may differ.

The execution control path is in-process Rust. The optional vLLM HTTP frontend
uses a process-local IPC protocol adapter because that crate is coupled to its
`EngineCoreClient`; no Python process or upstream inference HTTP hop is present.

## Source boundaries

```text
core/
  src/                    compiler, manager, RuntimeSession, checkpoint pool
executor/
  src/                    OrbitKV-to-Luminal lowering and transport contract
  tests/                  real-byte reference transport closures
  luminal/                complete pinned compiler/executor fork
server/
  src/                    local Engine, semantic requests, optional HTTP adapter
docs/                     current product contracts
results/                  compact reviewed current evidence, never active source
```

The Luminal submodule preserves its upstream history. OrbitKV-specific changes
are made in the fork and pinned here, rather than copied into model- or
hardware-specific directories. See [executor-upstream.md](executor-upstream.md)
for the fork delta and update procedure.

## Qualification boundary

The current tree proves compiler, manager, lifecycle, and executor-metadata
contracts on the host. Real-device tests additionally cover external block-page
attention, stream-ordered token relocation followed by packed-page decode, and
a minimal released full-attention checkpoint completing prefill plus repeated
decode through one precompiled two-bucket runtime. Relocation evidence is gated
by a real CUDA event; ordinary model-step completion still relies on the
embedding runtime's completion assertion. Generic stable-input outer-graph
replay and a released-checkpoint prefill/capture/replay lifecycle pass on H20.
The current parent graph preserves Luminal's searched executables as child
graphs and orders persistent-state D2D epilogues after them. Narrow fixed-step
matched tests improved decode wall time by 5.8-8.3%; broader performance remains
unqualified. The tree does not yet prove matched output equivalence against a
reference engine, throughput, capacity, model-backed HTTP execution,
cancellation cleanup, continuous batching, or long-running behavior.
The multi-class graph path has additionally executed a short synthetic
Full/Sliding policy on H20 using released dense weights. Because the checkpoint
was not trained with that policy and the window did not cross its boundary, this
is plumbing evidence rather than hybrid-model correctness or benefit evidence.
A separate same-graph H20 mechanism check crossed a 64-token Sliding boundary
with an 80-token prefill and a following decode. Compiled and request-lifetime
residence produced byte-identical logits and token IDs in both phases; after
decode, the Sliding arena held 5 pages / 491,520 bytes versus 6 pages / 589,824
bytes for the conservative baseline. This is a single synthetic-policy
execution and establishes only physical-attribution plumbing, not throughput,
capacity at workload scale, or a released hybrid-model benefit.
