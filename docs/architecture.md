# Architecture

OrbitKV is one native inference stack with three ownership layers.

```text
HTTP / SSE / WebSocket
        |
        v
server/                 admission, tokenization, scheduling, sampling
        | BatchIntent (logical request state only)
        v
core/                   compile visibility and own the KV lifecycle
        | prepared pages, copies, views, retirement rules
        v
executor/               Luminal graph compilation and device execution
        | completion evidence
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
and output events; they contain no page or device-buffer identity.

## Native data flow

1. The compiler turns model attention semantics into a fingerprinted
   `RuntimeManifest`.
2. The executor derives an `ExecutorPlan` directly from that manifest.
3. `RuntimeSession` prepares append, Prefix/COW, relocation, or release work.
4. The executor lowers manager-selected pages into Luminal metadata and runs the
   graph on the owning stream.
5. Completion evidence advances the Execution Frontier.
6. RuntimeSession publishes new request heads, retires unreachable generations,
   validates cleanup acknowledgement, and only then permits reuse.

The control path is in-process Rust. There is no C boundary, Python bridge,
upstream inference HTTP hop, generic engine adapter, or parallel allocator.

## Source boundaries

```text
core/
  src/                    compiler, manager, RuntimeSession, checkpoint pool
executor/
  src/                    OrbitKV-to-Luminal plan lowering
  luminal/                complete pinned compiler/executor fork
server/
  src/                    async local Engine and semantic request contracts
docs/                     current product contracts
results/                  append-only provenance, never active source
```

The Luminal submodule preserves its upstream history. OrbitKV-specific changes
are made in the fork and pinned here, rather than copied into model- or
hardware-specific directories.

## Qualification boundary

The current tree proves compiler, manager, lifecycle, and executor-metadata
contracts on the host. Real-device tests additionally cover external block-page
attention, stream-ordered token relocation followed by packed-page decode, and
a minimal released full-attention checkpoint completing prefill plus one decode
step. Relocation evidence is gated by a real CUDA event; ordinary model-step
completion still relies on the embedding runtime's completion assertion. The
tree does not yet prove matched output equivalence against a reference engine,
throughput, capacity, cancellation, continuous batching, or long-running
behavior.
