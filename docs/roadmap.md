# Roadmap

OrbitKV targets one native Rust inference process. `orbitkv` compiles and owns
attention-state lifetimes, the Luminal fork compiles and executes model graphs,
`orbitkv-engine` coordinates requests and device work, and `orbitkv-server`
provides the client protocol. Planned work is not a current capability.

## Current baseline

- `RuntimeManifest` is the shared source of truth for lifecycle and execution.
- OrbitKV is the sole KV authority: it owns pages, generations, snapshots,
  Prefix/COW, retirement, acknowledgement, and reuse.
- Full, Sliding, Full+Sliding, and exact Chunked lifetimes compile and pass host
  lifecycle tests. Token-level relocation and live-token compaction are not part
  of the product.
- One symbolic decoder graph is searched once into decode and prefill buckets.
  Dynamic inputs use stable addresses and each attention class has one persistent
  K/V arena.
- Persistent K/V updates are required aliases during search and artifact load.
  Candidates that materialize incompatible state fail closed.
- The current direct paged-attention node uses FlashInfer. Luminal searches the
  surrounding decoder graph, but does not yet select among multiple attention
  implementations or jointly derive a KV layout.
- Released H20 closures exist for a dense Full checkpoint and an interleaved
  Full+Sliding checkpoint. Exact Chunked, MLA, recurrent/linear attention,
  convolution state, MoE, quantization, and multi-device execution are not
  released-model-qualified.
- The single-process Rust engine and OpenAI-compatible server pass bounded
  batching, cancellation, streaming, shutdown, and final-drain tests.
- The best recorded matched product comparison remains negative: 0.598x stock
  SGLang output throughput on the recorded C2 trace. The same configuration used
  40.4% less K/V tensor payload because the reference disabled hybrid Sliding
  memory. This is not a replacement or superiority claim.
- External export, restore, deletion, and failure semantics pass through the
  host-memory reference transport. Mooncake, NIXL, remote leases, and network
  benefit remain open.

## Searchable attention execution

This is the immediate milestone. It turns the current FlashInfer integration
from a fixed custom-op implementation into a genuine Luminal compiler choice.

1. Define one backend-neutral paged-attention semantic op carrying OrbitKV
   class identity, visibility, page geometry, dtype, head geometry, and bucket
   dimensions.
2. Provide at least two semantically equivalent implementations for an admitted
   geometry: initially FlashInfer and a Luminal-native CUDA implementation.
3. Express matching and selection through egglog and Luminal extraction/search.
   Do not add model-name, release-name, or GPU-name dispatch in product code.
4. Preserve OrbitKV's required K/V aliases for every candidate and persist the
   chosen implementation and kernel identity in the decoder artifact.
5. Prove Full, Sliding, decode, and causal-prefill equivalence against an
   independent reference before interpreting performance.

The milestone closes only when a real search containing more than one legal
attention implementation selects by measured device cost and the selected
artifact replays correctly on a released checkpoint. Merely wrapping FlashInfer
in a Luminal custom op does not satisfy this gate.

## Additional attention-state families

Coverage advances by state family rather than checkpoint-name branches:

1. Independently qualify exact Chunked token KV on device.
2. Integrate recurrent/linear attention plus convolution state through the
   generation-safe fixed-state pool and one atomic engine step.
3. Add latent/RoPE-aware MLA attention and bind component-aware latent state.
4. Add MoE routing and then quantized linear operators only as required by the
   selected released checkpoints.
5. Defer sparse, tree, cross-attention, speculative decoding, and multi-device
   claims until their visibility and completion contracts are explicit.

The first heterogeneous target should be a released Full + linear-attention +
convolution checkpoint because the core already compiles that state shape. The
second should exercise MLA. Each family must pass plan compilation, randomized
lifecycle checks, executor lowering, operator parity, released-checkpoint
end-to-end correctness, and then a matched benefit experiment.

## Serving performance

After attention becomes a real compiler choice, optimize the complete warm path:

1. Attribute TTFT and TPOT to attention, graph dispatch, scheduler, metadata
   upload, sampling, and frontend overhead.
2. Remove exact-shape CUDA Graph recapture cliffs with stable capacity
   signatures and measured capture policy.
3. Add on-device temperature, top-k, and top-p sampling.
4. Run fixed-model, fixed-weight, fixed-dtype, fixed-memory, and fixed-trace
   comparisons with SGLang and vLLM through the same benchmark client.
5. Promote a claim only when output, completion, final drain, p95/p99 latency,
   throughput, and admission-capacity gates all pass.

No date or performance target is promised before the attribution profile shows
which layer owns the current gap.

## External KV transports

Implement Mooncake behind `ExternalKvTransport` and reuse the host adapter's
conformance suite. Add remote lease epochs, renewal, eviction intent, active
restore pins, exact deletion acknowledgement, timeout reconciliation, shared
Prefix restore, and node-failure recovery. Add NIXL only after Mooncake
semantics are stable, then compare cold prefill, local retention, and external
restore with identical workloads.

Dynamo may provide routing, topology, discovery, events, and telemetry. Do not
import `kvbm-logical`, Dynamo `KvBlockManager`, lifecycle pins, or another page
allocator: OrbitKV remains the sole KV authority.

## Formal and production closure

State the Minimum Persistent State Realization objective and constraints
formally. Prove optimality for bounded-window and exact-chunked subclasses and
compare generated plans with a small exact oracle for randomized instances.

Then add authenticated completion envelopes, metrics, tracing, crash recovery,
long soak tests, multi-device placement, release artifacts, and a supported
combination matrix. Production claims require every applicable correctness,
pressure, cancellation, and long-running gate.
