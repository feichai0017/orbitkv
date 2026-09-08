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
- Recurrent and convolution classes now have stable per-class CUDA arenas,
  generation-checked byte-range lowering, shared Luminal required aliases, and
  runtime-identity- and event-gated completion evidence. Dynamic slot metadata
  selects manager-authored destinations while the arena address stays fixed;
  graph search uses a private scratch arena and cannot mutate live OrbitKV
  state. The explicit real-device gate has not run in the current environment.
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

## Product model strategy

The primary release target is the Qwen3.8 27B block-FP8 checkpoint. Its text
decoder is the forcing function for the product architecture: 64 layers with a
3:1 Gated DeltaNet/Full-attention schedule, persistent recurrent and causal
convolution state, partial rotary dimensions, and dynamic block-FP8 linear
operators. OrbitKV already compiles the checkpoint into 16 Full token-KV layers
plus 48 recurrent and convolution layers. The executor now parses the nested
text configuration and carries fixed-state geometry into its compiler contract,
but GDN execution and checkpoint FP8 loading are still explicit fail-closed
gaps.

Use the structurally equivalent small BF16 checkpoint as the bring-up target.
It must exercise the same 3:1 layer schedule and state transitions before the
27B FP8 qualification. Existing dense Full and Full+Sliding checkpoints remain
regression and lifecycle witnesses; they are not parallel product targets.
Model support stays structural, so no checkpoint name may select an operator or
physical layout in product code.

The second architecture target is the latest openly released DeepSeek family.
At this roadmap revision that is DeepSeek V4 Flash Vision, whose text path adds
sparse retrieval/index state, low-rank projections, MoE, mixed FP8/FP4 storage,
and speculative heads. It follows the Qwen target because it needs several of
the same quantized-linear and persistent-state foundations but is not a
single-device bring-up model.

## Primary model closure

1. Run the stable fixed-state arena, shared-alias, and stream-ordered completion
   gate on the qualification GPU; keep it as a regression prerequisite for all
   recurrent/convolution kernels.
2. Extend the backend-neutral gated-delta recurrence now represented by an
   independent f32 oracle and a pure Luminal single-token HLIR graph. Add a
   production CUDA state-update candidate derived from an attributed,
   license-compatible mature implementation, then select it through Luminal's
   egglog/search pipeline. The unfused HLIR expression remains the semantic
   fallback and parity authority.
   The first Luminal-native in-place state-update candidate now exists and is
   introduced only by an exact rank-four egglog match. A generic graph arena
   gathers and commits manager-selected request slots in manifest layer order,
   while initialization copies the prior published slot before execution.
   Remaining work is to embed this path in the complete decoder, qualify it on
   device, and add fused token readout plus a chunked-prefill candidate.
3. Qualify recurrent decode, chunked prefill, causal-convolution history,
   cancellation, Prefix boundaries, and state-slot reuse on the small BF16
   checkpoint.
4. Add partial RoPE and text-only nested checkpoint loading without admitting
   unimplemented image/video inputs.
5. Add block-FP8 weights, dynamic activation scaling, scaled matrix products,
   and safetensors scale validation. Luminal already has FP8 dtypes, quantization
   and cuBLASLt primitives; the model loader and block-scale graph remain open.
6. Run the 27B FP8 text path on H20, first for deterministic token parity and
   complete state drain, then for continuous batching and long-context pressure.

## Searchable attention execution

This is the first performance-compiler milestone after the primary hybrid model
can execute. It turns the current FlashInfer integration from a fixed custom-op
implementation into a genuine Luminal compiler choice.

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

1. Bind recurrent/linear attention plus convolution state to stable device
   arenas and execute the host-qualified atomic engine step for the primary model.
2. Independently qualify exact Chunked token KV on device.
3. Add sparse retrieval/index state and low-rank attention components for the
   second architecture target.
4. Add MoE routing, mixed low-precision experts, and multi-device placement only
   after the dense primary target closes.
5. Defer tree, cross-attention, speculative decoding, and vision execution until
   their visibility and completion contracts are explicit.

Each family must pass plan compilation, randomized lifecycle checks, executor
lowering, operator parity, released-checkpoint end-to-end correctness, and then
a matched benefit experiment.

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

The product goal is to beat both current vLLM and SGLang on the primary 27B FP8
checkpoint. A win requires the lower confidence bound of output throughput to
exceed both baselines while p95 TTFT and p95 TPOT are no worse, under the same
weights, quantization, device budget, request trace, scheduler limits, output
semantics, and benchmark client. Long-prompt, decode-heavy, concurrency, and
memory-pressure suites are reported separately; winning one selected point is
not an overall claim. No date or speedup is promised before the attribution
profile shows which layer owns the current gap.

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
