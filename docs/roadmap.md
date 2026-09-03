# Roadmap

The target product is one Rust inference process: OrbitKV owns attention-state
lifetimes, the Luminal fork compiles and executes model graphs, and the Rust
server exposes client protocols. Planned work is not a current capability.

## Current checkpoint

- The active workspace contains only `core`, `executor`, and `server`.
- The compatibility tree, C ABI, Python runtime, packaged target, and numbered
  wire contracts have been removed.
- `ExecutorPlan` consumes `RuntimeManifest` directly.
- The Luminal fork accepts externally owned paged-attention metadata.
- The server has an async local `Engine` stream boundary with no physical-page
  types.
- Compiler, manager, RuntimeSession, and plan lowering have host tests.
- Complete model execution and current-architecture real-device evidence remain
  open.

## R1: Complete the native execution transaction

Connect one Luminal model graph to the complete RuntimeSession lifecycle:
prepare, COW copies, KV writes, attention metadata, forward, sampling, event
recording, completion, publication, retirement acknowledgement, and reuse. The
same stream or an explicitly synchronized dependency graph must own all effects.

## R2: Build the scheduler and API

Add tokenizer workers, request queues, continuous batching, sampling state,
cancellation, and backpressure above the local `Engine`. Then add
OpenAI-compatible Responses/chat endpoints and SSE/WebSocket translation.
Conversation persistence and tool loops should remain optional API concerns and
must not enter KV ownership.

## R3: Close a minimal real-device model path

Start with one released architecture whose full-attention graph already works in
the fork. Establish deterministic outputs, eager execution, stream/event
provenance, cancellation, final drain, and repeated requests before enabling
graph replay or more attention classes.

## R4: Qualify compiled lifetimes

Run independent end-to-end closures for Full, Sliding, Full+Sliding, and exact
Chunked attention. Each must exercise its actual visibility boundary, retirement
trace, generation reuse, and final drain. Latent and fixed state require their
own component-aware kernels and transactions.

## R5: Measure benefit

Compare against an unchanged reference engine using the same model, weights,
dtype, kernels, batching policy, request trace, and device budget. Report output
correctness, TTFT, inter-token latency, throughput, tail latency, resident and
reserved memory, capacity, copy cost, allocator work, and long-running pressure.
Predeclare pass/fail gates and retain failed runs.

## R6: Production hardening

Add CUDA Graph address stability, overlapping streams, bounded queues, failure
containment, metrics, tracing, soak tests, distributed ownership, release
artifacts, and supported-combination matrices only after the eager single-device
path is closed.
