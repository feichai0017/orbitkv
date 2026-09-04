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
- Manager-authored token moves lower to stream-ordered K/V copies and are
  published only after CUDA event completion.
- The server has an async local `Engine` stream/cancellation boundary with no
  physical-page types. An optional vLLM Rust frontend supplies
  OpenAI/tokenizer/chat/SSE code through a narrow Add/Abort adapter; a
  test-engine HTTP closure passes.
- Compiler, manager, RuntimeSession, and plan lowering have host tests.
- A minimal released full-attention checkpoint passes real-device prefill and
  decode; packed relocation followed by paged decode also passes.

## R1: Complete the native execution transaction

Connect one Luminal model graph to the complete RuntimeSession lifecycle:
prepare, COW copies, KV writes, attention metadata, forward, sampling, event
recording, completion, publication, retirement acknowledgement, and reuse. The
Full token-KV correctness path and relocation event gate now exist; remaining
work is scheduler-owned sampling/cancellation, batched execution, and a unified
event envelope for ordinary model steps.

## R2: Build the scheduler and API

Implement the concrete local scheduler/engine behind the existing server
contract: request queues, continuous batching, greedy sampling state,
cancellation cleanup, and backpressure. Then exercise the optional vLLM Rust
HTTP/tokenizer/chat/SSE frontend end to end. Extend sampling, logprobs, tool
parsing, and multimodal fields only as their local semantics become real; the
adapter rejects them today. Conversation persistence and tool loops remain API
concerns and must not enter KV ownership.

## R3: Extend the minimal real-device model path

The first released full-attention checkpoint now completes prefill and one
decode step. Extend this to deterministic reference parity, repeated requests,
cancellation, final drain, continuous batching, and ordinary-step event
provenance before claiming a complete engine path.

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
