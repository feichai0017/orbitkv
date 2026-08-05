# Loom Infer roadmap

Loom Infer targets the operator surface required by production LLM inference.
FlashInfer defines the broad comparison surface. FlashAttention defines the
attention-kernel comparison surface.

## 1. Permanent RMSNorm provider

**State:** active.

- keep the H20-qualified F32 path in `loom-infer-cuda`.
- support F32, FP16, and BF16 with scalar and packed paths.
- bind caller-supplied memory and non-default streams.
- share read-only inputs through `Arc` and transfer writable buffers until
  completion.
- keep one stream-ordered command scope for Rust and vendor providers.
- replace generated argument vectors with reusable fixed argument packs.
- pass correctness, CUDA Graph, sanitizer, and matched H20 performance gates.

The current owned-binding revision passes F32, FP16, and BF16 correctness across
scalar, packed, exact-length, non-default-stream, and chained cases. Memcheck,
racecheck, and synccheck report no errors. Fixed argument packs and matched
performance gates remain open.

Exit: all three dtypes use the common operator lifecycle and have reproducible
H20 correctness and performance records.

## 2. First vendor GEMM provider

**State:** active. The owned-binding and Graph revision passed its H20
correctness and sanitizer gates.

- define contiguous BF16 `D[M,N] = A[M,K] * W[N,K]^T` with F32 accumulation.
- plan one explicit cuBLASLt algorithm before enqueue.
- take caller-supplied output and workspace into checked ownership with validated
  alignment and spans, then return them after completion.
- run GEMM through the same command scope without tuning or fallback.
- verify real model shapes before any performance or default-provider claim.

Exit: a fixed BF16 GEMM plan passes correctness, graph, and matched-provider
gates without a tensor copy.

The fixed BF16 cuBLASLt plan passes H20 correctness and command-lifecycle gates.
Its RMSNorm-to-GEMM Graph replays twice with a bit-exact final output.
The first matched eager-provider result covers the fixed M=1 shape and records
1.33x lower median latency than the matched FlashInfer path. Other shapes,
isolated kernel and Graph timings, real-model shapes, and engine invocation
remain open.

## 3. Fixed-address CUDA Graph execution

**State:** complete for the first fixed-address contract.

- consume a one-shot `GraphQueue` to capture one checked command scope on its
  private non-default stream.
- retain fixed buffer ownership, CUDA functions, and vendor plans across replay.
- instantiate and launch through owned `CapturedGraph` and `GraphExec` values.
- retain one external completion event and record it once per replay.
- reject rebinding, node updates, cross-stream launch, and concurrent replay in
  the first contract.
- validate RMSNorm to BF16 GEMM capture, replay, cleanup, and sanitizer paths.

Result: the final output after two fixed-address replays matches the CPU oracle.
The runner drops its external resource owners before replay.

The Loom replay path has no planning or explicit allocation. Compute Sanitizer
reports no errors or device leaks on H20. See the
[machine-readable record](results/h20-owned-bindings-cuda-graph-correctness-20260803.json).

## 4. Attention core

**State:** active. The first narrow single-decode contract passed its H20
correctness and sanitizer gates.

- retain the BF16 NHD D128 single-request MHA/MQA/GQA correctness baseline.
- retain the matched eager-provider benchmark against the pinned FlashInfer
  release and add an isolated Graph/kernel timing gate.
- retain the backend-independent BF16 NHD D128 page-size-16 batch-decode
  contract and CPU oracle.
- implement its checked cuda-oxide plan and permanent MHA/MQA/GQA provider,
  then implement ragged prefill.
- retain split-K execution, stable F32 state merge, and its H20 correctness
  gate.
- support common causal and sliding-window contracts.
- integrate RoPE, KV append, and page-table access.
- replay fixed plans through CUDA Graphs.

Exit: a real model invokes Loom attention without tensor copies and preserves
tokens or declared numerical quality.

The first slice also rejects five short buffers and duplicate bindings before
CUDA submission. Split-K partial and merge kernels pass H20 correctness and
sanitizer gates with checked workspace and two-command admission. They lower
Loom median eager latency by 5.39x at GQA KV length 127 and 38.19x at KV length
4096 relative to the recorded direct baseline. FlashInfer remains 1.17x and
2.09x lower-latency. CUPTI activity timing now separates partial and merge
kernel duration.

The paged batch-decode contract validates FlashInfer-compatible `i32`
`indptr`, page indices, and last-page lengths before mapping logical KV tokens
to NHD physical pages. CPU tests establish numerical equivalence to contiguous
decode. A CUDA plan, H20 correctness and sanitizer evidence, matched eager and
Graph performance, and real model invocation remain open.

## 5. Decode and KV operations

- add logits processing, penalties, filtering, logprobs, and deterministic
  sampling.
- add speculative verification and token compaction.
- add KV gather, scatter, compaction, and remapping.
- add FP8 and INT8 KV storage with an explicit quality gate.

Exit: decode-tail and KV operations use the same lifecycle and show value in a
real engine workload.

## 6. Quantization, MoE, and matrix work

- add scale, pack, unpack, dequantize, and layout conversion.
- add expert routing support, permutation, and weighted combine.
- call vendor dense, quantized, and grouped GEMM through fixed plans.
- fuse adjacent work only when the complete matched path improves.

Exit: dense and MoE workloads record separate operator, engine, and serving
results.

## 7. Integration and hardware coverage

- expose a stable Rust API after the first provider passes admission.
- add a checked C ABI only when an external engine requires it.
- publish hashed device artifacts for supported architectures.
- qualify Hopper first, then add Blackwell as a separate matrix row.
- add collectives only for measured distributed workloads.

Exit: each supported hardware row has reproducible correctness, performance,
and integration evidence.

## Admission rule

A faster microbenchmark does not prove a faster model or server. Each result
must state whether it covers a kernel, graph, engine, or serving boundary.
