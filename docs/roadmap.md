# Loom Infer roadmap

Loom Infer targets the operator surface required by production LLM inference.
FlashInfer defines the broad comparison surface. FlashAttention defines the
attention-kernel comparison surface.

## 1. Permanent RMSNorm provider

**State:** active.

- keep the H20-qualified F32 path in `loom-infer-cuda`.
- support F32, FP16, and BF16 with scalar and packed paths.
- bind caller-owned memory and non-default streams.
- keep one stream-ordered command scope for every provider.
- replace generated argument vectors with reusable fixed argument packs.
- pass correctness, CUDA Graph, sanitizer, and matched H20 performance gates.

F32, FP16, and BF16 correctness gates pass across scalar, packed, exact-length,
non-default-stream, and chained cases. Typed heterogeneous bindings hold mixed
tensor and workspace types in one scope. Fixed argument packs, CUDA Graph,
sanitizer, and matched performance gates remain open.

Exit: all three dtypes use the common operator lifecycle and have reproducible
H20 correctness and performance records.

## 2. First vendor GEMM provider

**State:** next.

- define contiguous BF16 `D[M,N] = A[M,K] * W[N,K]^T` with F32 accumulation.
- plan one explicit cuBLASLt algorithm before enqueue.
- require caller-owned output and workspace with checked alignment and spans.
- run GEMM through the same command scope without tuning or fallback.
- verify real model shapes before any performance or default-provider claim.

Exit: a fixed BF16 GEMM plan passes correctness, graph, and matched-provider
gates without a tensor copy.

## 3. Attention core

- implement ragged prefill and paged MQA/GQA decode.
- add split-K execution and stable state merge.
- support common causal and sliding-window contracts.
- integrate RoPE, KV append, and page-table access.
- replay fixed plans through CUDA Graphs.

Exit: a real model invokes Loom attention without tensor copies and preserves
tokens or declared numerical quality.

## 4. Decode and KV operations

- add logits processing, penalties, filtering, logprobs, and deterministic
  sampling.
- add speculative verification and token compaction.
- add KV gather, scatter, compaction, and remapping.
- add FP8 and INT8 KV storage with an explicit quality gate.

Exit: decode-tail and KV operations use the same lifecycle and show value in a
real engine workload.

## 5. Quantization, MoE, and matrix work

- add scale, pack, unpack, dequantize, and layout conversion.
- add expert routing support, permutation, and weighted combine.
- call vendor dense, quantized, and grouped GEMM through fixed plans.
- fuse adjacent work only when the complete matched path improves.

Exit: dense and MoE workloads record separate operator, engine, and serving
results.

## 6. Integration and hardware coverage

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
