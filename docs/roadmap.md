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
- retain its checked cuda-oxide MHA/MQA/GQA provider and H20 correctness and
  sanitizer gate.
- retain eight-warp paged MQA/GQA token parallelism and its matched eager and
  CUPTI gates, then add fixed-address Graph gates.
- retain the BF16 NHD D128 ragged prefill contract, CPU oracle, checked
  cuda-oxide provider, and H20 correctness/sanitizer gate.
- retain tiled GQA4 QK/online-softmax/PV, eight-way split-K, stable F32 merge,
  and sixteen-warp MQA with their matched eager gates.
- expand query tiling beyond the admitted GQA4 shape and close the remaining
  long-GQA gap before expecting FlashInfer-class performance.
- retain split-K execution, stable F32 state merge, and its H20 correctness
  gate.
- support common causal and sliding-window contracts.
- retain the standard BF16 D128 NeoX RoPE explicit-position provider and its
  correctness, sanitizer, and matched eager gates.
- retain the first one-token/request standard RoPE plus paged KV append
  contract and its correctness, sanitizer, and matched eager gates.
- expand KV append to arbitrary batch indices, multi-token updates, and
  additional RoPE/storage variants.
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
to NHD physical pages. Its direct one-warp-per-request-head cuda-oxide provider
passes MHA/MQA/GQA H20 correctness and all four Compute Sanitizer tools. A
eight-warp block-local token parallelism lowers Loom MQA/GQA eager latency by
3.78x and 3.32x relative to the direct record. Loom is now 4.41x lower-latency
for MHA and 2.35x lower-latency for MQA than FlashInfer. Batch-4 GQA still has
no stable ranking because FlashInfer's order delta is 60.62%. Fixed-address
Graph replay and real model invocation remain open.

The ragged prefill slice uses separate query/KV `indptr` arrays and
FlashInfer-compatible bottom-right causal alignment. Short requests retain one
warp per query-row/head; long single-KV-head MQA uses sixteen warps; admitted
long GQA4 uses fused tensor-core QK/online-softmax/PV over eight KV partitions
and a caller-owned F32 merge workspace. Other declared long requests use eight
warps. All paths pass MHA/MQA/GQA H20 correctness plus all four Compute
Sanitizer tools.

The matched eager result lowers Loom long-GQA latency to `48.232`
microseconds. Unrolled 16-byte `cp.async` staging is `1.148x` faster than the
previous tiled path and the complete path is `7.729x` faster than direct.
FlashInfer remains `2.206x` lower-latency on stable long GQA. Broader query
tiling and real model invocation remain open. Fixed-address Graph correctness
now passes for the tiled partial-plus-merge path after two replays and external
owner teardown. The matched single-replay Graph result records Loom at `50.480`
microseconds and FlashInfer at `32.640` microseconds, with FlashInfer `1.547x`
lower-latency on the admitted long-GQA shape. Engine invocation remains open.

The first standalone RoPE slice accepts explicit I32 position IDs for BF16
NHD D128 Q/K tensors, rotates all 128 dimensions in NeoX split-half style, and
uses CUDA libdevice full-range math. It passes positions through 32,767 and all
four sanitizer tools. On the admitted 96-token Q16/K4 suffix shape, Loom records
`3.997` microseconds versus FlashInfer's `5.077` microseconds, or `1.270x`
lower latency.

The first fused paged mutation slice accepts one BF16 Q/K/V token per request,
derives each position from the extended page table, rotates Q/K, and writes
rotated K plus unmodified V into the final physical NHD slot. It passes
bit-exact H20 correctness, duplicate-slot and invalid-page sentinel guards,
and all four sanitizer tools. On the admitted batch-4 Q16/K4 D128,
page-size-16 case, its one-kernel eager path records `3.989` microseconds
versus `11.735` microseconds for FlashInfer's two-kernel composition, or
`2.942x` lower latency under fixed host affinity. Interleaved, ragged-offset,
Llama 3.1, arbitrary-index, multi-token, cached, and quantized variants remain
open.

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
