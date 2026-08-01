# MoE Movement Around Vendor Grouped GEMM

## Boundary

Loom owns the memory-bound movement on either side of an engine-selected
grouped GEMM:

```text
router IDs/weights -> Loom permute -> vendor grouped GEMM -> Loom combine
```

The grouped GEMM is never implemented or wrapped as a private Loom matrix
kernel. The first K5 slice begins at the production vLLM boundary after
`fused_topk`: the engine already owns routing scores, top-k selection, and F32
routing weights. Top-k routing remains a separate future slice.

## Permutation Contract

For hidden states `[tokens, hidden]` and int32 expert IDs
`[tokens, top_k]`, let `assignments = tokens * top_k`.
`moe_permute` returns:

- expert-major activations `[assignments, hidden]` in the input dtype;
- int64 expert offsets `[num_local_experts + 1]` for grouped GEMM;
- int32 inverse permutation `[tokens, top_k]` for the combine step;
- int32 flattened assignment IDs `[assignments]` in permuted order.

All tensors are contiguous inference tensors on one CUDA device. Permuted
activations may be F32, FP16, BF16, or FP8 E4M3FN; the FP8 route is an exact
byte movement contract with no scale arithmetic. Expert IDs and the optional
global-to-local expert map are int32. The flattened assignment count and
global expert count must fit int32.

Without expert parallelism, assignments are stably ordered by expert ID. With
expert parallelism, a non-negative map entry selects the local expert. A `-1`
entry marks a remote route. Local assignments come first and are stably
ordered by local expert; remote assignments follow, stably ordered by global
expert. This matches vLLM's production `MoEPermuteScratch` metadata order.

Remote activation rows are explicitly zero-filled. Their assignment IDs use
`assignments` as the sentinel. `expert_offsets[-1]` is the number of valid
local rows. vLLM leaves its scratch activation tail unspecified, so only valid
local activation rows are an equivalence boundary; offsets, inverse indices,
and assignment IDs are compared exactly across the full tensors.

## Combine Contract

`moe_combine` consumes F32, FP16, or BF16 grouped-GEMM output
`[assignments, hidden]`, contiguous F32 routing weights `[tokens, top_k]`, the
inverse permutation, and expert offsets. Routes whose inverse row is at or
beyond `expert_offsets[-1]` are remote and contribute zero. Valid routes are
multiplied and accumulated in F32 in route order, then converted once to the
output activation dtype. FP8 combine is deliberately rejected because it is
arithmetic rather than byte movement and needs an explicit scale contract.

The operator performs no routing softmax, no renormalization, no collective,
and no matrix multiplication. Expert-parallel communication stays with the
engine or its selected transport.

## One Vertical Implementation Path

The operator follows the same path as every other Loom operator:

| Layer | Implementation |
| --- | --- |
| Contract and oracle | `crates/loom-kernels/src/moe.rs` |
| Safe CUDA dispatch | `crates/loom-cuda/src/moe_dispatch.rs` |
| Checked borrowed-memory bridge | `crates/loom-cuda-bridge/src/cuda/moe_bridge.rs` |
| Raw launch ABI and CUDA | `crates/loom-cuda-sys` and `crates/loom-cuda-sys/cuda/src/moe.cu` |
| Stable ABI PyTorch dispatcher | `python/csrc/moe.cpp` |
| Public Python API | allocating `loom_kernels.moe_permute` / `loom_kernels.moe_combine` plus caller-owned `torch.ops.loom_kernels.*.out` |
| vLLM adapter | `loom_kernels.vllm.moe` |

The raw and checked Rust boundaries accept caller-owned output and workspace
storage. The public PyTorch convenience API allocates its outputs and a byte
workspace on the current stream. Its standard `.out` overloads expose the same
launch helpers to engines with existing scratch allocations. There is no
ctypes path, direct C++ CUDA launch, fallback implementation, or legacy schema.

## CUDA Schedule

Permutation initializes unsigned radix keys and flattened assignment IDs,
uses stable CUB radix sort, derives local expert offsets, then gathers
activations. The radix range ends at the bits required by
`2 * num_experts - 1`; this covers local IDs and vLLM-compatible remote keys
without sorting unused high bits.

Aligned rows use 16-byte activation loads and stores. One row block writes the
inverse/assignment metadata while gathering, removing a separate finalize
kernel. Non-aligned rows use the same contract through a scalar fallback.
Combine uses the same aligned 16-byte row schedule and F32 lane accumulators,
with a scalar fallback for arbitrary row widths.

## Current Evidence And Admission

The H20 admission baseline is vLLM 0.25.1 `MoEPermuteScratch` plus a reused
unpermute output, matching its Cutlass grouped-GEMM boundary. The candidate is
the allocating public Loom permute-plus-combine pipeline; neither side times a
GEMM.

For BF16, hidden size 4096, top-k 2, and 64 experts, seven-repeat CUDA Graph
pipeline ratios at 1/8/32/128/512/2048 tokens are
`1.032/0.962/1.014/1.077/1.163/1.124x`. Eager ratios are
`1.24-1.37x`. At 512 tokens, the isolated permutation and combine ratios are
`1.225x` and `1.104x`.

With 64 global and 32 local experts, CUDA Graph pipeline ratios at
32/128/512/2048 tokens are `1.138/1.190/1.191/1.013x`; eager ratios are
`1.58-1.59x`. All valid local activations and all metadata match vLLM, Loom's
remote tail is zero, and the measured combine maximum error is zero.

The source matrix passes on PyTorch 2.11 with vLLM 0.24 and 0.25, and the same
Stable ABI libraries pass on PyTorch 2.10 without vLLM. These results qualify
the direct movement boundary, not an end-to-end MoE model speedup.

The explicit vLLM adapter is enabled with
`LOOM_KERNELS_ENABLE_MOE_MOVEMENT=1`. It patches the production movement
wrappers and already-imported Cutlass/Humming consumers, preserves vLLM's
per-token FP8 scale reorder, uses caller-owned `.out` tensors, and leaves every
grouped-GEMM function untouched. Unsupported contracts call the original
wrapper before Loom admission; an admitted Loom call is fail-closed.

An isolated vLLM 0.25.1 `LLM.generate` gate over a synthetic two-layer
Qwen2-MoE checkpoint selects `VLLM_CUTLASS`, preserves every generated token,
and records 48 FP8 permutation plus 48 BF16 combine hits with no rejection.
The first admitted tensor is `[320, 512]`, top-k 2 over 8 experts, using
caller-owned output. Baseline and Loom median batch latencies are `17.453 ms`
and `17.103 ms`, a `1.0205x` ratio. This closes engine admission, not a
production-model or serving-speedup claim; K5 remains open for a pinned
production-representative MoE workload and profile-driven routing decision.

Raw evidence:

- [all-local H20 result](../results/h20-moe-movement-20260801.json)
- [expert-parallel H20 result](../results/h20-moe-movement-ep-20260801.json)
- [vLLM engine result](../results/h20-vllm-engine-moe-movement-20260801.json)
