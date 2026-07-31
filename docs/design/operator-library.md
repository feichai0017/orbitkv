# Loom Kernels Architecture

## Product Boundary

Loom Kernels is an operator backend that can be embedded into Rust-native or
existing LLM inference engines. It does not own model scheduling, weights,
tokenization, KV lifetime, or request serving.

## Layers

| Layer | Responsibility |
| --- | --- |
| `loom-kernels` | dtype, shape/layout, aliasing, capability, and reference contracts |
| `loom-cuda-sys` | internal launch ABI, CUDA compilation, and packaged handwritten kernels |
| `loom-cuda` | safe owned/borrowed CUDA resources, validation, dispatch, and benchmarks |
| `loom-cuda-bridge` | panic-contained checked C entrypoints into borrowed Rust dispatch |
| engine adapters | translate engine tensors/streams without owning engine policy |

CPU references never call accelerator code. Backends report unsupported
contracts explicitly; they do not silently copy, cast, or fall back.
The concrete module map and rules for extending these layers are documented in
[code layout](code-layout.md).

## Engine-Owned CUDA Resources

`loom-cuda` supports two execution modes behind the same checked operator
methods:

| Resource | Standalone Rust | Embedded engine |
| --- | --- | --- |
| stream | owned `CudaStream` | non-owning `CudaStreamRef` |
| read-only memory | owned `DeviceBuffer<T>` | borrowed `DeviceSlice<'a, T>` |
| writable memory | owned `DeviceBuffer<T>` | exclusive `DeviceSliceMut<'a, T>` |

`CudaBackend` is generic over the stream handle. All safe operator entrypoints
accept sealed read/write device-memory traits, so owned and borrowed storage use
the same dtype, length, and shape validation. Borrowed handles do not allocate,
copy, synchronize, or destroy framework resources.

Raw stream and device-pointer construction is intentionally `unsafe`: an
adapter must prove the active CUDA device/context, storage dtype, pointer
alignment, lifetime, exclusivity, and stream ordering. Once that narrow
boundary is crossed, ordinary callers cannot invent alternative trait
implementations or pass read-only storage as a mutable output. Kernel launches
remain asynchronous.

The pure-Rust H20 smoke test exercises this zero-copy path. Every framework
operator uses the same mechanism: the boxed LibTorch Stable ABI dispatcher
targets PyTorch 2.10 and passes tensor pointers, physical storage spans, layout
strides, and PyTorch's current stream through `loom-cuda-bridge`, which
constructs borrowed Rust views and calls safe `CudaBackend` methods. The bridge
validates lengths, alignment, address overflow, layouts, and non-overlap,
contains Rust panics behind a versioned status ABI, and keeps a thread-local
detailed error. It does not copy, synchronize, free, or destroy framework
resources. The paged-decode bridge may request an exact caller-owned split-K
workspace, which PyTorch allocates before launch.

The raw CUDA ABI exists only below safe Rust as an internal kernel-launch
layer. Framework code cannot bypass Rust validation, and no parallel ctypes,
ATen, unchecked, or direct-CUDA adapter is retained.

## Add+RMSNorm Contract

The fused normalization path follows the double in-place convention used by
LLM inference engines:

1. materialize `residual = input + residual` in the storage dtype;
2. compute the RMS statistic from that materialized residual;
3. overwrite `input = RMSNorm(residual, weight, epsilon)`.

The safe Rust entrypoint requires separate exclusive borrows for `input` and
`residual`, plus a shared borrow for `weight`. Owned `DeviceBuffer` values are
not cloneable, so ordinary safe callers cannot accidentally alias these three
allocations. Raw C ABI callers must obey the documented non-overlap rule.

FP16 and BF16 fused kernels use aligned 128-bit, eight-element packs when the
row shape and all pointers permit it. An aligned two-element path handles other
even sizes, while odd or unaligned shapes select the scalar implementation.
Launches are asynchronous on the caller-selected backend stream.

Because repeated in-place calls change nonzero operands, the standalone
benchmark separates correctness from timing. Correctness uses one nonzero
launch; latency uses zero-valued input/residual buffers, which are a stable
fixed point and execute the same branch-free kernel path. Reset copies are not
included in the kernel latency.

## Optional-Residual RMSNorm+Dynamic-FP8 Contract

The quantized normalization path consumes contiguous F32, FP16, or BF16 input,
a matching one-dimensional weight, and an optional mutable residual with the
same shape and dtype as the input. It writes FP8 E4M3FN values with the same
logical shape plus one F32 dequantization scale per flattened row. The scale is
`max(absmax / 448, 1 / (448 * 512))`, so zero rows remain valid.

Without residual, all three passes read `input`. With residual, each pass
independently forms the raw F32 sum `input + residual`; only the final
normalization/quantization pass stores that sum rounded to the input dtype back
to residual. FP16 and BF16 then round both the normalized intermediate and
weighted value at the input scalar boundary before FP8 conversion. This exact
ordering matches the vLLM dynamic per-token fusion and is part of Loom's public
compatibility contract.

The boxed PyTorch schema is intentionally identical to vLLM's fusion target:

```text
rms_norm_dynamic_per_token_fp8(
    Tensor(a!) result,
    Tensor input,
    Tensor weight,
    Tensor(b!) scale,
    float epsilon,
    Tensor? scale_ub=None,
    Tensor(c!)? residual=None
) -> ()
```

Loom rejects a non-null `scale_ub`; both registered vLLM FP8 fusion keys pass
`None`. Result, scale, and optional residual are caller-owned, pairwise
disjoint mutable buffers. The CUDA kernel uses three passes—RMS reduction,
weighted absmax reduction, then residual store plus quantization—and follows
the caller's stream without synchronizing. The convenience Python API
allocates result and scale once per call; engine and benchmark paths use the
out variant.

## Optional-Residual RMSNorm+Dynamic-INT8 Contract

The INT8 path keeps the same contiguous F32, FP16, or BF16 input, matching
one-dimensional weight, and optional mutable residual contract. It produces
signed INT8 values plus one F32 dequantization scale per flattened row. The
scale is `absmax / 127`; an all-zero row writes scale zero and INT8 zeros.

With residual, the raw F32 `input + residual` sum drives the RMS statistic,
while the residual output stores that sum rounded to the input dtype. The
normalized value is rounded to the input/weight dtype before multiplication,
the weighted result is rounded again, and quantization uses
round-to-nearest-even with signed saturation. These rounding points match the
vLLM native W8A8 IR boundary and are part of the public contract.

The boxed PyTorch schema is:

```text
rms_norm_dynamic_per_token_int8(
    Tensor(a!) result,
    Tensor input,
    Tensor weight,
    Tensor(b!) scale,
    float epsilon,
    Tensor(c!)? residual=None
) -> ()
```

Result, scale, and optional residual are caller-owned, pairwise-disjoint
mutable buffers. One CUDA kernel performs the RMS, weighted-absmax, and
quantization passes; aligned shapes use four-element input/output packs and
size the block from the pack count. The vLLM adapter adds plain and fused-add
patterns to its existing normalization-quantization compiler pass while
leaving Cutlass scaled GEMM unchanged.

This is a source-ABI10, explicit-opt-in candidate rather than a qualified
default path. H20 proves the operator and real W8A8 invocation, but held-out
one-step output quality is not exact, dual-order engine latency crosses parity,
and no ABI10 matrix wheel is qualified. The
[admission result](../results/h20-vllm-int8-quant-admission-20260729.json)
records those limits.

## SiLU-And-Mul Contract

The SwiGLU activation consumes a contiguous tensor whose final dimension is
`2 * width`. The first half is the gate and the second half is the up branch:

```text
output[..., index] = silu(input[..., index]) * input[..., width + index]
```

F32, FP16, and BF16 are supported. Low-precision compatibility includes the
storage-dtype rounding point used by vLLM: the SiLU activation is rounded to
the input dtype before multiplication, then the product is rounded into the
output dtype. The output is separately allocated, has the same prefix shape,
and has final dimension `width`; input/output overlap is forbidden.

Aligned rows use 16-byte packs (four F32 or eight FP16/BF16 elements), while
odd widths and unaligned pointers use a scalar path. Both safe Rust and
PyTorch launch asynchronously on the caller's current stream.

The vLLM out-of-tree layer replacement is explicitly opt-in because its graph
latency is currently at parity with vLLM's native CUDA operator. Compatibility
and engine integration are useful coverage, but do not justify silently
changing an installed engine. The next performance-motivated boundary is
SiLU-and-Mul fused with dynamic output quantization.

## SiLU-And-Mul+Dynamic-Block-FP8 Contract

The fused quantized path accepts contiguous FP16 or BF16 input with the same
split-half `[rows, 2 * width]` layout. It produces FP8 E4M3FN output with shape
`[rows, width]` and one F32 dequantization scale for every 64 or 128 adjacent
output elements. `width` must be divisible by the selected group size.

Unlike the standalone compatibility operator, this fusion does not materialize
or round a FP16/BF16 SiLU intermediate. Gate activation, multiplication by the
up branch, group absmax, and division by scale use F32 before the final FP8
conversion. Each scale is:

```text
max(min(absmax / 448, optional_scale_upper_bound), 1 / (448 * 512))
```

The public Rust and Python APIs emit contiguous row-major scales with logical
shape `[rows, width / group_size]`. The vLLM compatibility operator additionally
accepts the same logical shape backed by group-major strides and its optional
same-device F32 scale upper bound. Output, scales, and input storage must not
overlap; all launches use the caller's current CUDA stream.

The CUDA mapping assigns one thread block to each row/group pair and holds one
fused value per thread in a register across the absmax reduction. This removes
the temporary low-precision activation tensor and the second kernel launch of
the composed path. Because the composed path rounds that temporary tensor, it
is a useful performance comparison but not an exact semantic baseline; vLLM's
own fused per-block operator is the compatibility baseline.

## Fused Logits Preprocessing Contract

`logits_preprocess_` mutates rank-2 F32 logits with unit vocabulary stride and
an explicit, possibly padded row stride. It applies four stages in one fixed
order: a dense bool or uint8 blocked-token mask, unique sparse additive bias,
sparse token suppression to negative infinity, then division by one F32
temperature per row. Temperatures below `1e-5` use divisor one so greedy and
random rows can share one launch. Bias and suppression metadata are optional
only as complete groups; partial groups are rejected before submission.

The handwritten CUDA path assigns 256 threads and four vocabulary partitions
to each row. It scans every token once and consults the sparse metadata without
materializing another vocabulary-sized tensor. Safe Rust owns physical-span,
aliasing, index, uniqueness, and optional-group validation. The checked bridge
and boxed PyTorch mutation schema preserve the caller's current stream,
padded rows, `torch.compile`, FakeTensor/opcheck, and CUDA Graph replay.

The vLLM 0.24/0.25 registration is deliberately narrower than the public
operator. It admits only mixed greedy/random sampler batches with known masks,
biases, min-token or active bad-word suppression, and no penalties or thinking
state. Min-token and active bad-word suppression do not share an admitted call;
all other requests retain vLLM's original preprocessing. On H20 the fused pass
is exact and `3.26–7.30x` faster for
1–32 rows at a 151,936-token vocabulary. Order-reversed Qwen2.5-0.5B runs
preserve every token and show order-stable `1.010–1.084x` TPOT ratios, while
batch latency crosses parity at batch 32; the evidence therefore does not
claim a stable model-level batch-latency win.

## Sampling And Selected-Logprob Contracts

The decode-tail operator consumes finite rank-2 F32, FP16, or BF16 logits with
a unit vocabulary stride and an explicit, possibly padded row stride. For each
row it returns the lowest token index attaining the maximum, that token's F32
raw log-softmax value, and an `int64` sampled-token rank. The rank deliberately
matches vLLM 0.24/0.25: it counts values greater than or equal to the selected
value, so tied maximum logits produce a rank greater than one.

One CUDA block performs first-index argmax, online logsumexp, and maximum-tie
counting in the same vocabulary pass. This avoids materializing the full F32
logprob tensor and replaces vLLM's separate log-softmax, argmax, gather, and
rank work. Launches follow the caller's current stream; token IDs, logprobs,
and ranks are separately allocated outputs.

The vLLM adapter is intentionally narrower than the CUDA primitive. It only
intercepts vLLM 0.24/0.25 requests where every row is greedy,
`max_num_logprobs` is
zero, raw logprobs are requested, and masks, penalties, bad words, thinking
state, and argmax-changing processors are inactive. Other requests execute
vLLM's original sampler unchanged.

The complementary `selected_token_logprobs` contract accepts one caller-owned
int64 token ID per row and returns only that token's F32 raw logprob plus its
tie-aware int64 rank. One CUDA block loads the selected raw logit, computes an
online logsumexp over the row, and counts logits greater than or equal to the
selected value. It never materializes `[rows, vocab_size]` F32 logprobs.

Its vLLM 0.24/0.25 adapter deliberately does not own sampling policy. vLLM still
converts logits to F32, applies masks/processors/penalties and temperature,
runs greedy or random top-k/top-p selection, and consumes RNG in its original
order. Loom runs afterward against the preserved BF16/FP16 raw logits only for
`raw_logprobs` requests with `max_num_logprobs == 0`; all-greedy batches retain
the narrower fused argmax path. F32 logits, top-k logprob lists, specific-token
lists, processed-logprob modes, and version-mismatched vLLM builds fall back
from this registration.

The direct `topk_sampled_logprobs` contract returns the sampled token followed
by up to 32 top tokens, one shared F32 normalization, and the sampled-token
rank. A two-stage handwritten CUDA reduction writes partition-local
logsumexp/top-k states into caller-owned byte workspace and merges them in
descending-logit, ascending-token-ID order. This deterministic tie rule is part
of Loom's public operator contract.

vLLM exposes `torch.topk`'s otherwise unspecified tie order through returned
token ranks, so its exact adapter deliberately keeps that engine operation. It
reuses vLLM's mandatory F32 sampling logits for the small top-k list and calls
Loom's selected-token reduction on preserved raw logits to recover the shared
normalizer and rank without a full-vocabulary raw-logprob tensor. Thus the
direct fused operator and the vLLM adapter share normalization semantics but
do not pretend to share tie-order policy.

The separate `top_k_filter_` contract mutates rank-2 F32, FP16, or BF16 logits
in place from one contiguous int32 `top_k` value per row. Values strictly below
the kth-largest threshold become negative infinity; every value equal to that
threshold remains finite. This tie-preserving rule matches vLLM's PyTorch
full-sort path and is intentionally distinct from positionally trimming to
exactly `k` entries.

CUDA first radix-sorts independent 4,096-value vocabulary partitions. A
device-only selector then binary-searches the ordered float-key domain and
counts each midpoint with parallel binary searches over those sorted
partitions. The same algorithm handles every `top_k` in `[1, vocab_size]`;
there is no `top_k > 256` fallback and no device-to-host decision. Safe Rust
and the checked bridge require explicit uint32 workspace. The PyTorch operator
keeps its two-tensor mutation schema and owns that temporary internally on the
caller's current stream, including CUDA Graph capture.

The opt-in vLLM 0.24/0.25 registration replaces only top-k-only filtering for
one through seven rows, where vLLM otherwise performs a full vocabulary sort.
Top-p stays native. Eight or more rows stay on vLLM's Qrita Triton path because
that implementation exposes a different duplicate-threshold rule by selecting
exactly `k` positions. The H20 operator gate uses that exact registered
boundary rather than comparing unrelated semantics.

## Fused Top-P Renormalization Contract

`top_p_renorm_` accepts rank-2 F32, FP16, or BF16 logits and one contiguous
F32 `top_p` value in `(0, 1]` per row. It mutates every token outside the
nucleus to negative infinity and returns a new contiguous F32 probability
matrix already renormalized over the retained prefix. Equal logits are ordered
by descending token ID, making the discrete boundary deterministic. Existing
negative infinity is valid; NaN, positive infinity, and an all-masked row are
outside the contract.

CUDA radix-sorts 4,096-token partitions by a composite ordered-float/token key.
A device-only 64-bit threshold search sums the low-probability tail in the same
direction as vLLM's small-batch PyTorch implementation, then one final pass
filters logits and emits probabilities. There is one public algorithm for
every valid shape and `top_p`; no host readback, size-specific implementation,
or separate filter/softmax API exists. Safe Rust owns validation and requires
an eight-byte-aligned caller workspace. PyTorch owns that temporary below the
public boundary and submits every launch to its current stream.

The explicit vLLM 0.24/0.25 registration is narrower than the operator. It
admits top-p-only F32 sampling logits for rows 2–7 and vocabulary at least
32,768, where H20 measurements beat the native full sort plus softmax. Row one,
smaller vocabularies, joint top-k/top-p, non-F32 logits, and eight or more rows
remain native. vLLM still consumes the original per-request generators and
performs random selection. Because each implementation accumulates a long F32
tail in a different parallel order, the cutoff can differ by one token when
the threshold lands within rounding error; the qualified probability contract
uses a per-row L1 tolerance of `1e-4` instead of claiming bitwise mask identity.

## Deterministic Categorical-Sampling Contract

`categorical_sample` consumes normalized contiguous F32 probabilities and
caller-owned contiguous int64 `[rows, 2]` `(seed, counter)` state. One
successful call emits one int64 token per row and advances every counter once.
Canonical Philox4x32-10 and a fixed 1,024-logical-lane F32 CDF tree define the
same stream in the Rust oracle and handwritten CUDA. There is one kernel, no
probability-shaped noise tensor, no implicit generator, and no seedless
variant.

The direct ABI8 Rust/CUDA/PyTorch, persistent vLLM request-state, real-engine,
and repository-free wheel gates are complete. The engine-lifetime opt-in
requires an explicit seed on every random request and rejects speculative
engines; it never changes an in-flight request's RNG stream based on batch
size. The complete numerical, state-lifecycle, and evidence boundary is
documented in [counter-based sampling](counter-based-sampling.md).

## Greedy Speculative-Verify Contract

The deterministic speculative boundary consumes flattened int32 draft IDs,
matching int64 target argmax IDs, one int32 bonus ID per request, and inclusive
int32 cumulative draft lengths. It emits a `-1`-padded int32
`[requests, max_draft_tokens + 1]` matrix plus accepted and emitted lengths.
Each row contains the accepted draft prefix followed by the first target
mismatch, or the bonus token after full acceptance.

The public Python call uses one combined caller-owned allocation for all three
outputs. The bridge validates disjoint physical spans and launches one warp per
request on the current stream; no host synchronization or hidden workspace is
introduced. The explicit vLLM 0.24/0.25 registration replaces only the
all-greedy rejection branch. Stochastic residual sampling, RNG, tree masks,
KV-cache policy, attention, and GEMM remain engine/vendor-owned.

The complete contract and evidence boundary are documented in
[greedy speculative verification](greedy-speculative-verify.md).

## Operator Contract

Every operator contract must make these properties explicit:

- input/output dtype and accumulation dtype;
- logical shape, physical layout, strides, and alignment;
- aliasing and in-place mutation rules;
- stream and synchronization semantics;
- temporary workspace ownership and lifetime;
- supported shape range and deterministic fallback behavior.

## Admission Gates

An operator joins the supported surface only after closing six independent
gates:

1. validated contract and invalid-input tests;
2. deterministic CPU or high-precision oracle;
3. accelerator correctness over edge and representative shapes;
4. warmed repeated measurements against a named baseline;
5. invocation from a real inference-engine execution path;
6. TTFT, TPOT, throughput, memory, or goodput benefit on the motivating workload.

Kernel latency closes gate 4 only when the baseline and measurement protocol
are equivalent. It does not close engine integration or end-to-end value.

## Implementation Policy

- Handwrite memory-bound and fusion-sensitive kernels.
- Never implement dense, quantized, sparse, or grouped GEMM; use the
  engine-selected vendor backend and own only measured memory-bound work around
  it.
- Keep tuning decisions keyed by device, dtype, layout, and shape.
- Preserve one stable Rust contract across CUDA implementations.
- Add another backend only after a real consumer and benchmark justify it.

The complete product surface is tracked in the
[LLM inference operator catalog](../operator-catalog.md); items in that catalog
still have to pass these gates one by one.
