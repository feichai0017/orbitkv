# H20 validation

NVIDIA H20 is the first Loom Infer device target. Permanent providers live in
`crates/loom-infer-cuda`; hardware runners live in
`crates/loom-infer-validation`.

## Repository identity

The local checkout and the H20 validation checkout must have the same source
manifest and lockfile. Do not test a remote-only patch or copy generated
artifacts into the source tree.

Record these values before the run:

- source-manifest and `Cargo.lock` hashes.
- Rust nightly and `cuda-oxide` revision.
- CUDA toolkit, driver, LLVM, and Clang versions.
- GPU model and compute capability.
- PTX or cubin target and content hash.

## Device compiler

Run the permanent crate from the H20 validation checkout:

```bash
ssh <h20-validation-host>
cd <loom-infer-checkout>
make cuda-doctor
make cuda-check
make cuda-test
make h20
```

The Make targets are the canonical local entry points. Evidence records still
list the expanded commands so each compiler and runner invocation remains
auditable.

The run writes `loom_infer_cuda.ptx` at the workspace root with the pinned
cargo-oxide revision. The PTX must target `sm_90`. Assemble it with the recorded
CUDA toolkit before accepting the artifact.

## RMSNorm correctness

Validate these F32 shapes against the `loom-infer` CPU reference:

```text
(rows, hidden) = (1, 1), (3, 127), (8, 4096), (16, 8192)
```

All outputs must be finite. Maximum absolute error must not exceed `5e-5` for
F32, `4e-3` for FP16, or `4e-2` for BF16. Low-precision results must also stay
within two storage-format ULPs.

Validate FP16 and BF16 on these shapes:

```text
scalar:  (1, 1), (3, 127), (3, 4097)
packed:  (1, 2), (32, 256), (8, 4096), (16, 8192), (1, 11008)
```

The packed path requires four-byte-aligned input, weight, and output buffers.
Both paths use F32 arithmetic and nearest-even output conversion.

The current gate covers odd and even hidden sizes, all three short-buffer
positions, signed zero, typed heterogeneous bindings, and a caller-owned
non-default stream. It also covers two chained commands, queue reuse, and one
partial F32 scope. CUDA Graph replay remains a separate gate.

## BF16 GEMM correctness

Validate the fixed cuBLASLt contract:

```text
D[M,N] = A[M,K] * W[N,K]^T
A, W, D: contiguous row-major BF16
accumulation: F32
```

The gate covers `(M,N,K) = (1,4096,4096)` across two reusable scopes and a
transpose-sensitive `(2,3,4)` case. It rejects short A, W, and D buffers and a
second command when queue capacity is one. It also chains BF16 RMSNorm
`(1,4096)` into GEMM under one completion with no intermediate host wait.

Record the selected algorithm's actual workspace requirement. Test a short
workspace only when that requirement is nonzero. The current H20 selection
requires zero workspace, so that rejection case is not applicable.

## BF16 single-decode correctness

Validate the first fixed attention contract:

```text
Q, O: [query_heads, 128] BF16
K, V: [kv_len, kv_heads, 128] BF16 NHD
LSE:   [query_heads] F32 log2-domain
scale: 1 / sqrt(128)
```

The gate covers MHA `(1,8,8)`, MQA `(33,8,1)`, GQA `(127,16,4)` and
`(4096,32,4)`, plus one large-logit stability case. Tuples use
`(kv_len, query_heads, kv_heads)`.

Compare against the `loom-infer` CPU reference. BF16 output maximum absolute
error must not exceed `0.015625`. Log2-LSE maximum absolute error must not
exceed `0.01`. Initialize output buffers with NaN sentinels.

Reject Q, K, V, O, or LSE that is one element short. Reject duplicate operand
bindings before CUDA submission. Run memcheck, racecheck, synccheck, and
initcheck over the complete runner.

## BF16 paged batch-decode correctness

Validate the fixed FlashInfer-compatible paged contract:

```text
Q, O:          [batch_size, query_heads, 128] BF16
K/V pages:     [max_num_pages, 16, kv_heads, 128] BF16 NHD
page_indptr:   [batch_size + 1] I32
page_indices:  [page_indptr[batch_size]] I32
last_page_len: [batch_size] I32
LSE:           [batch_size, query_heads] F32 log2-domain
```

Cover MHA, MQA, and GQA with mixed request lengths, partial and full tail
pages, non-sequential physical pages, and physical-page reuse. Compare against
the paged CPU reference with the same `0.015625` BF16 output and `0.01`
log2-LSE absolute limits.

Reject short fixed metadata spans before submission. Since page-table contents
remain device-resident, exercise an out-of-range physical page under Compute
Sanitizer and require the invalid request to preserve NaN output sentinels
without preventing a valid request in the same batch from completing. This
guard is a memory-safety behavior, not asynchronous metadata error reporting.

Validate split-K with a non-divisible KV range and the tuned H20 configurations:

```text
(kv_len, query_heads, kv_heads, partitions)
  = (7,8,1,3), (33,8,1,12), (127,16,4,16), (4096,32,4,64)
```

Each partial state is F32
`[max_score_log2, normalizer, weighted_value[128]]`. The merge kernel must
produce the same final BF16 output and F32 log2-LSE contract. Require one
completion over both kernel submissions. Reject a one-command queue and a
workspace that is one F32 element short before either kernel reaches CUDA.

## BF16 ragged prefill correctness

Validate the first fixed ragged causal contract:

```text
Q, O:        [nnz_qo, query_heads, 128] BF16
K, V:        [nnz_kv, kv_heads, 128] BF16 NHD
qo_indptr:   [batch_size + 1] I32
kv_indptr:   [batch_size + 1] I32
LSE:         [nnz_qo, query_heads] F32 log2-domain
causal mask: kv_index <= kv_len - qo_len + query_index
```

Cover MHA, MQA, and GQA with equal and mixed query/KV lengths. The first
contract requires every request to satisfy `1 <= qo_len <= kv_len`. Compare
against the ragged CPU reference with the same `0.015625` BF16 output and
`0.01` log2-LSE absolute limits.

Long group-size-four GQA uses a caller-owned F32 workspace with shape
`[nnz_qo, query_heads, 8, 130]`. Each partial state is
`[max_score_log2, normalizer, weighted_value[128]]`. Require one completion
over the fused tensor-core partial kernel and the F32 merge kernel. Reject a
missing workspace before either kernel reaches CUDA.

Reject short fixed metadata spans before submission. Since indptr contents
remain device-resident, exercise invalid endpoints under Compute Sanitizer and
require output NaN sentinels to remain unchanged. This guard is a memory-safety
behavior, not asynchronous metadata error reporting. Run memcheck, racecheck,
synccheck, and initcheck over the complete runner.

## Fixed-address CUDA Graph

The BF16 GEMM runner also captures this exact chain:

```text
BF16 RMSNorm (1,4096)
  -> fixed BF16 cuBLASLt GEMM (1,4096,4096)
```

The graph gate prepares buffers on the ordinary validation stream, establishes
input readiness, then consumes a one-shot `GraphQueue` to capture on its private
stream. It requires two accepted replays with the same device addresses and one
retained external completion event recorded after each replay.

RMSNorm and GEMM must execute without a host wait between the two graph nodes.
The first replay waits before the second starts. The final output must match the
CPU oracle bit for bit.

One replay must settle through `wait()` and one through completion Drop.
Capture and replay must retain the kernel module, cuBLASLt plan, workspace, and
all bound allocations. The runner drops its external kernel-plan and read-buffer
owners after capture and before graph instantiation.

Run Compute Sanitizer over capture, both replays, and graph destruction before
changing the graph state from open. A successful compiler check alone is not a
device or graph result.

### Open checks

The pinned compiler fails the debug-profile `DisjointSlice` MIR check but passes
the release device test. Keep this boundary until the pinned revision changes.

Run Compute Sanitizer before full provider admission, or record that the host
does not provide it.

## RMSNorm performance

Measure `(1,4096)`, `(8,4096)`, `(64,4096)`, and `(16,8192)` with identical
preallocated buffers and streams. Use CUDA events around launches only.

Run both provider orders. Record warm-up count, samples, launches per sample,
raw timings, median, dispersion, clocks, and power policy. The timed region
must not allocate, copy, compile, tune, or synchronize the host.

Compare against the current named provider on the same revision and device.
Do not reuse timings from a deleted implementation.

The current FlashInfer `v0.6.16.post1` RMSNorm source fails to compile
unmodified on the declared CUDA 13.1 host, so no matched RMSNorm performance
gate has passed. Loom-only timings in the eager record are diagnostic, not a
provider comparison.

## GEMM performance

Benchmark the fixed BF16 contract against the same cuBLASLt contract through a
named baseline. Keep algorithm, layouts, workspace, stream, and timing region
identical.

The correctness runner is not a benchmark. The first matched eager-provider
gate covers only `(M,N,K) = (1,4096,4096)` with fixed tactic 0, preallocated
buffers, CUDA events, and both provider orders. It does not establish isolated
kernel, Graph, engine, serving, or other-shape performance.

## Single-decode performance

The first matched eager-provider gate covers BF16 NHD D128 MHA, MQA, and GQA
with identical operand bit patterns. It records 200 warm-up launches, 100
launches per sample, 50 samples per provider order, and both provider orders.

The CUDA event interval contains 100 sequential eager provider calls and is
divided by 100. Host-submission gaps can leave the GPU idle inside that
interval, so the result is not isolated kernel duration or CUDA Graph
performance. Preserve that measurement name and boundary when comparing future
runs.

## Ragged prefill performance

Match BF16 NHD D128 query, key, value, `qo_indptr`, and `kv_indptr` bits across
Loom and the pinned FlashInfer FA2 ragged wrapper. Use bottom-right causal
alignment, caller-owned output and LSE, preplanned providers, CUDA events, 200
warm-up calls, 100 calls per sample, 50 samples per provider order, and both
provider orders.

Keep short MHA, mixed append-style MQA, and long GQA as separate cases. Record
the Loom plan algorithm and every fixture digest. Exclude a provider ranking
when either provider's order median changes by more than five percent. Compare
an optimization against an immutable direct Loom record rather than a
working-tree timing.

The measurement remains eager provider latency. It does not establish isolated
kernel, CUDA Graph, engine, or serving performance.

## Current device state

The [shared-command regression](../results/h20-shared-command-regression-20260803.json)
qualifies the current source projection. It reruns RMSNorm, BF16 GEMM, the
fixed-address Graph, and BF16 single decode. The maximum F32 RMSNorm absolute
error is `4.768371582e-7`.

The F32 two-command chain reaches `7.152557373e-7` maximum absolute error. The
FP16 chain reaches two ULPs, and the BF16 chain reaches one ULP. Generated PTX
uses scalar and packed instructions, targets `sm_90`, and assembles with CUDA
13.1. The GEMM correctness cases are BF16 bit-exact against their declared CPU
fixtures. The fixed Graph replays twice and produces a bit-exact final output.

Compute Sanitizer memcheck reports no errors or leaked device allocations for
the RMSNorm and Graph runners. RMSNorm racecheck and synccheck report no errors.
Graph-runner initcheck also reports no errors.

The [single-decode record](../results/h20-bf16-single-decode-correctness-20260803.json)
qualifies the narrow attention slice. Its largest output absolute error is
`7.629394531e-6`; its largest log2-LSE error is `1.907348633e-6`. All four
Compute Sanitizer tools report zero errors.

The [split-K correctness record](../results/h20-bf16-single-decode-split-k-correctness-20260805.json)
passes the same H20 numerical limits for the declared partition choices. Its
tuned KV-length-4096 case has `9.536743164e-7` output maximum absolute error and
`3.814697266e-6` log2-LSE maximum absolute error. All four Compute Sanitizer
tools report zero errors, and both kernels assemble without stack or spills.

The [matched eager-provider record](../results/h20-flashinfer-v0.6.16.post1-eager-performance-20260805.json)
compares the pre-split-K Loom source against FlashInfer `v0.6.16.post1`.
FlashInfer has 6.29x lower median latency at GQA KV length 127 and 80.08x lower
median latency at KV length 4096 under the declared eager metric. Loom has
1.33x lower median latency for the fixed M=1 cuBLASLt GEMM case. The record
retains both provider orders and all 1,400 raw samples.

The [matched split-K record](../results/h20-flashinfer-v0.6.16.post1-split-k-eager-performance-20260805.json)
retains the same semantic fixtures and adds execution metadata. Split-K lowers
Loom median latency by 3.79x at GQA KV length 127 and 26.79x at KV length 4096
relative to the recorded direct baseline. FlashInfer remains 1.69x and 3.00x
lower-latency under the declared eager metric.

The [parallel-merge profiling record](../results/h20-bf16-single-decode-parallel-merge-profiling-20260805.json)
uses Nsight Systems CUPTI activity timing. At KV length 4096, the partial kernel
records `31.104` microseconds and the parallel merge records `5.056`
microseconds, down from `20.192` for the serial merge. Nsight Compute
hardware-counter metrics remain unavailable because the host sets
`RmProfilingAdminOnly=1`.

The [parallel-merge matched record](../results/h20-flashinfer-v0.6.16.post1-parallel-merge-eager-performance-20260805.json)
raises the complete Loom speedup over the direct baseline to 5.39x at GQA KV
length 127 and 38.19x at KV length 4096. FlashInfer remains 1.17x and 2.09x
lower-latency.

The [paged batch-decode record](../results/h20-bf16-paged-batch-decode-correctness-20260806.json)
passes MHA, MQA, and GQA batches with bit-exact BF16 output and maximum
log2-LSE error `4.768371582e-7`. It covers mixed request lengths, page
reordering and reuse, exact metadata spans, and a device-side invalid-page
guard. All four Compute Sanitizer tools report zero errors.

The current [ragged prefill record](../results/h20-bf16-ragged-prefill-tiled-split-k-correctness-20260806.json)
passes direct, eight-warp, sixteen-warp, and tiled eight-partition MHA, MQA,
and GQA batches. Maximum BF16 output error is `4.8828125e-4` and maximum
log2-LSE error is `2.861022949e-6`. It covers equal and mixed query/KV lengths,
bottom-right causal alignment, exact metadata spans, missing tiled workspace,
and a device-side nonmonotonic-indptr guard. All four Compute Sanitizer tools
report zero errors.

The [ragged matched eager record](../results/h20-flashinfer-v0.6.16.post1-ragged-prefill-cp-async-eager-performance-20260806.json)
retains both provider orders and all 600 raw samples. Unrolled 16-byte
`cp.async` K/V staging lowers Loom long-GQA latency to `48.232` microseconds,
`1.148x` below the previous tiled split-K result and `7.729x` below direct.
FlashInfer remains `2.206x` lower-latency on stable long GQA. Short-MHA and
mixed-MQA rankings are excluded because FlashInfer's provider-order deltas are
`10.643%` and `14.097%`.

The [ragged Graph correctness record](../results/h20-bf16-ragged-prefill-cuda-graph-correctness-20260806.json)
captures the tiled partial and merge kernels on one private stream. Two
fixed-address replays preserve the standalone output and log2-LSE digests after
external owner teardown. Memcheck, racecheck, initcheck, and synccheck report
no errors or leaks.

The [matched ragged Graph performance record](../results/h20-flashinfer-v0.6.16.post1-ragged-prefill-graph-performance-20260806.json)
measures one replay and one completion event per CUDA-event sample. Loom and
FlashInfer combined medians are `50.480` and `32.640` microseconds, and their
provider-order deltas are `0.127%` and `0.344%`. Capture, instantiation,
planning, allocation, fixture copies, and correctness reads are excluded. Do
not compare this single-replay metric directly with the eager provider record.

The first [matched paged eager record](../results/h20-flashinfer-v0.6.16.post1-paged-batch-decode-eager-performance-20260806.json)
retains both provider orders and all 600 raw samples. Loom is 4.21x
lower-latency for batch-1 MHA at KV length 1. FlashInfer is 1.62x
lower-latency for the mixed-length batch-3 MQA case. The batch-4 GQA shape is
excluded from stable ranking because FlashInfer's order delta is 52.49%.

The current [token-parallel record](../results/h20-flashinfer-v0.6.16.post1-paged-token-parallel-eager-performance-20260806.json)
uses the same fixtures and protocol. Eight-warp block-local state merge lowers
Loom MQA/GQA eager latency by 3.78x and 3.32x. Loom is now 4.41x
lower-latency for MHA and 2.35x lower-latency for MQA than FlashInfer. GQA
remains excluded because FlashInfer's order delta is 60.62%.

CUPTI records current Loom MHA/MQA/GQA kernel medians of `2.176`, `4.864`,
and `4.928` microseconds. These diagnostics guide optimization but are not a
provider-ordered isolated-kernel gate.

Fixed Rust-kernel argument packs, matched RMSNorm, hardware-counter profiling,
provider-ordered paged kernel timing, and non-ragged Graph performance gates
remain open.

The [standard RoPE correctness record](../results/h20-bf16-rope-pos-ids-correctness-20260806.json)
qualifies BF16 NHD D128 NeoX split-half rotation with explicit I32 positions
through 32,767. The query/key maximum absolute errors are `0.00390625` and
`0.001953125`; memcheck, racecheck, initcheck, and synccheck report no errors
or leaks.

The [matched RoPE eager record](../results/h20-flashinfer-v0.6.16.post1-bf16-rope-pos-ids-eager-performance-20260806.json)
uses the same Q/K bits and positions for both providers. Loom records `3.997`
microseconds and FlashInfer records `5.077` microseconds, making Loom `1.270x`
lower-latency. Both provider-order deltas are below five percent. Output bits
differ because Loom uses full CUDA libdevice math while FlashInfer uses
fast-math intrinsics; both remain within the shared BF16 reference limit.

The [fused RoPE paged KV append correctness record](../results/h20-bf16-rope-paged-kv-append-correctness-20260806.json)
qualifies one BF16 Q/K/V token per request at `request_kv_len - 1`. The full Q
output and K/V page pools are bit-exact with the CPU oracle. Duplicate final
slots and an invalid non-final physical page preserve all output sentinels;
all four Compute Sanitizer tools report no errors or leaks. The fused SM90
kernel uses 23 registers with no stack, spills, barriers, or static shared
memory.

The [matched fused append eager record](../results/h20-flashinfer-v0.6.16.post1-bf16-rope-paged-kv-append-eager-performance-20260806.json)
compares one Loom kernel with FlashInfer's standard RoPE plus paged append
composition. Both processes are pinned to CPU 40 on the GPU-local NUMA node,
with one OMP and MKL thread. Loom and FlashInfer combined medians are `3.989`
and `11.735` microseconds, making Loom `2.942x` lower-latency on the admitted
batch-4 Q16/K4 D128, page-size-16 case. Provider-order deltas are `0.128%` and
`3.159%`. Unrestricted-affinity diagnostic samples are excluded because CPU
migration produced non-admissible eager host-path drift.
