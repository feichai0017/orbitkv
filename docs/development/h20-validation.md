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

Validate split-K with a non-divisible KV range and the tuned H20 configurations:

```text
(kv_len, query_heads, kv_heads, partitions)
  = (7,8,1,3), (33,8,1,6), (127,16,4,10), (4096,32,4,64)
```

Each partial state is F32
`[max_score_log2, normalizer, weighted_value[128]]`. The merge kernel must
produce the same final BF16 output and F32 log2-LSE contract. Require one
completion over both kernel submissions. Reject a one-command queue and a
workspace that is one F32 element short before either kernel reaches CUDA.

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

The current split-K source projection passes the same H20 numerical limits for
the declared partition choices. Its tuned KV-length-4096 case has
`9.536743164e-7` output maximum absolute error and `3.814697266e-6` log2-LSE
maximum absolute error. A new immutable correctness and sanitizer record
remains open.

The [matched eager-provider record](../results/h20-flashinfer-v0.6.16.post1-eager-performance-20260805.json)
compares the pre-split-K Loom source against FlashInfer `v0.6.16.post1`.
FlashInfer has 6.29x lower median latency at GQA KV length 127 and 80.08x lower
median latency at KV length 4096 under the declared eager metric. Loom has
1.33x lower median latency for the fixed M=1 cuBLASLt GEMM case. The record
retains both provider orders and all 1,400 raw samples.

Fixed Rust-kernel argument packs, matched RMSNorm, isolated kernel timings, and
Graph performance gates remain open.
