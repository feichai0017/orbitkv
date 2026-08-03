# H20 validation

NVIDIA H20 is the first Loom Infer device target. The permanent device source
lives in `crates/loom-infer-cuda`.

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
cd <loom-infer-checkout>/crates/loom-infer-cuda
cargo oxide doctor
cargo oxide run rms_norm_h20 --bin rms_norm_h20 --features cuda --arch sm_90
cargo oxide run bf16_gemm_h20 --bin bf16_gemm_h20 --features cuda --arch sm_90
cargo oxide test --arch sm_90 -- --workspace --features cuda --release
```

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

## GEMM performance

Benchmark the fixed BF16 contract against the same cuBLASLt contract through a
named baseline. Keep algorithm, layouts, workspace, stream, and timing region
identical.

The correctness runner is not a benchmark. No GEMM performance gate has passed.

## Current device state

The [2026-08-03 record](../results/h20-owned-bindings-cuda-graph-correctness-20260803.json)
qualifies the current owned-binding and fixed-address Graph source projection.
The maximum F32 RMSNorm absolute error is `4.768371582e-7`.

The F32 two-command chain reaches `7.152557373e-7` maximum absolute error. The
FP16 chain reaches two ULPs, and the BF16 chain reaches one ULP. Generated PTX
uses scalar and packed instructions, targets `sm_90`, and assembles with CUDA
13.1. The GEMM correctness cases are BF16 bit-exact against their declared CPU
fixtures. The fixed Graph replays twice and produces a bit-exact final output.

Compute Sanitizer memcheck reports no errors or leaked device allocations for
both runners. RMSNorm racecheck and synccheck report no errors. Graph-runner
initcheck also reports no errors.

Fixed Rust-kernel argument packs and performance gates remain open.
