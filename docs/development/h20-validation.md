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
non-default stream. It also covers two chained launches, queue reuse, and one
partial F32 scope. CUDA Graph replay remains a separate gate.

### Open checks

The pinned compiler fails the debug-profile `DisjointSlice` MIR check but passes
the release device test. Keep this boundary until the pinned revision changes.

Run Compute Sanitizer before full provider admission, or record that the host
does not provide it.

## Performance

Measure `(1,4096)`, `(8,4096)`, `(64,4096)`, and `(16,8192)` with identical
preallocated buffers and streams. Use CUDA events around launches only.

Run both provider orders. Record warm-up count, samples, launches per sample,
raw timings, median, dispersion, clocks, and power policy. The timed region
must not allocate, copy, compile, tune, or synchronize the host.

Compare against the current named provider on the same revision and device.
Do not reuse timings from a deleted implementation.

## Current state

The permanent F32, FP16, and BF16 providers pass their declared H20 correctness
gates. The maximum F32 absolute error is `4.768371582e-7`.

The F32 two-launch chain reaches `7.152557373e-7` maximum absolute error. The
FP16 chain reaches two ULPs, and the BF16 chain reaches one ULP. Generated PTX
uses scalar and packed instructions, targets `sm_90`, and assembles with CUDA
13.1. Compute Sanitizer, CUDA Graph, fixed argument packs, and performance gates
remain open.
