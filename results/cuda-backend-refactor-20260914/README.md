# CUDA backend refactor qualification

The refactored backend passes H20 provider/kernel checks and the fixed
Qwen3.8-27B-FP8 decoder workload. All **152** request/step logits comparisons pass,
with maximum absolute error **0.8125** under the unchanged **1.0** gate.
All five processes drain state successfully and use the same selected artifact.

This is an unreleased architecture change. Crate versions remain `0.1.0` and
native provider commits are unchanged. There is no serving-throughput or
cross-engine performance claim.

## Changes under qualification

- Rust provider contracts, egglog rules and CUDA sources have separate owners.
  Thirty CUDA sources and thirty-nine egglog rule assets were extracted while
  preserving their original literal contents; explicit interpolation is checked
  by the Rust compiler.
- One provider lock records exact source and per-provider CUTLASS commits.
  Source caches bind the complete source contract, including dependency-only
  pin changes. Native builds share cache keys, staging, publication, deadlines
  and bounded diagnostics. Model compilation never downloads sources.
- Runtime device facts join executor state constraints before saturation.
  Missing, unsupported and conflicting targets have regression coverage.
  Native JIT uses the execution context. DeepGEMM retains its required
  `compute_90a` to `sm_90a` compilation contract, and preparation errors preserve
  the full native compiler diagnostic.
- The full/Lite compatibility surface, unused helpers/dependencies and GPU-name
  peak-performance table were removed. Decoder identity binds the provider lock;
  old decoder artifacts require fresh search without a serialization-version bump.

See [backend architecture](../../docs/cuda-backend.md) and
[joint compilation](../../docs/joint-compilation.md). Outer KV layout/placement
search and persistent megakernel generation remain planned.

## Checks

- 503 default host tests, 11 optional frontend tests and 5 doctests passed.
- 148 backend test executions passed, including H20 attention/DeepGEMM references,
  shared FP8 schedule replay, cuBLASLt capture identity/recapture, generated-kernel
  contracts and module-artifact validation. One cuBLASLt host identity case is
  intentionally repeated by its dedicated suite (147 unique tests).
- Rust 1.98 formatting and Clippy passed for host and CUDA/executor/server targets.
- 46 Python tool tests, source-layout verification, website checks and build passed.

The numerical/model executables were built with Rust 1.97.1. Exact binary and
source-file digests are in [source.json](source.json), provider provenance is in
[providers.json](providers.json), and commands/counts are in [checks.json](checks.json).
The CUDA test binary is identified independently from the model test binary.

## Model execution

One H20, official local FP8 checkpoint, existing independent reference, batch
capacity 8, graph residency 2 and unchanged `benchmarks/hotspot-search.json`:
seven workload representatives, eight measured graphs and the existing hotspot
budget. B1 performs fresh search; B1 replay/profile and B8 replay/profile load the
same artifact. Provider/CUDA caches were retained, so fresh search does not mean
an empty machine cache.

| Batch | Phase | Compile/load seconds | Comparisons | Median decode-2…7 wall ms |
| --- | --- | ---: | ---: | ---: |
| B1 | cold-search | 943.102 | 8 | 25.040 |
| B1 | strict-replay | 17.507 | 8 | 24.721 |
| B1 | profile | 17.515 | 8 | 60.691 |
| B8 | strict-replay | 17.485 | 64 | 37.063 |
| B8 | profile | 17.428 | 64 | 76.988 |

These are diagnostic wall times, including the logits path and synchronization.
The profile phases add CUDA Graph instrumentation. They are not comparable to a
serving TPOT benchmark. The cold-search cost remains substantial; this change
establishes maintainable compilation boundaries rather than a speedup result.

Artifact SHA-256: `93c9fe1f3fd55189fe88693e2d04c6bdabe1b1a0c2bd247a5f6aa086a3a0bf78`.
[model.json](model.json) retains every comparison and observed phase timing.
Raw logs, traces, binaries and the artifact are under
`.qualification/cuda-refactor-20260914/`; failed development runs remain separate
from these final qualified binaries. Historical evidence directories are unchanged.
