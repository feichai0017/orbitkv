# Integrated compiler workspace on H20

OrbitKV now owns its model compiler, operation contracts, CUDA backend and
tracing crates in one Cargo workspace. The Luminal gitlink is removed. The
import preserves the original MIT/Apache-2.0 licenses and records its source
revision in [compiler maintenance](../../compiler-maintenance.md).

## Scope

- `luminal` → `orbitkv-compiler`
- `luminal_nn` → `orbitkv-ops`
- `luminal_cuda_lite` → `orbitkv-cuda`
- `luminal_tracing` → `orbitkv-tracing`

Internal dependencies use root workspace bindings and one lockfile. All four
crates follow the product's module/test layout: 27 inline suites and 24 test
source files moved under their owning `tests/` trees. Two unused operation-test
dependencies were removed. Inherited large modules have explicit non-growing
size limits pending responsibility-based extraction.

The provider ABI names, environment variables, trace namespace and default
caches use OrbitKV names. Decoder schema 11 rejects earlier decoder artifacts.
The migration does not add graph rewrites, attention algorithms, layouts or
megakernels. The state manager remains independent of the compiler and CUDA.

## Validation

- Host workspace: **502 tests passed**, three intentionally ignored.
- Rust documentation: **5 passed**, one ignored.
- Frontend feature: **11 passed**, one hardware-gated test ignored.
- Python tools: **46 passed**.
- H20 search: **2 passed**, covering measured GEMM alternatives and persistent state.
- H20 state arenas: **2 passed**, covering stable addresses, event order and ragged state kernels.
- CUDA module artifacts: **7 passed**, including strict replay without NVRTC and rejection of missing cached modules.
- Formatting, host/executor/CUDA Clippy, server compilation, source-layout checks, and website checks/build pass.

The Qwen3.8-27B-FP8 model qualification uses one H20 with 97,871 MiB reported
memory. Nine processes pass **296 full-vocabulary reference comparisons** and
all state drains. Maximum absolute logit error is **0.6875**, within the
unchanged 1.0 gate. Model weights, reference logits, sampling semantics and
numerical tolerances are unchanged. Every replay preserves the same artifact.

Cold search ran with binary
`80782e9b420330040440d9ea89398e9651c116eddddf98d56d18dd97ed420a72`.
Subsequent documentation lint fixes and removal of unused test dependencies
changed the rebuilt binary hash. The final binary,
`af27b48e4ab5a3860d08999f639d88dcf2c9c080ec2ac88d49edcfc76ea4d8d2`,
therefore received its own B1/B8 strict-replay and profiling qualification:
**144 of the 296 comparisons use this final binary**. The source stayed fixed
through each qualification. [Source identities](source.json) record all 391
final build-input files and the ten cold-to-final file differences.

Fresh compilation took about 1,010 seconds in this run; strict replay's
`compile_or_load` intervals were approximately 18 seconds. Provider/cache
identities changed and the search snapshots are not a paired baseline. These
are migration diagnostics, not a serving-performance comparison or a speedup
claim. [Model evidence](model.json) retains per-process identities and checks;
[validation checks](checks.json) record the remaining gates.
