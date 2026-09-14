# Checkpoint/semantic boundary and inference fork qualification

The fork retains four packages: `luminal`, `luminal_nn`, `luminal_cuda_lite` and
`luminal_tracing`. Sixteen unused application/backend/training packages and their
associated CI jobs/hooks were removed. The dependency inventory is recorded in
[package-closure.json](package-closure.json).

Checkpoint import now resolves named serialization/architecture conventions into
explicit decoder semantics. Attention and block-scaled FP8 linear graph APIs
belong to NN, with CUDA providers supplying egglog implementations. Unlowered
semantic nodes are not executable search candidates. The portable CUDA FP8
oracle lives only under tests. The final executor boundary also rejects invalid
attention scales; test callers resolve the old zero-for-default convention to
`head_dim^-0.5` explicitly. See [the contract](../../docs/checkpoint-import.md).

## Evidence

| Gate | Result |
| --- | --- |
| Root workspace, default features | 247 passed, 2 external tests ignored |
| Engine frontend feature | 11 passed, 1 external test ignored |
| Luminal core | 210 passed, 1 external test ignored |
| NN + tracing, including backend-free semantic tests | 23 passed; 1 tracing doctest ignored |
| Final executor CUDA-feature unit suite | 71 passed, 2 external checkpoint tests ignored |
| Python qualification-tool tests | 34 passed |
| Directed CUDA/provider suites | 35 passed on H20: attention search/reference, FP8 equivalents/shared quantization, graph residency and module artifacts |
| Initial schema 7 search + replay | 96 full-vocabulary row comparisons, all lifecycle drains passed |
| Final build replay at cache capacities 2 and 1 | 64 full-vocabulary row comparisons, max absolute error 0.53125 under the unchanged 1.0 gate; all drains passed |
| Old schema rejection | Real schema 6 decoder artifact rejected before weight loading, on initial and final builds |
| Final HTTP replay | Independent three-token output `&!@`; live-stream cancellation and full token/fixed-state drain passed; artifact unchanged |
| Strict module-image replay | 420 module-image hits, 0 module-image compiles; stage trace completed with no open spans |
| Source/result preservation | All 338 final frozen build inputs and 102 pre-existing result files verified unchanged |

Formatting and Clippy with `-D warnings` passed for the owned workspace, CUDA
executor/server and retained Luminal packages. The layout checker includes the
four-package fork boundary and forbids NN/core dependencies on CUDA. Linux host
Cargo metadata resolves exactly the four retained Luminal packages.

The initial full core run exposed a source lint that still expected the removed
examples directory; its scan now covers retained source roots, with its suite
moved to core `tests/`. The initial full executor run exposed nine uses of the
zero scale convention. Those failure logs are retained; the corrected final
suites above passed. Numerical model definitions and parity tolerance were not
changed to accommodate a failure.

## Builds and artifacts

[build.json](build.json) identifies the initial and final server binaries, model
and CUDA test binaries, source manifests, toolchain and GPU. The final server
SHA-256 is `0ab813896aac352af8b2460caf11d713b34c66c44bc7a465d403188dd313172d`.
[files.json](files.json) records raw evidence and artifact hashes;
[checks.json](checks.json) records commands and gate receipts. Large source
snapshots, model artifacts and raw logs remain under
`.qualification/semantic-boundaries-20260914/`.

The final build adds explicit invalid-scale admission to the initially qualified
boundary. It replays the newly generated schema 7 schedules without modifying
them; valid model graph semantics and generated code remain identical. Both
final model replay configurations and the final HTTP run use the final frozen
binaries. CUDA provider checks precede this executor-only admission adjustment.

## Limits

This is correctness and architecture qualification on one H20 SM90 and the local
`qwen3.8-27b-fp8` directory, whose config declares Qwen3.5 text architecture.
The independent reference remains Transformers 5.12.1, with vocabulary 248320,
four prompt tokens and seven teacher-forced decode steps per sequence. Search
used one candidate per bucket and seed 7. It does not establish optimal search,
a throughput improvement, long-context support, serving-scale concurrency or
production soak readiness. Initial HTTP generation took about 208 seconds to
readiness, including compilation; replay timings include loading/preparation
and are not a matched performance comparison.

The current attention operation still specifies paged KV storage. Logical
Attention/KVView lowering, joint layout competition, additional providers and
megakernel generation remain separate work. Decoder schema 7 requires fresh
artifacts; old schema 6 artifacts have no compatibility conversion.
