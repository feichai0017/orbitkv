# Compiler boundary and search identity validation

The executor now separates tuning policy, feasible bucket planning, profiling
input ownership, and compile/load lifecycle. The DeepGEMM provider separates its
numerical/storage ABI, scratch ownership, tile policy, and JIT. This record
checks that the refactor remains executable and that CUDA search evidence refers
to the program actually stored and replayed.

The frozen compiler-refactor release binary runs Qwen3.8-27B-FP8 on NVIDIA H20 (SM90).
Fresh schedule search, strict artifact replay, and a separate instrumented replay
each pass eight independent-reference logit comparisons and final state drain.
The maximum absolute error is **0.5423088**, below the unchanged **1.0** gate.
All three phases use the same artifact. The reference is the existing
Transformers 5.12.1 model implementation with local DeepGEMM 2.6.1; it is not
independent of every underlying math library.

| Check | Result |
| --- | --- |
| Frozen build inputs | 338 files match their recorded hashes and source at the compiler-refactor checkpoint |
| Evidence references | Independent audit verifies 698 file references |
| Candidate trace | 52 direct outcomes: 50 resource rejections, 2 measured candidates |
| Program identity | Both buckets correlate direct measurement, deployment measurement, accepted validation, selection, and stored artifact |
| Full-model phases | Search, strict replay, and instrumented replay each pass 8 reference steps and drain |
| Focused regressions | FP8 shared preparation/reference, scratch capture lifetime, semantic identity, and strict replay pass on GPU; host checks and Clippy pass |

An earlier run passed the numerical gate but failed the identity audit:
cuBLASLt runtime caches and context stream counts appeared in semantic `Debug`.
The final code excludes those fields, and also excludes FlashInfer planning
scratch. A focused real cuBLASLt matmul regression proves identity stability
through preparation, execution, capture, artifact storage, and strict replay.
Semantic fields remain included. Old cache-dependent fingerprints require fresh
search; they are not treated as compatible by suppressing validation.

`summary.json` is a compact projection of the qualification and focused-check
receipts. `trace-audit.json` preserves the independent audit. `environment.json`
records hardware, model metadata, source, binary, and provider identities.
`SHA256SUMS` covers the promoted files. Large traces, source snapshots, binaries,
and per-phase logs remain in the raw directories referenced by the JSON.

## Final code and test placement

The subsequent layout pass keeps production files in `src/` and puts all owned
crate test source under `tests/`, with private suites in `tests/unit/`. Maintained
CUDA provider and search-trace suites follow the same placement. Test-only module
path declarations preserve private access and test namespaces; the production
API is unchanged. See [code and test layout](../../docs/code-layout.md).

`layout-checks.json` records the final placement checks: workspace, server,
engine/CLI, executor model tests, provider tests, Clippy, formatting, and the
source-layout gate. A release binary built after the initial module moves also
strictly replays the same artifact and passes a separate instrumented replay,
each with eight reference steps, maximum error 0.5423088, and drain. The final
centralization of test files changes only `cfg(test)` module bridges among
non-test inputs; all seven executor integration inputs remain byte-identical to
that GPU-replayed build. The source comparison and both build receipts are
retained rather than relabeling an older frozen binary as a new build.

## Scope

This is a B1, two-bucket correctness and trace-identity check. Search retains one
timed candidate per bucket; resource rejections can still require many attempts.
It does not qualify search quality, compilation speed, serving throughput, wider
batch/context geometry, or a performance gain. Existing CUDA/provider caches
were reused; “fresh search” means a new selected schedule, not empty caches.
Per-node profiling perturbs execution and is not serving TPOT.

The shared-FP8 option is enabled in this diagnostic manifest to exercise the
provider path; its production default remains off. Sampling tie semantics remain
unchanged. The separate [frozen-v3 logit diagnosis](../fp8-logit-diagnosis-20260912/README.md)
explains earlier serving output differences and remains distinct from this
current-source validation.

See [compiler boundaries](../../docs/compiler-boundaries.md) for the layout and
extension rules. Next work should measure compilation stages and repeated
rejected work, use measured region costs to guide exploration, and budget
resident executables together with KV and workspace before expanding residency.
