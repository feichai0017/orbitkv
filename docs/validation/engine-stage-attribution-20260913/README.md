# Engine merge and compiler/runtime stage attribution

`orbitkv-engine` now owns the logical protocol, model coordinator, optional HTTP
frontend, and `orbitkv-serve` executable. The separate server crate is removed;
its public contracts are re-exported from `orbitkv_engine`. Protocol/frontend
modules remain independent of physical execution types. All owned crate test
source lives in each crate's `tests/`, and the HTTP feature still compiles and
tests without CUDA.

The new buffered CPU stage layer separates compiler preparation, egglog work,
candidate generation/evaluation, provider JIT, CUDA Graph materialization, and
execution. Five final-build H20 processes pass eight independent-reference
logit steps and token/fixed-state drain: fresh search, strict replay, device
profiling, and two fixed-artifact replays with stage tracing disabled. Maximum
absolute error is **0.5078125**, below the unchanged **1.0** gate. The reference
uses Transformers 5.12.1 with local DeepGEMM 2.6.1; its model implementation is
independent, but not every underlying math library is independent.

| Observation | Final measurement | Interpretation |
| --- | --- | --- |
| Fresh schedule process | 299.7 s | Existing provider/CUDA caches reused; Rust build excluded |
| Search-space construction | 200.6 s, including 180.8 s egglog schedules | Largest observed cold-compile stage |
| CUDA search | 70.9 s | Includes candidate preparation, profiling, selection and installation |
| Candidate generation | 3.0 s across 122 calls | Generation/extraction alone does not explain startup |
| Rejected candidate evaluation | 37.3 s across 118 rejections | Significant preparation cost remains before rejection |
| Strict replay process | 37.3 s | Includes 20.9 s weight loading, 4.3 s graph construction, 11.1 s schedule loading |
| Generated-kernel compilation during replay | 425 NVRTC calls | Schedule replay still recompiles generated module images on this decoder path |
| First decode graph materialization | 89.3 ms | A large component of the first prefill-to-decode transition |
| Stage-off warm diagnostic decode | 24.2 ms median across six subsequent steps | Same artifact; includes diagnostic logits, not serving TPOT |

The last instrumented decode profile totals 41.625 ms. Grouped DeepGEMM calls
account for about 17.5 ms; fused regions 8.4 ms, gather 4.9 ms, cast 2.8 ms, and
FlashInfer attention 0.157 ms. These measurements prioritize investigation of
projections and small generated regions for this short-context workload.
Per-node events perturb the graph: they are not an additive decomposition of
the 24.2 ms uninstrumented diagnostic step.

The next work is to reuse provably bucket-independent compiler work, connect
generated module images to identity-checked artifact replay, and budget retained
decode/prefill graphs with KV and workspace before increasing residency. Split
weight conversion/upload timing before choosing a loader optimization. Use
measured region costs and preserve state/numerical contracts for subsequent
search improvements. See the [roadmap](../../roadmap.md) and
[stage tracing contract](../../stage-tracing.md).

`identity-audit.json` checks 869 file references and correlates both selected
programs through direct measurement, deployment measurement, validation, and
stored artifact. The stage audit additionally joins all 120 direct outcomes and
two deployment measurements with their program spans. The fixed-artifact
controls preserve the exact artifact and per-step numerical results.
`environment.json` records the final 420-input frozen build, binary, model,
oracle, provider and cache identities. `checks.json` records 254 host tests,
30 engine/server tests, a tokenizer-to-mock-engine HTTP test, three stage-layer
tests, 30 Python tests, Clippy, formatting and the source-layout gate. Some suites
overlap; these are not additive unique-test counts. Released-model engine/HTTP
tests remain opt-in; this run's real GPU closure is the decoder harness.

An initial diagnostic exposed a span-lifetime defect: a temporary in a
`while let` condition survived through evaluation. The final source bounds its
guard to generation, and the audit checks that generation/evaluation intervals
do not overlap on the same thread. All 26 invalid overlaps in the initial trace
serve as a negative control. That initial source, binary and raw data remain in
`.qualification/engine-stage-attribution-20260913`; only the corrected final
run in `.qualification/engine-stage-attribution-final-20260913` is promoted here.

Scope is B1 and two buckets with one timed candidate retained per bucket. Fresh
search selected different programs from earlier records, so startup differences
are not optimization gains. Stage bookkeeping has CPU overhead, nested and
parallel intervals overlap, and provider caches were not emptied. This is
attribution and correctness evidence, not a serving-performance or search-quality
qualification. Shared FP8 preparation remains opt-in and sampling tie semantics
are unchanged. Large traces, binaries and frozen source stay in the raw run.
