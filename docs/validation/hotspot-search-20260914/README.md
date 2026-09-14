# Profile-directed local search on H20

Luminal can now use measured execution regions to explore existing egraph
choices while retaining a valid parent's other bindings. The single-H20
Qwen3.8-27B-FP8 qualification passes **296 full-vocabulary reference comparisons**
and every final state drain. This is an optional search mechanism, not a
whole-model megakernel or a serving-performance qualification.

## Mechanism and checks

Extraction records operation/choice provenance separately from semantic program
identity. Loop copies retain their origins; compiled fusion regions retain their
constituent LLIR nodes. CUDA profiles kernels and library islands in a separate
execution. Region costs order single-choice neighbors of measured parents;
whole-program direct scores and final CUDA Graph measurements still rank them.
State-alias, cycle, layout and resource checks remain mandatory. No provider is
forced by checkpoint name, dimensions or an LLIR rewrite in Rust.

The [workload policy](../../../benchmarks/hotspot-search.json) sets one initial seed,
32 local attempts, two deployment finalists and three trials. The runner allows
eight measured graphs per bucket, with seven buckets, seed 7, graph residency 2
and batch capacity 8. Local attempts include duplicate or invalid extractions;
trace evaluation counts exclude those that never reach CUDA preparation.
The default `hotspot_candidates` remains zero.

Core regressions check exact unaffected bindings, finite neighbors, fixed-feedback
replay, rejected feedback, shared-region accounting and loop provenance. The H20
regression requires an actual local generated-GEMM/cuBLASLt transition beside a
persistent scatter branch. It compares independent CPU matrix products and exact
updated/untouched state rows across two requests. The default coverage regression
also passes. Core tests (222), tuning tests (5), root host tests, both Clippy checks,
formatting and source layout pass; [checks.json](checks.json) includes the initial
test-only Clippy line-limit failure and its final passing rerun.

## What was explored

The complete trace records **116 direct evaluations: 56 measured and 60 rejected**,
14 deployment measurements and seven selected programs. All **49 local candidates
were measured**, with no state/resource rejection. All 60 rejections occur during
initial seed discovery and violate a required state alias. Every measured region
maps to nodes in its exact candidate LLIR.

Two local transitions change the vocabulary projection from generated GEMM to
cuBLASLt inside a measured valid parent:

| Representative workload | Parent whole-program direct score | Neighbor score |
| --- | ---: | ---: |
| B1 prefill, 4 query tokens | 54.21 ms | 33.36 ms |
| B8 prefill, 32 query tokens | 229.79 ms | 59.15 ms |

These are direct search scores, not serving latency. The full parent, e-class,
old/new e-node and program identities are in [hotspots.json](hotspots.json).
The vocabulary extent is used only for this post-run audit. All other bindings
stay fixed, although a shared e-class can affect multiple operations.

**B8 decode starts from a cuBLASLt seed in this run.** Its faster selected artifact
therefore does not prove that hotspot exploration found that provider. B1 decode
retains Gemv. Three other buckets also begin with cuBLASLt vocabulary projections.
Fresh saturation is not canonicalized; all seven snapshot digests differ from
the preceding preflight record. Fixed seeds alone do not create a paired ablation.

## Runtime and compilation observations

One frozen release binary runs nine qualification processes: fresh search,
B1/B8 strict artifact replay and separate event profiles, followed by stage-disabled
replay/profile pairs. The maximum absolute logit error is **0.8125**, within the
unchanged 1.0 gate. Weights, model semantics and the independent reference remain
unchanged. Strict replay preserves the selected artifact and every run drains state.

The following observations disable stage tracing and CUDA timing events. Each
step requests full diagnostic logits. Prefill has one observation; decode is the
median of six subsequent steps. These are not TPOT or throughput measurements.

| Diagnostic wall time | Previous preflight record | This run |
| --- | ---: | ---: |
| B1 prefill, 4 query tokens | 134.55 ms | 103.28 ms |
| B1 subsequent decode | 24.67 ms | 24.85 ms |
| B8 prefill, 32 query tokens | 76.15 ms | 77.30 ms |
| B8 subsequent decode | 78.77 ms | 37.29 ms |

The separate final B8 event profile measures the cuBLASLt vocabulary projection
at 0.679 ms; the preceding artifact's generated projection took 43.024 ms.
The complete event-profiled execution takes 55.12 ms. Those device intervals
cannot be subtracted from uninstrumented wall times. Other selected choices,
snapshots, caches and measurement feedback also differ between records.

| Overlapping CPU wall accounting | Previous preflight record | This run |
| --- | ---: | ---: |
| Decoder compile call | 700.24 s | 667.29 s |
| Enclosing egglog runs | 522.37 s | 536.06 s |
| `glumoe` search/apply counters | 341.44 s | 358.31 s |
| CUDA bucket-search spans | 130.12 s | 83.97 s |

Do not add these overlapping accounting views. Existing provider/JIT caches were
reused; compact JSON analysis overlapped artifact replays, with no competing
model/GPU workload. Final static checks and the default-path GPU regression ran
after all model processes. These are independent observations, not an empty-cache
compiler speedup or a matched serving comparison.

## Next and reproduction

Restructure the costly `glumoe` joins around semantic anchors in egglog. Preserve
fixed-snapshot tests and independent complete-model gates while extending local
search to dependency closures where one-choice moves are insufficient. A
same-snapshot search ablation and matched serving run are needed before enabling
this policy by default or making serving claims.

Use [run_decoder_qualification.py](../../../tools/run_decoder_qualification.py) with
a release `model_execution` binary, test
`quantized_decoder_batched_reference_and_drain`, the profile above,
`--search-graphs 8`, and independent checkpoint/reference directories. Save
`LUMINAL_SEARCH_TRACE`, reuse the artifact for B1/B8 replay and keep event profiling
separate from uninstrumented runs. [environment.json](environment.json) records
the frozen binary, checked source inputs and test-only post-run cleanup;
[files.json](files.json) identifies raw evidence. Historical result files are unchanged.
