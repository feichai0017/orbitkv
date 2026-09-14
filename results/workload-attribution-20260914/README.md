# Compiler and runtime attribution

Status: correctness and diagnostic attribution pass on NVIDIA H20 (SM90).
Performance remains an open target. This record identifies search and compiler
costs; it does not establish a serving speedup.

## Changes

- Structured tracing retains complete egglog rule identities, per-schedule
  reports, workload dimensions, program identities and GPU execution steps.
  CPU span time and device event time remain separate.
- FlashInfer now carries an explicit request-count expression alongside query
  tokens. Retained-bucket planning no longer infers geometry from whichever CSR
  buffers happen to be installed. Execution validates logical metadata lengths.
  Decoder schema 10 requires regeneration of older artifacts.
- Ordinary compilation uses egglog `TimeOnly` reports. Verbose query-plan
  reports remain available with `EGGLOG_LOG=1 EGGLOG_DEBUG=1`.
  This changes diagnostic materialization, not the legal rewrite space.

No model-name dispatch, dimension-specific provider override, mathematical
change, or tolerance relaxation was introduced. New tests live in their owning
crate's `tests/` directory.

## Qualification

The Qwen3.8-27B-FP8 text decoder uses the
[bounded workload manifest](../../benchmarks/workload-attribution.json): seven
valid buckets spanning request capacities 1–8 and query-token capacities 1–32.
Search uses seed 7, one candidate graph per bucket, and graph residency capacity
two. This is a bounded correctness and diagnostic search, not a claim of optimal
selection or coverage of every shape in those intervals.

Two frozen release binaries each run nine model processes: B1 cold search,
B1/B8 strict replay and event profiles, followed by stage-disabled strict replay
and event profiles. Each build passes 296 full-vocabulary teacher-forced logit
comparisons and final resource drain. Across both builds, all 592 comparisons
pass the unchanged maximum-absolute-error gate of 1.0. The observed maxima are
0.6748047 for the full-plan control and 0.8125 for the final time-only build.

Regression checks pass: 247 root host tests, 243 fork host tests, 78 executor
CUDA tests, 52 directed release GPU tests, and 40 Python tests. These suites
overlap; their counts are not a count of unique test cases. The directed GPU
checks include B1→B8→B3→B1 retained-bucket execution and mixed provider recapture.
Formatting, source-layout checks, Clippy and the website check/build pass.

## Compiler observations

| Measurement | Full-plan control | Time-only final |
| --- | ---: | ---: |
| Cold diagnostic process | 744.69 s | 657.04 s |
| Decoder compile call | 742.58 s | 654.83 s |
| Enclosing egglog runs | 615.70 s | 535.07 s |
| `glumoe` ruleset search/apply | 433.61 s | 354.19 s |

These are single observations with existing provider/CUDA caches reused. All
seven selected program fingerprints differ despite the same seed and search
configuration. The difference therefore does not isolate the effect of report
materialization. The final `glumoe` cost is still the main compiler target.

Egglog's individual rule timers can omit asynchronously spawned join work,
while the enclosing ruleset timer waits for that work. A low per-rule timer or
zero match count cannot explain away the enclosing ruleset cost. The full
identities and raw counters are retained in [compiler.json](compiler.json);
interpret their scopes using [stage tracing](../../docs/stage-tracing.md).

## Runtime observations

Each number below measures one output-projection step with opt-in CUDA events.
Both phases perform full-vocabulary diagnostic logits. The implementations were
selected by the ordinary compiler search, without provider overrides.

| Workload | Full-plan control | Time-only final |
| --- | --- | --- |
| B1 prefill, 4 query tokens | GenericMatmul, 21.52 ms | GenericMatmul, 21.52 ms |
| B1 final decode, 1 query token | GenericMatmul, 5.36 ms | cuBLASLt, 0.68 ms |
| B8 prefill, 32 query tokens | cuBLASLt, 0.72 ms | GenericMatmul, 171.15 ms |
| B8 final decode, 8 query tokens | GenericMatmul, 43.03 ms | cuBLASLt, 0.68 ms |

The generic BF16 operation multiplies `[queries, 5120]` by the transposed
vocabulary weights to produce `[queries, 248320]`. Saved egraph inspection finds
a `cublaslt` candidate in the same result equivalence class as the slow generic
selection. These dimensions describe the measured workload; they are not
selection rules. [selection.json](selection.json) preserves the alternatives
and their full expressions before loop unrolling.

In stage-disabled, event-disabled replay, the median of six subsequent decode
steps changes from 29.80 to 24.43 ms at B1 and 79.14 to 36.95 ms at B8. But B8
prefill wall time increases from 70.03 to 247.36 ms. Program choices changed;
these results establish neither a general runtime improvement nor serving
TPOT. Per-step CUDA events also add substantial overhead: use
[runtime.json](runtime.json) for attribution, not serving latency.

## Next work

1. Make candidate enumeration reproducible and preserve the identities and
   order actually evaluated. Allocate a bounded measurement budget to legal
   alternatives for expensive regions, with prefill and decode both represented.
2. Restructure expensive `glumoe` joins using semantic anchors in egglog.
   Preserve admitted implementations and state/alias contracts, then compare
   repeated compilations with controlled candidate coverage and cache state.
3. Re-run independent logits, persistent-state transitions and complete
   uninstrumented serving workloads before changing defaults on performance
   grounds. Gather/cast/fused-region costs follow after gross selection misses.

## Reproduction and records

Use [the qualification runner](../../tools/run_decoder_qualification.py) and
the [stage-tracing commands](../../docs/stage-tracing.md) with the workload
manifest above. Pin the model, independent oracle, provider sources, target,
artifact and binary identities. Strict replay must use its own build's artifact;
event profiles must be separate from uninstrumented timings.

- [summary.json](summary.json): every process, comparison count, timing and limit.
- [build.json](build.json): frozen binaries, source, model/oracle and artifact hashes.
- [source-control.json](source-control.json) and [source-final.json](source-final.json):
  source-input hashes for the measured builds. Recorded base commits precede the
  final commits; these snapshots identify the uncommitted inputs used to build.
- [checks.json](checks.json): regression counts, commands and resolved failures.
- [files.json](files.json): hashes of local raw records under the ignored
  `.qualification/workload-attribution-20260914/` directory. These paths identify
  retained local evidence; raw traces, weights and executables are not shipped
  in this repository.
- [SHA256SUMS](SHA256SUMS): integrity manifest for this curated record.

The original schema-9 binary failed aggregate retained-bucket resource planning
before replay. Its failure is preserved and is not a passing performance
baseline. A debug GPU-test run was stopped for its compile cost; all directed
groups subsequently passed in release mode. An initial tracing test using a
thread-local subscriber missed worker callsites; the final isolated integration
test uses the production global-subscriber lifecycle and rejects missing phases.
Earlier result records remain unchanged.
