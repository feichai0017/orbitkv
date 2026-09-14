# Persistent-state preflight on H20

Required state aliases are now checked before CUDA source generation and
compilation. The Qwen3.8-27B-FP8 qualification passes all **296 full-vocabulary
logit comparisons** and final drains. Rejected-candidate evaluation is much
cheaper in this run; warmed decode remains close to the preceding record.

## Change and correctness

Static LLIR validation resolves each required logical output through the
operation's storage-alias contract to its designated input. Copying data from
that input is insufficient. Search, finalist preparation, direct loads and
stitched artifact loads enforce the same preflight. The runtime retains its
compiled-bucket check before installation. No graph rewriting, provider forcing
or model-specific dispatch was added.

Seven new host regressions cover alias chains, argument order, copying,
missing/wrong/ambiguous endpoints, multiple arenas and mutation hazards. Probe
operations panic if preparation or compilation is reached on an invalid graph.
One new H20 regression verifies that invalid direct and stitched loads leave the
working executable intact and do not populate the CUDA kernel cache. The 20-test
resource suite and eight-test alias suite overlap. Root host tests (247), root
and fork Clippy, formatting, source layout and website check/build pass; see
[checks.json](checks.json) for recorded commands and evidence scope.

One frozen release binary executes nine model processes: B1 cold search,
B1/B8 strict artifact replay and event profiles, followed by stage-disabled
replay/profile pairs. All reference comparisons pass under the unchanged
maximum absolute error gate of 1.0; the observed maximum is **0.8671875**.
Weights, model semantics and the independent teacher-forced reference are
unchanged. Each replay verifies artifact identity and final state drain.

The experiment uses seed 7, seven workload buckets, eight measured candidates
per bucket, eight initial genomes, two deployment finalists and three trials.
Graph residency is two, batch capacity is eight, and the existing
[workload manifest](../../benchmarks/search-coverage.json) is unchanged.
Only the recorded short-input B1/B8 workloads are qualified.

## Search and compilation

The trace records **319 direct evaluations: 56 measured, 263 rejected**, plus
14 deployment measurements and seven selected programs. Every rejection is a
required state-alias failure, now detected by static preparation.

| CPU wall accounting | Previous coverage record | This run |
| --- | ---: | ---: |
| Rejected-candidate evaluation, total | 81.06 s | 8.00 s |
| Rejected-candidate evaluation, median | 196.80 ms | 29.53 ms |
| Enclosing CUDA bucket-search spans | 185.30 s | 130.12 s |
| Decoder compile call | 750.97 s | 700.24 s |
| Enclosing egglog runs | 518.28 s | 522.37 s |
| `glumoe` search/apply counters | 337.09 s | 341.44 s |

The accounting views overlap and must not be added together. These are two
independent searches, not a paired empty-cache speedup experiment. All seven
snapshot digests and selected program identities differ from the preceding
record, despite identical per-bucket outcome counts and vocabulary-projection
coverage. Existing provider/JIT caches were reused. About 16 seconds of host
checks overlapped cold saturation; no competing model/GPU benchmark ran.
The [snapshot comparison](historical-snapshot-comparison.json) and
[environment receipt](environment.json) retain these limits.

Earlier rejection does not improve candidate validity or provider coverage.
In B8 decode, all eight measured graphs still use the generic vocabulary
projection, while 46 graphs containing a cuBLASLt projection fail the state
contract elsewhere. The full coverage and rejection audit is in
[summary.json](summary.json) and [search.json](search.json). Identifying this
checkpoint's vocabulary extent is a post-run diagnostic, never a dispatch rule.

## Runtime observations

These strict-replay observations disable stage tracing and CUDA timing events.
Each step requests diagnostic full-vocabulary logits; prefill is one observation
and decode is the median of six subsequent steps. They are not serving TPOT or
throughput measurements.

| Diagnostic wall time | Previous coverage record | This run |
| --- | ---: | ---: |
| B1 prefill, 4 query tokens | 134.30 ms | 134.55 ms |
| B1 subsequent decode | 24.22 ms | 24.67 ms |
| B8 prefill, 32 total query tokens | 77.32 ms | 76.15 ms |
| B8 subsequent decode | 78.96 ms | 78.77 ms |

A separate event profile attributes **43.02 ms** to the final B8 decode's
generic vocabulary projection. That device duration cannot be subtracted from
the uninstrumented wall time to promise a speedup. Runtime throughput has not
been improved by this preflight change.

## Next work and reproduction

Explore expensive regions while preserving a valid surrounding graph, and
optimize `glumoe` joins before raising search budgets. Keep state constraints
and independent output/next-state gates mandatory. The
[model target matrix](../../docs/model-targets.md) records the parallel model
coverage roadmap under one H20 with quantization and CPU offload allowed.

Use the [qualification runner](../../tools/run_decoder_qualification.py) with a
release `model_execution` binary, test
`quantized_decoder_batched_reference_and_drain`, the workload manifest above,
`--search-graphs 8`, and independent model/reference directories. Retain the
fresh search trace and reuse that artifact for B1/B8 replay. Keep event profiling
separate from uninstrumented runs. Raw file identities, build receipts, checked
source inputs and compact compiler/runtime records accompany this report;
historical result files were verified unchanged.
