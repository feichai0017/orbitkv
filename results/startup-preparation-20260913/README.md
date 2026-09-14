# Startup preparation in serving — H20, 2026-09-13

The same verified `orbitkv-serve` binary and strict decoder artifact run with
`--prepare-execution false` or `true`. Capacity is fixed within each pair: two
for the first three profiles and one for the default-capacity check. The only changed
server argument is preparation; model weights, tuning, capacities, request
trace, sampling and compiled programs stay fixed.

| HTTP workload | Bucket capacity | First stream interval, off → on | P99 TPOT, off → on | Output throughput, off → on |
| --- | ---: | ---: | ---: | ---: |
| phase-switch-c1 | 2 | 138.92 → 29.93 ms | 37.69 → 24.48 ms | 35.60 → 36.80 token/s |
| decode-c1 | 2 | 137.91 → 31.41 ms | 25.86 → 24.80 ms | 40.23 → 40.48 token/s |
| context-growth-c1 | 2 | 141.98 → 30.05 ms | 25.06 → 24.79 ms | 40.76 → 40.75 token/s |
| default-capacity-c1 | 1 | 139.27 → 130.51 ms | 42.43 → 43.88 ms | 18.34 → 18.12 token/s |

Columns are medians of process measurements. `summary.json` also retains
paired changes, first-request TTFT, readiness, startup duration and steady
inter-token latency. Percentiles are client estimates within each small run,
not a claim about a production tail distribution. Short/decode profiles have
four processes per arm in alternating ABBA order; context growth and default
capacity have two per arm. All first generation requests are included; client warmups
and generation-based readiness checks are explicitly disabled.

All 24 timing processes complete 352 requests and
13,824 output tokens without client errors. Paired generated-text
digests match within every profile. Every process exits normally and reports
complete KV/fixed-state drain. At capacity two, prepared servers retain both
graphs at readiness and create no additional complete graph during these
workloads. Capacity one still requires ordinary phase-switch eviction.

The startup report records preparation separately from full initialization.
This moves reusable graph construction and planning before readiness;
it does not establish a kernel throughput improvement. Real CSR metadata is
still uploaded and provider islands can be refreshed on actual requests.

Three CUDA regressions verify that preparation does not execute an in-place
state write, respects eviction capacity, reuses dynamic buckets, and rejects
additional preparation after freezing. Existing warmup/freeze and eight
FlashInfer tests pass. Four fixed-artifact model processes compare 128
full-vocabulary output rows against an independent reference through 16
released request lifecycles, including capacity reduction and eviction;
the existing numerical gate is unchanged.
At capacity two the prepared model verifies no first-request full graph
reconstruction; at capacity one it verifies existing graphs are preserved.

A separate traced HTTP probe reproduces reference token IDs `[5, 0, 31]`,
then cancels an active SSE generation on SIGTERM and drains normally. It
loads all 418 saved CUDA images without NVRTC compilation.

The engine preserves existing materializations first, then fills unused slots
in artifact order up to the explicit capacity. It does not infer workload
frequency. The default cache capacity remains one; with insufficient slots,
a request can still evict a prepared bucket. Preparation is enabled by default
and can be disabled explicitly for controlled deployment comparisons.

The corrected default-capacity trace establishes the cache contract, but
its observed P99 TPOT is 42.43 → 43.88 ms and throughput is
18.34 → 18.12 token/s. These two-process-per-arm samples
do not establish a default-capacity performance benefit. The negative
measurements remain part of the final record.

Final review caught and corrected an initial artifact-prefix policy: at
capacity one, it evicted the already loaded prefill graph, raising the first
diagnostic prefill from 43.16 ms to 140.22 ms. The earlier logs, source snapshot
and provisional evidence remain archived under their original build identity.
The final code preserves the resident graph without an extra complete build;
all measurements in the table above were rerun on the corrected binary.

Production owners are `model_engine/startup.rs`, `model/residency.rs`,
`model/representative.rs`, and CUDA `runtime/residency.rs`. Test source follows
the owning modules under `tests/`. Readiness is published only after configured
preparation succeeds; the engine exposes a structured startup report.

All 441 frozen build inputs match the workspace. Server SHA-256:
`56ca46578bcc4a452799bcc1524974017875039763b26041709d543ed757b703`. Eight previous result packages retain
their checksums. Raw binaries, snapshots, benchmark arrays, reports and
reproduction scripts are under `.qualification/startup-preparation-final-20260913/`.

The context-growth profile uses short admitted prompts and 128 generated
tokens; actual client token lengths are preserved in the audit. An earlier
128-input-token attempt exceeded this fixed configuration’s four-token
prefill admission limit. All eight requests were rejected before admission.
The client incorrectly counted empty streams as successful; the output-length
gate rejected the run. Its complete evidence is retained and excluded from
performance samples. That attempt belongs to the earlier build, whose binary
and frozen source identity are retained in the audit. Long-prefill qualification
requires another capacity configuration and artifact. Independent logit
qualification remains the short reference fixture. Multi-request batching, longer sustained load,
joint residency optimization and comparisons with other engines remain open.

See [graph residency](../../docs/graph-residency.md) and
[benchmarking](../../docs/benchmarking.md) for the contracts and measurement rules.
