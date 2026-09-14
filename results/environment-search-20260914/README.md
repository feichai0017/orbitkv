# Execution environment and query compilation

Selected decoder artifacts now validate their CUDA execution environment.
The legacy MXFP4 MoE matcher uses smaller staged queries. The final H20 binary
passes **152 request/step reference comparisons** and all five state drains.
Maximum absolute logit error is **0.8125**, within the unchanged **1.0** gate.
Provider pins, search budgets and crate versions are unchanged.

## Rule compilation

One frozen executable compares recorded rule text from `cd1c7242` with the final
rules. Each arm has two observations. Inputs and operation-constructor counts
match across arms; private query relations add no implementation alternatives.

| Workload | Original median seconds | Final median seconds |
| --- | ---: | ---: |
| Normalized 27B decoder fixture | 102.905 | 27.035 |
| Normalized SwiGLU MoE | 1.478 | 1.583 |
| Gemma GELU MoE | 1.750 | 1.989 |
| Small dense graph | 0.870 | 0.853 |

The decoder fixture improves **73.7%**; small routed graphs pay additional
relation/setup cost. The fixture comes from checkpoint metadata, default graph
normalization and the first decode interval. It has no weights, live state or
GPU candidate measurements. Native provider source overrides were absent in
these CPU processes, so source-dependent native alternatives are absent too.
It is an isolated query experiment, not the complete model search space.

Only `fused_moe_rewrite.egg` changes in production. A broader development
experiment also split GLUMoE activation matching; the final change keeps those
activation rules byte-identical to the baseline. Predicate composition and final
actions are preserved under the existing fixed-point schedule. The old MXFP4
kernel's narrow geometry remains unchanged and is not released-MoE-qualified.

[rules.json](rules.json) records every observation, input/rule/binary digest and
constructor count. Constructor counts alone are not a proof that every extracted
schedule is equivalent; separate H20 numerical gates remain necessary.

## Complete model

One H20, the existing Qwen3.8-27B-FP8 checkpoint and independent oracle, batch
capacity 8, graph residency 2 and unchanged `benchmarks/hotspot-search.json`:
seven workload representatives, eight measured graphs and 32 hotspot attempts.
B1 performs fresh search; B1/B8 replay and profile load one frozen artifact.

| Batch | Phase | Compile/load seconds | Reference comparisons | Median decode-2…7 wall ms |
| --- | --- | ---: | ---: | ---: |
| B1 | cold-search | 350.159 | 8 | 24.542 |
| B1 | strict-replay | 17.522 | 8 | 24.441 |
| B1 | profile | 17.586 | 8 | 60.338 |
| B8 | strict-replay | 19.319 | 64 | 37.600 |
| B8 | profile | 19.137 | 64 | 78.368 |

The preceding backend qualification recorded **943.102 s** cold compilation.
The `glumoe` ruleset's aggregate search/apply counter falls from **431.094 s** to
**0.824 s**. These complete-model observations use independent search snapshots
and selected programs, with retained provider/CUDA caches. They are not a paired
same-candidate or empty-cache speedup claim. The same-executable rule experiment
above isolates the query-text change more narrowly.

Diagnostic execution includes logits and synchronization; profile phases add
CUDA Graph instrumentation. This does **not** establish a serving-throughput or
TPOT improvement. Remaining cold costs include `kernel_specialize` (67.866 s
including merge/rebuild), seven egglog setup runs (63.372 s), native provider JIT
spans (53.174 s) and NVRTC spans (50.929 s). Their scopes must not be added as
independent whole-process costs. See [compiler.json](compiler.json) and
[model.json](model.json).

## Artifact admission

The selected environment covers all seven buckets and nested CUDA Graph calls:
H20/SM90, device geometry, CUDA driver API 13010, NVRTC 13010/options, actual
cuBLASLt 130200, selected DeepGEMM/FlashInfer/FlashAttention source identities,
native compiler/environment identity and provider inventory.

Four modified artifact copies are rejected:

- Changed cuBLASLt version: recompilation diagnostic before weight loading.
- Changed driver API version: retuning diagnostic before weight loading.
- Missing environment: deserialization failure.
- Omitted cuBLASLt dependency and its autotune setting: rejected by the installed
  program's dependency check, after loading weights and before execution.

[admission.json](admission.json) retains diagnostics and observed stage names.
Both recompilation and retuning requirements reject strict replay; neither
silently triggers search. This is conservative provenance, not a hermetic CUDA
installation fingerprint. See [artifact contracts](../../docs/module-artifacts.md).
Earlier unreleased schema-11 artifacts without the environment must be regenerated.

## Checks and provenance

- 503 default host tests, 11 optional frontend tests and 5 doctests pass.
- 152 CUDA test executions pass, covering environment, MoE numerical equivalence,
  provider contracts, mixed-library capture, DeepGEMM, module replay and kernels.
  One cuBLASLt host case appears in both its parent and dedicated suite.
- 71 executor unit tests pass with the provider source environment configured.
- Rust 1.98 formatting/Clippy, source layout, 46 Python tool tests and website
  check/build pass.

The numerical executables use Rust 1.97.1. [source.json](source.json) identifies
all three frozen binaries and 501 source/configuration files;
[checks.json](checks.json) records commands, counts and a corrected test-run
environment omission.

Artifact SHA-256: `25bdeedb9eab2a9ce73d156b21b9493b9f0e070d8aea9e69dadb9e144faece8f`.
Raw logs, traces, binaries, fixture and artifact copies remain under
`.qualification/environment-search-20260914/`. Historical evidence is unchanged.
