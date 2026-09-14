# Model inference performance

This directory publishes measured model inference performance: the exact
checkpoint and precision, device, input/output lengths, request concurrency,
TTFT, TPOT, output throughput and memory observations. Each report binds its
numbers to a source revision, executable, execution artifact and workload.

The website reads published performance data directly. Model support is tracked
separately in the [capability matrix](../docs/capability-matrix.md) and
[model targets](../docs/model-targets.md). A planned model has no performance
claim until it has its own measured report.

## Current measurements

[Qwen3.8-27B-FP8 on one H20](qwen3.8-27b-fp8-h20-20260914/README.md):
four bounded HTTP workloads, three runs each, on source `4b6c990`. The report
includes C1/C8 throughput, TTFT, TPOT, sampled memory and per-run ranges. It
retains the first-run latency outlier and the short-C8 output-repeatability
limit. Prior measurements below describe their own recorded revisions.

## Historical model measurements

| Model | Report | Scope |
| --- | --- | --- |
| Qwen3.8 27B FP8 | [Startup preparation](startup-preparation-20260913/README.md) | C1 HTTP latency and throughput; short input and context growth |
| Qwen3.8 27B FP8 | [Bucket residency](bucket-serving-20260913/README.md) | C1 HTTP latency and throughput with two graph-cache capacities |
| Qwen3.8 27B FP8 | [FP8 region tuning](fp8-region-tuning-20260912/README.md) | Limited serving observations; no qualified speedup |
| Qwen3.5 27B FP8 | [Engine comparison](deepgemm-luminal-bringup-20260909/README.md) | C1 comparison with SGLang and vLLM; OrbitKV is slower |
| Gemma 3 270M | [Serving load](serving-load-qualification-20260907/README.md) | C1/C2/C4/C8 fixed request traces |
| Gemma 3 270M | [SGLang comparison](sglang-product-comparison-20260907/README.md) | Matched HTTP workloads; OrbitKV is slower |
| Gemma 3 270M | [Schedule selection](compiler-constrained-schedule-benefit-20260907/README.md) | Same-engine HTTP improvement; SGLang comparison remains negative |

## Publication policy

Run experiments under ignored `.qualification/`, then publish one compact
model report after numerical preflight, complete-request and state-drain checks.
Keep per-run measurements and enough configuration to reproduce them. Report
medians across runs explicitly; do not label a median of run percentiles as a
pooled percentile. Distinguish startup, instrumented diagnostics and warm HTTP
inference. Sampled GPU memory is an observation, not an allocation peak.

Compiler benchmarks, provider tests and refactor checks belong in development
documents or `.qualification/`; older records are preserved in the
[validation archive](../docs/validation/README.md). See the
[benchmark method](../docs/benchmarking.md) for measurement details.
