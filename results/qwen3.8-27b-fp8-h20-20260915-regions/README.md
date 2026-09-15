# Qwen3.8-27B-FP8 on one H20: serving rerun

Measured on 15 September 2026 with **official vllm bench serve 0.29.0**.
OrbitKV `435b17b` includes attention-plan validity fixes and gather/cast
regions. **Extended numerical qualification remains open.** The measurements
below are diagnostic; generated outputs differ across engines.

## Performance

| Input / output | C | Engine | Output tokens/s (median; range) | TTFT median / P95 ms | TPOT median / P95 ms | Device GiB* | Repeatable text |
| --- | ---: | --- | ---: | ---: | ---: | ---: | :---: |
| 4 / 64 | 1 | OrbitKV | 40.09; 39.95–40.29 | 31.49 / 45.55 | 24.68 / 25.09 | 34.93 | yes |
| 4 / 64 | 1 | vLLM | 85.91; 85.79–85.98 | 61.70 / 65.38 | 10.82 / 10.83 | 31.16 | yes |
| 4 / 64 | 1 | SGLang | 55.36; 55.36–55.41 | 61.72 / 62.60 | 17.35 / 17.42 | 32.53 | yes |
| 4 / 64 | 8 | OrbitKV | 202.81; 199.80–211.73 | 208.74 / 217.30 | 36.69 / 36.73 | 34.90 | no |
| 4 / 64 | 8 | vLLM | 548.74; 544.80–549.13 | 165.53 / 169.30 | 12.24 / 12.24 | 31.41 | yes |
| 4 / 64 | 8 | SGLang | 404.70; 404.69–404.79 | 83.53 / 111.62 | 18.66 / 18.71 | 32.53 | yes |
| 32 / 128 | 1 | OrbitKV | 38.51; 38.43–38.63 | 71.31 / 73.29 | 25.61 / 25.76 | 34.93 | yes |
| 32 / 128 | 1 | vLLM | 86.74; 86.57–86.74 | 98.34 / 103.84 | 10.84 / 10.84 | 31.41 | yes |
| 32 / 128 | 1 | SGLang | 56.52; 56.45–56.56 | 61.67 / 62.17 | 17.34 / 17.40 | 32.53 | yes |
| 32 / 128 | 8 | OrbitKV | 217.15; 217.11–218.10 | 223.42 / 355.04 | 35.19 / 35.69 | 34.90 | no |
| 32 / 128 | 8 | vLLM | 580.95; 579.62–581.41 | 209.62 / 211.66 | 12.26 / 12.26 | 31.41 | yes |
| 32 / 128 | 8 | SGLang | 414.09; 413.47–414.12 | 90.69 / 114.63 | 18.68 / 18.72 | 32.56 | yes |

**192 new OrbitKV requests** complete across three fresh processes, each running
short-C1, short-C8, context-C1 and context-C8 in that order. All processes exit
normally and drain token KV and fixed state. The vLLM/SGLang rows retain **384
requests from the [earlier baseline](../qwen3.8-27b-fp8-h20-20260915/README.md)**; those engines
were not rerun. [runs.json](runs.json) preserves every per-run metric and raw
client result. Medians aggregate per-run metrics; P95s are not pooled.

Temperature is zero, EOS is ignored, and each trace has sixteen requests with
exact 4/32-token inputs and 64/128-token outputs. Fixed token IDs bypass the
unresolved text-tokenizer parity issue. `PYTHONHASHSEED=0`, no client warmups and
no generation-based readiness request are used. Later workloads share the
process warmed by preceding ones. These short traces do not establish a general
performance advantage or long-context support.

*Memory is the largest device-wide sample during serving, at a requested
one-second interval. Reservations differ between engines; this is not a
comparison of KV-memory efficiency.

## Numerical scope

The frozen [checkpoint](checkpoint.json) is official Qwen3.8-27B-FP8 revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`; its exported architecture class is
`Qwen3_5ForConditionalGeneration`. [source.json](source.json) binds the server
binary and production files to the implementation commit.

Using the same retained seven-bucket baseline schedule, identical teacher-forced
C8 histories preserve all **1024 full-vocabulary rows exactly** between
one-bucket residency and prepared multi-bucket residency. Reordering the eight
rows also preserves complete logits with the ten-bucket serving schedule.
The earlier retained
FlashInfer plan could reuse stale query/page segmentation and fault when a
prefill bucket was revisited; a multi-tile independent attention regression
fails before and passes after the fix.

This does **not** close the wider numerical gate. On sixteen distinct short
histories, each with 64 teacher-forced output rows, the exact serving artifact's
serial/reference maximum absolute error is **17.99609375**; **164/1024 rows exceed
the unchanged 1.0 limit**. Serial/B8 logits also differ. The reference uses
independent Transformers model code and the same DeepGEMM library. Tolerances,
histories and the existing highest-index sampling tie policy are unchanged.
[validation.json](validation.json) records scopes, failures, drain results and
checksums. Throughput improvement does not substitute for numerical acceptance.

The independent attention memcheck passes. The full-model memory diagnostic
uses a context fence immediately before teardown, with all memory checks enabled.
Without that diagnostic fence, Compute Sanitizer reports potential async-free
races also reproduced by a standalone CUDA-only program with two independent
child graphs. Both tested sanitizer versions reproduce that limitation. This is
not an unconditional clean teardown check; it is recorded in the validation
scope. Production code and the HTTP timings use ordinary execution/teardown.

## Reproduction

OrbitKV readiness took **22.53, 24.03, 22.53 seconds**, excluded from HTTP timings.
The server strictly replays a ten-bucket artifact; native provider disk caches
persist. This is not an empty-cache startup measurement. GPU diagnostics and
compiler traces stay in `.qualification/c8-consistency-20260915/` and are excluded
from the performance table.

Build commit `435b17ba4e21f738938e9922916ed1a6c71eefcb` and use [servers.json](servers.json), the unchanged
[tuning profile](../../benchmarks/model-serving-tuning.json) and
[common comparison command](../qwen3.8-27b-fp8-h20-20260915/README.md#reproduction).
For this rerun, retain only the OrbitKV server entry and run three epochs of the
same four traces. Limits are one GPU, eight active requests, 512 model tokens,
32-token prefills and 256 aggregate query tokens, with block-FP8 weights and
BF16 activations/KV; CPU offload is unused. [inputs.json](inputs.json) records
expanded prompt IDs; [environment.json](environment.json) records identities
checked before and after measurement.

[performance.json](performance.json) feeds the website and
[checksums.json](checksums.json) binds this report. The
[benchmark method](../../docs/benchmarking.md) defines promotion gates.
