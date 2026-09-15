# Qwen3.8-27B-FP8 on one H20: prepared execution

Measured on 15 September 2026 with **vllm bench serve 0.29.0**.
OrbitKV was rerun at `f4190d2` after fixing provider preparation and
attention-plan reuse. **vLLM and SGLang retain the [earlier baseline](../qwen3.8-27b-fp8-h20-20260915/README.md);
they were not rerun in these epochs.** The checkpoint and fixed token-ID workloads
are unchanged. Text differs across engines, so these are diagnostic timings.

## Performance

| Input / output | C | Engine | Output tokens/s (median; range) | TTFT median / P95 ms | TPOT median / P95 ms | Device GiB* | Repeatable text |
| --- | ---: | --- | ---: | ---: | ---: | ---: | :---: |
| 4 / 64 | 1 | OrbitKV | 40.03; 39.98–40.18 | 30.84 / 38.81 | 24.73 / 25.28 | 35.13 | yes |
| 4 / 64 | 1 | vLLM | 85.91; 85.79–85.98 | 61.70 / 65.38 | 10.82 / 10.83 | 31.16 | yes |
| 4 / 64 | 1 | SGLang | 55.36; 55.36–55.41 | 61.72 / 62.60 | 17.35 / 17.42 | 32.53 | yes |
| 4 / 64 | 8 | OrbitKV | 187.28; 186.65–200.03 | 304.22 / 316.53 | 38.51 / 38.60 | 35.13 | no |
| 4 / 64 | 8 | vLLM | 548.74; 544.80–549.13 | 165.53 / 169.30 | 12.24 / 12.24 | 31.41 | yes |
| 4 / 64 | 8 | SGLang | 404.70; 404.69–404.79 | 83.53 / 111.62 | 18.66 / 18.71 | 32.53 | yes |
| 32 / 128 | 1 | OrbitKV | 38.45; 38.43–38.48 | 76.51 / 77.65 | 25.62 / 25.78 | 35.13 | yes |
| 32 / 128 | 1 | vLLM | 86.74; 86.57–86.74 | 98.34 / 103.84 | 10.84 / 10.84 | 31.41 | yes |
| 32 / 128 | 1 | SGLang | 56.52; 56.45–56.56 | 61.67 / 62.17 | 17.34 / 17.40 | 32.53 | yes |
| 32 / 128 | 8 | OrbitKV | 210.09; 209.74–210.26 | 285.19 / 412.97 | 36.01 / 36.21 | 35.19 | no |
| 32 / 128 | 8 | vLLM | 580.95; 579.62–581.41 | 209.62 / 211.66 | 12.26 / 12.26 | 31.41 | yes |
| 32 / 128 | 8 | SGLang | 414.09; 413.47–414.12 | 90.69 / 114.63 | 18.68 / 18.72 | 32.56 | yes |

Compared with the earlier OrbitKV baseline, context-C8 narrows from
26.88–211.65 to **209.74–210.26 tokens/s**; the median per-run P95 TTFT falls
from 30,220 to 413 ms. Its previous best throughput was already comparable:
the improvement is removal of the observed stalls. Context-C1 throughput falls
from 40.27 to **38.45** tokens/s, and short-C8 from 191.65 to **187.28**.
Short-C1 remains near 40 tokens/s. This is not a general throughput improvement;
C8 generated-text variation also remains unresolved.

The new OrbitKV matrix contains **192 complete requests**, three fresh processes
with four workloads and 16 requests per workload. All sessions exit normally and
drain token KV and fixed state. The 384 baseline requests are retained observations,
not additional runs of this change. All per-run metrics and raw client results are
in [runs.json](runs.json). Medians aggregate per-run values; P95s are not pooled.

Each process runs short-C1, short-C8, context-C1 and context-C8 in that order.
Temperature is zero; EOS is ignored; output lengths are exactly 64 or 128 tokens.
The official `timed_trace` dataset uses one token per hash and `PYTHONHASHSEED=0`.
There are no client warmups or generation-based readiness probes. Later workloads
share the process warmed by earlier workloads. These short traces do not establish
long-context, production or output-equivalent performance.

*Memory is the largest sampled device-wide usage after readiness, at a requested
one-second interval. Engine reservations differ; this is not a KV-memory efficiency
comparison. Generated-text repeatability is reported per workload above.

## Identity and validation

The fixed [official checkpoint revision](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a)
is recorded in [checkpoint.json](checkpoint.json). Its official architecture class
remains `Qwen3_5ForConditionalGeneration`. Source inputs and frozen binaries are
bound to the implementation commit in [source.json](source.json).

The exact ten-bucket serving artifact passes eight B1 teacher-forced,
full-vocabulary steps on `[1,2,3,4]` with maximum absolute logit error
**0.625**, within the unchanged tolerance **1.0**, and successful drain.
A separate seven-bucket B8 artifact passes independent-reference search, strict
replay and profiling checks, with maximum absolute error **0.59375** and the same
1.0 tolerance. References and histories are unchanged. These probes
do not cover every HTTP prompt, changing batch history or serving bucket.
[environment.json](environment.json) retains the precise scopes and hashes.
Tokenizer parity and identical-history concurrent diagnosis remain follow-up gates.

## Preparation and reproduction

OrbitKV readiness took **22.03, 22.53, 22.53 seconds**, excluded from HTTP metrics.
Serving uses strict replay of the newly searched artifact. Provider disk caches
persist across attempts; this is not an empty-cache startup comparison.
The earlier baseline's native preparation and memory policies remain as recorded.

[attempts.json](attempts.json) identifies the excluded instrumented runs, including
the intermediate implementation with lower C1 throughput and the rejected B8
search whose input descriptors lost backing capacity. Raw compiler/provider
traces stay in `.qualification/`; their timings are not pooled into this table.
The earlier report retains its first-use outliers and rejected random-text input
comparison. No observations are discarded from an accepted epoch.

Build commit `f4190d2ae574a76b32412cc6f5014ff56871ab6b`, prepare the pinned provider sources, then generate a new
artifact using [servers.json](servers.json) and the unchanged
[tuning profile](../../benchmarks/model-serving-tuning.json). Old provider selections
must be regenerated. The serving manifest uses one GPU, 512 maximum model tokens,
eight active requests, 32-token prefills and a 256-query-token aggregate budget,
with BF16 activations/KV and block-FP8 weights. No CPU offload is used.

Use the [common comparison command](../qwen3.8-27b-fp8-h20-20260915/README.md#reproduction)
with this report's paths. For the OrbitKV-only rerun, retain only its server entry
and use three epochs with the same four fixed-token traces. [inputs.json](inputs.json)
retains expanded token IDs; input hashes are verified before and after measurement.

[performance.json](performance.json) feeds the website; [checksums.json](checksums.json)
binds every published file. See the [benchmark method](../../docs/benchmarking.md)
for promotion gates. These measurements do not establish an advantage over vLLM or SGLang.
