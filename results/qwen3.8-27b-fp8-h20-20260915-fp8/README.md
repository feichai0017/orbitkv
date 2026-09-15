# Qwen3.8-27B-FP8 on one H20: FP8 preparation

Measured on 15 September 2026 with **official vllm bench serve 0.29.0**.
OrbitKV `a905ed1` uses warp FP8 preparation with corrected F32 rounding.
**Full-model numerical qualification remains open.** Request-final projection
is available to the executor but is not enabled in this serving measurement.

## Performance

| Input / output | C | Engine | Output tokens/s (median; range) | TTFT median / P95 ms | TPOT median / P95 ms | Device GiB* | Repeatable text |
| --- | ---: | --- | ---: | ---: | ---: | ---: | :---: |
| 4 / 64 | 1 | OrbitKV | 40.03; 39.67–40.07 | 31.73 / 42.81 | 24.73 / 25.28 | 35.47 | yes |
| 4 / 64 | 1 | vLLM | 85.91; 85.79–85.98 | 61.70 / 65.38 | 10.82 / 10.83 | 31.16 | yes |
| 4 / 64 | 1 | SGLang | 55.36; 55.36–55.41 | 61.72 / 62.60 | 17.35 / 17.42 | 32.53 | yes |
| 4 / 64 | 8 | OrbitKV | 202.58; 202.21–203.33 | 215.23 / 220.42 | 36.59 / 36.71 | 35.37 | yes |
| 4 / 64 | 8 | vLLM | 548.74; 544.80–549.13 | 165.53 / 169.30 | 12.24 / 12.24 | 31.41 | yes |
| 4 / 64 | 8 | SGLang | 404.70; 404.69–404.79 | 83.53 / 111.62 | 18.66 / 18.71 | 32.53 | yes |
| 32 / 128 | 1 | OrbitKV | 38.30; 38.30–38.45 | 73.34 / 75.00 | 25.71 / 26.02 | 35.47 | yes |
| 32 / 128 | 1 | vLLM | 86.74; 86.57–86.74 | 98.34 / 103.84 | 10.84 / 10.84 | 31.41 | yes |
| 32 / 128 | 1 | SGLang | 56.52; 56.45–56.56 | 61.67 / 62.17 | 17.34 / 17.40 | 32.53 | yes |
| 32 / 128 | 8 | OrbitKV | 216.99; 215.56–217.46 | 224.31 / 443.03 | 34.79 / 36.06 | 35.50 | no |
| 32 / 128 | 8 | vLLM | 580.95; 579.62–581.41 | 209.62 / 211.66 | 12.26 / 12.26 | 31.41 | yes |
| 32 / 128 | 8 | SGLang | 414.09; 413.47–414.12 | 90.69 / 114.63 | 18.68 / 18.72 | 32.56 | yes |

All **192 new OrbitKV requests** complete across three fresh processes. Each
process runs short-C1, short-C8, context-C1 and context-C8 in that order and
drains token KV and fixed state on exit. vLLM/SGLang retain **384 requests from
the [earlier baseline](../qwen3.8-27b-fp8-h20-20260915/README.md)**; they were not rerun.
[runs.json](runs.json) preserves all per-run metrics and raw client results.
Medians aggregate per-run metrics; P95s are not pooled. Throughput remains close
to the [preceding OrbitKV run](../qwen3.8-27b-fp8-h20-20260915-regions/README.md); this batch does not
establish a model-level performance win.

The 32/128-token C8 workload's median per-run P95 TTFT increases from **355.04
to 443.03 ms** versus that preceding run. Similar throughput does not remove
this tail-latency regression; the performance-promotion gate is not met.

Each fixed token-ID trace contains sixteen requests, temperature zero and EOS
ignored, with exact 4/32-token inputs and 64/128-token outputs. Token IDs bypass
the unresolved text-tokenizer parity issue. `PYTHONHASHSEED=0` is retained.
There are no client warmups or
generation-based readiness checks. Later workloads share earlier process
warmup. Outputs differ across engines, so these are diagnostic timings rather
than an output-equivalent speed comparison.

*Memory is the largest device-wide serving sample at a requested one-second
interval. Reservations differ; this does not compare KV-memory efficiency.

## Numerical scope

The [checkpoint](checkpoint.json) remains official Qwen3.8-27B-FP8 revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, exported as
`Qwen3_5ForConditionalGeneration`. The corrected quantizer matches frozen
independent Torch/DeepGEMM bits. First-layer QKV and Z projections now match the
same reference inputs exactly. Shared/combined FP8 fanout, independent GEMM
and quantizer memcheck pass.

The exact serving artifact still fails the unchanged full-vocabulary error
gate: **158/1024 rows exceed 1.0**, with a maximum absolute error of
**17.828125** against the independent reference.
The oracle, teacher-forced histories and sampling tie policy are unchanged.
[validation.json](validation.json) records serial, C8, reordered and ragged
comparisons, state drains and diagnostic hashes.

An isolated final-normalization/LM-head check preserves complete logits exactly
when selecting two of four reference hidden rows. Separately searched complete
programs still disagree above the existing gate; that output-row optimization
therefore remains disabled in serving. Remaining convolution, recurrence,
normalization and MLP boundaries need diagnosis before wider region promotion.

## Reproduction

Build `a905ed1ef675c91b76c74904711994df7c273440` and use [servers.json](servers.json), the unchanged
[tuning profile](../../benchmarks/model-serving-tuning.json) and the
[common comparison method](../qwen3.8-27b-fp8-h20-20260915/README.md#reproduction).
For this rerun, retain only the OrbitKV entry and run three epochs of the four
traces. The limits are one H20, eight active requests, 512 model tokens,
32-token prefills and 256 aggregate query tokens. Weights use block FP8,
activations/KV use BF16 and recurrent state uses F32; CPU offload is unused.

Readiness took **24.53, 25.03, 23.53 seconds**, excluded from HTTP timings.
Creating the new artifact took **442.52 seconds** with persistent provider disk
caches. Each server strictly replays the same ten-bucket artifact. This is not
an empty-cache startup test. Numerical probes replay that exact
artifact; the output-row experiments use separate seven-bucket artifacts.

[source.json](source.json) binds the binary to committed production files.
[environment.json](environment.json) records the artifact and verified inputs.
Raw numerical traces and compiler/kernel diagnostics remain under
`.qualification/kernel-migration-20260915/corrected/`.
[performance.json](performance.json) feeds the website, and
[checksums.json](checksums.json) binds the report.
