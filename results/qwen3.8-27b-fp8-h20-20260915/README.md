# Qwen3.8-27B-FP8 on one H20

Measured on 15 September 2026 (Asia/Shanghai) using **vllm bench serve 0.29.0**.
OrbitKV, vLLM and SGLang load the same checkpoint and receive the same token-ID
prompts. These are diagnostic timings of the recorded configurations; generated
text differs across engines, so this is not an output-equivalent speed comparison.

## Performance

| Input / output | C | Engine | Output tokens/s (median; range) | TTFT median / P95 ms | TPOT median / P95 ms | Device GiB* | Repeatable text |
| --- | ---: | --- | ---: | ---: | ---: | ---: | :---: |
| 4 / 64 | 1 | OrbitKV | 40.09; 40.05–40.09 | 30.67 / 62.46 | 24.71 / 24.99 | 34.84 | yes |
| 4 / 64 | 1 | vLLM | 85.91; 85.79–85.98 | 61.70 / 65.38 | 10.82 / 10.83 | 31.16 | yes |
| 4 / 64 | 1 | SGLang | 55.36; 55.36–55.41 | 61.72 / 62.60 | 17.35 / 17.42 | 32.53 | yes |
| 4 / 64 | 8 | OrbitKV | 191.65; 190.97–198.96 | 259.51 / 264.35 | 38.23 / 38.91 | 34.78 | no |
| 4 / 64 | 8 | vLLM | 548.74; 544.80–549.13 | 165.53 / 169.30 | 12.24 / 12.24 | 31.41 | yes |
| 4 / 64 | 8 | SGLang | 404.70; 404.69–404.79 | 83.53 / 111.62 | 18.66 / 18.71 | 32.53 | yes |
| 32 / 128 | 1 | OrbitKV | 40.27; 40.27–40.38 | 45.55 / 46.33 | 24.62 / 24.90 | 34.84 | yes |
| 32 / 128 | 1 | vLLM | 86.74; 86.57–86.74 | 98.34 / 103.84 | 10.84 / 10.84 | 31.41 | yes |
| 32 / 128 | 1 | SGLang | 56.52; 56.45–56.56 | 61.67 / 62.17 | 17.34 / 17.40 | 32.53 | yes |
| 32 / 128 | 8 | OrbitKV | 51.79; 26.88–211.65 | 3054.51 / 30220.13 | 36.20 / 36.83 | 34.91 | no |
| 32 / 128 | 8 | vLLM | 580.95; 579.62–581.41 | 209.62 / 211.66 | 12.26 / 12.26 | 31.41 | yes |
| 32 / 128 | 8 | SGLang | 414.09; 413.47–414.12 | 90.69 / 114.63 | 18.68 / 18.72 | 32.56 | yes |

**576 complete requests; 55,296 output tokens; no client errors.**
Three epochs rotate the three engines through every run position. Each fresh
engine process runs short-C1, short-C8, context-C1 and context-C8 in order, with
16 requests each. Medians aggregate per-run metrics; P95 values are not pooled.
OrbitKV exits normally and drains token KV and fixed-state ownership in all
three sessions. Baselines do not expose the OrbitKV state census.

The official `timed_trace` dataset sends pre-tokenized inputs. Each trace hash
expands to one token; `PYTHONHASHSEED=0` fixes that expansion in every client
process. `--no-self-timed` applies infinite request rate with the recorded
concurrency. [inputs.json](inputs.json) preserves the expanded token IDs; the
[trace files](../../benchmarks/traces/) are hashed before and after measurement.
Sampling uses temperature 0, ignored EOS and exactly 64/128 output tokens.
Client warmups and generation-based readiness probes are disabled. These short
synthetic traces do not establish long-context or production performance.

*GPU memory is the largest sampled **device-wide** usage after readiness, at a
requested one-second interval. Missing process attribution remains null.
Reservations differ across engines; this is not a KV-memory efficiency comparison.

## Identity and correctness

All 66 indexed weight shards and five configuration/tokenizer files match the
[official fixed revision](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a).
[checkpoint.json](checkpoint.json) records those hashes. The official config
retains the `Qwen3_5ForConditionalGeneration` architecture class.

OrbitKV runtime source and the serving executable are unchanged from `4b6c990`.
The exact ten-bucket serving artifact passes an eight-step B1 teacher-forced
full-vocabulary probe on `[1,2,3,4]`, with maximum absolute logit error
**0.625** against the existing independent reference and unchanged tolerance 1.0.
The earlier seven-bucket B1/B8 preflight remains separately identified. Both
retain artifact/reference hashes and state-drain checks in
[environment.json](environment.json). Neither probe covers every benchmark
prompt or serving bucket. Generated texts/digests are retained per request;
within-engine variation is marked above and requires identical-history logit
diagnosis. Cross-engine output equality is not established.

## Preparation and excluded attempts

| Engine | Readiness seconds for each epoch |
| --- | --- |
| OrbitKV | 22.03, 24.03, 22.03 |
| vLLM | 37.57, 38.12, 37.56 |
| SGLang | 40.56, 41.06, 40.06 |

Readiness is excluded from HTTP timings. OrbitKV replays a prepared artifact;
baselines perform their recorded native preparation. Compiler/provider disk
caches persist across all attempts. SGLang uses
`SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` to skip exhaustive shape precompilation after
three prior default-precompile starts populated the cache. DeepGEMM execution,
default overlap and prefill graph behavior remain enabled. This is an explicit
startup setting, not a default or empty-cache startup comparison. Later profiles
share the same process warmed by earlier profiles. All timed observations in
the accepted matrix are retained.

[attempts.json](attempts.json) preserves two earlier attempts:

- Setup completed eight OrbitKV/vLLM measurements before SGLang failed on an
  occupied rendezvous port. It includes OrbitKV context-C8 at 34.20 tokens/s
  with a 50.52-second maximum TTFT and vLLM short-C8 at 111.36 tokens/s.
- A full random-text matrix completed 576 requests but failed the input-length
  gate: OrbitKV reported 24–32 context input tokens, versus 32 in the baselines.
  Its timings are excluded from the accepted comparison. Fixed token-ID traces
  remove this tokenizer ambiguity; the original raw measurements remain intact.

The accepted matrix uses caches left by those attempts. This does not show
that first-use compilation delays are fixed. The earlier
[OrbitKV-only report](../qwen3.8-27b-fp8-h20-20260914/README.md) also retains a
first-process outlier; its random-text inputs differ from this fixed-token trace.

## Reproduction

[servers.json](servers.json) contains full argument arrays. Package versions and
wheel-source integrity receipts are in [environment.json](environment.json).
Adjust executable, checkpoint, provider and artifact paths to your machine.
All engines use one GPU, block-FP8 weights, BF16 activations/KV, a 512-token
sequence limit and eight active requests. Prefix/radix reuse is disabled.
OrbitKV admits 32-token prefills with a 256-query-token aggregate budget;
vLLM reserves 2560M of KV memory with 256-token batches; SGLang uses FA3, a
4096-token pool and 256-token prefill chunks. There is no CPU offload or
speculative decoding. Native kernels and allocation policies differ.

Build OrbitKV source `4b6c990`, prepare its pinned providers, and create the
artifact with the recorded capacity and [tuning profile](../../benchmarks/model-serving-tuning.json).
Run the common client against those servers:

```sh
python tools/run_serving_comparison.py \
  --servers-file /absolute/path/servers.json \
  --model qwen38-27b-fp8 --tokenizer /absolute/path/checkpoint \
  --profiles-file benchmarks/model-serving.json --trace-dir benchmarks/traces \
  --profile short-c1 --profile short-c8 \
  --profile context-c1 --profile context-c8 \
  --epochs 3 --vllm-command /absolute/path/vllm --client-style python \
  --memory-device 0 --startup-timeout-seconds 1200 \
  --shutdown-timeout-seconds 60 \
  --identity-file /absolute/path/orbitkv-serve \
  --identity-file /absolute/path/decoder.json \
  --bench-arg=--num-warmups --bench-arg=0 \
  --bench-arg=--ready-check-timeout-sec --bench-arg=0
```

[performance.json](performance.json) feeds the website. [runs.json](runs.json)
retains individual metrics, complete client data, lifecycle records and memory
scope. [checksums.json](checksums.json) binds every published file. See the
[benchmark method](../../docs/benchmarking.md) for interpretation and promotion gates.
