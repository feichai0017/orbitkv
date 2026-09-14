# Qwen3.8-27B-FP8 on one H20

Measured on 14 September 2026 through OrbitKV's OpenAI-compatible streaming
completions endpoint, using `vllm bench serve` 0.29.0. This is an OrbitKV model
baseline; the client does not run a vLLM reference engine.

| Input tokens | Output tokens | Concurrency | Output token/s (median; range) | TTFT median / P95 ms | TPOT median / P95 ms | Sampled device GiB | Repeatable text |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 4 | 64 | 1 | 40.10; 40.07–40.16 | 30.65 / 61.44 | 24.70 / 24.84 | 34.84 | yes |
| 4 | 64 | 8 | 189.94; 188.76–190.96 | 264.49 / 314.52 | 38.23 / 38.84 | 34.78 | no |
| 24–32 | 128 | 1 | 39.86; 39.76–39.88 | 45.79 / 143.33 | 24.66 / 24.96 | 34.78 | yes |
| 24–32 | 128 | 8 | 211.74; 76.10–212.06 | 235.87 / 257.78 | 36.22 / 36.93 | 34.78 | yes |

Each workload has three independent server processes and 16 requests per process.
The table uses medians of per-run metrics, including each run's P95 values;
these are not pooled percentiles. The throughput range includes all three runs.
Native/provider disk caches were retained between processes. The first context-C8
run measured 76.10 output token/s: its first eight requests had about 17.5 s TTFT,
versus about 0.2 s for the next eight. Later runs measured 211.74 and 212.06 token/s.
Attribution of the first-run delay remains open; the median hides this event.

Input lengths are the client's observed lengths after tokenization, with
requested lengths of 4 or 32. Outputs contain
exactly 64 or 128 tokens. Greedy decoding uses temperature zero and ignores EOS.
All first requests are included. Readiness checks generate no requests, and
client warmups are disabled; decoder execution preparation happens before readiness.

GPU memory is the maximum sampled **device-wide** usage after readiness, not an
allocation peak or process attribution. The host/container PID mismatch prevents
reliable NVML process matching. Both scopes and missing values remain explicit in
the individual run records. Memory is sampled at a one-second requested interval.

A non-repeatable text entry means the generated-text digest differs between runs.
All measured requests completed and the final KV and fixed-state ownership drained.
These observations do not establish matched-output performance improvements,
production readiness, long-context capability or an advantage over another engine.
The independent B1/B8 numerical preflight uses eight teacher-forced reference
steps, the existing absolute-logit tolerance of 1.0, and a separate seven-bucket
artifact. It is not an evaluation of arbitrary generated text or every program
in the ten-bucket serving artifact.

## Configuration and reproduction

The model uses block-FP8 projection weights, BF16 attention/KV, and FP32 recurrent
state. There is no CPU weight offload. Limits are eight active requests, 32 input
tokens per request, 256 tokens per batch, 512 total tokens per request, 16 tokens
per KV page and 256 physical pages. The graph cache permits 12 retained buckets;
the tuning profile produces ten feasible buckets. The batch token budget admits
all eight maximum-length prefills together. Initial server startup is excluded;
any work triggered after readiness remains in the HTTP measurements.

Build the source revision in `environment.json`, prepare the pinned providers,
and set their source directories as described in
[provider setup](../../docs/attention-providers.md). Use the checkpoint identified
by the checksums in `environment.json`. Create the artifact with the same server
command before timing:

```sh
cargo build --locked --release -p orbitkv-engine --features server --bin orbitkv-serve
python tools/run_model_serving.py \
  --server-command 'target/release/orbitkv-serve --model /path/to/model --served-model qwen38-27b-fp8 --decoder-artifact /path/to/decoder.json --tuning-profile benchmarks/model-serving-tuning.json --page-counts 256 --max-model-tokens 512 --max-prefill-tokens 32 --max-batch-tokens 256 --max-active-requests 8 --max-queued-requests 64 --search-graphs 8 --search-seed 7 --graph-cache-capacity 12 --port 8010' \
  --base-url http://127.0.0.1:8010 --model qwen38-27b-fp8 \
  --tokenizer /path/to/model --profiles-file benchmarks/model-serving.json \
  --profile short-c1 --runs 3 --seed 0 --memory-device 0 \
  --vllm-command /path/to/vllm --client-style python \
  --bench-arg=--num-warmups --bench-arg=0 \
  --bench-arg=--ready-check-timeout-sec --bench-arg=0 \
  --startup-timeout-seconds 300 --shutdown-timeout-seconds 30 \
  --output-dir .qualification/model-serving/short-c1
```

Repeat with `short-c8`, `context-c1` and `context-c8`, using fresh output directories.
`performance.json` supplies the website table. `runs.json` retains per-run and
per-request timings, output digests, memory observations and final state reports.
`environment.json` binds the source, executable, artifact, checkpoint and settings.
