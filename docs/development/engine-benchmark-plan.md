# Engine benchmark and evidence plan

Oxide Infer needs two independent performance proofs:

1. matched kernel comparisons against FlashInfer where contracts overlap;
2. complete inference and serving comparisons against vLLM and SGLang, plus
   the pinned source-reference engine.

A faster kernel does not prove a faster engine. A faster unloaded server does
not prove production goodput. Correctness, kernel timing, model execution, and
serving results remain separate records.

## Evidence ladder

```text
operator reference and negative cases
  -> device correctness and sentinels
  -> sanitizer and lifetime checks
  -> CUDA Graph qualification or exclusion
  -> matched kernel benchmark
  -> full-model output and provider trace
  -> offline engine benchmark
  -> online serving and SLO goodput benchmark
```

Each record pins source commits, container or environment hashes, model files,
tokenizer, kernel artifacts, CUDA and driver versions, hardware identity,
clock policy, command line, warmup, samples, and exclusions.

## Kernel comparison with FlashInfer

Compare semantic contracts, not wrapper names. Oxide and FlashInfer receive
the same tensor bits, page tables, sequence metadata, dtype, layout, stream,
and caller-owned outputs. Compilation, tuning, allocations, page
materialization, and correctness copies stay outside the timed interval.

The primary matrix covers:

| Family | Workloads |
| --- | --- |
| Attention | single and paged decode; ragged and paged prefill; MHA, GQA, short and long context |
| KV cache | RoPE, paged append, gather, scatter, and compaction as implemented |
| Sampling | logits transforms, top-k/top-p/min-p, and token selection as implemented |
| GEMM | model-census M=1 decode and larger prefill shapes; FlashInfer where available and cuBLASLt as a diagnostic baseline |

For each row:

- validate finite output against an independent reference before timing;
- report maximum absolute and relative error and stable output digests;
- use CUDA events on one caller-owned stream;
- run complementary balanced provider orders;
- report median, p95, dispersion, workspace, and effective bandwidth or FLOPS;
- fail the ranking if order drift exceeds 5%;
- record the loaded artifact hash and provider-hit identity.

The first TileLang spike is not a promotion result. On its fixed paged-prefill
shape, TileLang measured 31.83 microseconds, the current Oxide path 47.01
microseconds, and FlashInfer 25.90 microseconds. TileLang improved the current
path by 32.3% but remained 22.9% slower than FlashInfer. It proves that the
toolchain can express and run this contract, not that the engine migration has
already won.

Kernel promotion gates for the first release are:

- all admitted shapes pass correctness and safety gates;
- no critical-path row is more than 15% slower than its matched comparison;
- the latency geometric mean across the model-weighted matrix is no more than
  5% slower than the matched comparison;
- no hidden allocation, JIT, tuning, synchronization, or provider fallback
  occurs in enqueue;
- a slower admitted row has a measured engine-level reason to remain.

## Full-model correctness

Before performance ranking, run Oxide and the pinned source-reference engine
with the same local weights and tokenizer. The source-reference engine is an
independent process and build; no reference implementation path is linked into
the Oxide product.

The first gate uses deterministic greedy decoding with fixed prompt token IDs,
fixed output lengths, EOS ignored for the timed interval, and prefix caching,
speculation, and quantization disabled. It records:

- prompt and generated token IDs;
- selected logits slices and numerical limits at layer checkpoints;
- KV page hashes and page-allocation events;
- provider hits for every GPU operation;
- host-to-device and device-to-device copy counts;
- peak device memory and clean teardown.

The Oxide path must produce the same greedy tokens and zero unregistered GPU
provider hits. Numerical equality is contract-specific; bitwise equality is
not assumed unless the contract requires it.

## Neutral engine comparison

The primary vLLM and SGLang comparison uses one repository-owned workload
driver against each engine's OpenAI-compatible HTTP endpoint. A frozen JSONL
trace supplies prompt token IDs, requested output lengths, arrival times, and
request identifiers. This avoids ranking servers with different client-side
load generation or tokenization.

The official benchmark clients remain secondary reproduction checks:

- vLLM documents `vllm bench serve`, `vllm bench throughput`, and
  `vllm bench latency` in its benchmark CLI;
- SGLang documents `python -m sglang.bench_serving` as the HTTP and scheduler
  benchmark and recommends at least five times the maximum concurrency in
  prompt count.

Pinned references:

- <https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md>
- <https://github.com/vllm-project/vllm/blob/main/benchmarks/README.md>
- <https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/bench_serving.md>
- <https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md>

## Workload matrix

Start with one identical model, BF16 weights, BF16 KV cache, one GPU, the same
maximum context, greedy decoding, and no prefix cache or speculative decode.
Use local model files so network and cache state cannot change a run.

| Profile | Input / output tokens | Concurrency | Purpose |
| --- | ---: | ---: | --- |
| Interactive | 128 / 128 | 1, 4 | TTFT and decode latency |
| Balanced | 1,024 / 256 | 1, 4, 16, 32 | common serving throughput |
| Prefill-heavy | 8,192 / 256 | 1, 4, 8 | long-context prefill |
| Long context | 32,768 / 256 | 1, 2 | memory and attention scaling |
| Decode-heavy | 128 / 1,024 | 1, 8, 32 | KV and decode efficiency |
| Mixed trace | fixed distribution of all rows | arrival-rate sweep | scheduler behavior and goodput |

Run at least `5 * max_concurrency` requests after warmup and enough requests to
stabilize p99. Sweep offered load rather than reporting only the saturation
point.

## Metrics

Online records report:

- time to first token (TTFT) p50, p95, and p99;
- inter-token latency or time per output token p50, p95, and p99;
- end-to-end request latency p50, p95, and p99;
- request, input-token, output-token, and total-token throughput;
- goodput under declared TTFT and inter-token SLOs;
- error, cancellation, and timeout counts;
- peak and steady device memory, KV utilization, and host CPU usage;
- startup, weight-load, and artifact-load time separately from steady state.

Offline records report prefill tokens per second, decode tokens per second,
total tokens per second, per-step batch composition, peak memory, and Graph hit
rate.

## Fairness controls

All three engines use:

- the same GPU and exclusive host allocation;
- the same local model and tokenizer file hashes;
- the same dtype, KV dtype, maximum sequence length, and output token counts;
- the same prefix-cache, speculation, quantization, and tensor-parallel policy;
- fixed clocks or a recorded clock and power trace;
- pinned engine commits and container digests;
- identical warmup traces and a rotated engine run order;
- no compilation, model download, or tuning in the steady-state interval.

Engine-native optimizations may be enabled only in a separately named
"best configured" cohort. The primary cohort isolates the core engine under
matched semantics.

## Rust-native peer cohort

PegaInfer is a useful architectural and performance peer, but it is not forced
into the first release matrix unless it supports the same model profile,
weights, dtype, context, and semantics. Its current public measurements use
Qwen3-4B on an RTX 5090, which cannot be compared numerically with Oxide's
historical H20 operator records or the first Qwen2.5-1.5B target.

When both engines admit an identical profile, run PegaInfer through the same
neutral JSONL workload driver and fairness controls. Report it as a separate
Rust-native cohort until repeated results justify changing a primary release
gate. PegaFlow is not an engine baseline; external-KV comparisons belong to a
future cache-service matrix with fixed cache budget, hit trace, topology, and
transfer semantics.

Pinned peer references:

- <https://github.com/pegainfer-project/pegainfer>
- <https://openedinfer.com/blog/pegainfer-010/>
- <https://github.com/novitalabs/pegaflow>

## Competitive claim gate

The first production-shaped claim requires all of the following on the frozen
matrix:

- complete correctness and provider-coverage gates;
- no more than 10% regression in p50 TTFT, p50 inter-token latency, peak
  memory, or saturated output throughput against both pinned engines;
- at least one repeatable advantage outside the 5% noise band;
- no p99, error-rate, or goodput regression hidden by an average;
- results reproduced from a clean release artifact, not a development tree.

This gate supports a "competitive Rust inference engine" claim. A claim that
Oxide Infer is faster than vLLM or SGLang requires the named workloads and
metrics to show that result; it is never generalized beyond the evidence.
