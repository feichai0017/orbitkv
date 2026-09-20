# Single-node cache measurements

Use the same inference engine to compare cache backends. Comparing a vLLM run
directly against an SGLang run also measures their attention kernels, scheduler,
and frontend; it does not isolate the cache implementation.

The first reference backends are vLLM's `OffloadingConnector` with pinned CPU
memory and SGLang's HiCache with a CPU pool. Native HBM-only prefix caching is
the control for each engine. LMCache is the next independent cache-system
comparison; its engine, PyTorch, CUDA, and connector versions must be pinned
and validated together. FlexKV and Mooncake Store can extend that matrix.
Mooncake Transfer Engine alone is a transport library, so a raw transfer-engine
bandwidth result is not an end-to-end cache comparison.

References: [vLLM KV offloading](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/),
[SGLang HiCache](https://docs.sglang.io/docs/advanced_features/hicache),
[LMCache compatibility](https://docs.lmcache.ai/getting_started/compatibility.html),
[FlexKV](https://github.com/taco-project/FlexKV),
[Mooncake](https://github.com/kvcache-ai/Mooncake).

## Qwen3-8B on H20: initial measurements

Measured on one NVIDIA H20 (97,871 MiB reported memory), with vLLM 0.29.0,
SGLang 0.5.20, PyTorch 2.13.0, Transformers 5.12.1, and Python 3.11.2.
The model is `Qwen/Qwen3-8B` at revision
`b968826d9c46dd6066d109eabc6255188de91218`. OrbitKV uses release build
`1184b09a`. The fixed-capacity workload below produces 270 measured requests.

**Client TTFT p50 after GPU cache pressure, in milliseconds; lower is better:**

| Engine | Cache backend | 1,024 tokens | 4,096 tokens | 8,192 tokens |
| --- | --- | ---: | ---: | ---: |
| vLLM | Native HBM cache, now evicted | 117.69 | 475.21 | 1,005.71 |
| vLLM | Native CPU offload | 22.48 | 32.78 | 47.81 |
| vLLM | OrbitKV DRAM | 24.16 | 35.02 | 56.02 |
| SGLang | Native HBM cache, now evicted | 116.82 | 472.19 | 1,000.07 |
| SGLang | HiCache CPU | 31.85 | 32.99 | 44.81 |
| SGLang | OrbitKV DRAM | 39.31 | 50.51 | 62.04 |

All native-HBM samples in this table were misses. All CPU/OrbitKV samples were
verified external restores with no HBM hit. The two engines have different
restore boundaries: for these aligned prompts vLLM restores the full prefix,
while SGLang restores all but the final 64-token page. Compare backends within
each engine; the two OrbitKV rows are not a transport-only comparison.

OrbitKV avoids expensive recomputation, but both engines' built-in CPU caches
are faster in this experiment. At 8K, OrbitKV adds 8.21 ms over vLLM CPU offload
and 17.23 ms over HiCache. Manager load-task p50 is 28.79 ms and 26.46 ms,
respectively; this includes descriptor construction, H2D submission and stream
synchronization, not just PCIe transfer. Profiling transfer batching, completion
observation and overlap with inference is the next performance task.

For resident prefixes, OrbitKV TTFT p50 was 20.10/22.27/25.34 ms in vLLM and
29.66/29.87/30.91 ms in SGLang. vLLM's resident phase includes a small external
tail-page restore and is therefore classified as a mixed hit.

Output comparison is retained for every sample. The performance runs do not
enable deterministic inference: two of five 1K prefixes changed output on
vLLM reuse, including native HBM reuse and native CPU offload. OrbitKV and
vLLM CPU offload nevertheless matched on **all 45 corresponding requests**.
SGLang's three backends matched on all 45 corresponding requests, and every
reuse matched its cold output. These observations do not replace the separate
deterministic correctness gates.

The first vLLM OrbitKV run exposed a real bug: 4K/8K multi-layer Publish metadata
exceeded the default 64 KiB descriptor slot, so those prefixes were not saved.
The shared client now splits metadata at complete per-page layer boundaries
and retains GPU sources until every chunk finishes. The table uses the full
rerun after this fix. The earlier failed-offload run and an SGLang CLI startup
failure are preserved separately in the local benchmark directory.

The [270 request measurements](benchmarks/qwen3-8b-h20.csv) and
[launch manifests and summaries](benchmarks/qwen3-8b-h20.json) are checked in.
Complete JSONL responses, counter deltas, and engine/manager logs are retained
under `/workspace/benchmarks/orbitkv-qwen3-8b` on the measurement host.
LMCache, FlexKV, and Mooncake Store have **not** been measured in this experiment.

## Reproduce the latency experiment

Build a **release** wheel using [the single-node setup](single-node.md), and
install it into the two engine release environments. From the repository root:

```bash
for engine in vllm sglang; do
  for backend in native cpu orbitkv; do
    ".venv/${engine}-release/bin/python" examples/bench_single_node.py \
      --engine "$engine" --backend "$backend" \
      --model /workspace/models/qwen3-8b \
      --output "/workspace/benchmarks/qwen3-8b/${engine}-${backend}"
  done
done
```

Each output directory must be empty. The script starts and stops its own engine
and, for OrbitKV, its own manager. It preserves engine/manager logs, launch
commands, versions, GPU details, raw samples, cache-source evidence, and a
summary. Do not run other GPU workloads alongside these measurements.

The initial experiment uses dense Qwen3-8B in BF16, one GPU, TP=1, 64-token
pages, 16,384 GPU KV tokens (2.25 GiB), and a 16 GiB external payload budget.
Metadata, pinned staging buffers, and process RSS are outside that payload
budget. It uses 1K/4K/8K synthetic token inputs, 16 output tokens, concurrency
one, and five independent prefixes per input length. Model startup, kernel
warmup, and cache-pressure traffic are excluded from measured request latency.

Every prefix is measured in three states:

1. **Cold:** a new prefix with no intentional cache reuse.
2. **HBM hit:** the exact same request while its GPU KV is resident.
3. **After pressure:** two disjoint 12,288-token requests exceed the 16,384-token
   GPU budget; repeat the original request and inspect where its KV came from.

This avoids flushing HiCache's CPU pool while preserving OrbitKV's pool, which
would give the two systems different starting conditions. A request after
pressure counts as an external-cache sample only if cache counters or manager
H2D bytes establish a restore without an HBM hit. Cold misses and mixed hits
remain in the raw results and must not be relabeled as CPU restores.

TTFT is measured at the HTTP client when the first nonempty streamed text
arrives. Engine counters and OrbitKV load bytes identify the cache source;
output strings are compared with the corresponding cold request. Output
differences are recorded rather than silently dropped. Numerical correctness
still has a separate deterministic E2E gate.

This is a serial latency experiment. Five observations do not establish a
production tail-latency SLO. It does not measure concurrent goodput, SSD
performance, multi-node transfers, restart recovery, or a production prompt
distribution. Those require additional workloads and capacity sweeps.
