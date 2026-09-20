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
