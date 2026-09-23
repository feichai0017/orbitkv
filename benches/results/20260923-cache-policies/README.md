# Cache policies: final pressure controls

H20, BF16 Qwen3-8B at revision
`b968826d9c46dd6066d109eabc6255188de91218`, TP=1, vLLM 0.29.0 and SGLang 0.5.20.
The Manager and harness sources are
`69334e027260b41bf34680f57ad8c147e1e8fd34`. Every configuration uses the same
release binary. The existing source-built Python extension is unchanged;
this change does not alter its client protocol.

`summary.csv` retains final aggregates only. Raw manifests, samples, timelines,
storage evidence and service logs remain in ignored
`benches/results/runs/20260923-policies-*` directories. Read the
[policy implementation and results](../../../docs/cache-policies.md) together
with the [preceding SSD allocation controls](../../../docs/ssd-performance.md).

## Workload

Each window uses fresh engine and Manager processes. Thirty-two alternating
4K/8K prefixes contain 27 GiB of nominal KV, exceeding 9 GiB GPU KV plus 4 GiB
Manager DRAM. SSD capacity is 64 GiB, query admission is 3 GiB, client
concurrency is eight, the prefill batch limit is 8K and requests generate
16 tokens. The reuse-selection probability is 75%; other requests create new
prefixes. Reuse selection is not proof of a cache hit.

The short comparison runs exactly 192 requests per configuration, three times,
with the middle repetition's order reversed. Its 180-second admission cutoff
is a safety limit, never reached. Each run contains the same 142 reuse choices
and 50 cold requests. The extended comparison runs exactly 768 requests once
per arm with a 600-second cutoff. It isolates write admission after more SSD
turnover; one extended pair per engine does not establish a production default.

Reference-prefix preparation is outside request timing. Its committed SSD
writes are recorded separately: `all` initially populates SSD while `reuse`
initially skips those pages. The CSV includes before/after write counters so
window-only traffic does not hide that difference. All runs use direct I/O on
an overlay filesystem; they do not qualify physical NVMe bandwidth.

| Configuration | Protected percent | SSD writes |
| --- | ---: | --- |
| control | 0 | all |
| protected | 80 | all |
| selective | 0 | reuse |
| combined | 80 | reuse |

The short matrix includes all four configurations. The extended comparison
uses `control` and `selective`, in that order. Automatic warming, consumer
preparation and read cutoffs are disabled; transfer tracing is enabled equally.

## Reproduce

Build the ordinary release Manager before starting any runtime:

```bash
cargo build --release -p orbitkv-server --bin orbitkv-cache-manager \
  --no-default-features --features cuda-13,mooncake
```

Run from the repository root with the matching engine environment:

```bash
ORBITKV_CACHE_MANAGER_BINARY="$PWD/target/release/orbitkv-cache-manager" \
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --workload sustained --lengths 4096 8192 --concurrencies 8 \
  --gpu-tokens 65536 --prefill-tokens 8192 --host-gib 4 --ssd-gib 64 \
  --query-budget-gib 3 --working-set 32 --output-tokens 16 \
  --duration-seconds 180 --max-requests 192 \
  --cache-protected-percent 0 --ssd-write-policy all \
  --read-batch-mib 0 --queue-warmup off --prepare-requests off \
  --trace-transfers --seed 20260923 \
  --output benches/results/runs/policies-vllm-control-1
```

Use a fresh output directory for every run. Repeat in the table's order,
reverse that order for repetition two, then repeat the first order. Use the
SGLang environment and `--engine sglang` for its matrix. For the extended
comparison, set `--max-requests 768 --duration-seconds 600` and run the first
and third configurations once per engine. Run all GPU workloads sequentially;
do not build or restage native libraries while they are mapped by a runtime.

## Aggregation

`python -m benches.report RUN... --output EMPTY_DIRECTORY` validates and
rebuilds individual reports. The retained CSV groups short runs by engine and
configuration and retains each extended run separately. Means and min/max are
over per-window measurements, including each window's latency quantiles;
they are not pooled request quantiles or confidence intervals. Byte rates are
window deltas per completed request. Counter totals sum repetitions, sampled
budget fields take the maximum, and final resource fields take the maximum
across runs. Admission skips and full-queue drops are separate counters.

`reuse_requests_without_hit` counts chosen reuse requests with zero cached
tokens. `reuse_partial_hit_requests` counts positive hits shorter than
`prompt_length - 64`, allowing each engine's final incomplete page. Output
differences are retained diagnostics from nondeterministic serving and are
separate from deterministic output and exact GPU-byte correctness gates.
