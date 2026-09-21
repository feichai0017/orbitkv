# Qwen3-8B concurrent query ownership baseline

Measured September 21, 2026 at source commit `3a64513f`, after adding versioned
queries, byte admission, shared backing reads, and the SGLang shared-prefix
admission fix. Both measured runs had no uncommitted source diff. This is a
bounded burst baseline for further scheduling work, not a before/after speedup
claim or a steady-state tail-latency qualification.

## Configuration and workload

- One NVIDIA H20, Qwen3-8B revision
  `b968826d9c46dd6066d109eabc6255188de91218`, BF16, TP=1, 64-token pages.
- vLLM 0.29.0 and SGLang 0.5.20 in separate environments; services run sequentially.
- GPU KV capacity: 16,384 tokens. Manager pinned pool: 16 GiB. SSD ring: 32 GiB.
- Global query ownership budget: 2 GiB; per-instance budget defaults to that limit.
  A single 8K demand fits, while several concurrent long demands exceed it.
- SSD descriptors verified `O_DIRECT`; io_uring on the workspace overlay mount.
  Device inventory cannot identify the physical backing drive of that mount.
- Concurrency 1, 4, and 8; three bursts per pattern and phase; 16 output tokens.
  Shared bursts use one prefix, cycling 1K/4K/8K over the three repetitions.
  Mixed bursts use independent prefixes with rotating 1K/4K/8K lengths.

Each burst is measured cold, after two disjoint 12K GPU-pressure requests, and
then after fresh GPU pressure plus explicit Manager DRAM eviction. Writes are
observed idle before DRAM eviction. Preparation is outside request timing;
scheduler waiting is inside it. The host pool is deliberately cleared, so this
is not a natural host-pressure workload. Shared cold bursts may reuse another
request's newly computed HBM pages before the burst finishes.

Each engine completed 234 measured requests in 54 bursts. The
[request CSV](../benches/results/qwen3-8b-query-budgets.csv),
[batch counters](../benches/results/qwen3-8b-query-budgets-batches.csv),
[summary CSV](../benches/results/qwen3-8b-query-budgets-summary.csv), and
[manifests, storage evidence, and summaries](../benches/results/qwen3-8b-query-budgets-summary.json)
are retained. Counters belong to a whole burst and are never copied into each
request as independent evidence. Cached-token reports alone do not establish
which tier supplied an individual overlapping request.

## Observed latency and burst throughput

Client TTFT is time to the first nonempty streamed text. Median milliseconds
across all three lengths; each phase has 3/12/24 samples at concurrency 1/4/8.
Throughput divides requests by measured burst wall time, excluding preparation.

| Engine | Pattern | Concurrency | Cold TTFT | After GPU pressure | After DRAM eviction | SSD-phase requests/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| vLLM | Shared | 1 | 474.63 | 42.48 | 129.78 | 4.17 |
| vLLM | Shared | 4 | 521.95 | 78.92 | 182.79 | 13.01 |
| vLLM | Shared | 8 | 556.24 | 109.35 | 216.50 | 22.35 |
| vLLM | Mixed | 1 | 474.43 | 38.38 | 156.81 | 4.10 |
| vLLM | Mixed | 4 | 1,634.16 | 99.48 | 295.25 | 7.02 |
| vLLM | Mixed | 8 | 2,364.82 | 229.01 | 383.11 | 7.92 |
| SGLang | Shared | 1 | 473.99 | 43.42 | 162.39 | 3.84 |
| SGLang | Shared | 4 | 1,147.81 | 78.55 | 164.74 | 13.02 |
| SGLang | Shared | 8 | 1,188.35 | 118.78 | 214.27 | 21.15 |
| SGLang | Mixed | 1 | 473.39 | 41.95 | 170.18 | 3.75 |
| SGLang | Mixed | 4 | 1,622.90 | 109.76 | 381.46 | 6.68 |
| SGLang | Mixed | 8 | 3,245.89 | 322.62 | 398.28 | 7.40 |

The machine-readable summaries also contain descriptive p95/p99, end-to-end
latency, output-token throughput, and client decode milliseconds per token.
These are small, correlated burst samples; they do not establish production
P99, sustained goodput, fairness, or isolated inter-token latency. The decode
value is `(end_to_end - TTFT) / (output_tokens - 1)`, not a token-arrival trace.
There is no concurrent native-engine/LMCache/FlexKV latency comparison here.

## Resource bounds and sharing

| Engine | Sampled query peak | Sampled occupied-pool peak | Budget-wait attempts | Coalesced reads | Oversize bypasses | Bursts retaining query bytes after settling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM | 2,016 MiB | 13.641 GiB | 3,720 | 14 | 0 | 0/54 |
| SGLang | 1,980 MiB | 13.641 GiB | 3,102 | 14 | 0 | 0/54 |

All 18 forced-SSD bursts per engine recorded positive SSD reads and GPU loads.
No measured burst recorded an SSD read/write failure. Ownership includes
preparing pages, ready leases, and GPU consumers; it returned to zero after
every burst. Budget waits count failed reservation attempts, including repeated
polls, not distinct requests. Shared pages are charged conservatively per owner.

Memory metrics were sampled every 25 ms. Their peaks are lower bounds, not
proofs of a hard limit. Separate Rust reservation/lease tests and native-client
integration tests check the hard global/per-instance accounting and cleanup.
The occupied-pool gauge is physical payload occupancy; it is neither process
RSS nor the total pinned allocation, which remains 16 GiB.

For the 8-concurrent 1K shared-prefix SSD burst:

| Engine | SSD read | H2D load | Joined backing reads |
| --- | ---: | ---: | ---: |
| vLLM | 144 MiB | 1,152 MiB | 7 |
| SGLang | 135 MiB | 135 MiB | 7 |

Backing-read sharing does not automatically share engine GPU destinations.
vLLM restored eight destinations here; SGLang subsequently reused its restored
radix prefix. SGLang's restorable prefix excludes the last 64-token page for
these aligned prompts. Reducing vLLM's duplicate H2D traffic is a useful next
profiling target, but any destination sharing must respect engine allocation,
mutable tails, and completion ownership.

## Correctness findings and controls

The first SGLang run at `6f2dd41c` stopped at 4-concurrent shared-prefix SSD
recovery. One request restored the prefix, while others kept stale pending
queries after that prefix became an HBM hit. The old admission check missed
the aligned last-page boundary; its timeout also depended on another lookup.

Admission now uses the pinned engine's actual maximum match boundary, including
logprob limits, cancels unused queries and ready leases, and observes expiry
without requiring another lookup. Controlled admission regressions pass. The
GPU E2E now restores four identical page-aligned requests simultaneously after
engine restart, for both DRAM and SSD, against a deterministic cold control.
The incomplete run remains in
`benches/results/runs/query-budgets-sglang-failed-shared-prefix/` and is excluded
from latency summaries.

Ordinary performance runs retain **13 vLLM and 3 SGLang output differences**
from their cold-phase counterparts. All concern the same 8-concurrent 1K shared
prefix, alternating between repeated `A` and `and`. Greedy sampling alone does
not establish batch-invariant output. The additional
[output controls](../benches/results/qwen3-8b-query-budgets-output-controls.json)
retain that exact input and all observed responses.

Both native engines reproduced the same two outputs while varying cold and
warm batch sizes (43 diagnostic requests per engine, without OrbitKV). With
SGLang deterministic inference or vLLM batch invariance plus FLASH_ATTN, all
40 OrbitKV requests per engine produced one consistent output: 8 cold, 8 DRAM,
and 24 forced-SSD requests. Every restored burst recorded GPU-load bytes; each
SSD burst also recorded SSD-read bytes. These controls demonstrate the output
variation without OrbitKV and support a batching explanation for this prefix.
They do not make ordinary serving universally deterministic. Control timings
are excluded from the performance table.

The original mismatches remain in the published performance data. The
correctness gates additionally include exact GPU-byte restoration, cancellation
of one owner of a shared SSD read, delivered-lease cleanup on disconnect,
versioned-query retirement, and multi-consumer reservation lifetime. Multi-rank
serving, sustained contention, and delivery-loss/restart fault coverage remain
separate work.

## Reproduce

Build the release wheel and use the isolated release environments from the
[single-node guide](single-node.md). From the repository root:

```bash
.venv/sglang-release/bin/python -m benches.single_node \
  --engine sglang --backend orbitkv --model /workspace/models/qwen3-8b \
  --workload concurrent --concurrencies 1 4 8 --repeats 3 \
  --ssd-gib 32 --query-budget-gib 2 \
  --output benches/results/runs/query-budgets-sglang

.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --workload concurrent --concurrencies 1 4 8 --repeats 3 \
  --ssd-gib 32 --query-budget-gib 2 \
  --output benches/results/runs/query-budgets-vllm

python -m benches.report \
  benches/results/runs/query-budgets-vllm \
  benches/results/runs/query-budgets-sglang \
  --output benches/results/runs/query-budgets-report
```

Use empty output directories and run one engine at a time. Raw responses,
metrics, logs and control scripts remain under `benches/results/runs/` on the
measurement host. These measurements support byte-bounded restoration on this
workload. The subsequent [queued-warming implementation](queued-warming.md)
is outside these measurements. First-use deadlines, fair scheduling, and
calibrated restore-versus-recompute policy still require implementation and
independent measurements.
