# Qwen3-8B concurrent query ownership baseline

For artifact locations and verification limits, see [benchmark evidence](benchmark-evidence.md).
The [pre-migration report](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/concurrent-performance.md) retains full tables and historical run details.

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

Each engine completed 234 measured requests in 54 bursts. Request rows, batch
counters, manifests and storage evidence remain in the
[historical dataset snapshot](https://github.com/feichai0017/orbitkv/tree/44c1e5f9a253aa7378c6187b2aeea9bff93df304/benches/results).
Counters belong to a whole burst and are never copied into each
request as independent evidence. Cached-token reports alone do not establish
which tier supplied an individual overlapping request.

## Observed latency and burst throughput

Client TTFT is time to the first nonempty streamed text. Median milliseconds
across all three lengths; each phase has 3/12/24 samples at concurrency 1/4/8.
Throughput divides requests by measured burst wall time, excluding preparation.

[Full historical measurement table](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/concurrent-performance.md)

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
The earlier report identifies the incomplete run as
`historical run label: query-budgets-sglang-failed-shared-prefix/` and is excluded
from latency summaries.

Ordinary performance runs retain **13 vLLM and 3 SGLang output differences**
from their cold-phase counterparts. All concern the same 8-concurrent 1K shared
prefix, alternating between repeated `A` and `and`. Greedy sampling alone does
not establish batch-invariant output. The additional
[archived output controls](https://github.com/feichai0017/orbitkv/tree/44c1e5f9a253aa7378c6187b2aeea9bff93df304/benches/results)
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
  --output /var/tmp/orbitkv-bench/query-budgets-sglang

.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --workload concurrent --concurrencies 1 4 8 --repeats 3 \
  --ssd-gib 32 --query-budget-gib 2 \
  --output /var/tmp/orbitkv-bench/query-budgets-vllm

python -m benches.report \
  /var/tmp/orbitkv-bench/query-budgets-vllm \
  /var/tmp/orbitkv-bench/query-budgets-sglang \
  --output /var/tmp/orbitkv-bench/query-budgets-report
```

Use empty external output directories and run one engine at a time. The fixed
Git snapshot above preserves tracked responses and controls; missing historical
raw logs are described in the evidence inventory. These measurements support byte-bounded restoration on this
workload. The subsequent [queued-warming implementation](queued-warming.md)
is outside these measurements. First-use deadlines, fair scheduling, and
calibrated restore-versus-recompute policy still require implementation and
independent measurements.
