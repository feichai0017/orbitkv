# Preparing queued requests

vLLM 0.29.0 and SGLang 0.5.20 can announce an exact missing prefix when a
request enters their ordinary serving queue. OrbitKV uses that interval to
prepare DRAM pages from SSD or a peer. This is the first P3 implementation;
it does not predict future prompts or decide engine admission.

## Request lifecycle

```mermaid
sequenceDiagram
    participant E as Engine queue
    participant M as Cache Manager
    participant B as SSD / peer
    E->>E: Inspect existing HBM prefix
    E->>M: Warm missing page hashes
    M->>M: Try bounded warmup admission
    M->>B: Shared backing read
    B-->>M: Pages ready in DRAM
    M->>M: Release warmup reservation; keep pages evictable
    E->>M: Query current demand at scheduling time
    M-->>E: Revalidated prefix and restore lease
    E->>M: Restore into engine-owned GPU pages
    M-->>E: GPU completion
    E->>E: Consume restored KV
```

The vLLM connector uses `on_new_request` and its existing block hashes. It
checks the bound GPU block pool without allocating or pinning pages. Whole-page
warmup includes the final full page even when admission later recomputes its
last token; this lets identical demand queries join the same backing read.
Hybrid cache groups are excluded from this initial enqueue path.

SGLang uses its plugin hook around `_add_request_to_queue`. Only accepted
ordinary-queue requests are warmed. The adapter matches HBM with `req=None`,
then uses the linker's existing tail-hash calculation, preserving `extra_key`,
`cache_salt`, page boundaries and the logprob recovery limit. It does not create
an external hit marker or allocate GPU destinations. Positional embedding
overrides and disaggregated bootstrap queues are excluded.

## Capacity and cancellation

Warmups reserve their registered, padded page footprint against the existing
global and per-instance query budgets. Each warmup class is capped at **one
quarter** of both budgets. They also have limits of 16 active operations per
session and 128 globally, within the existing 128/1024 total limits. Byte or
operation pressure skips the hint immediately; an oversized hint is skipped
as a whole. A new hint also skips while any foreground query owns preparing,
ready or restoring bytes. Already-submitted reads still drain; this guard is
conservative admission, not preemption or a calibrated deadline scheduler.
These caps reserve accounting headroom, not a bandwidth or latency
guarantee for foreground reads.

A warmup never returns a hit promise or restore lease. Its reservation ends
when preparation completes, even if the engine never polls again. Prepared
pages enter the existing reclaimable cache class, which pressure evicts before
retained pages. Warmup hits use a non-mutating peek: they do not refresh existing
pages' recency or frequency. Foreground demand promotes the matching warmed
page generation to the retained class; a query/lease alone does not count as
successful use in the metrics below. They can disappear
before use. Ordinary admission always revalidates the current hashes, obtains
an independent lease and retains the normal GPU ownership rules.

Each client retains at most 16 pending hint tickets. Hints expire after five
seconds at the Manager; the client retires stale tickets when submitting
further hints. Admission or cancellation retires the matching hint before a
fresh demand operation. An already-submitted read drains under its original
reservation and operation permit. It is not aborted when interest disappears.
The ticket timeout does not expire resident pages; normal cache eviction does.
The existing backing-read coalescer can serve independent warmup/demand owners;
only identical read plans coalesce.

Set `ORBITKV_QUEUE_WARMUP=1` in the **engine** environment to enable automatic
enqueue warming. It is experimental and disabled by default: the initial
pressure controls did not improve throughput and increased SSD reads.
Normal query/prefetch and restore remain available with warming disabled.
The explicit `CacheManagerClient.warm_prefix()` API always attempts the supplied
hint; the environment switch controls automatic engine enqueue hooks only.
Transport failures in this optional enqueue hint are logged by the adapters
without raising out of the already-accepted request's queue callback.
Manager and Python extension must both use channel ABI 5; ABI 4 is not supported.

## Observing the path

Set `ORBITKV_TRACE_TRANSFERS=1` for both Manager and engine. Their logs emit
`cache_timeline` JSON records containing request IDs, process IDs, stage and
timestamps, without tokens or cache keys:

| Stage | Observation |
| --- | --- |
| `queued` | Request accepted by the engine queue |
| `read_start` | Manager begins the admitted preparation operation |
| `host_ready` | Manager preparation returns; includes hit count, warmup flag and local elapsed microseconds |
| `restore_submit` | Adapter submits H2D for engine-owned destinations |
| `gpu_ready` | Adapter observes successful transfer completion |
| `first_use` | vLLM schedules the first compute step; SGLang passes the first-layer wait for a restored batch |

`first_use` is an engine callback observation, not a CUDA kernel timestamp.
SGLang records it for restored requests; cold/HBM-only requests have no external
layer wait. Python monotonic timestamps may be compared within one process.
Manager `elapsed_us` measures its own preparation interval. Wall timestamps
help inspect logs but are not a cross-host deadline or latency clock.

`orbitkv_query_reserved_bytes{phase="warming"}` separates warmup reservations
from preparing/ready/restoring demand. A sampled peak is a lower bound. Existing
cache-tier query counters include warmup probes; use actual SSD/TE/H2D byte
counters to establish transfers rather than interpreting probe counts as
request hits.

## Measuring whether preparation was useful

The following counters follow the actual `SealedBlock` allocation, not a hash
or request ID. The owner that starts a shared read determines its origin:
joining a demand-started read or hitting existing DRAM does not create warmup
bytes. Joining a warmup-started read does not count them again. Rereading an
evicted key creates a new cohort.

| Metric | Meaning |
| --- | --- |
| `orbitkv_warmup_prepared_bytes_total` | Page footprints returned by warmup-started backing reads |
| `orbitkv_warmup_restored_bytes_total` | Those footprints contributing to at least one successfully completed local H2D, once per physical page |
| `orbitkv_warmup_unused_bytes_total` | Those footprints released by the last owner before any successful local H2D |
| `orbitkv_warmup_pending_bytes` | Live footprints still awaiting a successful local H2D; includes cache, query and transfer owners |
| `orbitkv_warmup_wait_byte_seconds_total{outcome="restored\|unused"}` | Footprint times time from host readiness to first successful H2D or final unused release |
| `orbitkv_warmup_foreground_skips_total` | Hints rejected while foreground query ownership is active |

At quiescence, prepared bytes equal restored + unused + pending bytes. Query
success, lease creation and cancellation do not resolve a pending page. The
last-reference rule prevents an eviction from classifying a still-running
transfer as unused. Multiple layer/rank copies credit a page only once, after
GPU synchronization succeeds.

These are **page-footprint** counters. A partial-layer/rank transfer credits
the contributing page footprint; use `orbitkv_load_bytes_total` for actual H2D
bytes. A successful transfer does not prove the engine later consumed it, or
that warming saved latency. Remote serving alone is not local H2D use. Physical
pool usage can also include allocation padding and shared slab retention.

Byte-seconds settle only when an outcome occurs; live intervals are excluded.
Benchmark windows retain starting/ending pending bytes and a sampled peak, so
live pages and carry-in are visible instead of being called waste. A window's
restored/prepared ratio is not a cohort success rate when it includes carry-in.
The operation's quarter-budget limit ends at read completion; it does **not**
cap all prepared resident pages. Their replacement priority and the pinned
pool's physical limit govern subsequent retention.

Pressure reclaim stops each batch once the allocation's requested footprint
has been selected (still at most 512 pages), releases those references, then
checks the allocator's actual largest contiguous free region. Fragmentation or
shared slabs can require another batch. The previous count-only batch could
discard every eligible page in a small pool: a Qwen3-8B page in these controls
is 9 MiB, so a 4 GiB pool holds fewer than 512 pages. Byte-bounded reclaim applies
to ordinary demand and writes as well as warming; its effect must be separated
from the warming on/off comparison.

## Initial pressure controls

Measured September 21, 2026 at source `2c1b27e3`, using Qwen3-8B on one H20.
Both modes use the same build and tracing settings. Each engine runs a 30-second
admission window at concurrency 8, with 4,096 input tokens, 16 output tokens,
8,192 GPU-cache tokens, 4 GiB of host cache, a 3 GiB query budget and 16 GiB SSD.
Twelve reusable prefixes occupy 6.75 GiB of KV payload; the reuse probability
is 0.75. Admitted requests drain after the window, and throughput includes that
time. The SSD file uses `O_DIRECT`/`io_uring` on an overlay mount; this is not a
physical-NVMe qualification.

| Engine | Warmup | Requests | Requests/s | TTFT P50 / P95 (ms) | SSD read MiB/request |
| --- | --- | ---: | ---: | ---: | ---: |
| vLLM | Off | 126 | 3.933 | 2126 / 2750 | 254.8 |
| vLLM | On | 128 | 3.907 | 2050 / 2685 | 304.7 |
| SGLang | Off | 116 | 3.679 | 1997 / 3171 | 265.7 |
| SGLang | On | 116 | 3.606 | 2054 / 3179 | 427.2 |

The experiment demonstrates bounded preparation, not a throughput improvement.
All four runs drained query reservations to zero. Sampled warmup peaks were
576 MiB for vLLM and 567 MiB for SGLang, below the 768 MiB warmup limit.
SSD bytes per request increased about 20% and 61%, respectively. The small
vLLM latency changes are not evidence of a general gain from one trial. Automatic
warming therefore remains experimental and disabled by default. Useful-byte
and retention accounting, followed by better hint admission, is the next gate.

vLLM recorded no prepared-reference output differences. Both SGLang modes
recorded nine differences, all for prefix 11. A separate native SGLang probe
reproduced the same split: cold computation matched the prepared reference;
an immediate 4,032-token HBM hit matched all nine differing cached OrbitKV
responses. The one uncached OrbitKV response for that prefix matched native
cold computation. This reproduces the difference without OrbitKV and does not
establish batch-invariant output equality. Exact GPU-byte and engine recovery
gates remain separate correctness evidence.

See the [summaries and complete window counters](../benches/results/qwen3-8b-queued-warming-summary.json),
[CSV](../benches/results/qwen3-8b-queued-warming-summary.csv), and
[native output control, inputs and reproduction script](../benches/results/qwen3-8b-queued-warming-output-control.json).
Raw samples and logs remain under `benches/results/runs/queued-warming-*` on the
measurement host. Initial SGLang logs repeat the first-layer wait callback;
the collector pairs each enqueue with its first callback, and the adapter now
emits `first_use` once per restored batch.

Reproduce an experimental window from the repository root:

```bash
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --workload sustained --concurrencies 8 --duration-seconds 30 \
  --max-requests 1000 --working-set 12 --reuse-ratio 0.75 \
  --lengths 4096 --gpu-tokens 8192 --output-tokens 16 \
  --host-gib 4 --query-budget-gib 3 --ssd-gib 16 \
  --queue-warmup on --trace-transfers \
  --output benches/results/runs/queued-warming-vllm-on
```

Use a fresh output directory for each run; switch `on` to `off` for the control.
Use `.venv/sglang-release/bin/python` and `--engine sglang` for SGLang.

## Qualification and remaining P3 work

CPU gates cover foreground budget headroom, operation pressure, unpolled
completion cleanup, client cancellation and fresh demand tickets. The SGLang
pinned-release hook test checks salted keys, logprob boundaries and HBM skipping.
The GPU integration gate warms real SSD pages without polling, waits for query
bytes to return to zero, leases them through an immediate DRAM query and checks
exact restored GPU contents for both supported stored layouts.
For changes to enqueue warming, run both engine E2E commands from the
[test gates](../python/tests/README.md) with `ORBITKV_QUEUE_WARMUP=1`.

Use the [benchmark harness](../benches/README.md) with a working set exceeding
HBM and host cache. Compare `--queue-warmup on` and `off` at equal capacities;
`--trace-transfers` enables timeline capture. A quiet serial server may see no
benefit, and an overloaded backend may perform extra reads.

P3 remains open for calibrated priority/deadline hints, per-device/staging
reservations, engine-consumption-level usefulness and admission calibration,
and delayed-read/reordering/cancellation qualification under sustained serving.
The fixed warmup share is an initial admission policy, not a cost-aware scheduler.
Restore-versus-recompute decisions remain P4. See the
[implementation sequence](state-planning.md#implementation-sequence).
