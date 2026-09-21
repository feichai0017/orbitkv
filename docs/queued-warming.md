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
as a whole. These caps reserve accounting headroom, not a bandwidth or latency
guarantee for foreground reads.

A warmup never returns a hit promise or restore lease. Its reservation ends
when preparation completes, even if the engine never polls again. Prepared
pages enter the existing bounded, evictable read cache. They can disappear
before use. Ordinary admission always revalidates the current hashes, obtains
an independent lease and retains the normal GPU ownership rules.

Each client retains at most 16 pending hint tickets. Hints expire after five
seconds at the Manager; the client retires stale tickets when submitting
further hints. Admission or cancellation retires the matching hint before a
fresh demand operation. An already-submitted read drains under its original
reservation and operation permit. It is not aborted when interest disappears.
The existing backing-read coalescer can serve independent warmup/demand owners;
only identical read plans coalesce.

Set `ORBITKV_QUEUE_WARMUP=0` in the **engine** environment for a control run.
Normal query/prefetch and restore remain available. Warmup is enabled by default.
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

## Qualification and remaining P3 work

CPU gates cover foreground budget headroom, operation pressure, unpolled
completion cleanup, client cancellation and fresh demand tickets. The SGLang
pinned-release hook test checks salted keys, logprob boundaries and HBM skipping.
The GPU integration gate warms real SSD pages without polling, waits for query
bytes to return to zero, leases them through an immediate DRAM query and checks
exact restored GPU contents for both supported stored layouts.

Use the [benchmark harness](../benches/README.md) with a working set exceeding
HBM and host cache. Compare `--queue-warmup on` and `off` at equal capacities;
`--trace-transfers` enables timeline capture. A quiet serial server may see no
benefit, and an overloaded backend may perform extra reads.

P3 remains open for calibrated priority/deadline hints, per-device/staging
reservations, useful-prefetch-byte and unused-retained-byte-second accounting,
and delayed-read/reordering/cancellation qualification under sustained serving.
The fixed warmup share is an initial admission policy, not a cost-aware scheduler.
Restore-versus-recompute decisions remain P4. See the
[implementation sequence](state-planning.md#implementation-sequence).
