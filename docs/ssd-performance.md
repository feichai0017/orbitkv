# Single-node offload under SSD pressure

The September 23, 2026 Qwen3-8B experiment keeps DRAM and SSD enabled together
and uses a working set larger than their in-memory capacity. It measures
request latency, transfer stages, bytes and cleanup under natural eviction.
The H20 development container exposes an `O_DIRECT`/io_uring cache file on an
overlay mount; these are application measurements, not a physical NVMe or PCIe
bandwidth limit.

## Workload and evidence

Both engines use the same BF16 model revision, TP=1, 64-token pages, concurrency
eight and 16 output tokens. Thirty-two alternating 4K/8K prefixes occupy a
nominal **27 GiB** of KV. The budgets are **9 GiB GPU KV, 4 GiB Manager DRAM,
64 GiB SSD and 3 GiB query reservations**. A request has a 75% chance of choosing
one of the prepared prefixes; other requests use fresh tokens. Choosing a
prefix does not guarantee a cache hit.

Each fresh service prepares its reference prefixes outside measurement,
admits requests for 60 seconds, then finishes every admitted request. The
10,000-request safety cap is not reached. No manual DRAM cleanup runs during
the window: reads and background writes compete while cold traffic adds new
state. Warming and owned preparation are off, read batching uses the default,
and transfer tracing is on. Throughput includes the admitted-request tail but
excludes initialization, reference preparation and the later resource-drain
check. It is not an arrival-rate SLO experiment.

The [final CSV](../benches/results/20260923-offload/summary.csv) and
[reproduction instructions](../benches/results/20260923-offload/README.md)
identify revisions, configuration and counters. Raw samples, manifests,
process-local timelines and service logs stay in ignored `benches/results/runs/`.
The baseline uses vLLM 0.29.0 and SGLang 0.5.20 with the same engine releases
and model for every candidate run.

## Changes being measured

- vLLM records a CUDA event on the producing stream after the forward launch,
  outside CUDA graph capture. The save worker waits for its producer events;
  unrelated later GPU work no longer extends a device-wide synchronization.
  Source pages remain owned through the native D2H completion.
- Rust routes SSD reads across the existing read workers and reserves stable
  queues for writes. A read need not wait for a write's submission queue to
  drain. Thread count and in-flight I/O limits are unchanged.
- SSD-backed Managers allocate saved and restored page segments independently
  on their NUMA node. Read and write allocation sizes agree, and one surviving
  page cannot retain an unrelated multi-page batch. Page-first layouts already
  have one full page per stored segment. DRAM-only allocation keeps its existing
  batching option. The old multi-page staging planner and raw-pointer
  reconstruction helpers are removed.
- Total and speculative query reservations use independent counters updated
  under the admission lock. Phase-labelled samples remain diagnostic and
  must not be summed to enforce a budget: a scrape can overlap a transition.

The allocation change addresses a failure observed in the large-working-set
control: a 256 MiB allocation could fail with more than that much aggregate
pool space available. The retained-page GPU test checks both saved and
SSD-restored pages, with contiguous and split K/V layouts. It keeps one page
alive, verifies that eviction frees its three sibling pages, then restores the
held page and compares its exact GPU bytes.

## Final comparison

Milliseconds for TTFT; generated tokens per second for throughput.

| Engine | Configuration | Requests | TTFT P50 | P95 | P99 | Output tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| vllm | Before, 8K | 227 | 1,145.45 | 3,133.77 | 4,014.06 | 59.35 |
| vllm | Queue/fence control, 8K | 223 | 1,181.32 | 3,233.63 | 4,018.17 | 58.23 |
| vllm | Current, 8K | 232 | 1,178.52 | 3,157.50 | 3,650.18 | 59.39 |
| vllm | Current, 4K | 216 | 1,195.28 | 2,674.29 | 3,666.07 | 56.71 |
| sglang | Before, 8K | 226 | 1,240.44 | 3,432.28 | 4,568.53 | 59.01 |
| sglang | Queue/fence control, 8K | 224 | 1,487.66 | 3,339.70 | 3,902.59 | 58.18 |
| sglang | Current, 8K | 226 | 1,615.55 | 3,591.38 | 4,773.39 | 57.83 |
| sglang | Current, 4K | 217 | 1,450.97 | 3,623.50 | 4,145.95 | 55.44 |

At the unchanged 8K prefill limit, vLLM throughput is essentially flat
(59.35 → 59.39); SGLang is 2.0% lower (59.01 → 57.83). SGLang P95 is
4.6% higher. These observations do not establish an overall throughput gain;
the retained control rows also show run-to-run tail variation. No speedup
claim or hardware-limit claim follows from this comparison.

All final configurations recorded zero pool-allocation and SSD-read failures.
The old vLLM allocation path recorded one failure in the baseline and six
in the queue/fence control. Independent reclamation has a deterministic GPU
gate; zero failures in a short workload do not prove that all capacity choices
can admit every save.

With 8K prefill, vLLM read **107.97 GiB** and wrote **47.21 GiB** through SSD;
SGLang read **100.76 GiB** and wrote **39.39 GiB**. These are window totals,
including the final drain. SGLang still dropped **625 blocks** from its SSD
write queue (baseline: 639), so write admission remains unfinished work.

For vLLM, 4K prefill lowers P95 by 15.3% while reducing throughput by 4.5%.
For SGLang it reduces throughput by 4.1% without improving P95. Keep 8K as the
benchmark default; a latency-oriented deployment can test vLLM's 4K option
against its own throughput target. The current 8K runs retain four vLLM
output differences and zero SGLang differences; 4K retains five and one
respectively. They remain visible in the CSV and require the separate
deterministic controls below.

## Interpreting the stages

`demand_prepare_ms` includes queued work, allocation, SSD reads and rebuilding.
`manager_restore_ms` includes dispatch, the GPU worker queue and synchronized
H2D work. `completion_delivery_ms` runs from the Manager's completed GPU task
until it answers the engine's terminal poll. These are overlapping,
process-local observations; adding their percentiles does not produce TTFT.
Dense lookup combines candidate discovery and preparation in this workload.

The initial queue/fence comparison, before page-granular staging, left both
engines near 59 output tokens/s. vLLM's completion-delivery P95 stayed near
0.94 seconds despite a Manager-restore P95 near 50 ms. The completion signal
itself took about 0.13 ms. This locates an exposed wait in engine consumption,
rather than establishing a slow notification transport or PCIe limit.

A separate 2K prefill diagnostic cut vLLM's delivery P95 to about 234 ms, but
reduced throughput from 59.2 to 56.0 output tokens/s. Shorter compute quanta
can improve one stage while adding scheduling/compute overhead. The final
4K prefill rows are configuration controls; compare only the 8K rows when
attributing a change to OrbitKV code. Neither engine's default is changed.

## Correctness and operational limits

Performance runs use ordinary greedy inference and retain every output
mismatch from the serial prefix references. They do not enable deterministic
inference, so these counts neither prove cache corruption nor establish exact
output equivalence. Separate native GPU-byte checks and deterministic serving
E2E gates cover restoration correctness. Cancellation, lost notification,
stalled Publish and restart retain their page-lifetime gates.

The final implementation passes 164 Rust unit tests (one ignored), 13 native
SSD tests, 12 fault-lifetime tests and both engines' GPU-byte gates. Qwen3-8B
serving passes six vLLM cases and both SGLang DRAM/SSD cases; vLLM's HMA-only
case is skipped for this dense model. The producer-fence gate verifies exact
restored bytes while unrelated GPU work is still running.

The final candidate checks the independent total-query counter against 3 GiB
and the speculative counter against one quarter of that limit. Query, copy,
SSD-read and SSD-write activity must drain after requests finish. The explicit
drain interval includes a fixed 1.2-second settle; it is not a last-page-release
latency. Sampling every 25 ms can miss peaks, so ownership tests remain required.
The baseline's phase-summed peak is retained in a separate CSV column and is
not used as an exact budget observation.

The CSV also retains allocation failures, SSD read failures and write-queue
rejections. `orbitkv_ssd_write_queue_full_total` counts dropped **blocks**.
A completed D2H save confirms a DRAM replica; SSD writes are asynchronous and
may be dropped under pressure. A failed allocation can make a request
recompute. Neither outcome should disappear from a performance report.

This is one short window per final configuration, not a repeated tail-latency
qualification, a capacity soak or a competitor ranking. It does not show that
single-node offload has reached its limit. Repeat paired runs with order
reversal before selecting defaults. The remaining measured work is engine
completion/admission overlap, retention and SSD write admission under mixed
traffic, and physical-device scaling with a controlled storage mount. Keep
those changes separate from multi-host/RDMA qualification.

## Query-readiness follow-up

The September 21 SSD-readiness experiment remains a separate historical gate:
after forced DRAM eviction, both engines restored **15/15** requests from SSD,
with positive and matching SSD-read/H2D byte counts. At 8K, median TTFT was
244.47 ms for vLLM and 268.04 ms for SGLang, versus cold prefill near one second.
The old SGLang integration's 0/15 result predates readiness-aware admission.
See the [immutable experiment and its controls](https://github.com/feichai0017/orbitkv/blob/4712f780c900120719f178b2ea36c9e0ac7c135f/docs/ssd-performance.md).
These serial forced-eviction numbers must not be mixed with the concurrent
natural-pressure measurements above.
