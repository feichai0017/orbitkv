# Single-node offload under SSD pressure

For artifact locations and verification limits, see [benchmark evidence](benchmark-evidence.md).
The [pre-migration report](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/ssd-performance.md) retains full tables and historical run details.

## Native SSD host restore batching (2026-10-07)

**Implemented; native median and installed-engine local gates pass; independent review open.**
Production commit `50c6d58d` groups immutable io_uring restore sources by SSD
store, with at most 16 unique keys per reader batch. Repeated layer references
share one generation; different generations of the same key remain in separate
batches. All reads reach terminal completion before any GPU source is replaced
or an error returns. Existing query, extent and destination ownership is retained.
The change reduces batch admission/completion work, not physical READV operations.
It is consumed by the managed SSD host-restore lane, including the explicit
`--ssd-read-path uring` demand route tested here. Default host-prefetch queries
keep their existing batch reader; this is not a claimed speedup for every SSD
lookup. It changes no GPU backend default, cache policy or readiness boundary.

A frozen baseline/candidate cohort uses complete CUDA 13 wheels on one A100 SM80,
with `O_DIRECT` files on the `/tmp` ext4 NVMe mount. Twelve cells cover five
shapes, each with 20 warm-ups and 100 measured restores; three independent pairs
reverse order. They include 4 KiB blocks and representative 36-layer split K/V
layouts of 9 and 108 MiB. These layouts are not a captured engine histogram.
The client timer wraps native `start_restore` through `wait_restore` return,
including Python call/return overhead, host SSD materialization and H2D. It is
not a GPU-only or instrumented Rust-only timer; preparation queries and byte
checks are outside.

All 15 per-shape paired median ratios pass the predeclared 1.05 guard; the largest
is 1.0154. Twelve-block 4 KiB restores use one reader batch instead of twelve,
with a geometric mean median ratio of 0.645. Thirty-four-block restores use three
batches instead of 34, with ratio 0.903. The 9 MiB layout stays near parity.
The 108 MiB median ratio is 0.971, but its descriptive pair-bootstrap interval
[0.947, 1.011] crosses parity, so a stable model-shaped improvement is not
established. Every measured destination is filled with a sentinel first and
compared byte-for-byte; physical SSD reads equal completed H2D bytes. Managers
exit zero and pinned/read/write ownership counters drain. These are client
restore measurements, not TTFT, ITL, throughput or device-bandwidth qualification.

Only three independent pairs are available per shape. All samples and pair
bootstrap intervals remain in the external analysis; p99 at 100 samples is
descriptive. One 34-block paired p99 rises from 0.738 to 2.486 ms (3.37x), and one
12-block pair rises from 0.639 to 1.101 ms (1.72x). These tails remain visible;
the median guard does not qualify tail performance or inference contention.
Earlier manifest/controller errors, asynchronous-write preparation failure and
temporary-file provenance loss are recorded separately. The final cohort uses
one corrected harness throughout and retains raw evidence in `/workspace`,
without mixing cells from older attempts. `/cache/sync` waits for insertion,
not asynchronous SSD writes; the benchmark waits for SSD write/query ownership
before evicting DRAM. SSD data uses the same ext4/NVMe mount via a separate fresh
`--ssd-cache-path`, which cannot overwrite an existing path.

Reproduce with a prebuilt installed wheel and a fresh external directory:

```bash
/path/to/release/python -m benches.communication \
  --manager /path/to/installed/orbitkv-cache-manager-py \
  --tier ssd --ssd-cache-path /mnt/nvme/orbitkv-bench/restore-001 \
  --layout split --layers 36 --block-bytes 262144 \
  --payload-bytes 262144 3145728 --iterations 100 --warmup 20 \
  --repeats 1 --idle-ms 0 --idle-seconds 0.1 \
  --output /var/tmp/orbitkv-ssd/restore-001
```

The benchmark forces io_uring and an SSD-only source for every restore. Its
Manager, extension, shared libraries, commands, mount, counters and byte oracles
are recorded. Use the same harness, budgets and workload for both wheels. Current
raw results and the promotion contract live outside Git at
`/root/orbitkv-artifacts/s5-ssd-host-batching-20261007/`, with the new A100
cohort at `/workspace/orbitkv-s5-ssd-host-batching-20261007/`. Evidence and
SSD payload paths are separate. The first passing cohort lost part of its
raw evidence during unexpected `/tmp` cleanup; its aggregate observations
and surviving files are retained, but do not qualify independent acceptance.

The complete CUDA 13 wheel `8acd6a06…` also passes three fresh official-engine
SSD gates on A100: vLLM 0.31.0 V1 and SGLang 0.5.21 ordinary cold/HBM/full/partial
cache flows, plus shared-Manager serving, engine restart and drained Manager
cold rebuild. Each full/partial restore loads 113,246,208 bytes; concurrent
external restore loads 226,492,416 bytes with matching physical io_uring reads
and four reader batches. All 20 services exit zero without forced cleanup;
installed engine/OrbitKV files are unchanged and GPU postflight is empty. These
functional gates use `direct`, dense Qwen3-8B TP=1/PP=1 eager, with cache files
on the `/workspace` overlay; they do not measure NVMe throughput or close
sustained contention, graphs, H20, native P/D faults or S3 crash reclamation.

## Next SSD work and native GDS boundary

[LMCache's local disk design](https://docs.lmcache.ai/kv_cache/storage_backends/local_storage.html)
uses asynchronous writes and prioritizes prefetch over deletes and puts.
[FlexKV's current configuration reference](https://github.com/taco-project/FlexKV/blob/main/docs/flexkv_config_reference/README_en.md)
describes coalescing small scattered SSD I/O. These are design references inspected
on 2026-10-07, not matched performance controls or qualified OrbitKV behavior.
OrbitKV already has bounded read/write owners; extend those owners after measuring
mixed save/query pressure, per-instance progress and write admission.

Physical I/O merging is separate from host-reader batching. A future candidate
must use leased adjacent extents in the same file, bounded bytes/iovec counts,
correct short-read/error handling and validation for every generation. It must
avoid unrequested gaps, ring-wrap aliasing and extra CPU repacking. Multi-device
scaling needs actual device topology and matched budgets.

The existing cuFile path uses registered GPU staging before scatter/decode.
[NVIDIA's guide](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html)
distinguishes small batch I/O from stream-ordered operations and recommends
reusing registered staging where registration can be amortized. It also notes
higher execution latency for small stream-ordered I/O. GDS is therefore a measured
route candidate, not an automatic speedup. Use [the native-path gate](gds.md)
with compatibility disabled and physical native I/O statistics; library presence
or successful initialization is insufficient. Direct engine-page I/O and new
scatter/checksum kernels remain unqualified follow-ups in S4.

## Historical concurrent inference evidence

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

The [final CSV](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/benches/results/20260923-offload/summary.csv) and
[reproduction instructions](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/benches/results/20260923-offload/README.md)
identify revisions, configuration and counters. Raw samples, manifests,
process-local timelines and service logs stay in external artifact directories.
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

[Full historical measurement table](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/ssd-performance.md)

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
H2D work. In these recorded runs, `completion_delivery_ms` ran from the Manager's
completed GPU task until it answered the engine's terminal poll. The shared-memory
completion protocol has since removed that RPC and its metric; current reports
use the engine's `restore_ms` interval and Manager `completion_signal_ms`.
Reproduce the recorded reports with their original revision. These are
overlapping, process-local observations; adding their percentiles does not produce TTFT.
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
