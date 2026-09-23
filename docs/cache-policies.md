# Retention and SSD write admission

OrbitKV provides two optional Rust policies for workloads larger than GPU and
host memory: protect demand-reused pages from scans, and avoid writing every
new page to SSD. They use the existing cache and backing-store owners; Python
adapters do not maintain another policy queue. Defaults remain
`--cache-protected-percent 0 --ssd-write-policy all` until matched serving
measurements justify a change.

## Protect reused pages

Set `--cache-protected-percent 80` to give the protected replacement segment a
maximum footprint of 80% of the pinned pool. This is a replacement preference,
not an additional memory reservation. The cache can still use the full pool.

With protection enabled, newly saved and ordinarily restored pages enter the
probationary segment. A foreground prefix lookup promotes matching pages into
the protected segment. Promotion demotes its oldest pages first when necessary
to stay within the byte limit. A page larger than the limit stays probationary.
Speculative warming and consumer preparation do not count as foreground demand.

Pressure eviction visits reclaimable replicas, probationary pages, then
protected pages, using recency within each class. It skips pages held by query
leases or transfers. Protection does not prevent eventual pressure eviction or
override GPU completion ownership. Promotion validates the resident allocation;
a stale reference cannot protect a replacement allocation for the same key.
Catalog-driven replica demotion and memory cleanup update the protected count.

The existing `--enable-lfu-admission` option is independent. Leave it disabled
when measuring this policy so admission and replacement effects are separable.

## Admit SSD writes selectively

`--ssd-write-policy all` writes newly saved pages asynchronously as before.
`--ssd-write-policy reuse` admits a page when either:

- a foreground query returns that page; or
- the same state key is published again within the bounded reuse history.

The second condition permits a recomputed page to reach SSD after its original
DRAM replica was evicted. The history holds 16,384 exact state keys, including
their namespace, and owns no payload memory. Expired history may treat an old
key as cold again. Repeated keys within one batch do not manufacture reuse.

Both policies skip already resident SSD keys and pending writes. Queued writes
hold weak source references; the writer acquires strong references when it
prepares a batch and retains them through I/O completion. Queue rejection and
dead source references leave later writes retryable. The existing writer still
holds the prepared batch: its staging footprint is not a new per-I/O byte cap.

Foreground lookup is evidence of interest, not proof that the engine eventually
consumed the page. Pure HBM hits and prepared-lease handoff do not provide an
additional reuse signal. The first reuse after DRAM eviction may need
recomputation under `reuse`, because no SSD replica was written on the first
publication. Selective writes can therefore reduce write traffic while making
latency or throughput worse. Save completion still confirms only GPU-to-DRAM
copy completion, not an SSD replica or restart durability.

## Configure and measure

For a capacity-pressure experiment:

```bash
orbitkv-cache-manager \
  --pool-size 4gb --query-budget 3gb \
  --ssd-cache-path /data/orbitkv/cache.bin --ssd-cache-capacity 64gb \
  --cache-protected-percent 80 --ssd-write-policy reuse
```

Keep model revision, engine release, GPU/host/SSD capacities, prompt sequence,
concurrency, prefill batching and tracing fixed. Compare these configurations
with warming and consumer preparation disabled:

| Control | Protected percent | SSD write policy |
| --- | ---: | --- |
| Existing replacement and writes | 0 | all |
| Retention only | 80 | all |
| Write admission only | 0 | reuse |
| Combined policies | 80 | reuse |

Use a working set larger than GPU plus host capacity. Repeat windows with order
reversal before recommending a default. Record TTFT, output-token throughput,
decode timing, SSD read/write bytes per request, allocation failures, write
drops and resource drain. Intentional admission skips are distinct from full
write-queue drops. The benchmark accepts both Manager flags and checks sampled
protected bytes against the configured cap.

The [metrics reference](metrics.md) includes protected bytes, promotions,
demotions and SSD admission skips. The [benchmark guide](../benches/README.md)
describes the serving harness; the [SSD capacity comparison](ssd-performance.md)
provides the preceding allocation and queue controls.

## Reference designs and scope

The probationary/protected split follows the established segmented-LRU idea
described in [FlexKV's replacement policies at `738ddc14`](https://github.com/taco-project/FlexKV/blob/738ddc141a198b4e20de6c5d1f0128e387f7fdb2/docs/eviction_policy/README_en.md).
OrbitKV bounds its protected segment by bytes and preserves existing transfer
leases and replica classes.

[SGLang v0.5.20's HiCache controller](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/managers/cache_controller.py)
separates write-through, selective write-through and write-back. OrbitKV applies
selective admission to its DRAM-to-SSD path; the tiers and ownership contract
are different. This is not an implementation of HiCache's entire policy stack.

[LMCache v0.5.5's prefetch controller](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/distributed/storage_controllers/prefetch_controller.py)
distinguishes demand-held results from unowned warming. OrbitKV keeps that
distinction in its existing query/lease lifecycle and does not count warming
as demand for these policies.

These are reuse heuristics, not model execution prediction. Compiled recovery
requirements still determine which pages, windows and checkpoints are legal
to restore. Cost-based restore-versus-recompute decisions, execution-time
forecasting and finer copy/compute overlap remain separate work in the
[roadmap](roadmap.md).
