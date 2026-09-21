# Qwen3-8B SSD restoration measurements

Measured September 21, 2026 at source commit `45caecfb`, with the same H20,
Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, vLLM 0.29.0,
SGLang 0.5.20, BF16, TP=1, and 64-token pages as the
[DRAM experiment](single-node-performance.md). The release Cache Manager has
16 GiB pinned DRAM and a 32 GiB SSD ring; GPU KV capacity is 16,384 tokens.

The cache file lives on the workspace overlay filesystem. The Manager's open
file descriptor was verified to use `O_DIRECT`; its SSD backend uses io_uring.
The host exposes Solidigm NVMe drives, but the container mount information does
not identify which physical drive backs the overlay. These are application
path measurements, not bare-device bandwidth or GPUDirect Storage results.

## Workload and evidence

Each engine served five independent prefixes at 1K, 4K and 8K, with 16 output
tokens and concurrency one. Each prefix has four measured requests:

1. Cold prefill.
2. Reuse while HBM is resident.
3. Reuse after two disjoint 12K requests evict the original GPU pages.
4. Apply fresh GPU pressure, wait for observed background writes to quiesce,
   remove Manager DRAM while preserving SSD, and request that prefix again.

Preparation and pressure traffic are excluded from request timings. Phase four
isolates the backing tier; it is not a natural host-capacity-pressure workload.
The random generator consumes additional pressure prompts compared with the
earlier three-phase benchmark, so compare phases within this run. SSD writes
remain enabled during the DRAM control phase. Services run sequentially, and
each cache payload file is removed after its Manager stops.

The [120 request measurements](../benches/results/qwen3-8b-ssd.csv),
[summary CSV](../benches/results/qwen3-8b-ssd-summary.csv), and
[manifests, storage evidence and summaries](../benches/results/qwen3-8b-ssd-summary.json)
are checked in. Full JSONL responses, cleanup responses, counter snapshots,
and logs are in `benches/results/runs/qwen3-8b-ssd-20260921/` on the measurement
host. Failed restores or speculative reads are not relabelled as hits.

## Client TTFT

Median milliseconds; five requests per cell:

| Engine | Measured path | 1K | 4K | 8K |
| --- | --- | ---: | ---: | ---: |
| vLLM | Cold prefill | 118.76 | 474.93 | 1,002.83 |
| vLLM | DRAM restore, SSD enabled | 23.78 | 37.07 | 57.71 |
| vLLM | SSD restore after DRAM eviction | 47.27 | 140.36 | 244.88 |
| SGLang | Cold prefill | 116.80 | 469.86 | 994.16 |
| SGLang | DRAM restore, SSD enabled | 32.13 | 42.17 | 56.80 |
| SGLang | After DRAM eviction: recomputation | 116.72 | 469.72 | 994.25 |

All 15 vLLM SSD-phase requests loaded the same number of bytes from SSD and
into the GPU: 144, 576, or 1,152 MiB per request. None had an HBM hit. SSD
restore was about 2.5x/3.4x/4.1x faster than cold prefill here, while taking
roughly 2.0x/3.8x/4.2x the DRAM restore time.

**SGLang had zero successful SSD restores in this experiment.** All 15 requests
read SSD data (135, 567, or 1,143 MiB per request) but loaded zero external
bytes into GPU memory and reported zero cached tokens. Its `lookup` returns an
empty match on `QueryLoading`, so the scheduler continues with prefill. The
last 64-token page is not part of SGLang's restore boundary for these prompts.
The vLLM connector can return an unresolved match to its scheduler and retry;
the SGLang linker does not currently have equivalent pending-query handling.

This is an integration readiness gap, not evidence that SSD copies are too
slow to help SGLang. Never advertise the SGLang recomputation row as an SSD
cache hit. An early-demand/readiness contract is required before its SSD and
other asynchronously fetched cache tiers can be considered qualified.

## Where the time goes

Median instrumented operation time per vLLM SSD-phase request, milliseconds:

| Prefix | SSD prefix prefetch | GPU load task | Client TTFT |
| --- | ---: | ---: | ---: |
| 1K | 23.54 | 4.66 | 47.27 |
| 4K | 99.58 | 18.48 | 140.36 |
| 8K | 182.79 | 36.83 | 244.88 |

SSD prefetch includes queueing, pinned allocation, reads, and reconstruction.
GPU load includes task construction, copies and synchronization. The remaining
client latency includes engine scheduling, remaining computation, and response
delivery; subtracting medians does not estimate any one of those stages.
Per-request read bytes divided by prefetch duration has a median of about
5.6-6.2 GiB/s; the analogous whole GPU-load task is about 30 GiB/s. These are
effective application-stage rates, not isolated SSD or PCIe bandwidth.

SGLang's unused prefetches took 24.26/92.66/186.79 ms. Even when those reads
finish well before cold prefill would finish, the current request does not
adopt the completed result. The first optimization is exposing readiness to
the scheduler, followed by starting reads earlier and overlapping layer loads.

At vLLM's final DRAM eviction, the SSD write counter had reached 112.32 GiB;
its 15 SSD restores totalled 9.14 GiB. Pressure requests intentionally have
little reuse, so this is not a production write-amplification estimate. It
does demonstrate why SSD admission should account for reuse and copy cost.
No write failures or write-queue drops were recorded at that checkpoint.

## Correctness and scope

All 15 vLLM SSD outputs match their corresponding DRAM outputs. One 4K prefix
has a different cold output; its HBM, DRAM and SSD outputs all agree with one
another. All 60 SGLang outputs match their respective cold outputs, including
the SSD-phase requests that recomputed. The runs did not enable deterministic
inference; these output observations are not a proof of byte integrity.

The separate GPU integration test exercises SSD write completion, DRAM
eviction, explicit polling to `QueryReady`, and restoration into poisoned GPU
destinations for both stored page layouts. Its explicit readiness polling is
not the current SGLang serving adapter and does not fix the gap above.

Five observations do not establish a tail-latency SLO. This experiment does
not measure concurrent goodput, sustained read/write contention, natural
host eviction, multi-disk scaling, restart durability, or competitor SSD paths.
The next experiments and implementation boundaries are in
[state demand and transfer planning](state-planning.md).

## Reproduce

Build the release wheel as described in [single-node setup](single-node.md).
From the repository root, run each engine sequentially:

```bash
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --ssd-gib 32 --output benches/results/runs/ssd-vllm

.venv/sglang-release/bin/python -m benches.single_node \
  --engine sglang --backend orbitkv --model /workspace/models/qwen3-8b \
  --ssd-gib 32 --output benches/results/runs/ssd-sglang

python -m benches.report benches/results/runs/ssd-vllm \
  benches/results/runs/ssd-sglang --output benches/results/runs/ssd-report
```

The GPU byte gate runs with the SGLang release environment:

```bash
cd python
../.venv/sglang-release/bin/python -m pytest -m integration \
  tests/integration/test_sglang_direct_transfer.py -k ssd
```
