# Qwen3-8B SSD restoration measurements

The original experiment below records the pre-admission implementation. The
[query-readiness follow-up](#query-readiness-follow-up) describes the current
serving path; keep the baseline results when comparing revisions.
The later [concurrent baseline](concurrent-performance.md) measures versioned
queries and byte admission with shared and mixed 1/4/8-request bursts.

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

The 120 request measurements, storage evidence and summaries are in the
[historical dataset snapshot](https://github.com/feichai0017/orbitkv/tree/44c1e5f9a253aa7378c6187b2aeea9bff93df304/benches/results).
Full JSONL responses, cleanup responses, counter snapshots,
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
the baseline SGLang linker had no equivalent pending-query handling.

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
destinations for both stored page layouts. Its explicit readiness polling was
separate from the baseline serving adapter. The subsequent serving gate now
checks forced-SSD recovery through the plugin admission hook; see the follow-up
results below.

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

## Query-readiness follow-up

The SGLang 0.5.20 plugin now uses `HookRegistry` admission to keep a pending
request queued and consume its ready lease on a later prefix match. Its
five-second preparation budget allows recomputation before GPU submission;
GPU restores still require a confirmed completion.

Both DRAM and forced-SSD serving recovery pass with Qwen3-8B at TP=1 across
engine restart. The SSD gate checks positive disk-read and H2D byte counters,
cached tokens, and equal deterministic output against a cold identity. Real
GPU-buffer tests separately verify cancellation and disconnect during SSD
reads leave no unconsumed lease, and validate exact restored bytes for both
stored page layouts. Controlled admission tests cover delayed completion and
other-request progress; they do not qualify concurrent goodput or multi-rank
serving. The old 0/15 SGLang measurement remains a baseline, not a description
of the new serving path.

The repeat experiment on September 21, 2026 used source commit `e3c819a8`,
with a clean source tree recorded at launch and the same model, engine releases,
budgets, seed, and four-phase workload described above. Both engines completed
all 60 requests. Median client TTFT in milliseconds, five requests per cell:

| Engine | Measured path | 1K | 4K | 8K |
| --- | --- | ---: | ---: | ---: |
| vLLM | Cold prefill | 118.70 | 476.35 | 1,006.73 |
| vLLM | DRAM restore, SSD enabled | 23.79 | 36.69 | 56.79 |
| vLLM | SSD restore after DRAM eviction | 55.24 | 132.05 | 244.47 |
| SGLang | Cold prefill | 117.90 | 473.41 | 1,000.45 |
| SGLang | DRAM restore, SSD enabled | 33.07 | 43.16 | 57.97 |
| SGLang | SSD restore after DRAM eviction | 58.94 | 150.37 | 268.04 |

**Both engines restored from SSD in 15/15 forced-SSD requests.** Each request
read exactly as many bytes from SSD as it loaded into the GPU. vLLM restored
144/576/1,152 MiB; SGLang restored 135/567/1,143 MiB and reported
960/4,032/8,128 cached tokens, respecting its final-page boundary. Neither
engine reported an HBM hit during this phase. SGLang's SSD TTFT is now about
2.0x/3.1x/3.7x faster than its cold prefill in this run.

Instrumented median stage times in milliseconds:

| Engine | Stage | 1K | 4K | 8K |
| --- | --- | ---: | ---: | ---: |
| vLLM | SSD prefix prefetch | 31.16 | 90.78 | 181.62 |
| vLLM | GPU load task after SSD read | 4.65 | 18.47 | 36.74 |
| SGLang | SSD prefix prefetch | 22.79 | 93.78 | 184.33 |
| SGLang | GPU load task after SSD read | 6.19 | 25.73 | 51.56 |

The vLLM 1K SSD median increased from 47.27 to 55.24 ms; its SSD-prefetch
median increased from 23.54 to 31.16 ms while its GPU-load and DRAM-control
medians stayed essentially unchanged. This locates the observed difference
in the storage stage, but five samples on the shared overlay cannot establish
whether code changes or storage conditions caused it. The 4K SSD median
decreased and the 8K median remained close to the baseline. These results do
not establish a latency improvement for every workload.

SGLang's GPU-load task after SSD reads took longer than its DRAM-control load
(2.90/12.55/26.46 ms). Allocation, layout reconstruction, and copy behavior need
separate profiling before attributing this gap or attempting layer overlap.
Both engines recorded zero SSD read/write failure deltas in measured requests.

All 15 SSD outputs per engine match their respective DRAM outputs. All SGLang
outputs match cold controls. As in the baseline, one vLLM 4K prefix has a
different cold output while its HBM, DRAM, and SSD outputs agree. The performance
run does not enable deterministic inference; exact GPU-buffer tests and the
separate deterministic serving gates provide the integrity checks.

The [historical readiness dataset](https://github.com/feichai0017/orbitkv/tree/44c1e5f9a253aa7378c6187b2aeea9bff93df304/benches/results)
retains these 120 requests, source manifests, storage evidence and summaries
separately from the baseline. Full responses and service logs
remain in `benches/results/runs/query-readiness-{vllm,sglang}/` on the measurement
host; gate logs are in `benches/results/runs/query-readiness-validation/`.
Use the reproduction commands above with fresh output directories to repeat
the experiment. This qualifies single-rank full-attention recovery; concurrency,
multi-rank coordination, natural memory pressure, and tail latency remain
separate experiments.
