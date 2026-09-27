# Local communication measurements

## Raw descriptors and split K/V coalescing (current increment)

On 2026-09-27, compiling raw resident Restore directly into owned copy
batches and ordering independent descriptors by GPU address reduced the
36-layer, 18 MiB split K/V Restore p50 from **24.70 ms to 1.36 ms** (18.1×,
94.5% lower). Split Publish p50 fell from **24.18 ms to 2.95 ms** (8.2×).
The same-size contiguous Restore improved **14.7%**, while 4 KiB Restore
remained approximately unchanged. These are matched local communication
microbenchmarks; CUDA submission still runs in the Manager. They do not
establish serving TTFT/ITL, compute overlap, or superiority to native engines.

### Implementation and comparison

The baseline is commit `76cbde7cffd75446ba5e402d7ab4c9d797135066`, frozen as
`prepared-production`. The candidate is `descriptor-production`. Both bundles
contain matching release Managers and Python extensions without test hooks.
There is no runtime old/new implementation switch.

Raw Restore holds each selected leased source once, builds its checked copy
ranges directly, and keeps those owners and query reservations through GPU
drain. It removes the layer × block source-reference expansion, worker-side raw
descriptor reconstruction, raw codec/SSD scans, temporary target-range array,
and repeated raw cost-shape computation. Encoded, SSD and mixed routes keep
the layer information their execution requires; raw errors never retry through
a second implementation. Host subranges are checked before pointer arithmetic.

The old split order was K0, V0, K1, V1, which prevented adjacent K ranges or V
ranges from coalescing. Sorting paired descriptors groups those ranges while
preserving each source/destination association. Raw admission rejects overlapping
destinations before GPU submission. Publish and mixed restores share the same
checked descriptor builder and ordering. Merging remains restricted to the
same host and GPU allocation identities. In this dense split workload, each
layer's K and V slabs can coalesce separately; scattered allocations or
page-first host layouts do not promise the same gain.

### Matched workload and results

The [harness](../benches/communication.py) adds `--layout split` as a workload
shape. Each logical 4 KiB block has separate 2 KiB K and V regions with distinct
expected bytes. Logical payload, hashes, destination pages and H2D/D2H counters
are equal to the contiguous shape. Every run checks both regions on the GPU.

All eighteen fresh-Manager sessions passed byte/counter validation. The H20 and
CPU affinity `8,10,12,14` match the environment described below. Each matrix used
`A1 B1 B2 A2 A3 B3`, 150 measured samples after 20 warmups, zero and 1 ms prescribed
idle, and a 256 MiB pool. The three matrices were:

- One contiguous layer, one lease, 4 KiB / 256 KiB / 4 MiB.
- 36 contiguous layers × 128 blocks × 4 KiB = 18 MiB, 32 leases.
- The same 36-layer payload and 32 leases with split K/V storage.

Lease acquisition stays outside Restore timing. Submit-to-ready includes client
encoding, IPC, source preparation/admission, CUDA work and terminal observation.
No build or qualification test ran during measurements. Values are **medians of
three per-run percentiles**, not pooled samples; no timing overhead is subtracted.

All times below are µs, with no prescribed idle:

| Operation / shape | Baseline p50 / p99 | Candidate p50 / p99 |
| --- | ---: | ---: |
| Restore, contiguous 4 KiB | 26.90 / 66.96 | 27.07 / 53.39 |
| Restore, contiguous 256 KiB | 40.87 / 57.17 | 38.98 / 59.17 |
| Restore, contiguous 4 MiB | 233.90 / 267.03 | 222.29 / 255.03 |
| Restore, contiguous 18 MiB / 32 leases | 952.77 / 1051.64 | 812.88 / 866.41 |
| Restore, split 18 MiB / 32 leases | 24699.86 / 25063.66 | 1362.71 / 1492.91 |
| Publish, contiguous 4 MiB | 765.29 / 875.05 | 772.59 / 843.55 |
| Publish, contiguous 18 MiB | 2374.63 / 2537.23 | 2385.71 / 2525.68 |
| Publish, split 18 MiB | 24184.95 / 24337.91 | 2953.76 / 3217.27 |

The three split Restore p50s were 24643.48–24746.57 µs for the baseline and
1354.78–1369.55 µs for the candidate. Contiguous 18 MiB Restore was
951.44–953.68 µs versus 811.64–813.42 µs. These gains repeat across all pairs.
Small-request and contiguous Publish results do not show uniform improvement:
4 KiB Restore p50 increased 0.17 µs, 256 KiB Restore p99 increased 2.00 µs,
and contiguous 4 MiB / 18 MiB Publish p50 increased 7.29 / 11.08 µs. The measured
4 KiB Publish p50 was 75.74 → 63.99 µs, but baseline runs ranged 64.09–85.01 µs;
that median alone is not evidence of a reliable tiny-Publish gain.

Descriptor compilation now occurs before worker enqueue. Split Restore's
submission-only p50 increased **334.60 → 372.71 µs**, despite the much larger
complete-operation gain. Contiguous 4 MiB submission increased 103.88 → 108.92 µs;
contiguous 18 MiB submission decreased 140.13 → 137.92 µs. Phase percentiles must
not be added or subtracted to infer GPU time or pure IPC overhead.

With 1 ms idle, contiguous 18 MiB Restore p50/p99 was 964.63/1022.02 →
820.94/860.39 µs; split Restore was 24711.82/25085.60 → 1368.56/1451.94 µs.
Contiguous 18 MiB Publish regressed from 2327.25/2474.66 to 2383.69/2508.44 µs.

Manager CPU seconds per 150-sample, zero-idle cohort fell from **3.76 to 0.25**
for split Restore and **3.73 to 0.52** for split Publish. Client CPU for those
cohorts was 0.098 → 0.099 s and 0.319 → 0.307 s, respectively. Contiguous 18 MiB
Restore Manager CPU was 0.19 → 0.16 s, while Publish was 0.43 → 0.44 s.
These are medians of cohort totals, including preparation queries and cleanup,
not per-RPC CPU measurements; Manager accounting has 10 ms `/proc` resolution.

### Artifacts, reproduction and qualification

Results, raw samples, byte/counter checks, manifests and `summary.json` are under
`/workspace/.orbitkv-tools/communication-microbench/runs/raw-descriptors`.
Each bundle retains its source patch and SHA-256 manifest. The native extension
hash is unchanged because the changed execution code is linked into the Manager.

| Measured artifact | SHA-256 |
| --- | --- |
| Harness | `06b0d99c105f8d8ff04d818d963c8031ad00a0aec95cd50aa26ad01fc975f040` |
| Candidate Manager | `0ea5dc4df3f0afd0c1e6e645eee8b95e190152105b5cf25f818b24c4ed38bfc8` |
| Candidate extension | `69e1b911af8ae1d8abf986e6f61fe146dd1c0a9ea367760c69a154916c78ae5a` |

Use the established CUDA/Python environment, select the frozen
`prepared-production` or `descriptor-production` bundle, and use a fresh output
path. Repeat in the paired order above:

```bash
PYTHONPATH="$BENCH_BUNDLE/python" taskset -c 8,10,12,14 \
  python3 -m benches.communication \
  --manager "$BENCH_BUNDLE/orbitkv-cache-manager" \
  --label "$BENCH_LABEL" --output "$BENCH_OUTPUT" \
  --iterations 150 --warmup 20 --repeats 1 --layout contiguous
```

For the layered matrix append `--layers 36 --payload-bytes 524288
--restore-batch-size 32`; repeat with `--layout split` for K/V separation.
Keep the default idle matrix and quiet window. The local runner is
`/workspace/.orbitkv-tools/run-descriptor-comparison.py`.

Validation: release workspace 470 passed / 37 ignored; after the equivalent
Clippy boundary-check cleanup, worker tests 22 passed / 2 ignored and workspace
all-target Clippy passed. Python unit tests passed 374; benchmark tests passed
199. Native integration/fault qualification passed 30 distinct cases, with 30
configuration-dependent skips. The initial ANS setup failure was resolved by
installing the documented `nvidia-libnvcomp-cu13==5.3.0.16` into an isolated tools
directory; the formerly blocked ANS case then passed. The separate
`descriptor-fault` bundle contains test hooks and is excluded from performance.
The gates include actual partial GPU submission, dropped completion receiver,
source/budget retention, split target permutation/skipped pages, host bounds,
SSD/codec recovery, timeouts, lost notification and Manager restart. No cuFile
qualification, engine serving E2E, engine-local executor or cross-host result is
claimed by this increment.

## Earlier restore preparation increment

On 2026-09-27, the preparation/geometry refactor reduced the measured submission
p50 for a 32-lease, 36-layer, 18 MiB Restore from **176.25 to 141.14 µs** (19.9%).
The complete submit-to-ready p50 decreased from **987.73 to 955.09 µs** (3.3%).
Single-lease 4 KiB Restore increased from **26.20 to 27.33 µs** (4.3%, 1.13 µs).
This increment improves the measured batch path; it is not a universal latency
improvement or a serving TTFT/ITL result. CUDA submission still runs in the Manager.

The baseline is `b0f953eb851add59f7c4f50b54f40f9307a1ed0c`, the frozen
`verified-final` bundle from the notification increment below. The candidate is
`prepared-production`, with one matching release Manager and extension and no
`test-hooks` feature on either side. It separates source preparation from local
GPU address binding, validates/consumes all leases under one lock, avoids a
whole-table expiry sweep per lease, and removes redundant clones/wrappers.

The same extended [harness](../benches/communication.py) ran twelve fresh-Manager
sessions on the H20, with CPU affinity `8,10,12,14`. Each matrix used
`A1 B1 B2 A2 A3 B3`, 150 measured samples after 20 warmups, zero and 1 ms prescribed
idle, one repeat, and a 256 MiB pool. The single-lease matrix used one layer and
4 KiB/256 KiB/4 MiB payloads. The batch matrix used 36 layers × 128 blocks ×
4 KiB = 18 MiB, partitioned into 32 nonoverlapping leases in one Restore command.
Native hash views and target chunks are prepared in advance; all lease queries
are outside Restore latency. Total payload and transfer counters are unchanged
by the number of leases. All twelve runs passed GPU byte and H2D/D2H counter checks.

Numbers below are **medians of three per-run percentiles**, not pooled samples.
Times are µs, with no prescribed idle:

| Operation / shape | Baseline p50 / p99 | Candidate p50 / p99 |
| --- | ---: | ---: |
| Restore, 4 KiB / 1 lease | 26.20 / 59.31 | 27.33 / 51.75 |
| Restore, 256 KiB / 1 lease | 40.43 / 58.42 | 40.83 / 59.34 |
| Restore, 4 MiB / 1 lease | 236.72 / 265.71 | 234.73 / 263.91 |
| Restore, 18 MiB / 36 layers / 32 leases | 987.73 / 1056.78 | 955.09 / 1047.27 |
| Submission portion of that 32-lease Restore | 176.25 / 213.97 | 141.14 / 179.07 |

The submission measurement includes client encoding, IPC, Manager preparation
and admission; it does not isolate one function. Phase percentiles must not be
added or subtracted to infer GPU copy time. One candidate 4 MiB run had a
277.30 µs p50; the other two were 233.42 and 234.73 µs, versus baseline runs
233.96–237.07 µs. The small median change is not evidence of a reliable 4 MiB gain.

With 1 ms prescribed idle, 32-lease Restore p50 was 991.48 → 959.56 µs while p99
was 1037.96 → 1063.49 µs. Single-lease 4 KiB p50 was 39.78 → 41.75 µs.
The batch median gain does not establish uniformly improved tail latency.
The benchmark does not qualify inference scheduling/compute overlap, native
engine parity, cross-host traffic, or a new engine-local executor.

The dataset is
`/workspace/.orbitkv-tools/communication-microbench/runs/prepared-production`,
with `single-a/b1..3`, `batch32-a/b1..3`, manifests, raw samples and `summary.json`.
An earlier `runs/prepared-restore` experiment used a candidate with test hooks,
was stopped on detecting that mismatch, and is excluded entirely. Its fault-test
bundle (`prepared-final`) remains separate from the production comparison.

| Measured artifact | SHA-256 |
| --- | --- |
| Harness | `5fc32e3f07cf0c99c9812e43b3c8296e0de269d5e15dddf98de1a5e99e564b4d` |
| Candidate production Manager | `19e9bf5499f517d64528a989bae8ed7e013f19049def452be29fceefb2c315e7` |
| Candidate production extension | `69e1b911af8ae1d8abf986e6f61fe146dd1c0a9ea367760c69a154916c78ae5a` |

Reproduce using the frozen `verified-final` and `prepared-production` bundles,
the established CUDA/Python environment, and a fresh output path for every run:

```bash
PYTHONPATH="$BENCH_BUNDLE/python" taskset -c 8,10,12,14 \
  python3 -m benches.communication \
  --manager "$BENCH_BUNDLE/orbitkv-cache-manager" \
  --label "$BENCH_LABEL" --output "$BENCH_OUTPUT" \
  --iterations 150 --warmup 20 --repeats 1
```

For the batch matrix append `--layers 36 --payload-bytes 524288
--restore-batch-size 32`. Retain the default idle matrix and two-second quiet
window; do not rebuild native libraries while measurements are running.

## Earlier request-notification increment

On 2026-09-27, replacing periodic Manager/Publish waits with notifications
reduced the measured 4 KiB Query-hit p50 from 108.42 to 6.87 µs and Restore
submit-to-ready p50 from 127.58 to 26.75 µs. A 36-layer Publish fell from
4.22 to 2.37 ms. The standalone channel ping's p50 and CPU cost increased;
its p99 improved. The tradeoff is reported below. These are local microbenchmarks,
not serving TTFT/ITL or an engine-local GPU executor qualification.

### Compared implementations

The baseline is `d3c156ea667819ac8964e6b05d15a0c3d0e3f5ae`, frozen as the
`baseline` artifact bundle. The candidate is the `verified-final` bundle built from
the final source of the communication changes accompanying this report, identified by SHA-256 below.
Each bundle supplies its own matching release Manager and native extension.
There is no runtime old/new implementation selector or mixed-client control.

The candidate replaces the Manager's 50 µs idle sleep with a required iceoryx2
request notification, bounded spin, and a maintenance-bounded wait. Publish uses
a dedicated reply eventfd instead of a 100 µs sleep; restore completion retains
its independent eventfd. Four bootstrap FDs, bootstrap version 4, and channel
ABI 7 require matching clients and Managers. Exact encoded-size planning and
borrowed Publish ranges remove repeated candidate cloning/encoding; redundant
forwarding helpers were removed. A sub-microsecond remaining wait returns
immediately: passing a timeout truncated to zero into `SO_RCVTIMEO` would disable
the timeout. Final measurements include that fix.

Shared restore results and shared pinned-pool backing already exist in the
baseline. Their earlier implementation is not a newly measured benefit here.
GPU restore still executes in the Manager. The
[engine-local design](engine-local-restore.md) remains subsequent work.

### Workload and controls

The [harness](../benches/communication.py) ran twelve fresh-Manager sessions:
three baseline/candidate pairs for one layer and three for 36 layers, ordered
`A1 B1 B2 A2 A3 B3` within each matrix. The dataset root is
`/workspace/.orbitkv-tools/communication-microbench/runs/verified`. Only its
`final-a/b1..3`, `final-layers-a/b1..3`, and `raw-final-baseline/optimized.json`
contribute here; all earlier datasets are excluded. All twelve completed.

One-layer cases use 4 KiB blocks, 1/64/1024 hashes, and 4 KiB/256 KiB/4 MiB
payloads: 300 measured samples after 30 warmups per case per run. The layered
case uses 36 × 128 × 4 KiB = 18 MiB, with 100 samples after 10 warmups. Both use
0 and 1 ms prescribed inter-operation idle, a 256 MiB host pool, raw uint8
storage, and the direct copy backend. Query budgets are 16 MiB and 36 MiB,
respectively. Empty Restore submits registered layer names with no loads.

The host has an Intel Xeon Platinum 8457C and NVIDIA H20. Client and Manager
launch/main-thread affinity is `8,10,12,14`, four distinct physical cores on
NUMA node 0. Manager GPU workers can override this through NUMA pinning; this
is not a hard four-core budget for every thread.
Python is 3.11.10; torch is `2.10.0a0+git89fc82e.aml`, built for CUDA 13.1.
Both sides use the same CUDA compatibility libraries and private override
`CUDA_MPS_PIPE_DIRECTORY=/workspace/.orbitkv-tools/mps-probe` to avoid a broken
shared MPS setup. That environment value does not establish an active MPS
service; these measurements make no active-MPS qualification claim.

Every payload first passes poison/restore/GPU-byte validation; layer contents
differ. Restore cohorts check final bytes, and measured H2D/D2H counters must
equal requested bytes. Warmup and byte checks are outside timing. Query covers
submission through `QueryReady`, including any polling; lease release is outside
its timer. Restore excludes lease acquisition and includes native terminal
observation. Publish uses fresh hashes to force D2H; sealing synchronization and
cache cleanup occur between samples, outside latency. Thus zero prescribed idle
does not mean uninterrupted Publish traffic.

### Latency

Cells are **median of three per-run p50 / median of three per-run p99**, in µs.
Samples are not pooled. Main-harness quantiles use linear interpolation.
For Query, payload size describes the logical KV demand: the request carries
32-byte hashes and metadata, not those KV payload bytes.

| Operation | Logical payload / shape | Baseline p50 / p99 | Candidate p50 / p99 |
| --- | --- | ---: | ---: |
| Query miss | 4 KiB / 1 layer | 105.82 / 115.93 | 6.15 / 18.39 |
| Query hit | 4 KiB / 1 layer | 108.42 / 117.71 | 6.87 / 15.99 |
| Restore | 4 KiB / 1 layer | 127.58 / 142.35 | 26.75 / 49.71 |
| Publish | 4 KiB / 1 layer | 184.90 / 201.94 | 64.26 / 105.13 |
| Query miss | 256 KiB / 1 layer | 117.50 / 132.44 | 26.49 / 37.80 |
| Query hit | 256 KiB / 1 layer | 118.95 / 126.18 | 20.20 / 33.79 |
| Restore | 256 KiB / 1 layer | 140.71 / 155.19 | 40.49 / 58.19 |
| Publish | 256 KiB / 1 layer | 191.27 / 352.65 | 107.48 / 149.53 |
| Query miss | 4 MiB / 1 layer | 327.12 / 341.43 | 284.59 / 346.75 |
| Query hit | 4 MiB / 1 layer | 305.87 / 323.15 | 248.78 / 265.99 |
| Restore | 4 MiB / 1 layer | 304.04 / 345.50 | 235.46 / 249.59 |
| Publish | 4 MiB / 1 layer | 862.30 / 1015.55 | 757.79 / 842.72 |
| Empty Restore | 1 layer | 105.29 / 116.02 | 14.49 / 26.41 |
| Query miss | 18 MiB / 36 layers | 130.57 / 134.18 | 42.99 / 53.68 |
| Query hit | 18 MiB / 36 layers | 130.14 / 140.41 | 38.84 / 50.89 |
| Restore | 18 MiB / 36 layers | 1050.11 / 1157.27 | 959.51 / 1021.24 |
| Publish | 18 MiB / 36 layers | 4215.03 / 4522.17 | 2367.53 / 2484.76 |
| Empty Restore | 36 layers | 110.67 / 125.35 | 21.66 / 36.12 |

The table uses zero prescribed idle. Representative **1 ms idle** cases are:

| Operation / payload | Baseline p50 / p99 | Candidate p50 / p99 |
| --- | ---: | ---: |
| Query hit / 4 KiB | 97.21 / 108.48 | 17.50 / 29.05 |
| Restore / 4 KiB | 115.85 / 145.02 | 41.23 / 71.22 |
| Publish / 4 KiB | 182.66 / 198.08 | 64.62 / 86.10 |
| Restore / 4 MiB | 275.44 / 346.27 | 241.84 / 266.83 |
| Publish / 18 MiB, 36 layers | 4202.46 / 4394.17 | 2325.43 / 2474.19 |

Wakeup cost remains visible after idle. Across the three no-idle runs, candidate
4 KiB Publish p50 was **81.89, 64.26, 63.35 µs**. The 1024-hash
Query-miss p99 varied **295.49–444.10 µs**, with its median slightly above
baseline despite lower p50. Candidate 36-layer Publish p50 was
2336.28–2379.43 µs. Three runs do not provide confidence intervals;
a 100-sample p99 is close to the largest observations. The results do not
separate each code change's effect.

### CPU and the raw-channel tradeoff

In two-second quiet windows, baseline Manager use was 0.045–0.050 CPU cores;
candidate one-layer windows were 0–0.005 cores. None of the three layered
candidate windows recorded a CPU tick. `/proc` CPU counters have **10 ms
resolution**, so zero means no observed tick increase, not zero work.
Whole-cohort Manager CPU includes preparation queries, release, Publish cleanup,
and idle intervals; it is not precise per-RPC CPU. Caller-process p50 for
no-idle 36-layer Publish fell from 1989.73 to 377.42 µs.

The standalone 64-byte two-process ping uses 100,000 measured requests after
10,000 warmups, echo on CPU 8 and client on CPU 10, with three runs per bundle.
Median per-run p50 **regressed 2.325 → 2.444 µs (+5.1%)**, while p99 improved
**3.686 → 3.391 µs (-8.0%)**. Combined child-process CPU per batch rose
**0.547608 → 0.576289 s (+5.2%)**, including startup/warmup/shutdown accounting.
The pair occupied approximately two cores during this saturated workload.
The notification path improves the real Manager workload and idle CPU here;
it does not improve the tightest raw ping's median or establish universally lower CPU.

### Artifact identity and reproduction

SHA-256 identifies the measured files; it does not assert that subsequent
rebuilds are byte-identical. The frozen candidate hashes remain authoritative.

| File | SHA-256 |
| --- | --- |
| Harness, all twelve runs | `279fbb4bf37f3447ed8d81fa7c3410d2d658402f5b61df457d80d874089e2fd1` |
| Baseline Manager | `503cf3c97e9cc2aafebfa58c29c5e970b147694fdbe5f9a2c6803571a5465875` |
| Baseline extension | `7fe446ea03420124b08ea0e23f5c4df46926b7a4693358981a8250c3b9db445b` |
| Candidate Manager | `b43f2adbc3232b6cd928910ba372484f543cb92acf30a5e74a3738c7290d40c6` |
| Candidate extension | `44fad7c809446f1f4700012a5b272a1052715434b7c707691bb52d45a3c738a6` |

With frozen matching bundles and the same CUDA/Python/path overrides, run from
the harness checkout; set `BENCH_ARTIFACT_ROOT` to their parent directory:

```bash
for run in a1 b1 b2 a2 a3 b3; do
  case "$run" in a*) bundle=baseline ;; b*) bundle=verified-final ;; esac
  PYTHONPATH="$BENCH_ARTIFACT_ROOT/$bundle/python" taskset -c 8,10,12,14 \
    python3 -m benches.communication \
    --manager "$BENCH_ARTIFACT_ROOT/$bundle/orbitkv-cache-manager" \
    --label "final-$run" --output "benches/results/runs/verified/final-$run" \
    --iterations 300 --warmup 30 --repeats 1
done
```

Repeat with labels/output `final-layers-$run`, `--iterations 100 --warmup 10
--layers 36 --payload-bytes 524288`. The default idle matrix and two-second
quiet window match this report. For raw ping use the prebuilt
[channel echo/bench pair](../crates/orbitkv-channel/README.md), affinity 8/10,
and 100,000 iterations; aggregate both children's user/system CPU separately.
Keep raw manifests, samples, logs and `final-summary.json` under the run directory;
accept only complete runs. Do not rebuild libraries while services are alive.
