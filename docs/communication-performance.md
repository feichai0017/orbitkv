# Local communication measurements

## Restore preparation (current increment)

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
