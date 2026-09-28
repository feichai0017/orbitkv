# Local communication measurements

## Completion evidence and copy-path diagnosis

The September 28 increment (`fb2c3b62`, `5b1fb1ff`) measures the native
caller-to-drain interval independently of Manager retirement. Cost samples are
opt-in; full tracing additionally records six native stage offsets and result
consumption. These observations do not enable cross-route selection. Subsequent
cost-key isolation (`c1d8d43c`) separates source domain, destination GPU and copy
backend; the measurements below precede that key-only correction.

### Instrumentation overhead

Nine fresh-Manager runs use one H20, CPU set `8,10,12,14`, a 256 MiB pool,
150 samples after 20 warmups, and no prescribed inter-operation idle. The order
is `off cost trace trace cost off off trace cost`. Cost mode enables only
`ORBITKV_COST_OBSERVATIONS=1`; trace mode also enables
`ORBITKV_TRACE_TRANSFERS=1`. Every Restore passes exact GPU-byte and copy-counter
checks. A 100 ms process audit observed no Cargo/rustc/cc1plus activity. This
is a local process audit, not proof of exclusive machine use.

Cells are **median of three per-run p50 / median of three per-run p99**, in
microseconds; samples are not pooled.

| Contiguous Restore | Both disabled | Cost only | Cost and tracing |
| --- | ---: | ---: | ---: |
| 4 KiB | 54.39 / 101.07 | 53.40 / 91.72 | 61.53 / 108.54 |
| 256 KiB | 64.59 / 102.68 | 60.83 / 100.88 | 73.11 / 126.26 |
| 4 MiB | 217.66 / 256.62 | 220.86 / 277.49 | 231.37 / 291.72 |

Cost collection does not establish a small-payload speedup: negative overhead
is measurement variation. At 4 MiB it adds 1.5% to median latency and its tail
is worse. Full tracing adds about 7–14 us to these medians. Empty-Restore p50 is
37.87 / 36.42 / 46.31 us in the same mode order. Both switches remain off by
default, and tracing must match across performance comparisons.

### Real vLLM serving decomposition

Two diagnostic cohorts use Qwen3-8B, vLLM 0.29.0, H20, TP=1, CPU set `8–23`,
16,384 GPU KV tokens, 8,192 prefill tokens, 16 GiB host capacity, and 16 output
tokens. Each length has five cold/resident/pressure repetitions. Cost collection
and tracing are enabled in both; the transfer backend is the only intended
configuration difference. These are single cohorts, without the separate
external-build monitor used above, and are not repeated speedup acceptance.

| Prompt tokens | Direct pressure TTFT / E2E, ms | Kernel pressure TTFT / E2E, ms |
| --- | ---: | ---: |
| 1,024 | 25.05 / 114.34 | 25.20 / 111.78 |
| 4,096 | 35.92 / 127.00 | 42.09 / 131.03 |
| 8,192 | 54.86 / 151.24 | 64.84 / 157.95 |

The kernel control regresses the larger shapes; keep the existing direct
backend. Both cohorts have two cold-versus-pressure output differences at 1K
and none at 4K/8K. They use ordinary greedy execution and do not qualify output
parity. Direct and kernel nevertheless match all 45 corresponding request
outputs across cohorts. Use the separate deterministic serving gates for
correctness.

All 45 measured direct requests have connector traces. All 30 linked physical
Restore batches have native completion and consumption evidence, including
small partial restores in the resident phase. No local batch requires a
Manager-to-engine completion notification. For the five **pressure** restores
per length, native stage medians are:

| Tokens | Readiness | Dispatch | Queue | Grant wait | Plan/enqueue | Drain wait | Native total | Consumer wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.0174 | 0.1722 | 0.0138 | 0.0499 | 0.8933 | 3.8082 | 4.9629 | 0.5149 |
| 4,096 | 0.0170 | 0.2187 | 0.0156 | 0.0438 | 0.7127 | 11.5864 | 12.7325 | 0.4383 |
| 8,192 | 0.0172 | 0.5863 | 0.0131 | 0.1922 | 15.2132 | 13.6298 | 29.6335 | 0.6109 |

Times are milliseconds. Each interval is computed from that operation's native
monotonic offsets before aggregation; medians must not be subtracted or summed
to attribute another median. Plan/enqueue includes validation, descriptor work,
driver backpressure and any concurrent DMA; it is not pure CPU submission time.
Consumer wait is outside native total. This evidence prioritizes copy batching
at 8K over another control-transport rewrite.

A separate synthetic 2D DMA probe copies 72 rows per block with 128 KiB row
width. For 16/64/128 blocks, 1D total medians are 6.19/24.97/50.00 ms and 2D
medians are 2.97/11.20/22.15 ms, with exact output bytes checked. The allocation
is deliberately regular; it does not represent fragmented production grants.
No 2D production path was applied and these numbers are not serving gains.
Production integration still needs bounds/gap preservation, allocation identity,
partial-enqueue drain and matched serving qualification.

### Artifacts and reproduction

| Measured artifact | SHA-256 |
| --- | --- |
| Manager | `6fbd684265cf88408191ad1914e92510f68797d82726ef4f54e286377af8cf57` |
| Native extension | `5cfef266efa208318cc0e8595dd7f06f335aa179a86ab06a8bdd573e5f39ecce` |

Raw runs, process audits, commands and probe sources are retained under
`benches/results/runs/completion-evidence-20260928/` on the measurement host.
Use a matching Manager/extension pair and the pinned release environment:

```bash
# Repeat in off/cost/trace order as specified above.
ORBITKV_COST_OBSERVATIONS=1 ORBITKV_TRACE_TRANSFERS=0 \
RUST_LOG=warn,orbitkv_common::timeline=info taskset -c 8,10,12,14 \
  .venv/vllm-release/bin/python -m benches.communication \
  --label cost --output /path/to/empty-output \
  --iterations 150 --warmup 20 --repeats 1 \
  --payload-bytes 4096 262144 4194304 --idle-ms 0 --idle-seconds 0.5

ORBITKV_COST_OBSERVATIONS=1 RUST_LOG=info taskset -c 8-23 \
  .venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/Qwen3-8B \
  --output /path/to/empty-serving-output --lengths 1024 4096 8192 \
  --repeats 5 --gpu-tokens 16384 --prefill-tokens 8192 --host-gib 16 \
  --output-tokens 16 --seed 20260920 --orbitkv-transfer-backend direct \
  --trace-transfers
```

Repeat the serving command with `kernel` for its control. Do not compare these
traced cohorts with earlier untraced backends as a cache speedup. The next
acceptance gate remains repeated, order-reversed native HBM/native CPU/OrbitKV/
LMCache cohorts. The separate [deterministic C4 gate](sustained-performance.md#deterministic-c4-qualification-after-completion-evidence)
now passes for both engines; it does not retroactively qualify ordinary-mode
cohorts or establish a repeated performance gain.

## Compacted raw plans and idle-stream readiness

The next increment after `d25abcf7` removes two measured costs from engine-local
raw Restore. The executor queries the actual engine stream before preparation:
an idle stream already establishes readiness, while a busy stream records and
waits on one reusable event. The Manager sorts destination pages and traverses
K/V separately, merging consecutive source and destination ranges within the
same layer and allocation identity/bounds **before encoding**. That avoids
per-page descriptor/string construction and repeated transport/decode work for
contiguous runs. The raw plan wire format and grant/drain ownership are unchanged;
no old executor, compatibility switch, or forwarding wrapper was added.

A separate 1,000-sample stage probe (100 warmups) measured idle-stream readiness
p50 at 11.44 → 1.39 us. Reusing an event while still recording/waiting took
10.76 us, so the idle query matters more than event reuse alone. These are
isolated stage measurements, not an additive attribution of the end-to-end gains.
The probe sources and logs are retained under
`/workspace/.orbitkv-tools/local-restore-stages-{baseline,compact}.{rs,log}`.

### Three-way matched comparison

The production bundles are `identity-production` (Manager executor,
`e36161d8`), `local-executor-production` (initial local executor, `d25abcf7`),
and `compact-restore-production` (this increment). All use the same H20, CPU
set `8,10,12,14`, 256 MiB pool, 150 samples after 20 warmups, and prescribed
idle of zero or 1 ms. Per workload the order is `A1 B1 C1 C2 B2 A2 A3 C3 B3`,
with a fresh Manager each time. The Manager bundle uses its old API harness;
both local bundles use the identical current harness, including tensor binding
and timed readiness handling. Lease lookup stays outside Restore timing.

All **18 accepted runs** passed exact GPU-byte and copy-counter checks. Cargo/
rustc was sampled every 100 ms, with a five-second quiet period before each
attempt. No build activity was observed in these runs, so no attempt was
excluded; the exclusion rule was independent of measured latency. No build or qualification run from this
task overlapped these measurements. This process audit does not establish
exclusive use of the whole machine.

Values below are medians of three per-run percentiles, not pooled samples.
Times are microseconds with no prescribed idle:

| Operation / shape | Manager executor p50 / p99 | Initial local executor p50 / p99 | Optimized local executor p50 / p99 |
| --- | ---: | ---: | ---: |
| Restore, contiguous 4 KiB | 26.06 / 53.17 | 69.59 / 128.51 | 51.90 / 93.58 |
| Restore, contiguous 256 KiB | 38.39 / 66.51 | 93.39 / 144.58 | 59.05 / 99.23 |
| Restore, contiguous 4 MiB | 223.18 / 246.77 | 454.53 / 506.36 | 217.33 / 282.20 |
| Restore, split 18 MiB, 32 leases | 1,363.76 / 1,424.39 | 3,062.31 / 3,264.71 | 1,186.18 / 1,300.72 |
| Publish, contiguous 4 KiB | 66.64 / 105.03 | 64.28 / 98.45 | 80.48 / 124.24 |
| Publish, contiguous 256 KiB | 109.97 / 162.36 | 109.04 / 139.44 | 109.03 / 137.10 |
| Publish, contiguous 4 MiB | 772.23 / 825.07 | 755.53 / 808.42 | 766.42 / 853.02 |
| Publish, split 18 MiB | 2,947.07 / 3,076.04 | 2,942.34 / 3,149.52 | 2,909.64 / 3,100.77 |

Relative to the initial local executor, Restore p50 falls by
25.4%, 36.8%, 52.2% and 61.3% respectively.
Against the pre-migration Manager executor, the 4 MiB result is close
(223.18 → 217.33 us), and the 18 MiB batch improves
(1363.76 → 1186.18 us). **Small operations remain slower than
that Manager baseline**: 26.06 → 51.90 us at 4 KiB and
38.39 → 59.05 us at 256 KiB. This increment recovers much of the
cutover regression; it does not make every shape faster than the earlier design.
The 18 MiB Restore p50 ranges are 1363.46–1363.90 us for the Manager and
1186.01–1193.18 us for the optimized executor. The 4 MiB p50 ranges overlap
(223.02–225.27 versus 216.97–224.00 us), and its p99 is still worse than the
Manager: 246.77 → 282.20 us. Treat that shape as recovery to similar median
latency, not a demonstrated tail improvement.

Publish keeps its existing executor, but its 4 KiB no-idle p50 rises from
64.28 to 80.48 us against the initial local bundle. Its per-run p50 ranges are
62.96–64.89 versus 64.86–83.07 us; with 1 ms idle, the medians are 65.31 and
65.68 us. The cause is not isolated. Retain this adverse control result when
evaluating the increment rather than claiming an all-path latency win.

For the 150-sample 18 MiB Restore cohort, initial/optimized client CPU time is
0.482/0.223 seconds and Manager CPU time is 0.180/0.070 seconds. These figures
include untimed lease setup; Manager accounting has clock-tick resolution.
The engine process still uses more CPU than the old Manager-executor client
(0.101 seconds), despite the combined CPU reduction in this cohort.

With 1 ms prescribed idle, the Manager / initial local / optimized Restore p50
values are 43.32 / 92.50 / 62.08; 53.04 / 108.17 / 78.87;
228.93 / 436.24 / 227.32; and 1371.95 / 3058.82 / 1191.95 us
in the same four-shape order. Submission-only medians and every per-run p50/p99
remain in `summary.json`. These short samples do not establish stable serving
tails. Compaction requires contiguous source and destination runs in the same
allocation; fragmented layouts need separate performance qualification.

### Bounds, correctness and reproduction

The 1 MiB cap now applies to the compacted plan. A 32,768-page dense unit fixture
encodes one 71-byte plan; adjacent distinct allocation IDs cannot merge.
Oversized fragmented plans still reject before consuming leases. The existing
`cpu_path/load_submit_wait/32768` GPU smoke passes Criterion `--test` mode,
proving submission/drain rather than byte validation or a throughput result.
Automatic bounded partitioning remains open.

The release workspace passed 488 tests (38 ignored), the explicit CUDA context
gate passed, Python units passed 374 and benchmark units 199, and all-target
Clippy denied warnings. The production native suite passed seven cases; the
separate fault suite passed 38 with 30 cuFile configuration skips. Busy-stream
event reuse and same-host raw/encoded peer GPU bytes passed. See
[artifact-specific fault qualification](fault-qualification.md#engine-local-raw-restore-gates).

| Optimized production artifact | SHA-256 |
| --- | --- |
| Manager | `3435bf33ada1927a419420fa104cc2e8cf75fce3ca8e294416253614aa6da638` |
| Native extension | `978bd219c77e544100e87f0add4ee39a919766e59cbb83af055439d02dbb2db4` |

The base revision is `d25abcf70c72d3635ebcc61c929766ddd527a425`;
the frozen source-patch hash is
`bc1010b8d606c0ca86904e51eb105fd0675fecd2ac4ee96900901d1bc3a01fbb`.
Code/test source hashes match the fault and production bundles; subsequent edits are documentation only. Baseline binary identities
are retained in the preceding-cutover evidence below. Harness hashes are
`06b0d99c105f8d8ff04d818d963c8031ad00a0aec95cd50aa26ad01fc975f040` and
`3f91d3d63da4d0eed403fcd80d9873b74767ec9bd624c480c2b28bce3bb0b9f5`.

Raw samples, manifests, per-run CPU accounting, process audits, and `summary.json`
are under `/workspace/.orbitkv-tools/communication-microbench/runs/compact-restore-final/`.
Workspace reproduction uses `/workspace/.orbitkv-tools/run-compact-restore-final.py`
and `/workspace/.orbitkv-tools/summarize-compact-restore-final.py` with the named
frozen bundles and the CUDA environment described below. The next measurements
should isolate small-operation queue handoff and fragmented-plan costs before
changing execution overlap. Pinned serving environments were unavailable during
these microbenchmarks. The subsequent
[Qwen3-8B serving qualification](single-node-performance.md#engine-local-restore-serving-qualification)
uses the same production artifacts. These microbenchmarks themselves do not
prove TTFT/ITL gains or superiority to an engine's resident GPU KV cache.

## Engine-local raw Restore: functional cutover, measured latency regression

On 2026-09-27, unencoded resident Restore moved CUDA submission into the
inference process. The Manager retains source grants and query reservations;
local stream drain releases the engine's destination fence before asynchronous
Manager source retirement. vLLM and SGLang now provide retained tensors and an
explicit readiness stream. The obsolete Manager raw execution branch is removed,
with no compatibility switch or fallback. SSD, codec and mixed routes retain
workers that own their actual I/O and decode operations. See the
[execution and lifetime contract](engine-local-restore.md).

**This initial cutover is functionally qualified on the tested single-GPU process paths,
but it regresses serial Restore latency. It is not a performance win.** Moving
submission alone adds readiness, plan transport and native scheduling costs.
The measurements below do not isolate the contribution of each stage, and do
not demonstrate inference serving, compute overlap, or superiority to an
engine's native KV cache.

### Matched comparison

The baseline is `e36161d8500bb00dd7fe616a39d64fc03bd0ab2e`, frozen as
`identity-production`; the candidate is `local-executor-production`. Each has a
release Manager and matching extension without test hooks. H20, CPU affinity
`8,10,12,14`, a 256 MiB pool, 150 samples after 20 warmups, and zero/1 ms
prescribed idle match the preceding experiment. Each workload used three pairs
in `A1 B1 B2 A2 A3 B3` order, with a fresh Manager per run:

- One contiguous layer and one lease: 4 KiB, 256 KiB and 4 MiB.
- 36 split K/V layers and 32 leases: 18 MiB logical payload.

The candidate harness adds the required `tensors` registration and
`ready_stream` submission arguments; the baseline uses its revision's API.
Current-stream lookup and native readiness handling remain inside candidate
Restore timing. Lease lookup remains outside that timing. Thus these are
matched workloads across the API cutover, not identical harness binaries.
No source-retirement wait was added to the local destination fence.

All twelve accepted runs passed exact copy-counter and GPU-byte checks. An
external main-checkout build interrupted two preliminary matrices; neither is
included. The final runner observed Cargo/rustc every 100 ms, waited for five
seconds without an observed build before starting, and excluded/retried four
runs with observed build activity, regardless of their latency. Attempt logs
and process audits are retained. No build was observed during any accepted
run; this is sampled process evidence, not a system-wide isolation guarantee.
No qualification tests or builds from this task ran during measurement.

Values are medians of three per-run percentiles, not pooled samples. Times are
microseconds, without prescribed idle:

| Operation / shape | Manager executor p50 / p99 | Local executor p50 / p99 |
| --- | ---: | ---: |
| Restore, contiguous 4 KiB | 26.51 / 62.94 | 67.13 / 125.78 |
| Restore, contiguous 256 KiB | 38.63 / 54.15 | 88.79 / 137.25 |
| Restore, contiguous 4 MiB | 222.78 / 238.12 | 415.42 / 500.45 |
| Restore, split 18 MiB, 32 leases | 1,362.37 / 1,413.61 | 3,037.25 / 3,248.23 |
| Publish, contiguous 4 KiB | 64.83 / 91.41 | 63.53 / 99.03 |
| Publish, contiguous 256 KiB | 109.42 / 143.05 | 111.36 / 163.92 |
| Publish, contiguous 4 MiB | 768.76 / 819.54 | 761.66 / 861.00 |
| Publish, split 18 MiB | 2,940.39 / 3,160.90 | 2,892.54 / 3,124.70 |

The four Restore p50 values regress by approximately 2.53x, 2.30x, 1.86x and
2.23x respectively. With 1 ms prescribed idle they are 42.33 → 90.20 us,
51.39 → 106.78 us, 229.07 → 436.66 us and 1,368.07 → 3,068.71 us.
Submission-only p50 without idle is 8.94 → 41.03 us, 14.48 → 46.73 us,
109.04 → 177.74 us and 376.41 → 679.34 us. Submission is not GPU completion.
An empty single-layer Restore also regresses, 17.75 → 59.07 us, showing fixed
operation overhead even without payload copies. Publish is a control path and
remains broadly similar; its tail variation is included rather than interpreted
as a change to that unchanged executor.

Across the three runs, 4 KiB Restore p50 ranges from 25.79–29.35 us at baseline
and 64.30–70.65 us in the candidate. Split 18 MiB ranges from
1,356.28–1,365.10 us and 3,027.07–3,109.93 us. The regression is present in all
pairs, while these short runs remain insufficient for stable tail estimates.
For the 150-sample split Restore cohort, client CPU time grows from 0.101 to
0.479 seconds; Manager CPU falls from 0.250 to 0.180 seconds. Those cohort CPU
figures include untimed lease setup; Manager accounting has clock-tick
resolution. They show work moving to the engine process, not a total CPU saving.

### Evidence and next work

| Bundle | Manager SHA-256 | Extension SHA-256 |
| --- | --- | --- |
| `identity-production` | `308dc6b2590ed2445af4575ee0bf94c0bff9b4429414020467d34b28ed49b7cb` | `ee569e0eba40b7953afc785f24a2d0e30d5d5631151dee414406f37101f4b734` |
| `local-executor-production` | `ddf2a7d61e8220923a25dcb9c9bbd4bdabe396988deb1d042694a0ac73a20b02` | `861ae3bb224ad5f7b1ac27b6ef623b0be20b102e3a95bb2da09e03a258b9eeb1` |

The candidate source patch SHA-256 is
`c65318e0f36f1bcae491f5448a5c3bbcb5301aa0431a4138df1d8886c913ce57`.
Its manifest also hashes and retains the new source files absent from that
tracked-file patch. The baseline harness SHA-256 is
`06b0d99c105f8d8ff04d818d963c8031ad00a0aec95cd50aa26ad01fc975f040`;
the candidate harness SHA-256 is
`3f91d3d63da4d0eed403fcd80d9873b74767ec9bd624c480c2b28bce3bb0b9f5`.
Post-freeze changes are documentation only. Per-run samples, manifests, CPU
accounting, accepted-run links, excluded attempts and `summary.json` remain in
`/workspace/.orbitkv-tools/communication-microbench/runs/local-executor-audited/`.
The workspace reproduction scripts are
`/workspace/.orbitkv-tools/run-local-restore-audited.py` and
`/workspace/.orbitkv-tools/summarize-local-restore-comparison.py`; they use the
same command shape and CUDA setup documented below, paired with these bundles.

The final workspace release gate passed 486 tests, with 38 ignored; the new
CUDA context gate was also run explicitly and passed. Python units passed 374
cases and benchmark units passed 199. All-target Clippy denied warnings. The
production native channel/client suite passed seven cases. The separate frozen
fault build passed 38 cases, with 30 cuFile configuration skips. Raw peer READ
round trips passed with both pipeline modes, and the encoded peer round trip
passed on same-host Mooncake TCP. See [fault qualification](fault-qualification.md#engine-local-raw-restore-gates)
for binary identities and the exact process-death evidence boundary.

At this initial revision, readiness created an event per call, layer/allocation
metadata repeated in every copy, plan transport used 64-bit atomics, and local
copy scratch was rebuilt each time. The 1 MiB limit rejected the existing
32,768-block CPU benchmark before consuming leases. The subsequent compaction
and readiness measurements above supersede these performance observations;
fragmented-plan partitioning and native handoff overhead remain open.

No pinned vLLM/SGLang serving environments or model artifacts were available.
Serving correctness/TTFT/ITL, group overlap, graph replay, multiple GPUs,
huge-page imports and cross-host RDMA remain unqualified. The new executors
provide the ownership boundary needed for further work; these results do not
justify calling the single-node path fully optimized.

## Client-reserved Restore identity (preceding increment)

On 2026-09-27, the Restore submission protocol moved identity reservation to the
client and added atomic Manager claim versus client cancellation. An accepted
operation keeps its handle after a lost or malformed submission ACK. The ACK no
longer carries an encoded RestoreResponse; terminal results and preparation
errors use the shared completion mapping. See [the implementation contract](communication-plan.md#operation-identity-and-source-retirement).

This increment establishes recoverable submission identity, **not a measured
throughput improvement**. Median Restore latency is close to the preceding
revision, with a small 4 KiB regression and some higher p99 values. Removing
response encoding did not establish an overall latency win. In these two measured revisions, CUDA submission
still belongs to the Manager; engine-local execution and model serving speedups
are not demonstrated by these measurements.

### Comparison and results

The baseline is `2c4041bac57c536986d769ea3343c4ae0b8e6fa1`, frozen as
`descriptor-production`; the candidate is `identity-production`. Both contain
release Managers and matching Python extensions without test hooks. The same
unchanged harness, H20, CPU affinity `8,10,12,14`, 256 MiB pool, 150 samples after
20 warmups and zero/1 ms prescribed idle were used. Each matrix ran
`A1 B1 B2 A2 A3 B3`, with one fresh Manager per run:

- One contiguous layer, one lease, 4 KiB / 256 KiB / 4 MiB.
- 36 split K/V layers, 32 leases, 18 MiB logical payload.

All twelve sessions passed GPU-byte validation and exact copy-counter checks.
No builds or other qualification tests ran during measurement. Lease lookup is
outside Restore timing; submit-to-ready includes encoding, IPC, admission, CUDA
work and terminal observation. Values are medians of three per-run percentiles,
not pooled samples. These short runs do not establish tail-latency equivalence.

Times in microseconds, without prescribed idle:

| Operation / shape | Baseline p50 / p99 | Candidate p50 / p99 |
| --- | ---: | ---: |
| Restore, contiguous 4 KiB | 25.93 / 65.90 | 26.62 / 46.30 |
| Restore, contiguous 256 KiB | 38.50 / 60.25 | 38.38 / 66.75 |
| Restore, contiguous 4 MiB | 222.66 / 240.43 | 223.21 / 283.19 |
| Restore, split 18 MiB, 32 leases | 1,359.48 / 1,449.03 | 1,362.47 / 1,466.27 |
| Publish, contiguous 4 KiB | 64.11 / 105.18 | 66.93 / 110.53 |
| Publish, contiguous 256 KiB | 108.76 / 143.66 | 113.35 / 173.69 |
| Publish, contiguous 4 MiB | 770.60 / 833.74 | 773.80 / 853.97 |
| Publish, split 18 MiB | 2,909.23 / 3,117.83 | 2,931.24 / 3,123.90 |

With 1 ms prescribed idle, Restore p50 was 39.13 → 40.84 us at 4 KiB,
51.54 → 52.00 us at 256 KiB, 229.93 → 230.07 us at 4 MiB, and
1,361.61 → 1,367.31 us for split 18 MiB. Zero-idle submission-only p50 was
8.87 → 8.69 us, 14.81 → 14.56 us, 109.38 → 109.10 us and
371.92 → 379.77 us respectively. Submission-only timing is not GPU completion.
Publish remains a control measurement; its data path was not optimized here.

Frozen production artifacts:

| Bundle | Manager SHA-256 | Extension SHA-256 |
| --- | --- | --- |
| Baseline | `0ea5dc4df3f0afd0c1e6e645eee8b95e190152105b5cf25f818b24c4ed38bfc8` | `69e1b911af8ae1d8abf986e6f61fe146dd1c0a9ea367760c69a154916c78ae5a` |
| Candidate | `308dc6b2590ed2445af4575ee0bf94c0bff9b4429414020467d34b28ed49b7cb` | `ee569e0eba40b7953afc785f24a2d0e30d5d5631151dee414406f37101f4b734` |

The candidate manifest records base `2c4041ba` and source patch SHA-256
`c493aa60944de176ebd9cf33d0c7b40bb5cd9b9f9c633c66dd068358c9499cca`.
The unchanged harness SHA-256 is
`06b0d99c105f8d8ff04d818d963c8031ad00a0aec95cd50aa26ad01fc975f040`.
This workspace retains per-run samples, manifests, CPU accounting and the
three-pair aggregate in
`/workspace/.orbitkv-tools/communication-microbench/runs/restore-identity/`.
Use the same reproduction commands and environment below, selecting these two
release bundles and the two workload shapes above. Clients and Managers must be
paired by revision because bootstrap v5 / channel ABI 8 replace v4 / ABI 7.

### Correctness qualification

Channel tests cover cancellation before claim, lost ACK after claim, mapped
completion after UDS closure, actual process exit, bounded slot reuse and
operation-ID collisions across clients and separately configured Managers.
The final channel gate passed 55 tests (one direct child-helper invocation is
ignored), and the Manager endpoint gate passed four tests. Workspace Clippy
with warnings denied and 374 Python unit tests passed.

The final production extension also passed 32 real Manager/native/GPU tests
against the matching-protocol fault Manager; 30 cases require the unselected
cuFile configuration and were skipped. The new cases corrupt the ACK after
claim, delay completion and lose its notification, then verify the original
handle, actual restored bytes and exactly one copy. Preparation rejection uses
the same handle with a Failed result. Existing SSD, codec, Publish and restart
fault gates still pass. No pinned vLLM/SGLang release environments or model
artifacts were available for serving E2E; these results do not qualify the
future engine-local grant lifecycle.

## Raw descriptors and split K/V coalescing

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
