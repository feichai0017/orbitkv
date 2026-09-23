# Sustained single-node serving

Both adapters complete bounded mixed-prefix traffic with native HBM caching,
OrbitKV DRAM and OrbitKV SSD backing. This extends the
[concurrent burst baseline](concurrent-performance.md) with ongoing arrivals
and final drain checks. It also exposed and fixed a vLLM restore-admission stall.

## Workload and controls

Measured on 2026-09-21 with Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, one NVIDIA H20, vLLM 0.29.0 and
SGLang 0.5.20. Runtime source is commit
`d75e3ff019942643f71ba9a3771cf840b4694ba5`; manifests retain source diffs for runs
started while documentation was being edited.

- Each configuration starts fresh services. Native prefix caching stays enabled.
- HBM KV capacity is limited to 16,384 tokens; requests use 64-token pages and
  BF16 state. This intentionally makes a 53,248-token prepared working set
  exceed GPU cache capacity.
- Twelve prepared prefixes alternate 1,024, 4,096 and 8,192 tokens. Each new
  request has a 75% chance of reusing one and a 25% chance of using a new prefix.
  The random sequence is reproducible by seed and request index.
- Concurrency is eight, with at most eight client requests outstanding. Admission
  stops after 60 seconds, then all submitted requests complete. Throughput uses
  the full request window, including that completion drain. A separate post-run
  check waits for query bytes, GPU work and SSD queues to become idle.
- Outputs contain 16 generated tokens. Native has no external cache; DRAM has a
  16 GiB host pool; SSD has a 4 GiB host pool plus 16 GiB backing. OrbitKV query
  preparation/lease budgets are 2 GiB in both cases.

These configurations compare serving under different cache capacities. They do
not isolate the SSD device's bandwidth or hold the number of completed requests
constant. A faster closed-loop server reaches more of the same generated request
sequence within the time limit. There is one measurement per configuration,
without confidence intervals or a long-duration soak.

## Results

| Engine | Cache | Completed | Requests/s | TTFT P50 (ms) | P95 (ms) | P99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| vLLM | Native HBM | 117 | 1.89 | 3,884 | 6,318 | 7,048 |
| vLLM | OrbitKV DRAM | 260 | 4.30 | 1,418 | 3,623 | 4,339 |
| vLLM | OrbitKV SSD | 235 | 3.72 | 1,505 | 4,504 | 5,244 |
| SGLang | Native HBM | 111 | 1.77 | 4,270 | 6,698 | 7,347 |
| SGLang | OrbitKV DRAM | 276 | 4.45 | 1,683 | 3,385 | 4,039 |
| SGLang | OrbitKV SSD | 226 | 3.68 | 1,780 | 4,434 | 5,053 |

There are 1,225 completed requests and no request failures in these six runs.
Relative to each engine's native control, DRAM throughput is 2.28× for vLLM and
2.52× for SGLang; SSD throughput is 1.97× and 2.08× respectively. These ratios
apply to this deliberately constrained HBM workload, not arbitrary serving.

| Engine / tier | SSD read (GiB) | GPU load (GiB) | Sampled query peak (GiB) |
| --- | ---: | ---: | ---: |
| vLLM DRAM | 0 | 91.79 | 1.995 |
| vLLM SSD | 63.76 | 66.22 | 1.995 |
| SGLang DRAM | 0 | 93.47 | 1.934 |
| SGLang SSD | 51.20 | 55.48 | 1.934 |

Transfer counters cover the full measured window and post-request drain, not
individual requests. SSD reads and GPU loads confirm that external restoration
occurred. They cannot attribute every hit to one tier or prove that every
prefetched byte was consumed. Memory peaks are sampled lower bounds. All final
query-byte, GPU-inflight and SSD queue gauges drained to zero; no query-budget
bypass was recorded.

Both SSD runs used `O_DIRECT` and the manager's io_uring path. The cache files
resided on an overlay filesystem, so the device inventory does not establish
which physical SSD backed them. Do not interpret these numbers as standalone
NVMe throughput.

## Restore-admission stall found by the workload

Before the fix, vLLM could leave all requests waiting despite completed GPU
restoration. A captured deferred queue had a cold 8,192-token request at its
head requiring 128 GPU blocks, with only 127 free. Behind it, an already
completed remote restore held 128 blocks. The scheduler stopped on the head's
allocation failure before visiting the completed restore, preventing either
request from making progress. No active SSD or GPU copy explained the wait.

The vLLM adapter now retains an admission gate from external allocation through
the request's first scheduled compute step. Other queries can prepare bounded
results, but their lookups continue to defer admission until that step. Copy
completion alone does not release the gate. Cancellation releases admission
while outstanding save/copy holds retain their own lifetime rules.

This is a conservative gate: it can reduce concurrent restore admission.
Future optimization must preserve progress through the engine's allocation
queue, with measured evidence, rather than remove the gate based on copy
completion alone. Five focused unit cases cover ready hits, misses, short
prompts, compute completion and cancellation with retained save holds.

The initial failed runs remain on the measurement host as
`runs/sustained-vllm-smoke/` and `runs/sustained-vllm-diagnose8/`. The
[archived regression evidence](https://github.com/feichai0017/orbitkv/tree/44c1e5f9a253aa7378c6187b2aeea9bff93df304/benches/results)
contains the queue snapshot and the fixed C4/C8 follow-up. Those follow-up
windows reused one service instance and are not mixed into the fresh-service
comparison above.

## Output and correctness evidence

All reused prompts in the six fresh-service runs matched their serial prepared
reference text. This is diagnostic evidence, not a batch-invariant correctness
proof. The earlier fixed C4 regression window retained three ordinary-mode
output differences; its following C8 window had none. They remain in the report.

Separately, the vLLM Qwen3-8B GPU correctness gate passed **6 tests, with 1
skipped**, using its native/deterministic controls, cold/partial/warm requests
and restart restoration checks. Existing SGLang recovery gates are described
in the [SSD report](ssd-performance.md). These checks do not qualify every
model, multi-rank serving or cross-engine state interchange.

## Reproduce and inspect

Install the matching engine environment and built OrbitKV package as described
in the [single-node guide](single-node.md). From the repository root:

```bash
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --workload sustained --concurrencies 8 --duration-seconds 60 \
  --max-requests 1000 --working-set 12 --query-budget-gib 2 \
  --host-gib 4 --ssd-gib 16 \
  --output benches/results/runs/sustained-vllm-ssd-new
```

For the DRAM control use `--host-gib 16 --ssd-gib 0`. For native use
`--backend native --ssd-gib 0`. For SGLang, change the interpreter, `--engine`
and output directory. Run sequentially on an otherwise idle GPU. Do not rebuild
native libraries while a Cache Manager is running. The manifests retain exact
commands, engine versions and environment details.

Each run writes prepared token sequences, per-request samples, window counters,
a manifest, summaries and service logs. Offline reporting rejects failed,
incomplete, duplicated or out-of-budget windows:

```bash
python -m benches.report \
  benches/results/runs/sustained-vllm-ssd-new \
  --output /tmp/orbitkv-sustained-report
```

Request rows, manifests, window counters and summaries remain in the
[historical dataset snapshot](https://github.com/feichai0017/orbitkv/tree/44c1e5f9a253aa7378c6187b2aeea9bff93df304/benches/results).
Large raw logs remain in ignored `benches/results/runs/sustained-{engine}-{tier}/`
on the measurement host.

## Next gates

Profile duplicate vLLM H2D restores and query/save interference, and evaluate
the subsequent [queued-warming implementation](queued-warming.md) before adding
restore-versus-recompute decisions. The measurements above predate that change. Extend sustained
loads with varied arrival rates, longer output, cancellation, delayed I/O and
engine/manager restarts. Keep these separate from a two-host independent-replica
cache-sharing gate, followed by P/D plus reusable cache. Catalog replication,
cross-host TP/PP and routing remain later milestones.
