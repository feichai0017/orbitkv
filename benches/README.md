# OrbitKV benchmarks

Run performance experiments from the repository root. Benchmark code, harness
tests, curated measurements, and local raw runs all live here. Runtime Python
code belongs in `python/orbitkv/`; correctness gates belong in `python/tests/`.

## Layout

| Path | Responsibility |
| --- | --- |
| `client.py` | Admitted-query polling overhead with a held byte budget; no storage or model compute in the timed loop |
| `catalog.rs` | Rust directory cleanup microbenchmark, run through Cargo |
| `single_node.py` | Fixed-capacity cold, HBM-hit, and post-pressure experiment |
| `launch.py` | Engine/backend commands and matched memory budgets |
| `runtime.py` | Owned process groups, readiness, teardown, and launch manifest |
| `workload.py` | Token-exact requests, streaming timings, and pressure traffic |
| `concurrent.py` | Closed-loop bursts, shared/mixed prefixes, batch counters and sampled memory peaks |
| `sustained.py` | Bounded continuous traffic mixing reusable prefixes with cold requests |
| `metrics.py` | Cache-source evidence and statistical summaries |
| `report.py` | Offline CSV/JSON reports from complete raw runs |
| `serving.sh` | vLLM serving measurements against an already running endpoint |
| `sharegpt.py` | Multi-turn workload using the pinned vLLM benchmark scripts |
| `tests/` | CPU-only checks for measurement and report correctness |
| `results/` | Reviewed CSV/JSON measurements committed to Git |
| `results/runs/` | Ignored raw runs: manifests, responses, counters, logs, and failures |

## Single-node comparison

Use separate environments for the validated vLLM 0.29.0 and SGLang 0.5.20
releases. Follow the [deployment guide](../docs/single-node.md) to build/install
the release wheel. The harness needs `requests`; the selected engine provides
its tokenizer and GPU runtime. LMCache and FlexKV are comparison dependencies,
not OrbitKV package dependencies.

```bash
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b

.venv/sglang-release/bin/python -m benches.single_node \
  --engine sglang --backend orbitkv --model /workspace/models/qwen3-8b
```

By default, each run gets a new timestamped directory below `results/runs/`.
Use `--output benches/results/runs/<name>` for an explicit empty directory. The
harness owns the engine and the OrbitKV/LMCache process; do not start other GPU
workloads during measurement. An OrbitKV source run uses the staged Cache Manager
binary in `python/orbitkv/` and the Python adapters from this checkout.
Listener ports, including SGLang's rendezvous port, are chosen outside Linux's
outgoing ephemeral range to reduce startup conflicts during GPU initialization.

Use `--backend native`, `cpu`, `orbitkv`, `lmcache`, or `flexkv`. Compare within
one engine, using identical model, GPU token capacity, host capacity, request
sequence, and dependencies. Supported model scope is dense Qwen3 in BF16, TP=1.
FlexKV compatibility failures observed on these releases are documented in the
[measurement report](../docs/single-node-performance.md); accepting a backend
option does not mean its current integration can start successfully.

All outputs and cache-source evidence are retained, including mixed hits,
misses, and generated-text differences. A startup/request failure writes
`failure.json` and is not a latency result. The serial workload measures TTFT;
it does not establish concurrent goodput or production tail latency.

## SSD restoration

Add `--ssd-gib 32` with `--backend orbitkv` for either engine. This creates a
dedicated cache file inside the empty result directory, verifies `O_DIRECT` on
the Manager's open descriptor, and records the filesystem and device inventory
in `storage.json`. The owned payload file is removed after services stop; all
measurement evidence remains. An overlay mount does not identify its physical
backing device, and these results must not be presented as bare-device bandwidth.

The workload adds `after_host_eviction` after the original three phases. It
applies fresh GPU pressure, waits for observed SSD writes to become idle, and
evicts Manager DRAM through its HTTP administration endpoint while preserving
SSD copies. Eviction responses and counters are stored with each sample. These
preparation operations are outside request latency. This is a controlled tier
experiment, not a natural host-memory-pressure workload or a durability test.

SSD read bytes alone do not prove an inference cache hit. Reports retain GPU
load bytes, engine cache-hit evidence, and `ssd_reads_without_gpu_restore` so
a late prefetch followed by recomputation stays visible. `ssd_prefetch_p50_ms`
includes allocation, queueing, reads and reconstruction; `load_task_p50_ms`
includes H2D task construction and synchronization. Histogram sums describe
instrumented operations and are not an additive decomposition of client TTFT.

```bash
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --ssd-gib 32 --output benches/results/runs/qwen3-8b-ssd-vllm
```

## Sustained mixed traffic

For sustained mixed traffic, use `--workload sustained`. Each concurrency level
prepares an independent working set, then continuously replaces finished client
requests until the admission duration or request cap is reached. The reusable
prefix fraction is probabilistic; other requests have new token sequences and
exercise saves alongside restores. No explicit cache cleanup or pressure request
runs inside the measured window. A working set larger than HBM creates natural
GPU pressure; a smaller host pool with SSD enabled can exercise disk reads.
Verify the counters: a working-set configuration alone does not prove a tier hit.

Automatic queued warming is experimental and disabled by default. For comparisons, run the same OrbitKV workload with
`--queue-warmup on` and `--queue-warmup off`. Add `--trace-transfers` to record
request-correlated `cache_timeline` JSON in both service logs. Keep tracing equal
between controls. Hints exceeding a quarter of the query budget are skipped;
choose and report the budget explicitly. See [the lifecycle and current
limits](../docs/queued-warming.md). Streaming samples also retain the engine's
request ID for correlating measured traffic with the logs.
`timeline.jsonl` extracts only measured requests; `timeline-summary.json` reports
stage coverage and preparation/restore/queue intervals. Durations use one
process's monotonic clock or Manager-local elapsed time, never a subtraction
of clocks on different hosts. Missing stages are not counted as zero latency.
The [initial Qwen3-8B pressure controls](../docs/queued-warming.md#initial-pressure-controls)
increased SSD bytes per request without a throughput gain. These results also
retain a native HBM control for SGLang's prepared-reference output differences.
The [page-use/reclamation controls](../docs/queued-warming.md#page-use-and-reclamation-controls)
retain complete starting/ending counters and show why warming stays opt-in:
vLLM admits few hints, while SGLang releases most prepared pages unused.
Reports also retain warmup prepared/restored/unused byte deltas, completed
byte-seconds by outcome, and pending bytes before/after the window and at the
sampled peak. These count physical page footprints, not exact layer-copy bytes.
Carry-in and live pending pages prevent treating a window ratio as a completed
cohort hit rate. Enabling warming or transfer tracing for another backend is
rejected rather than recorded as an ineffective control.

```bash
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --workload sustained --concurrencies 1 4 8 --duration-seconds 60 \
  --working-set 12 --reuse-ratio 0.75 --max-requests 10000 \
  --host-gib 4 --ssd-gib 16 --query-budget-gib 2 \
  --output benches/results/runs/sustained-vllm-ssd
```

Use the SGLang environment and `--engine sglang` for its gate. For DRAM coverage,
omit `--ssd-gib` and choose a host budget that fits the working set. Run a separate
`--backend native` control with the same workload and GPU budget. Each backend
executes the same deterministic request-generation recipe, but duration-based
closed-loop runs may complete different numbers of requests.

`prefixes.jsonl` retains prepared token inputs and serial reference outputs.
`samples.jsonl` retains every measured response, client admission/start/finish
times, cached tokens and reference comparison. Cold requests can be regenerated
from the recorded seed, concurrency and request index. `windows.jsonl` stores
each complete window's counters, sampled memory peaks, stop reason and post-run
resource state. The cap bounds retained samples; capped windows are explicitly
marked and must not be presented as completing the configured duration.

Throughput divides completions by wall time including the final admitted
requests' completion, excluding preparation and the subsequent cache-drain check.
The drain check waits up to 30 seconds after the settling interval for query,
copy and SSD work to become idle; it fails a run that retains query ownership.
Cache counters include this drain, so they are window-level evidence, not
per-request tier attribution. Memory peaks are sampled every 25 ms and are
lower bounds. `decode_ms_per_token_p50` is a response-level estimate from text
TTFT and completion counts, not a measured inter-token arrival distribution.
Output differences from serial preparation are retained as diagnostics: greedy
sampling does not establish batch-invariant correctness. Use the engine E2E
gates and native/deterministic controls to investigate differences.

The [sustained report](../docs/sustained-performance.md) records fresh-service
native/DRAM/SSD controls for both engines and the vLLM admission-stall regression.

## Concurrent bursts

For concurrent qualification, use `--workload concurrent --concurrencies 1 4 8`.
Each burst uses shared prefixes or independent mixed-length prompts; cold,
post-GPU-pressure, and optional post-host-eviction phases have fresh pressure
preparation. `samples.jsonl` contains individual responses and
`batches.jsonl` contains each burst's aggregate tier counters and wall time.
Counters are never attributed to individual overlapping requests. Engine cached
token reports alone do not distinguish HBM, DRAM, and SSD in these summaries.

```bash
.venv/sglang-release/bin/python -m benches.single_node \
  --engine sglang --backend orbitkv --model /workspace/models/qwen3-8b \
  --workload concurrent --concurrencies 1 4 8 --repeats 3 \
  --ssd-gib 32 --query-budget-gib 2 \
  --output benches/results/runs/query-budgets-sglang
```

Run vLLM with its release environment and `--engine vllm`. The query budget can
be smaller than aggregate demand while every individual prefix still fits.
Reports include TTFT p50/p95/p99, burst throughput, client decode milliseconds
per output token, budget waits/bypasses, coalesced reads, and memory peaks sampled
every 25 ms. These peaks are lower bounds on the true peak; lifecycle gates
separately check hard accounting limits. The fixed number of bursts is not a
steady-state load, natural memory-pressure experiment, or tail-latency SLO.
The [recorded concurrent baseline](../docs/concurrent-performance.md) includes
the discovered SGLang admission regression and native/deterministic controls
for output differences; ordinary greedy output is not assumed batch invariant.

## Report existing runs

```bash
python -m benches.report \
  benches/results/runs/<cpu-run> benches/results/runs/<orbitkv-run> \
  --output benches/results/runs/<report-name>
```

This needs only `requests`, not torch, either inference engine, or the native
extension. It writes `summary.csv` and `summary.json` without mixing samples
across runs. Incomplete workloads, duplicate samples, and failed runs are
rejected. Copy reviewed exports into `results/` when publishing a measurement;
keep raw logs and dataset downloads in the ignored `results/runs/` directory.

## Additional workloads and harness checks

```bash
BASE_URL=http://127.0.0.1:8000 MODEL=/path/to/model LABEL=orbitkv \
  bash benches/serving.sh

git submodule update --init third-party/vllm
.venv/vllm-release/bin/python -m benches.sharegpt \
  --model /path/to/model --dataset-path /path/to/sharegpt.json

uv run --isolated --no-project --with pytest --with requests pytest benches/tests
```

The ShareGPT workload requires the dependencies listed by the pinned vLLM
`benchmarks/multi_turn/requirements.txt`. These endpoint workloads have their
own workload definitions; do not combine their numbers with the fixed-capacity
single-node comparison.

## Catalog cleanup

```bash
cargo bench -p orbitkv-catalog --bench unregister_node
```

This benchmark registers one million keys, then removes an owner of 10,000.
Inventory population and destruction of the remaining directory are outside
the timed section. It measures owner-index cleanup, not remote discovery or
end-to-end serving latency. Criterion writes local raw output to
`target/criterion/`; copy reviewed reports into `benches/results/`.

## Native client polling

```bash
PYTHONPATH=python .venv/sglang-release/bin/python -m benches.client \
  --label rust-client --output benches/results/runs/client-poll
```

Requires built source artifacts in `target/release`, the native extension, and
one CUDA GPU for registration. It publishes 1024 pages, leases them to occupy
the 64 MiB instance budget, then measures admitted pending queries with
64/256/1024 hashes. A native `BlockHashes` batch is constructed outside the
timed loop and reused across polls. The script owns its Manager process and
records three batches of 1000 calls per size, wall percentiles and caller thread
CPU time. This isolates the client/control path; it is not a TTFT, SSD throughput,
or production concurrency measurement. Do not run it alongside Cargo builds or
other GPU workloads.

See the [controlled client measurements](../docs/client-performance.md) for the
baseline, final path and old-client/new-Manager control. Raw per-batch summaries
are in `results/client-control-2026-09-22.json`.
