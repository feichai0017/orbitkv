# OrbitKV benchmarks

Run performance experiments from the repository root. Benchmark code, harness
tests, curated measurements, and local raw runs all live here. Runtime Python
code belongs in `python/orbitkv/`; correctness gates belong in `python/tests/`.

## Layout

| Path | Responsibility |
| --- | --- |
| `client.py` | Admitted-query polling overhead with a held byte budget; no storage or model compute in the timed loop |
| `catalog.rs` | Rust directory cleanup microbenchmark, run through Cargo |
| `cpu_codec.rs` | Production scalar/AVX2/AVX-512/auto CPU FP8 conversion with an independent oracle before timing |
| `cost_observations.py` | Same-binary off/on observation overhead, three reversed-order pairs on both engines |
| `single_node.py` | Fixed-capacity cold, HBM-hit, and post-pressure experiment |
| `shared_cache.py` | Independent-replica serving requests with remote-byte, GPU-copy, output and reservation-drain evidence |
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
| `results/` | Maintained final reports; new output is ignored by default |
| `results/runs/` | Ignored raw runs: manifests, responses, counters, logs, and failures |

## CPU codec benchmark

`cpu_codec.rs` measures the production CPU E4M3FN encode/decode paths for BF16
and FP16 at 4 KiB, 256 KiB and 16 MiB of logical 16-bit data. It includes the
actual `codec/cpu.rs` inside its benchmark module to access private forced
backends; it adds no production API. `auto` uses the normal runtime dispatcher.
Unsupported AVX2/AVX-512 paths are recorded as `unsupported`, with no timing.

Run builds separately from live Managers or GPU benchmarks. Once the machine
is idle, use an available physical CPU for a repeatable affinity, and keep raw
CSV and logs outside tracked results:

```bash
cargo bench -p orbitkv-core --bench cpu_codec --no-default-features \
  --features cuda-13 --no-run
# Use the executable path printed by Cargo; choose a CPU in your allowed affinity.
taskset -c "$BENCH_CPU" /path/to/cpu_codec-executable --check
taskset -c "$BENCH_CPU" /path/to/cpu_codec-executable --seconds 0.25 --samples 3 \
  > /tmp/orbitkv-cpu-codec.csv 2> /tmp/orbitkv-cpu-codec.log
```

Before any timing, an independent nearest-value/ties-to-even oracle checks all
65,536 BF16/FP16 input patterns, all 256 FP8 decode codes, rejected nonfinite
and out-of-range inputs, SIMD tails, unaligned slices and every timed buffer
against each available backend. A mismatch fails the process. The deterministic
timed corpus samples finite in-range 16-bit patterns, including signed zero and
subnormals; it is synthetic data, not captured model activations.

Each CSV row contains the backend requested and selected, operation, sample,
actual iteration count, elapsed seconds, input/output bytes per iteration,
nanoseconds per iteration and logical GiB/s. The throughput numerator is the
uncompressed 16-bit byte count for both operations: decode reads half that
many encoded bytes. Tables and buffers are reused; allocation, oracle checks
and three warmup calls are outside timing. Backend order rotates between
samples. Keep the individual samples and report their spread; these measurements
describe CPU conversion only and do not measure GPU kernels, SSD transfer,
end-to-end cache recovery, TTFT or serving throughput.

## Shared-cache qualification

For shared replicas, use the [shared-cache qualification guide](../docs/shared-cache-qualification.md).
Its HTTP driver works with either engine on existing deployments. The model-serving
restart gate lives in `python/tests/e2e/test_shared_cache.py`; same-host results do
not qualify physical two-host or RDMA deployment.

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
Set `ORBITKV_CACHE_MANAGER_BINARY` to select an explicitly built Manager.
The extension and Manager must come from the same source revision and protocol.
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

Choose `--ssd-backend uring`, `auto`, or `cufile` for both engines. This benchmark
keeps `uring` as its reproducible default; the Manager itself defaults to `auto`.
The cuFile backend performs complete-group GPU writes and demand restores through bounded
GPU staging; fragmented writes and preparation still use DRAM. See
[GPU storage](../docs/gds.md) for configuration, ownership and hardware limits.
`python -m benches.gds --help` describes the complete bare-metal acceptance
sequence, including native-path statistics and matched pressure workloads.

Add `--ssd-gib 32` with `--backend orbitkv` for either engine. This creates a
dedicated cache file inside the empty result directory, verifies `O_DIRECT` on
the Manager's open descriptor, and records the filesystem and device inventory
in `storage.json`. The owned payload file is removed after services stop; all
measurement evidence remains. An overlay mount does not identify its physical
backing device, and these results must not be presented as bare-device bandwidth.
To keep raw output elsewhere, pass `--ssd-dir /mnt/nvme/orbitkv-bench` pointing
to an existing writable directory. Only a new private subdirectory there holds
the cache; it is removed after services stop. DRAM runs create no SSD payload.

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
For cuFile restores, load duration also includes SSD reads and GPU scatter;
the explicitly selected io_uring host lane includes SSD reads and host
materialization before H2D or decoding. These timers overlap the SSD read timers.
Reports count both io_uring and cuFile read bytes; sustained/burst results retain
the individual counters. `manager-usage.json` records process CPU seconds and
Linux I/O accounting for the complete workload, including warmup and pressure.
Linux process I/O accounting is not a measurement of GPU DMA bytes.

```bash
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --ssd-gib 32 --output benches/results/runs/qwen3-8b-ssd-vllm
```

For an independent read-path control, keep cuFile initialized and direct GPU
writes enabled while restoring through io_uring host staging:

```bash
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --ssd-gib 16 --host-gib 1 --gpu-tokens 8192 --prefill-tokens 4096 \
  --lengths 1024 4096 --ssd-backend cufile --ssd-read-path uring \
  --output benches/results/runs/vllm-cufile-uring-reads
```

Use the SGLang environment and `--engine sglang` for its matching control.
`--ssd-read-path` defaults to unset; `uring` and `cufile` override only restores.
Explicit-path measurements require `--queue-warmup off --prepare-requests off`,
because aggregate read counters cannot separate demand from preparation reads.
The manifest records `ssd_backend` and `ssd_read_path` separately from the
`O_DIRECT` file flag. Verify positive cuFile write bytes and io_uring read bytes
alongside GPU restores; cuFile initialization does not prove the read route.
Use the [documented compatibility environment](../docs/gds.md#enable-and-qualify)
for this container. A forced io_uring read is not native GDS read qualification
and cannot use `--gds-stats`. These are reproduction commands, not new performance
results; the existing qualification matrix is unchanged.

## Storage codec comparisons

Both engines accept `--storage-codec none|ans|fp8|turboquant-4|turboquant-3`
and `--storage-codec-budget 64mb` for OrbitKV DRAM and SSD. The budget is per
transfer worker, accepts bytes or binary `kb`/`mb`/`gb`, and does not change the
engine KV capacity. Use matching prebuilt Managers/extensions and set
`ORBITKV_NVCOMP_LIBRARY` for ANS where needed. The manifest records the selected
Manager's SHA-256, codec library environment, configured HBM/DRAM/SSD budgets,
and logical working-set bytes so an old binary is distinguishable from the
current source checkout. Build artifacts before starting any serving run.

This matched matrix runs real streaming inference with synthetic token prompts.
Start each revision in a new output root; keep the engine environment, model,
capacities, prefill limit and sampling controls unchanged. Choose an existing
SSD mount for `--ssd-dir`; raw logs stay in the temporary output directory.

```bash
export ORBITKV_CACHE_MANAGER_BINARY=/opt/orbitkv/manager
export ORBITKV_NVCOMP_LIBRARY=/path/to/nvidia/libnvcomp/lib64/libnvcomp.so.5
run_root=$(mktemp -d /tmp/orbitkv-codecs.XXXXXX)
for engine in vllm sglang; do
  for ssd_gib in 0 8; do
    if [ "$ssd_gib" -eq 0 ]; then host_gib=8; else host_gib=1; fi
    for codec in none ans fp8 turboquant-4 turboquant-3; do
      .venv/"$engine"-release/bin/python -m benches.single_node \
        --engine "$engine" --backend orbitkv --model /workspace/models/qwen3-8b \
        --storage-codec "$codec" --storage-codec-budget 64mb \
        --host-gib "$host_gib" --ssd-gib "$ssd_gib" --ssd-backend uring \
        --ssd-dir /mnt/nvme/orbitkv-bench \
        --gpu-tokens 8192 --prefill-tokens 4096 --output-tokens 16 \
        --workload sustained --lengths 4096 --working-set 12 \
        --concurrencies 4 --duration-seconds 3600 --max-requests 64 \
        --reuse-ratio 0.75 --seed 20260920 \
        --output "$run_root/$engine-$ssd_gib-$codec"
    done
  done
done
```

For Qwen3-8B BF16, these 49,152 prefix tokens represent 6.75 GiB of logical KV.
The TQ4 payload estimate, including four exact layers, is about 2.32 GiB;
TQ3 is about 1.95 GiB before additional alignment and fallback. These estimates
exceed the 1 GiB host budget; actual SSD reads and GPU restores must still be
verified in each run. The engine retains at most 8,192 KV tokens (1.125 GiB).
Configured compression does not prove cache hits, complete working-set
residency, or faster inference.

The DRAM-only configuration has an 8 GiB host pool. This fits the prepared
prefixes, but the fixed cohort also introduces 15 cold prompts: total distinct
logical KV reaches about 15.19 GiB. DRAM results therefore include eviction and
the capacity effects of compression.

Require all 64 requests, `stop_reason=request_limit`, and 64/64 matching prompt
hashes against the same engine/tier's `none` control. Throughput uses actual wall
time, including completion of admitted requests. A duration-limited alternative
such as `--duration-seconds 20 --max-requests 512` can finish different request
counts; report comparison coverage instead of treating those cohorts as identical.
Repeat comparisons and reverse codec/revision order before drawing general
performance conclusions.

Summaries retain labelled logical/stored publication bytes, D2H/H2D payload
bytes, encode/decode durations, skip reasons and decode failures. The
`encoded_publication_stored_fraction` applies only to slots that contain
encoded segments, including aligned raw siblings; it excludes raw-only slots
and is not whole-cache compression. Codec durations include transfers and
fallback work, so they are not isolated kernel times or additive TTFT stages.
Workspace allocations, encode/decode batch counts and segment histogram
sums/counts expose batching and scratch reuse. Sampled workspace peaks include
idle retained arenas; active reservation bytes must drain, retained arenas may
remain. Missing metrics from older binaries are absent/null, never invented
zero measurements. `manager-usage.json` also retains whole-workload counter
changes, including preparation, separately from measured serving windows.

Cold/prepared output differences remain in every summary. For a matched
`none` control, compare measured prompt hashes and exact generated text:

```bash
python -m benches.report "$run_root/vllm-8-ans" "$run_root/vllm-8-turboquant-4" \
  --reference-run "$run_root/vllm-8-none" --output "$run_root/vllm-ssd-report"
```

Use separate reports for each engine/tier. Reports retain mismatch counts and
uncompared requests; older runs without prompt hashes cannot establish this
comparison. Greedy text agreement is a workload diagnostic, not application
accuracy or general model-quality qualification. Concurrent batching can also
change greedy outputs. TTFT means client time to first nonempty streamed text;
throughput counts observed completion tokens over measured wall time. Serial
TTFT results do not imply sustained throughput.

For native GDS, use the existing prerequisite verifier and prebuilt test
artifacts on a bare-metal NVMe host:

```bash
.venv/vllm-release/bin/python -m benches.gds \
  --ssd-dir /mnt/nvme/qualification --model /workspace/models/qwen3-8b \
  --manager /opt/orbitkv/manager --fault-manager /opt/orbitkv/manager-faults \
  --core-test /opt/orbitkv/cufile-test --gds-tools /usr/local/cuda/gds/tools \
  --cufile-library /usr/local/cuda/lib64/libcufile.so.0 \
  --storage-codecs none ans fp8 turboquant-4 turboquant-3 \
  --storage-codec-budget 64mb --workloads serial sustained \
  --host-gib 1 --ssd-gib 8 --gpu-tokens 8192 --prefill-tokens 4096 \
  --length 4096 --working-set 12 --concurrencies 4 \
  --duration-seconds 20 --max-requests 512 --output-tokens 16 --seed 20260920
```

This covers both engines, both tiers and all selected codecs, with SSD controls
for io_uring/auto/cuFile. It retains bare-metal/NVMe prerequisites, disables
compatibility mode, requires per-process native reads and writes, and rejects
fallback. Whole-process native statistics do not identify the representation
of each transferred byte; the report preserves that attribution limit.
Strict DRAM/SSD correctness gates run for each selected codec. A failed gate
remains a qualification failure while the serving matrix continues to collect
diagnostics. The codec budget option controls the benchmarks; the existing
correctness fixtures retain their default 64 MiB budget. Keep raw artifacts in
the private run directory and publish only reviewed final summaries.

## Sustained mixed traffic

For [retention and SSD admission](../docs/cache-policies.md), compare
`--cache-protected-percent 0|80` and `--ssd-write-policy all|reuse`
independently. Both flags configure the Rust Manager; `reuse` requires SSD.
Keep warming and preparation off for this comparison. The report records
protected byte peaks, promotions/demotions and admission skip reasons separately
from full write-queue drops. Use the same capacities and request sequence,
reverse run order, and repeat before choosing defaults. A lower SSD write count
alone does not establish faster serving.

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

Consumer-owned preparation has a separate `--prepare-requests on|off` control;
it cannot be combined with `--queue-warmup on`. The Manager read controls are
`--read-batch-mib`, `--read-timeout-ms` and `--read-max-batches`, all disabled by
default for ordinary demand. Prepared reads always use at most 32 MiB per
batch, or one oversized page. Retained prepared pages remain budgeted.
See [ownership and stopping](../docs/request-preparation.md) and the
[ordinary recovery profile](../docs/recovery-performance.md).
The [paired reproduction script](results/20260922-preparation/reproduce.sh)
compares preparation on/off three times, reverses the middle pair's order,
and separates unbounded, deadline and one-batch controls. Fixed request caps
preserve the request sequence; duration-only runs can sample different traffic.
Additional DRAM-only runs record their larger host capacity explicitly.

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

For a larger cache working set, use `--working-set 32 --lengths 4096 8192
--gpu-tokens 65536 --host-gib 4 --ssd-gib 64 --query-budget-gib 3`.
Qwen3-8B BF16 needs about 27 GiB for these prepared prefixes, exceeding the
9 GiB GPU KV budget plus 4 GiB Manager DRAM. Sustained runs size the engine's
per-request context limit from the largest prompt plus 64 output tokens;
total GPU cache capacity can exceed one request's model context limit.
Use `--prefill-tokens` (default 8192) to compare compute scheduling quanta
separately from cache changes; it sets vLLM's `--max-num-batched-tokens` and
SGLang's `--chunked-prefill-size`. Keep it fixed in revision comparisons.
Keep `--max-requests` high enough to reach the configured duration. Compare
identical budgets, request seeds and read controls across revisions, and retain
both read and write counters along with resource-drain evidence.

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
Separate cold/reused-prefix TTFT P95 values expose scheduling tradeoffs. A
reused-prefix choice may still miss the cache; these fields are not per-tier
hit latencies.
Reports retain pool-allocation failures, SSD I/O failures and dropped write
queue submissions. The total and speculative query counters are independent
of phase diagnostics; sustained validation rejects observed budget overruns.

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

Add `--trace-transfers` to either engine's OrbitKV run for request-correlated
discovery, host-read, restore and completion observations in `timeline.jsonl`
and `timeline-summary.json`. Manager restore time includes dispatch and worker
queueing; the load histogram separately measures the H2D worker task including
stream synchronization. Completion signal and delivery intervals start at the
GPU worker's terminal timestamp. A signal event records the notification attempt,
whereas delivery records the terminal poll response. These are distinct from
the engine's own restore-submit to GPU-ready interval. Shared restore batches
are counted once, using the Manager epoch and operation ID. Dense ordinary
queries combine candidate discovery and reading; missing separate discovery
samples do not mean discovery takes zero time. No subtraction between process
clocks is used, and these overlapping intervals must not be added to obtain TTFT.

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
baseline, final path and old-client/new-Manager control, with a link to the
historical per-batch evidence.

## Cost observation overhead

Build and test native artifacts before running any serving process. The harness
never invokes Cargo and refuses to start alongside a build or another Manager.
It runs three off/on pairs for vLLM and SGLang in DRAM-only and pressured
io_uring-SSD modes, plus an ANS SSD control. Every second pair reverses order;
request seed, trace, capacity, engine release and Manager binary remain matched.
The process startup switch is `ORBITKV_COST_OBSERVATIONS=0|1`; observations are
off by default, and neither setting changes backend or recovery decisions.
The default cohort has 128 requests, concurrency
four and 1024/4096-token prefixes. DRAM capacity is 16 GiB, SSD-mode DRAM 1 GiB,
SSD capacity 16 GiB and engine KV capacity 8192 tokens.

```bash
export ORBITKV_NVCOMP_LIBRARY=/path/to/libnvcomp.so.5
.venv/vllm-release/bin/python -m benches.cost_observations \
  --model /path/to/qwen3-8b \
  --manager /absolute/path/to/prebuilt/orbitkv-cache-manager \
  --ssd-dir /path/to/ssd-test-directory \
  --output benches/results/runs/cost-observations
```

Predeclared budgets are 3% throughput loss, 3% TTFT p50 growth and 5% TTFT
p95/p99 growth, assessed as medians of paired ratios. TTFT p95 must be at most
2000 ms and response-average decode p95 at most 100 ms/token. Engine-reported
ITL histograms also require at least 95% of samples within the exact 100 ms
bucket boundary. Their p50/p95/p99 are bucket-interpolated estimates with bounds:
vLLM measures engine-core token intervals; SGLang measures tokenizer receipt
intervals, averaged over and weighted by newly received tokens. Neither uses
HTTP chunk spacing as token timing. Aggregate ITL cannot be associated with
individual requests, so request goodput uses TTFT and response-average decode.
A missing GPU restore, O_DIRECT SSD read/write, required encoded work, executed
prediction sample or raw-copy shadow sample fails the evidence gate. Exact
concurrent output differences are reported; independent correctness gates are
required because engine batching can affect generated text.

The output directory contains the predeclared `plan.json`, raw traces under
`runs/`, and only final aggregate JSON/CSV under `final/`. Use `--report-only`
with identical arguments to regenerate the summary. Keep raw data untracked.
The existing sampler scrapes Manager metrics every 25 ms. The paired delta
includes the additional cost-series export and sampling at that frequency;
it does not isolate observer hot-path cost. These finite cohorts measure
instrumentation overhead on the recorded host; they do not establish a throughput ceiling, dynamic-path benefit, native GDS
performance or distributed-cache qualification. See the final evidence in the
[implementation handoff](../docs/implementation-plan.md#p41-final-evidence).
