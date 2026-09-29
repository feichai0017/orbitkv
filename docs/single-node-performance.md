# Single-node cache measurements

Use the same inference engine to compare cache backends. Comparing a vLLM run
directly against an SGLang run also measures their attention kernels, scheduler,
and frontend; it does not isolate the cache implementation.

For GPU batching, encoded DRAM/SSD and CPU AVX-512 measurements, see
[storage codec qualification](storage-formats.md#qualification). Those pressure
workloads are separate from the ordinary restore measurements below.

The first reference backends are vLLM's `OffloadingConnector` with pinned CPU
memory and SGLang's HiCache with a CPU pool. Native HBM-only prefix caching is
the control for each engine. Independent LMCache and FlexKV comparisons must
pin and validate the engine, PyTorch, CUDA, and connector versions together.
Mooncake Store can extend that matrix.
Mooncake TENT alone is a transport library, so a raw transfer-engine
bandwidth result is not an end-to-end cache comparison.

References: [vLLM KV offloading](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/),
[SGLang HiCache](https://docs.sglang.io/docs/advanced_features/hicache),
[LMCache compatibility](https://docs.lmcache.ai/getting_started/compatibility.html),
[FlexKV](https://github.com/taco-project/FlexKV),
[Mooncake](https://github.com/kvcache-ai/Mooncake).

## Deterministic comparison controls

`benches.single_node --deterministic-inference` requests vLLM batch-invariant
execution with `FLASH_ATTN`, or SGLang's deterministic inference mode. Use it
for a separate output-parity cohort, including concurrent traffic. Apply the
same flag to every backend: it changes computation kernels and its latency must
not be compared with ordinary serving as if only the cache changed. The manifest
records the flag and command; output comparison rejects mismatched modes. Exact
text on these synthetic prompts remains distinct from task-quality evaluation.

## Engine-local Restore serving qualification

The current completion-evidence and cost-key builds passed the same direct DRAM
model gates on September 28: vLLM **6 passed / 1 skipped**, SGLang **1 passed /
1 deselected**. See [the final artifact hashes and scope](fault-qualification.md#completion-evidence-requalification).
The performance tables below still refer to their explicitly identified earlier
builds; passing a new correctness gate does not update historical latency data.


The September 28, 2026 rerun uses the `eb61d166` engine-local raw Restore
implementation and frozen `compact-restore-production` artifacts. Historical
latency tables below predate this cutover and are not measurements of this build.
The model is Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, with all five weight shards verified
against their repository SHA-256 values, on one NVIDIA H20 (97,871 MiB).

| Production artifact | SHA-256 |
| --- | --- |
| Cache Manager | `3435bf33ada1927a419420fa104cc2e8cf75fce3ca8e294416253614aa6da638` |
| Python native extension | `978bd219c77e544100e87f0add4ee39a919766e59cbb83af055439d02dbb2db4` |

The isolated release environments use Python 3.11.10, PyTorch 2.13.0+cu130,
NumPy 2.2.6, and LMCache 0.5.5. vLLM 0.29.0 uses Transformers 5.17.0;
SGLang 0.5.20 uses Transformers 5.12.1. Backends within an engine share its
environment. Cross-engine numbers would include these runtime differences.

The direct-transfer, unencoded DRAM correctness gates passed:

- vLLM: **6 passed, 1 skipped in 433.41 seconds**. Twelve native-path and
  OrbitKV outputs match exactly across cold/warm requests, prefix extension,
  rollback, multiple conversation rounds, and engine restart while retaining
  the Manager. Positive cache-save and cache-hit bytes are required. The
  recurrent-state case is inapplicable to this dense model. This deterministic
  gate uses a no-op connector in its native control to match computation shape;
  the performance native control has no external connector.
- SGLang: **1 passed, 1 deselected in 296.26 seconds**. The DRAM case checks
  flush/restart recovery, four concurrent restart requests, positive restored
  GPU bytes, and a cold control under a changed model fingerprint. Text and
  token IDs match the corresponding native cold/reuse computation shapes;
  output log probabilities use an absolute tolerance of 0.05. The SSD case
  was not selected.

Reproduce with the pinned release environments and matching production binaries:

```bash
cd python
../.venv/vllm-release/bin/python -m pytest -m e2e \
  tests/e2e/test_vllm_e2e_correctness.py \
  --model /workspace/models/Qwen3-8B --max-model-len 4096
../.venv/sglang-release/bin/python -m pytest -m e2e \
  tests/e2e/test_sglang_direct_e2e.py -k dram \
  --model /workspace/models/Qwen3-8B
```

Full stdout, engine/Manager logs, and environment freezes are retained on the
measurement host under `/workspace/.orbitkv-tools/e2e-vllm-correctness*`,
`e2e-sglang-correctness*`, and `e2e-{vllm,sglang}-requirements.txt`. These gates
establish the tested serving correctness; they do not establish performance,
multi-GPU behavior, arbitrary graph modes, or model task quality.

### Matched vLLM end-to-end comparison

The recorded communication-branch build completed **180 measured serial requests** across native HBM,
vLLM `OffloadingConnector` CPU offload, OrbitKV DRAM, and LMCache MP DRAM.
All use the environment above, CPU affinity 8–23, 16,384 GPU KV tokens
(2.25 GiB), a prefill batch limit of 8,192 tokens, 64-token pages, and 16 output
tokens. External pools are 16 GiB; native HBM has no host pool. OrbitKV uses
`direct`, with no storage codec, SSD, or speculative preparation. Each backend
starts fresh services and uses the same seed and prompt hashes.

Each of five independent prefixes per length is measured cold, resident, and
after two unrelated 12,288-token requests evict it from HBM. Startup, warmup,
pressure generation, and the 1.2-second settling waits are outside request
latency. TTFT measures time to the first nonempty streamed text at the client;
E2E measures time through all 16 output tokens. They include frontend and
scheduling costs, not just cache transfer.

**TTFT p50 after HBM pressure, milliseconds; five requests per cell:**

| Backend | 1,024 tokens | 4,096 tokens | 8,192 tokens |
| --- | ---: | ---: | ---: |
| Native HBM, evicted | 117.83 | 476.07 | 1,007.43 |
| Native CPU offload | 21.75 | 32.15 | 47.74 |
| OrbitKV DRAM, engine-local direct | 22.94 | 33.31 | 54.54 |
| LMCache MP DRAM | 25.74 | 39.42 | 55.45 |

Every pressure request is verified as a miss for native HBM and an external
restore for the other three backends. OrbitKV is 5.14×/14.29×/18.47× faster
than recomputation in this workload, but native CPU offload is still faster:
OrbitKV adds 1.20/1.16/6.79 ms, or 5.5%/3.6%/14.2%. Its TTFT is
10.9%/15.5%/1.7% lower than LMCache's. The 8K difference is small; five
observations do not establish a stable advantage.

**Complete request latency p50 after HBM pressure, milliseconds:**

| Backend | 1,024 tokens | 4,096 tokens | 8,192 tokens |
| --- | ---: | ---: | ---: |
| Native HBM, evicted | 207.57 | 568.12 | 1,103.55 |
| Native CPU offload | 110.99 | 123.53 | 143.74 |
| OrbitKV DRAM, engine-local direct | 112.22 | 124.84 | 150.00 |
| LMCache MP DRAM | 112.70 | 128.69 | 149.36 |

Decode time reduces the relative benefit seen in TTFT. At 8K, OrbitKV's complete
request median is slightly **higher** than LMCache's despite its lower TTFT.
This run does not support an across-the-board end-to-end win over LMCache.

Cold OrbitKV TTFT is 118.98/477.14/1,008.37 ms, versus native
117.99/475.82/1,007.62 ms. In the resident phase, native HBM is
20.08/22.26/26.47 ms and OrbitKV is 19.26/21.32/24.46 ms. The latter are
**mixed hits**: offload connectors can restore the tail page that native HBM
recomputes. They are not evidence that an external copy is faster than directly
using the same resident GPU state. The connectors also report different final
token boundaries; these are complete integrations, not isolated transports.

CPU offload, OrbitKV, and LMCache match on all **45 corresponding outputs**.
Each differs from native HBM on two 1K pressure requests. Native HBM itself
changes output between cold and resident execution for those prefixes. All
differences remain in the report. The performance configuration is not batch
invariant; use the deterministic gate above for its scoped correctness evidence.

The [fixed-cohort concurrent comparison](sustained-performance.md#engine-local-restore-fixed-cohort-comparison)
separately measures throughput. Historical Manager-owned Restore tables below
use different environments and are not a matched before/after control for the
engine-local optimization. This comparison ranks the current integrations; it
does not attribute a speedup to any one communication change.

The next optimization target is the 8K gap to native CPU offload. Profile source
preparation, grant/worker queueing, copy submission, GPU drain, and connector
admission before changing synchronization. The current aggregate load timer
does not isolate those costs, so this run cannot identify which one explains
the extra 6.79 ms. Any candidate should preserve the ownership/progress gates
and rerun this end-to-end matrix, rather than rely on a copy microbenchmark.

### Measurement evidence and reproduction

Raw runs, exact launch manifests, samples, package/source identities, and JSON/CSV
reports are retained under
`/root/orbitkv/benches/results/runs/cache-e2e-20260928/`.
These artifacts were moved when the communication worktree was retired; original
launch manifests retain the former path as historical provenance.
The accepted serial attempts are native **5**, CPU **1**, OrbitKV **2**, and
LMCache **1**. A 100 ms process monitor rejected entire runs containing observed
Cargo/rustc activity, including startup/cleanup. The four earlier native attempts
and first OrbitKV attempt remain excluded and preserved. Accepted runs contain
no sampled Cargo/rustc activity; this is not proof of an otherwise idle machine.
Runs were selected by that rule, not their latency. There is no confidence
interval, randomized backend order, or long-duration tail guarantee.

With the matching production Manager and extension installed, run from the
repository root in a quiet GPU/CPU window:

```bash
result_root=benches/results/runs/engine-local-comparison
for backend in native cpu orbitkv lmcache; do
  extra=()
  if [[ "$backend" == orbitkv ]]; then
    extra=(--orbitkv-transfer-backend direct)
  fi
  taskset -c 8-23 .venv/vllm-release/bin/python -m benches.single_node \
    --engine vllm --backend "$backend" --model /workspace/models/Qwen3-8B \
    --output "$result_root/vllm-serial-$backend" --workload serial \
    --lengths 1024 4096 8192 --repeats 5 --output-tokens 16 \
    --gpu-tokens 16384 --prefill-tokens 8192 --host-gib 16 \
    --seed 20260920 "${extra[@]}"
done
.venv/vllm-release/bin/python -m benches.report \
  "$result_root"/vllm-serial-{native,cpu,orbitkv,lmcache} \
  --reference-run "$result_root/vllm-serial-native" \
  --output "$result_root/report"
```

The report verifies matching model revision, runtime packages, Python, GPU,
CPU affinity, workload arguments and prompt hashes. It retains unpaired requests
and output differences. The manifest now correctly records zero host-cache
capacity for native HBM. No competitor integration was patched for these runs.

## Qwen3-8B on H20: initial measurements

Measured on one NVIDIA H20 (97,871 MiB reported memory), with vLLM 0.29.0,
SGLang 0.5.20, PyTorch 2.13.0, Transformers 5.12.1, and Python 3.11.2.
The model is `Qwen/Qwen3-8B` at revision
`b968826d9c46dd6066d109eabc6255188de91218`. OrbitKV uses release build
`1184b09a`. The fixed-capacity workload below produces 270 measured requests.

**Client TTFT p50 after GPU cache pressure, in milliseconds; lower is better:**

| Engine | Cache backend | 1,024 tokens | 4,096 tokens | 8,192 tokens |
| --- | --- | ---: | ---: | ---: |
| vLLM | Native HBM cache, now evicted | 117.69 | 475.21 | 1,005.71 |
| vLLM | Native CPU offload | 22.48 | 32.78 | 47.81 |
| vLLM | OrbitKV DRAM | 24.16 | 35.02 | 56.02 |
| SGLang | Native HBM cache, now evicted | 116.82 | 472.19 | 1,000.07 |
| SGLang | HiCache CPU | 31.85 | 32.99 | 44.81 |
| SGLang | OrbitKV DRAM | 39.31 | 50.51 | 62.04 |

All native-HBM samples in this table were misses. All CPU/OrbitKV samples were
verified external restores with no HBM hit. The two engines have different
restore boundaries: for these aligned prompts vLLM restores the full prefix,
while SGLang restores all but the final 64-token page. Compare backends within
each engine; the two OrbitKV rows are not a transport-only comparison.

OrbitKV avoids expensive recomputation, but both engines' built-in CPU caches
are faster in this experiment. At 8K, OrbitKV adds 8.21 ms over vLLM CPU offload
and 17.23 ms over HiCache. Manager load-task p50 is 28.79 ms and 26.46 ms,
respectively; this includes descriptor construction, H2D submission and stream
synchronization, not just PCIe transfer. Profiling transfer batching, completion
observation and overlap with inference is the next performance task.

For resident prefixes, OrbitKV TTFT p50 was 20.10/22.27/25.34 ms in vLLM and
29.66/29.87/30.91 ms in SGLang. vLLM's resident phase includes a small external
tail-page restore and is therefore classified as a mixed hit.

Output comparison is retained for every sample. The performance runs do not
enable deterministic inference: two of five 1K prefixes changed output on
vLLM reuse, including native HBM reuse and native CPU offload. OrbitKV and
vLLM CPU offload nevertheless matched on **all 45 corresponding requests**.
SGLang's three backends matched on all 45 corresponding requests, and every
reuse matched its cold output. These observations do not replace the separate
deterministic correctness gates.

The first vLLM OrbitKV run exposed a real bug: 4K/8K multi-layer Publish metadata
exceeded the default 64 KiB descriptor slot, so those prefixes were not saved.
The shared client now splits metadata at complete per-page layer boundaries
and retains GPU sources until every chunk finishes. The table uses the full
rerun after this fix. The earlier failed-offload run and an SGLang CLI startup
failure are preserved separately in the local benchmark directory.

The 270 request measurements, launch manifests and summaries are available in the
[historical dataset snapshot](https://github.com/feichai0017/orbitkv/tree/44c1e5f9a253aa7378c6187b2aeea9bff93df304/benches/results).
Complete JSONL responses, counter deltas, and engine/manager logs are retained
under `/workspace/orbitkv/benches/results/runs/orbitkv-qwen3-8b` on the measurement host.
LMCache, FlexKV, and Mooncake Store are not included in this initial experiment.

## Completion-notification experiment

The SGLang restore worker now consumes the process channel's existing
completion notification instead of sleeping 10 ms between polls. With the same
original release environment and workload, the before/after TTFT p50 values
were:

| SGLang OrbitKV | 1,024 tokens | 4,096 tokens | 8,192 tokens |
| --- | ---: | ---: | ---: |
| Fixed-interval polling | 39.31 ms | 50.51 ms | 62.04 ms |
| Completion notification | 32.25 ms | 44.33 ms | 60.12 ms |

All 15 pressure samples were verified external restores. All 45 outputs match
their corresponding cold outputs. The 4K notification run includes two
59–60 ms observations; they remain in the results. This is a small before/after
experiment, not a tail-latency guarantee. The code change removes an explicit
waiting interval; it does not make copies faster or add layerwise overlap.

## LMCache and FlexKV on the current engine releases

Measured on September 21, 2026, the follow-up uses the same model, capacity,
and request sequence, with vLLM 0.29.0 and SGLang 0.5.20. LMCache is 0.5.5,
using its official CUDA 13 /
PyTorch 2.13 wheel. FlexKV is a release build from commit
`738ddc141a198b4e20de6c5d1f0128e387f7fdb2`; its locally built package identifies
as `0.0.0+unknown`, so the source commit is the version identity for this test.

The independent-cache dependencies live in isolated comparison environments.
CPU offload and OrbitKV were rerun in those same environments: NumPy 2.2.6,
OpenTelemetry API 1.40.0, and prometheus-client 0.24.1, with the engine and
PyTorch builds unchanged. The earlier notification experiment deliberately
uses the original release environment. Its numbers should not be substituted
for these matched controls.

**Client TTFT p50 after pressure, milliseconds; five external restores per cell:**

| Engine | Cache backend | 1,024 tokens | 4,096 tokens | 8,192 tokens |
| --- | --- | ---: | ---: | ---: |
| vLLM | Native CPU offload | 21.72 | 32.30 | 47.17 |
| vLLM | OrbitKV DRAM, direct | 23.99 | 34.92 | 53.96 |
| vLLM | LMCache MP, DRAM | 26.88 | 39.61 | 56.65 |
| SGLang | HiCache CPU | 30.25 | 31.63 | 44.50 |
| SGLang | OrbitKV DRAM, direct | 31.44 | 42.31 | 56.15 |
| SGLang | LMCache MP, DRAM | 33.17 | 44.04 | 58.76 |

OrbitKV's medians are lower than LMCache's in this configuration, but the
native CPU caches remain faster. All 45 corresponding outputs match the CPU
control for OrbitKV and LMCache within each engine. Non-deterministic reuse
changes output for the same two 1K vLLM prefixes and one 1K SGLang prefix in
the CPU controls and external caches. These are retained, not discarded.

FlexKV has **no latency row** because the unmodified engine integrations fail:

- vLLM's built-in wrapper does not pass `kv_cache_config` to FlexKV. FlexKV
  consequently calls `FlashAttentionBackend.get_kv_cache_shape`, which is no
  longer present in vLLM 0.29.0.
- SGLang's built-in connector passes `is_mla` to `KVCacheLayout`; this FlexKV
  commit expects `kv_dim`. GPU registration repeatedly fails before serving
  requests. The test was stopped after confirming that error.

The latest published Git tag found, FlexKV `v1.2.1` at `24455633`, was also
inspected. Its SGLang configuration signature is older and it lacks the
`flexkv.transfer.layerwise` module imported by SGLang 0.5.20. It was not built
or benchmarked. A compatible FlexKV/engine adapter tuple must be established
before adding a performance result; these failures say nothing about FlexKV's
transfer speed. No engine or competitor adapter was patched for this matrix.

The first setup attempts exposed missing overlay dependencies, an incorrect
LMCache metrics port, and an embedded-Python package search path issue. Those
were corrected before the successful runs; failed attempts remain in the
manifest archive. Source logs and complete responses are under
`/workspace/orbitkv/benches/results/runs/orbitkv-qwen3-8b-comparisons`.

### Existing copy-kernel experiment

vLLM OrbitKV also ran the same workload with its existing mapped-host copy
kernel, selected through `orbitkv.transfer_backend`. The manager logs confirm
that both GPU workers used the requested backend.

| OrbitKV vLLM transfer backend | 1,024 tokens | 4,096 tokens | 8,192 tokens |
| --- | ---: | ---: | ---: |
| Direct DMA, TTFT p50 | 23.99 ms | 34.92 ms | 53.96 ms |
| Copy kernel, TTFT p50 | 24.31 ms | 42.58 ms | 63.40 ms |

Every pressure sample was an external restore. The kernel is slower here,
especially for long prefixes; it is not promoted to the default. Fewer driver
submissions do not by themselves establish a faster transfer path. This result
does not rule out the kernel on a different layout, fragmentation pattern,
GPU, or host topology.

The [historical comparison dataset](https://github.com/feichai0017/orbitkv/tree/44c1e5f9a253aa7378c6187b2aeea9bff93df304/benches/results)
contains 360 follow-up requests covering matched controls, LMCache, notification
and kernel experiments, with launch commands, dependencies and failed attempts.
The original 270 measurements remain a separate dataset in that snapshot.

The notification change passed 335 source-only Python tests and six real GPU
integration cases, including failure/timeout ownership and poisoned-page
round trips. The deterministic model gates passed: vLLM 6 passed, 1 skipped
(hybrid-only case on a dense model), and SGLang 1 passed. The existing kernel
test also verified exact H2D and D2H bytes against the direct backend. Applicable
local checks, including release Rust tests, passed. External-cache performance
runs are not a substitute for that backend's own correctness qualification.

## Reproduce the latency experiment

Build a **release** wheel using [the single-node setup](single-node.md), and
install it into the two engine release environments. From the repository root:

```bash
for engine in vllm sglang; do
  for backend in native cpu orbitkv; do
    ".venv/${engine}-release/bin/python" -m benches.single_node \
      --engine "$engine" --backend "$backend" \
      --model /workspace/models/qwen3-8b \
      --output "benches/results/runs/qwen3-8b/${engine}-${backend}"
  done
done
```

Each output directory must be empty. The script starts and stops its own engine
and, for OrbitKV or LMCache, its own cache service. It preserves service logs, launch
commands, versions, GPU details, raw samples, cache-source evidence, and a
summary. Do not run other GPU workloads alongside these measurements.

The same script accepts `--backend lmcache` and `--backend flexkv`. Install the
cache backend and its complete runtime dependencies in an isolated environment
with the pinned engine; do not let installation silently replace PyTorch or the
engine being compared. LMCache 0.5.5 uses its MP daemon for both engines here.
SGLang 0.5.20 requires `--lmcache-config-file` with `mp_host` and `mp_port`; the
script writes that file and uses the daemon's HTTP `/metrics` endpoint. vLLM
loads the external LMCache connector module explicitly.

The script sets FlexKV's host payload budget and disables SSD, GDS, MPS, and
optional metrics. It leaves the default SM-copy and non-layerwise transfer
settings in place. This is a specified configuration baseline, not an exhaustive
FlexKV tuning result. Successful installation does not establish compatibility
with an engine's KV APIs; a startup failure is not a cache miss or a latency
sample.

For the existing OrbitKV transfer-backend experiment, add
`--engine vllm --backend orbitkv --orbitkv-transfer-backend kernel`. The default
remains `direct`. This switch belongs to the benchmark and selects an existing
connector option; it does not introduce another cache API.

The initial experiment uses dense Qwen3-8B in BF16, one GPU, TP=1, 64-token
pages, 16,384 GPU KV tokens (2.25 GiB), and a 16 GiB external payload budget.
Metadata, pinned staging buffers, and process RSS are outside that payload
budget. It uses 1K/4K/8K synthetic token inputs, 16 output tokens, concurrency
one, and five independent prefixes per input length. Model startup, kernel
warmup, and cache-pressure traffic are excluded from measured request latency.

Every prefix is measured in three states:

1. **Cold:** a new prefix with no intentional cache reuse.
2. **HBM hit:** the exact same request while its GPU KV is resident.
3. **After pressure:** two disjoint 12,288-token requests exceed the 16,384-token
   GPU budget; repeat the original request and inspect where its KV came from.

This avoids flushing HiCache's CPU pool while preserving OrbitKV's pool, which
would give the two systems different starting conditions. A request after
pressure counts as an external-cache sample only if cache counters or manager
H2D bytes establish a restore without an HBM hit. Cold misses and mixed hits
remain in the raw results and must not be relabeled as CPU restores.

TTFT is measured at the HTTP client when the first nonempty streamed text
arrives. Engine counters and OrbitKV load bytes identify the cache source;
output strings are compared with the corresponding cold request. Output
differences are recorded rather than silently dropped. Numerical correctness
still has a separate deterministic E2E gate.

This is a serial latency experiment. Five observations do not establish a
production tail-latency SLO. It does not measure concurrent goodput, SSD
performance, multi-node transfers, restart recovery, or a production prompt
distribution. Runs are sequential rather than randomized. Those questions
require repeated experiments, additional workloads, and capacity sweeps.

## Single-node optimization order

The follow-up [SSD measurements](ssd-performance.md) cover 120 requests and
identify a SGLang pending-query gap: background SSD reads currently do not
become restores for those serving requests. vLLM successfully restores all 15
SSD-phase requests. Resolve this readiness boundary before comparing SGLang
SSD latency or adding speculative prefetch policies.

Benchmark code and committed results are maintained in [`benches/`](../benches/README.md).
The SGLang adapter now submits up to eight independent restores before waiting
for their completions. This bounds its contribution to the manager's operation
queue and avoids a Python submission gap after every completed request. Each
request retains its own descriptor and lease; combining all leases into a
single descriptor could exceed the process channel's descriptor-size limit.
The batch is acknowledged only when every restore has succeeded. This change
has GPU byte and failure-lifetime coverage, but its concurrent TTFT/goodput
benefit has not yet been measured. The tables above predate this change.
It still uses an all-layer completion barrier; true layerwise overlap remains
the next architectural performance step.

1. **Observe completion promptly.** The SGLang load worker used to sleep for
   10 ms between restore polls. It now waits on the existing completion
   notification, with a 50 ms fallback poll if a notification is lost. The
   shared client owns the deadline. Terminal polling now reads a shared result
   record directly, and an outcome waiter publishes and notifies without the
   endpoint's idle scan. Submission still uses iceoryx2 and its existing dispatch
   scheduling. The historical tables above do not measure this new path.
   Timeout or transport failure still leaves
   destination ownership unresolved and faults the engine; neither condition
   means GPU pages can be reused.
2. **Measure transfer fragmentation before choosing a backend.** Record
   descriptor construction, merged-copy count, submission, CUDA completion,
   and engine resumption separately. The existing load-duration counter
   includes construction, submission, and stream synchronization; it is not a
   pure PCIe bandwidth measurement. Compare the existing `direct` and `kernel`
   backends under identical requests, then under concurrent inference. SM copy
   kernels and DMA copies have different contention costs, so an isolated
   latency win is insufficient to change the default for every workload.
3. **Expose actual layer readiness.** SGLang's current
   `start_layer_wise_loading` entry point restores all layers before completing
   any layer future. Introduce manager-owned per-layer or per-group completion
   fences and let inference consume completed layers while later layers load.
   GPU dependencies must be established before acknowledging readiness, and
   all in-flight work must drain before source leases or destination pages are
   released. Keep a whole-operation terminal result for cleanup and failures.
   Validate the engine's CUDA graph mode as well: a Python layer wait that runs
   during capture does not automatically run on graph replay. Measure any
   graph-mode change alongside the transfer benefit.
4. **Bound work under load.** Sweep concurrency 1/4/8/16, partial-prefix hits,
   and a working set larger than the host pool. Measure TTFT/TPOT and goodput
   with restore and publish traffic together; bound outstanding bytes and
   apply backpressure. The current Rust GPU workers have separate load/save
   threads but unbounded queues. SGLang also waits for queued requests one at
   a time. Batch independent requests only within descriptor capacity and
   maintain cancellation and page-lifetime guarantees.
5. **Tune placement and retention from measurements.** Verify the existing
   NUMA placement on the target host. Measure offload write amplification,
   host-pool pressure, allocation cost, and hit reuse before adding admission
   or eviction policies. Preserve useful prefixes without allowing a long
   one-off request to monopolize transfer bandwidth. Add SSD and distributed
   cache comparisons after the DRAM path has stable correctness and load tests.

HBM allocation and eviction remain engine-owned. OrbitKV owns external copies,
transfer scheduling, and its completion contract. These changes do not require
a new central directory, a compatibility facade, or a second engine-facing
cache API.
