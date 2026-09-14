# FP8 region tuning

This is the first workload-driven region optimization in the
[joint compilation design](joint-compilation.md). Qwen3.8 27B is the primary
acceptance workload. Rules select implementations from input identity, shape,
quantization semantics, layout, and device capabilities.

The [reviewed result package](../results/fp8-region-tuning-20260912/README.md)
records the bounded correctness checks, source versions, and completed serving
diagnostic. Its performance-benefit gate did not pass.

The subsequent [compiler boundary refactor](compiler-boundaries.md) separates
policy, bucket planning, profiling fixtures, FP8 contracts and captured scratch.
It also adds structured candidate/program/score tracing without model-specific
operator selection. Its source identities differ from the frozen report above.

## Searchable activation preparation

The original DeepGEMM provider quantizes BF16 activations separately inside each
linear call. The experimental alternative exposes one graph-owned packed
activation to multiple GEMMs. A gate/up or QKV/Z fanout can therefore reuse the
same quantized values and scales.

Egglog introduces and shares this producer. Original combined providers remain
candidates. `enable_shared_fp8_quantization` is an explicit, artifact-bound opt-in;
bounded device and model correctness does not enable it by default.

The packed ABI contains row-major E4M3 values followed by FP32 scales indexed by
128-column block and an aligned row stride. Its byte capacity, scale offset,
alignment, producer identity, and consumer ABI are checked. The output is opaque
byte storage. Ordinary graph dependencies keep it alive through its consumers.
Both implementations use the same CUDA quantizer source, including clamping,
rounding, and deterministic padding.

Captured library calls also need explicit temporary-resource lifetime. The
`HostOp::cuda_graph_capture_resources` hook snapshots allocation owners after
preparation. Each captured child graph retains those owners, including when a
different shape becomes active in the resident graph cache. DeepGEMM acquires
scratch addresses outside capture and completes their allocation before
publishing them. Graph retirement waits for previous execution before releasing
the captured resources. This prevents scratch growth from invalidating an older
graph and avoids recording allocator bookkeeping events inside child captures.

## Workload profiles

`DecoderTuningProfile` is separate from executable capacity. Existing compilation
and engine entry points keep their default search settings; explicit callers use
`compile_or_load_with_tuning` or `ModelEngine::start_with_tuning`. The server
accepts `--tuning-profile PATH`.

| Field | Meaning |
| --- | --- |
| `batch_sizes` | Preferred representative request counts |
| `prefill_tokens` | Preferred total query-token counts, not per-request lengths |
| `context_pages` | Preferred flattened CSR page counts |
| `keep_best` | Finalists compared on the CUDA Graph deployment path |
| `initial_candidates` | Initial genomes from per-class coverage cycles, including the first executable seed; defaults to 1 and is clamped to the graph budget |
| `trials` | Profiling trials per candidate |
| `search_time_limit_ms` | Cooperative genetic-search budget, starting after graph saturation; synchronous compiler calls are not preempted |
| `maximum_buckets` | Bound on the proposed Cartesian bucket count before search-space construction |
| `enable_shared_fp8_quantization` | Admit the packed preparation/GEMM alternative |

[decoder-tuning.json](../benchmarks/decoder-tuning.json) is an example for an
executable admitting at least 8 requests and 128 query tokens. Representatives
must fit configured capacity, and `keep_best` must not exceed the graph-search
limit. Empty lists retain the existing bucket policy.

For broader initial exploration, [search-coverage.json](../benchmarks/search-coverage.json)
reserves eight initial genomes and two deployment finalists. Use it with
`--search-graphs 8` or a larger graph budget. After the first executable seed,
duplicate programs and rejected genomes consume the initial allowance; remaining
measurement budget then goes to mutation and restarts. This is bounded sampling
of admitted alternatives, not guaranteed measurement of every provider. See
[search coverage](search-coverage.md) for ordering and reproducibility limits.

The compiler chooses feasible joint representative shapes within bucket ranges.
The synthetic fixture supplies all query and page CSR rows, per-request write
slots, positions, and fixed-state slots before candidate preparation. These
inputs belong to profiling scratch state; request execution supplies its own
manager-authored metadata. This first fixture uses private physical pages.
It does not qualify shared-Prefix layout search. The entire tuning profile
participates in decoder artifact identity; changing it requires a fresh artifact.

The search timer excludes model loading, loop rolling and per-bucket e-graph
saturation. Initial extraction can attempt one candidate per bucket before the
retry deadline is checked; finalist graph preparation also has to finish.
Consequently this is not a hard startup deadline. The qualification runner's
separate process timeout bounds the complete phase.

CUDA supplies each custom op's existing deployment eligibility to the generic
extractor. Sampling excludes non-executable placeholders and dependencies that
cannot form a finite executable term. This leaves the e-graph and every legal
provider alternative intact. Initial-candidate retries obey the cooperative
deadline and a finite attempt bound, including cases that never reach a timed
GPU candidate. Buffer validation examines an operation's own inputs and output
without copying the entire graph's buffer table on every host launch.

## Reproducible qualification

Build the release `model_execution` test binary before timing. Use its executable
path with [run_decoder_qualification.py](../tools/run_decoder_qualification.py),
passing `--test-binary`, `--test-name`, `--model-dir`, `--reference-dir`, a fresh
`--output-dir`, and `--search-graphs 16`. CUDA and provider source paths must be
configured for that process.

The first phase creates a new schedule using the existing content-addressed JIT
cache; it does not claim an empty machine-code cache. The next phase loads the
same artifact in a fresh process. A third process independently collects CUDA
Graph attribution. The harness never downloads sources or clears shared caches.

Qualification requires eight independent reference comparisons, successful state
drain, the correct search/replay marker, and an unchanged artifact on replay.
Binary, configuration, reference, tuning-profile, and artifact hashes accompany
logs, exit status, and process times. Merely writing an artifact is insufficient.
Model configuration/index hashes do not fingerprint every weight byte; source
observations do not prove how an independently supplied binary was built.

For the shared-quantization ablation, repeat with a fresh output directory and
`--tuning-profile benchmarks/shared-fp8-ablation.json`. This changes candidate
eligibility while keeping default representative lists and finalist count.
Inspect selection: enabling the candidate does not force it to win.

The `quantized_decoder_batched_reference_and_drain` test additionally accepts
`ORBITKV_QUALIFICATION_BATCH_SIZE`, `ORBITKV_QUALIFICATION_BATCH_CAPACITY`, and
`ORBITKV_QUALIFICATION_RAGGED=1`. Capacity defaults to batch size. Setting capacity
to 8 allows actual B4 and B8 executions to use the same compiler configuration.
[batched-fp8-qualification.json](../benchmarks/batched-fp8-qualification.json)
provides seven feasible joint buckets for this capacity, with three finalists
and a cooperative 20-minute search budget. Runtime shapes inside each interval
are correctness witnesses; only the representative points are tuning samples.
The ragged case first fills a prefix for some requests, then appends remaining
tokens together with fresh requests. Every request is compared with the same
completed-prompt reference, followed by seven teacher-forced decodes and atomic
token/fixed-state completion. The harness checks every request's parity evidence.
To qualify another admitted runtime shape with an existing schedule, pass
`--replay-artifact PATH`. The harness copies it into the fresh output directory
and runs strict replay and profile only, checking that both source and copied
artifact contents remain unchanged.

These diagnostic tests read full logits; their wall times are not serving TPOT
or TTFT. Per-node CUDA events can substantially perturb graphs with many small
kernels. Use whole-graph timing and an uninstrumented serving comparison to judge
performance. Report workspace and compilation cost along with latency.

Schedule replay also needs first-use execution measurements. In the B8 ragged
v3 run, `compile_or_load` took 43.446 seconds, followed by a first `s=24` prefill
of 7.771 seconds. The separate later profile process measured that prefill at
0.300 seconds. The recorded trace does not split provider JIT, graph preparation,
and execution, so it cannot attribute the difference to one stage. Loading a
schedule avoids graph search; it does not certify that every admitted runtime
shape is already specialized and prepared.

## Observed search limitations

The seven-bucket B8 acceptance artifact passes strict replay and a separate
profile process for B4/B8, with both aligned and ragged prefill, on the v3
runtime. Every case completes eight reference steps per request and state drain.
The largest absolute logit errors are 0.625 for B4, 0.78125 for ragged B8, and
0.90625 for aligned B8, under the unchanged 1.0 gate. All requests use the same
fixed prompt and teacher-forced continuation; these cases do not qualify a
diverse-prompt workload or arbitrary continuous-batching interleavings.

The earlier v2 ragged failure exposed a captured DeepGEMM scratch-lifetime bug.
An isolated graph reproduced the deferred CUDA error when growing from 4 to
12 rows. The capture-resource fix passes independent numerical and owner-lifetime
checks across ordinary and resident execution, and the four full-model cases
reuse the unchanged original artifact.

Its BF16 output projection (`M=8`, `N=248320`, `K=5120`) still selects GenericMatmul.
That operation averages about 43 ms in the instrumented decode trace. The same
artifact's `M=32` prefill uses cuBLASLt; these are different shapes and do not
constitute an equal-shape speedup measurement. This motivates cost-guided
exploration of expensive operations as well as final candidate reranking.

The saved B8 output e-class also contains two BF16 cuBLASLt lowerings, including
a row-major lowering with the same input list and `M=s, N=248320, K=5120`.
Their presence proves that compiler alternatives exist; it does not qualify
their device execution for this schedule. The log retains total direct and
deployment scores but not each finalist's output-projection implementation,
so it cannot establish where those alternatives were lost. The subsequent
[candidate trace](compiler-boundaries.md) records program identity, complete
operation choices and both timing scores together for new searches. It cannot
recover missing candidate evidence from the historical run.

Reranking already changes a narrower decision: for the `B=1`, `s=4`, eight-page
representative, direct rank three measures 26.284 ms on the deployment graph,
versus 28.320 ms for direct rank one. These are candidate measurements for one
representative, not a serving result or a global optimality guarantee.

## Matched serving diagnostic

The final v3 server was run off/on/on/off in two alternating epochs on H20,
with four input tokens, eight output tokens, eight measured requests per run,
and configured concurrency one. Both profiles retain three deployment finalists
and differ only in admitting shared FP8 preparation. Each run completed all
requests with the requested output length. Each arm reproduced its generated
texts across process restarts, but the arms differed for two of eight prompts
in both epochs. At the time of that report no independent logit oracle covered
those prompts. A subsequent frozen-v3 probe using identical teacher-forced
inputs explains both first divergences: OFF logits tie at the maximum and
Luminal chooses the highest token ID; ON has a unique maximum matching the
independent reference. All measured selected tokens attain their own row's
maximum. This does not promote the serving result or qualify every internal
numerical path; see [logits diagnosis](logit-diagnosis.md).

| Metric | Median paired on/off change |
| --- | ---: |
| Output throughput | +1.82% |
| Median TPOT | -5.82% |
| Median TTFT | +8.26% |

These are completed diagnostic measurements. Output equivalence failed, and
two short epochs do not establish a performance benefit. The feature remains
disabled by default. Fresh server readiness took 287.4 seconds off and 232.3
seconds on; artifact-loaded readiness was about 38.0 seconds for both.

The selected decode output projection changes from GenericMatmul off to
cuBLASLt on. All selected on-mode decode quantizers have one consumer, so this
bucket does not actually share activation preparation across GEMMs. Prefill
does select shared consumers. Provider choice and the rest of the searched
graph therefore need to be considered alongside the feature flag. In prefill,
48 repeated qkv/z fanouts reduce selected quantization instances from 400 to
352. All 13 rolled FP8 variant groups change in decode, and 8 of 13 change in
prefill. These are selected operations with loop multiplicities, not a CUDA
launch-count measurement or a controlled quantizer-only ablation.

Client traces show an initial stream interval of roughly 110–130 ms followed
by intervals around 24–28 ms. The decoder currently retains at most one
materialized bucket. Rebuilding when switching between prefill and decode is
a plausible contributor to the initial cost; the client trace alone cannot
separate graph preparation, scheduling, and execution. A joint memory budget
for KV storage, workspace, and resident executables should be measured before
changing that policy.
