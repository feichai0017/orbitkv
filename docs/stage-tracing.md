# Compiler and runtime stage attribution

The optional `LUMINAL_STAGE_TRACE` diagnostic records synchronous CPU wall-time
spans through the existing `tracing` API. It separates graph construction,
weight inspection/loading, egglog preparation and schedules, candidate
generation, provider compilation, resource preparation, profiling, CUDA Graph
materialization, and execution/input preparation. It does not change the search
space, provider selection, numerical gates, or sampling policy.

`tools/run_decoder_qualification.py --stage-trace` assigns a fresh
`stages.jsonl` to each search/replay/profile process and produces
`stages-summary.json`. The bounded full-decoder integration harness installs the
layer and explicitly finishes it after numerical checks and lifecycle drain.
A requested trace that is missing, incomplete, or structurally invalid fails
qualification. Other harnesses must opt into the installer before using this
flag. Ambient trace settings are cleared when the flag is absent.

The `orbitkv-serve` executable also installs the layer when the environment
variable is present. Choose a fresh path, stop the process normally, then run:

```bash
python tools/summarize_stage_trace.py /tmp/stages.jsonl --output /tmp/stages-summary.json
```

Embedders can compose `orbitkv_executor::diagnostics::stage_trace_layer` with
their existing tracing subscriber. The environment installer is for standalone
processes: it installs a global subscriber for stage spans and returns an error
if a subscriber is already installed. Keep its guard until all worker spans
close and call `finish`; dropping it without finishing marks the trace
incomplete. The composable layer and file writer live in Luminal's
`luminal_tracing` crate; the executor only exposes the diagnostic entry point.

| Output | Meaning |
| --- | --- |
| `inclusive_wall_ns` | Sum of each span's lifetime, including children and host waits |
| `self_wall_ns` | Inclusive duration minus the union of direct child intervals on the same thread |
| `roots` | Top-level observed intervals, retaining their thread identities |
| `candidate_program_spans` | Preparation/profile intervals carrying the semantic program identity used by search records and artifacts |
| `non_profiled_executions` | `cuda.execute` calls with runtime profiling disabled; may include compile-time warmups, so these are not automatically serving steps |

`cuda.module_artifact.capture` encloses the selected-schedule image capture pass.
`cuda.module_image.hit` records a source digest and loaded image size;
`cuda.nvrtc.compile` remains exclusive to actual NVRTC compilation. Compatible
cached replay should emit module hits and zero NVRTC spans across initialization
and execution. See [module artifacts](module-artifacts.md) for its scope.

Weight loading separately records mapping, metadata, encoding validation,
conversion, device allocation, upload API calls and per-shard completion.
`cuda.weights.loaded` carries actual tensor and byte totals. Mapping time does
not include all first-touch page faults, and the upload span is a CPU API timer.
See [weight loading](weight-loading.md) for the ownership and timing contract.

`luminal.search.next_candidate` includes candidate mutation/extraction work;
it is not a pure extraction timer. Provider JIT records include a cache-hit
field; `cuda.nvrtc.compile` only wraps actual compilation after a cache miss.
Resource/provider preparation and compilation are nested inside candidate
evaluation, so adding their inclusive totals would double-count work.

Instrumentation must bound the span's lifetime to the intended operation. In
loop conditions, create an entered guard inside a block: a temporary span in a
`while let` expression can survive through the body even after `in_scope`
returns. The evidence audit checks generation/evaluation intervals for this
overlap; the initial diagnostic that exposed it is retained with the raw runs.

Spans do not add CUDA synchronization, CUDA events, or per-kernel stage writes.
The JSON header is written when the file is created; subsequent records stay
in memory until the guard finishes. This keeps filesystem writes outside the
measured stages, but span bookkeeping still adds CPU overhead. Memory use grows
with the number of recorded spans: use this for bounded diagnostic runs.
Durations are wall time, not CPU utilization or GPU kernel time. Threads can
overlap; their sums are not process elapsed time. Use the separate CUDA Graph
profile for device timing and an uninstrumented serving benchmark for TPOT/ITL.

The runner fingerprints the summarizer, input binary, model metadata, oracle,
tuning manifest, artifact, and per-phase traces. It labels stage instrumentation
and CUDA Graph instrumentation separately. Build time is excluded, existing
provider/CUDA caches are reused, and cold search means a fresh selected schedule,
not an empty provider cache. Retain the source snapshot and build receipt with
the raw run before promoting a compact report to `results/`.
