# Compiler boundaries and extension points

The executor describes model semantics and the state-manager contract. Luminal
owns rewrite alternatives, candidate measurement, and deployment. A checkpoint
name or a successful benchmark shape must not select an implementation in
application code.

## Code layout

The repository-wide [code and test layout](code-layout.md) governs file placement;
the table below assigns compiler responsibilities within that structure.

| Responsibility | Owner | Inputs and limits |
| --- | --- | --- |
| Persistent KV/fixed-state ownership and request transitions | `crates/orbitkv` | Manager plans, arena identities, completion evidence |
| Model topology and semantic graph | `orbitkv-executor/src/model.rs`, `model/import.rs`, `model/config.rs`, `model/topology.rs` | Parsed architecture and tensor metadata |
| Compile/load lifecycle | `model/compiler.rs` | Model, storage registrations, capacity, tuning, optional strict artifact |
| Serialized tuning policy | `model/tuning.rs` | Workload representatives and caller-selected search budgets |
| Feasible correlated buckets | `model/tuning/buckets.rs` | Query/request/page ranges and each physical arena's capacity |
| Profiling input fixture | `model/tuning/fixture.rs` | Typed graph bindings with explicit capacities; private-page CSR metadata |
| Graph alternatives and generic extraction | Luminal `src/graph.rs`, `src/egglog_utils`, `src/search` | Semantic rewrites and executable dependencies; no GPU measurement in core |
| CUDA evaluation and deployment ranking | `luminal_cuda_lite/src/search.rs` | Actual device measurements plus resource checks |
| Search evidence | `luminal_cuda_lite/src/search/trace.rs` | Candidate program identity, full operation manifest, outcomes and scores |
| CPU stage attribution | `luminal_tracing/src/stages.rs`, exposed by executor `diagnostics.rs` | Buffered synchronous wall spans; explicit completion, no added device synchronization |
| FP8 numerical/storage ABI | `host/deepgemm/contract.rs` | Block geometry, scale stride, alignment, checked sizes, quantizer source |
| Captured scratch ownership | `host/deepgemm/scratch.rs` | Exact allocation owners retained until graph retirement |
| Provider tile legality and initial ordering | `host/deepgemm/tiling.rs` | Supported device architecture, dimensions, pinned-provider constraints |
| Source generation/loading | `host/deepgemm/jit.rs` | Contract, tile configuration and content-addressed provider identity |

The CUDA provider paths in the last rows are relative to
`third_party/luminal/crates/luminal_cuda_lite/src`. Existing public decoder entry
points and the tuning JSON representation remain compatible with this layout.

The profiling fixture stores each input binding together with its maximum byte
capacity. Compiler setup no longer scans node IDs to guess which allocation
size applies. Its private pages are measurement inputs; live request metadata
still comes from OrbitKV. It does not model shared-prefix placement.

## Constants and policies

Three kinds of values have different owners:

* **Semantic or ABI requirements** stay with the operation contract. E4M3's
  finite range, the current 128-column scale block, F32 scale alignment, and
  i32 metadata bounds are not tuning knobs. Rust and CUDA quantization now use
  one definition of those requirements. A layout/numerical change needs an
  explicit ABI change and new correctness evidence.
* **Search/resource budgets** belong to the caller. `maximum_buckets` bounds the
  proposed Cartesian planning work; there is no additional unexplained 256
  ceiling. A graph-search budget of one is valid. Empty/zero budgets and
  overflowing geometry are rejected. Graph limits count timed candidates;
  pre-evaluation resource rejections can require additional extraction attempts.
  A one-graph limit is therefore not a one-attempt or hard startup-time limit.
* **Backend heuristics** belong to the provider. DeepGEMM's tile enumeration
  documents hardware restrictions separately from its nominal clock/bandwidth
  assumptions. Those assumptions only order candidates; they are not measured
  device properties or a global optimum claim.

Provider identity covers the rendered quantizer and tile-selection source as
well as the wrapper/dependencies. A changed mapping from variant index to tile
configuration must invalidate an old schedule. Old source-bound artifacts are
therefore expected to reject this provider refactor and require recompilation.

The executor still uses its named, conservative single-active-graph deployment
default. Making several graphs resident needs aggregate accounting for captured
resources, KV and workspace, followed by transition tests and serving evidence.
Exposing an unrestricted residency number alone would not implement that plan.

## Search trace

CUDA accepts `CompileOptions::search_trace(path)`. OrbitKV can use the existing
diagnostic environment boundary without adding a trace path to artifact-bound
tuning JSON:

```sh
LUMINAL_SEARCH_TRACE=/absolute/path/new-search.jsonl orbitkv-serve ...
```

The explicit builder option takes precedence over the environment variable.
The parent directory must exist. A trace is created only for a fresh search;
strict schedule replay does not fabricate candidate measurements. Existing
files are never overwritten, and requested trace write failures are surfaced.

Schema 1 is JSON Lines:

| Event | Evidence |
| --- | --- |
| `search_started` | Backend, search/trial budgets, schema and fingerprint scope |
| `program` | Semantic program identity, all operations, ordered inputs and available host-provider labels |
| `direct` | Candidate/bucket, actual dimensions, measurement/rejection, timeout decision, device and evaluation wall times |
| `deployment` | Same program identity, direct rank/score, deployment CUDA Graph score or rejection |
| `deployment_extraction_rejected` | Direct rank and available extraction failure reason |
| `finalist_validation` | Deployment rank and final resource-validation result |
| `aggregate_rejected` | Rejected set of bucket programs and aggregate constraint reason |
| `selected`, `search_completed` | Installed programs and successful end of search |

Program identity reuses the schedule's semantic fingerprint, including operation
parameters and ordered edges while ignoring LLIR node allocation order. It is
build-local, not a cryptographic digest of weights, toolchain or provider source.
Host-operation `Debug` is currently part of that semantic representation.
cuBLASLt handles/preparation caches and FlashInfer planning scratch are explicitly
excluded; preparing a library call or creating another CUDA stream must not
change a program's identity. Exhaustive field destructuring requires newly added
fields to make this semantic-versus-runtime-state decision explicitly. Older
fingerprints that included cache state must be regenerated through fresh search.
Use the accompanying artifact/environment receipts for those identities.
Operation counts are LLIR nodes, not CUDA launch counts. No model name, output
head shape, or preferred provider determines which operations are recorded.

Direct measurements may use early-stop/trial limits; those options and the
early-stop hint are retained, and a score does not certify that every trial
completed. Evaluation wall time includes synchronous preparation and profiling
(and finalist cleanup); it is not a stage-by-stage compiler trace. JSON writing
is outside device timing and the direct candidate timeout decision, but consumes the
cooperative overall search budget. Disable tracing for serving performance
comparisons. An interrupted trace has no `search_completed` marker.

The separate [stage trace](stage-tracing.md) measures preparation and execution
around those candidates. Its program identities join the search records without
changing their meaning. CPU wall time includes host waits; nested inclusive
durations and work on parallel threads cannot be added into a startup total.

## Checkpoint and operation boundary

Checkpoint syntax is normalized by an explicit importer; unrelated RoPE/gating
fields no longer infer the normalization convention. Attention and FP8 linear
semantics live in `luminal_nn::ops`, with CUDA providers supplying egglog
implementations. See [checkpoint import](checkpoint-import.md) for admitted
formats, extension rules and the current artifact migration.

## Extending the compiler

Add model structure to the configuration/topology layer. Add an operation's
semantic/layout/resource contract at its operation boundary, then express legal
provider alternatives and fusion in egglog. The runtime evaluates extracted
programs and reports evidence; it must not rewrite a selected LLIR to force a
particular backend. Keep model/device examples in benchmark manifests and test
fixtures rather than production selection branches.

For numerical diagnostics, [logit_probe.py](../tools/logit_probe.py) accepts
manifest-driven inputs and compares full-vocabulary teacher-forced traces.
It reports actual selected tokens separately from a canonical lowest-index
argmax: Luminal's current `argmax`/`argmin` contract chooses the highest index on
ties. A different sampling tie policy is a semantic compatibility change, not
an optimization or a way to qualify a numerically different schedule.

The [compiler-boundary validation](../results/compiler-boundaries-20260912/README.md)
records the final frozen build, focused host/GPU regressions, and full-model
search/replay checks. It also preserves the identity-audit failure that exposed
runtime cache state in cuBLASLt fingerprints and the evidence for its fix.
The subsequent [engine/stage validation](../results/engine-stage-attribution-20260913/README.md)
records the merged engine/frontend crate and a newly frozen stage-instrumented
build, with a fixed-artifact stage-off control and the measured next priorities.
