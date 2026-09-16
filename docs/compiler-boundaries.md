# Compiler boundaries and extension points

The executor describes model semantics and the state-manager contract. Compiler
core builds rewrite alternatives; the CUDA backend owns candidate measurement
and deployment. A checkpoint name or a successful benchmark shape must not
select an implementation in application code.

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
| Graph alternatives and generic extraction | OrbitKV compiler `src/graph.rs`, `src/egglog_utils`, `src/search` | Semantic rewrites and executable dependencies; no GPU measurement in core |
| Snapshot sampling and initial coverage | OrbitKV compiler `src/egglog_utils/sampling.rs`, `src/search/genetic.rs` | Existing admitted egraph alternatives, RNG state and caller budgets; no provider preference or graph rewriting |
| Local extraction and cost attribution | OrbitKV compiler `src/egglog_utils/neighborhood.rs`, `src/search/profile.rs`, `src/search/unroll.rs` | Exact extraction provenance, runtime region costs and bounded single-choice neighbors; no new rewrites |
| CUDA evaluation and deployment ranking | `orbitkv-cuda/src/search.rs` | Actual device measurements plus resource checks |
| Search evidence | `orbitkv-cuda/src/search/trace.rs` | Candidate program identity, full operation manifest, outcomes and scores |
| CPU stage attribution | `orbitkv-tracing/src/stages.rs`, exposed by executor `diagnostics.rs` | Buffered synchronous wall spans; explicit completion, no added device synchronization |
| FP8 numerical/storage ABI | `providers/deepgemm/contract.rs` | Block geometry, scale stride, alignment, checked sizes, quantizer source |
| Captured scratch ownership | `providers/deepgemm/scratch.rs` | Exact allocation owners retained until graph retirement |
| Egglog scalar extensions | `orbitkv-compiler/src/egglog_utils/primitives.rs` | Stateless typed functions; configuration comes from scalar facts; no device queries, graph traversal or hidden state |
| Explicit provider selections | `providers/deepgemm/selection.rs` | Complete tile, row bound, static dimensions and actual SM count; rank exists only during candidate generation |
| Provider tile legality and initial ordering | `providers/deepgemm/tiling.rs` | Supported device architecture, dimensions, pinned-provider constraints |
| Source generation/loading | `providers/deepgemm/jit.rs` | Contract, tile configuration and content-addressed provider identity |

CUDA provider paths in this table are relative to `crates/orbitkv-cuda/src`.

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

DeepGEMM rules call a pure scalar primitive with the bucket's row upper bound,
static N/K dimensions, actual SM count and a candidate rank. The result is an
explicit tile descriptor in the egraph. Extraction validates that descriptor;
`prepare_compilation` validates the target and loads exactly that native library.
`execute` accepts only rows in the recorded range and a prepared handle. It does
not re-rank tiles, invoke NVCC or load another library. A dynamic expression with
no finite supported bound, or a target with no SM-count fact, admits no DeepGEMM
candidate. Constant shapes do not require interval analysis.

Provider identity includes the descriptor schema, scalar selection/legality
source, rewrite text, rendered quantizer and wrapper/dependencies. Prior ordinal
schedules must be regenerated. Prepared handles and scratch ownership are
excluded from semantic fingerprints; preparing an operation cannot change its
selected program identity.

FlashAttention-3 rules similarly require an explicit context-page capacity,
proved by a constant shape or the bucket's upper bound. Its page-table stride,
scratch and native launch bound use that capacity; GPU CSR and last-page metadata
provide the actual sequence lengths. Context growth within a bucket therefore
does not change its capture key or prepared plan. The page-index allocation must
back the recorded capacity, while its logical length may be smaller. Query and
request counts remain capture dimensions. The plan and rewrite sources are part
of the provider identity, and captures retain their scratch owner until retirement.
The runtime's input descriptors preserve backing capacity through resolution and
caching; changing logical contents must not shrink that physical-capacity fact.

Local decoder artifacts use buffered streaming JSON for file input/output and
atomic publication without overwriting an existing file. Their size follows the
selected graph and embedded CUDA images; the engine does not impose a fixed
64 MiB file ceiling. Schema, image hashes, execution environment and selected
schedule validation remain mandatory.

The executor still uses its named, conservative single-active-graph deployment
default. Making several graphs resident needs aggregate accounting for captured
resources, KV and workspace, followed by transition tests and serving evidence.
Exposing an unrestricted residency number alone would not implement that plan.

## Search trace

CUDA accepts `CompileOptions::search_trace(path)`. OrbitKV can use the existing
diagnostic environment boundary without adding a trace path to artifact-bound
tuning JSON:

```sh
ORBITKV_SEARCH_TRACE=/absolute/path/new-search.jsonl orbitkv-serve ...
```

The explicit builder option takes precedence over the environment variable.
The parent directory must exist. A trace is created only for a fresh search;
strict schedule replay does not fabricate candidate measurements. Existing
files are never overwritten, and requested trace write failures are surfaced.

Schema 3 is JSON Lines:

| Event | Evidence |
| --- | --- |
| `search_started` | Backend, search/trial/initial-population/local-attempt budgets, schema and fingerprint scope |
| `bucket_started` | SHA-256 of the ordered serialized egraph and custom-op descriptors, format/scope, size and representative dimensions |
| `program` | Semantic program identity, all operations, ordered inputs, available host-provider and generated-kernel labels |
| `direct` | Candidate/bucket, sampling origin (`Coverage`, `Hotspot`, `Mutation`, `Restart`), actual dimensions, measurement/rejection, timeout decision, device and evaluation wall times; optional parent/class/from/to decision and separately measured LLIR regions |
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

Sampling orders class and node IDs within an existing snapshot. Its digest is
an input identity, not a canonical semantic hash across fresh saturation runs;
it excludes compiler binaries, input data and device state. Fixed-snapshot
sampling also needs the same RNG state, options and feedback to reproduce later
generations. GPU timing noise and time budgets can change rankings and stopping
points. [Search coverage](search-coverage.md) describes the scope of the initial
exploration policy and its regression checks.

Direct records include `targeted_mutation` and `profile_regions`. The former
names the measured parent and every changed snapshot e-class with its old/new
e-node; the latter gives source LLIR node IDs and CUDA event costs in seconds.
`hotspot_max_changes` in `search_started` bounds the proposal width. A fused
region's cost is apportioned across its distinct mutable choices for scheduling
exploration. These diagnostic costs never replace `device_duration_ns` or the
final deployment score. Disabled hotspot search emits no region measurements.

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
semantics live in `orbitkv_ops::ops`, with CUDA providers supplying egglog
implementations. See [checkpoint import](checkpoint-import.md) for admitted
formats, extension rules and the current artifact migration.

A retained attention graph also has a planning-content contract. FlashInfer's
explicit CSR plans depend on query and page segmentation, even when buffer
addresses, lengths and dynamic dimensions stay unchanged. Materialization
compares those contents with the prepared plan and replans affected captures.
Identical metadata retains the existing capture. Snapshots are shared across
users within one materialization and expire before the next execution; current
device-backed inputs require a synchronized read. Outer decode captures remain
guarded by the executor's exact query/page-indptr signature.

## Extending the compiler

Add model structure to the configuration/topology layer. Add an operation's
semantic/layout/resource contract at its operation boundary, then express legal
provider alternatives and fusion in egglog. The runtime evaluates extracted
programs and reports evidence; it must not rewrite a selected LLIR to force a
particular backend. Keep model/device examples in benchmark manifests and test
fixtures rather than production selection branches.

`KernelGatherCast` is one such region: egglog can replace either
`cast(gather(indices, data))` or `gather(indices, cast(data))`. Its addressing
retains source/index views and the cast's physical span; the CUDA kernel performs
the same conversion once per selected element. The original composition remains
available, including when other consumers share its intermediate. Current
admission covers BF16/F16 to FP32 and the reverse conversions. It does not change
reduction order or absorb an external provider call.

For numerical diagnostics, [logit_probe.py](../tools/logit_probe.py) accepts
manifest-driven inputs and compares full-vocabulary teacher-forced traces.
The executor's `model_execution` logit probe also accepts optional `batches`, an
array of submissions containing `{ "case": "case-id", "tokens": count }` queries.
Queries consume the next tokens from that case's fixed history; they can chunk
prefill, mix requests, change row order and reuse completed requests' state slots.
The complete plan is validated before CUDA execution. Traces retain both logical
output steps and actual per-submission tokens, positions and releases. An omitted
plan executes cases sequentially, chunking prompts to the configured capacity.
Optional `graph_cache_capacity` and `prepare_execution` reproduce deployment
residency and startup preparation; the trace records the effective capacity,
preparation report and final graph counters. They default to the decoder's
minimal residency and on-demand preparation, independently of environment knobs
used by other test harnesses.
Use the same schedule artifact and histories to compare reordered batches;
compare changing shapes and independent-reference errors separately.
It reports actual selected tokens separately from a canonical lowest-index
argmax: OrbitKV compiler's current `argmax`/`argmin` contract chooses the highest index on
ties. A different sampling tie policy is a semantic compatibility change, not
an optimization or a way to qualify a numerically different schedule.
