# Search coverage

OrbitKV compiler generates implementation alternatives in egglog. Core samples complete
genomes from those existing equivalence classes; the CUDA runtime checks their
resource contracts, prepares them and measures executable graphs on the GPU.
Deployment-mode measurement chooses among retained finalists. Neither sampling
nor the executor rewrites extracted LLIR to force a provider.

## Simplification target

Status: first implementation is present. Decoder profiles default to
`CompilePolicy::Default`; existing benchmark/search profiles explicitly select
`CompilePolicy::Tune`. Generic `CompileOptions::default()` remains Tune during
the migration so existing compiler tests and non-model callers do not silently
change behavior.

| Policy | Intended work | Shared output |
| --- | --- | --- |
| Default | Up to eight deterministic diagonal extractions per bucket, stable cycle repair, existing custom-op admission, per-bucket legality checks and aggregate retained-resource validation; first legal set wins, with no candidate execution or GPU timing | Validated `SelectedSchedule`, module images and the existing runtime |
| Tune | Existing measured whole-program search, including initial coverage, hotspot exploration, mutation, deployment reranking and bucket-lattice fallback | The same artifact and installation path |

The default selector must resolve equivalent choices from semantic structure and
explicit rule facts, rather than incidental e-class IDs, checkpoint names or a
Rust graph-replacement pass. Tuning can change a connected group such as shared
FP8 preparation plus its consumers; it must preserve unrelated decisions. Local
operator timing only prioritizes trials. Accept a winner on the complete CUDA
Graph deployment, including conversions, copies and provider preparation costs
where they recur during execution. Cold compilation remains a separate metric.

The first migration step is complete: stable extraction, aggregate validation,
artifact generation/replay and an H20 two-bucket smoke pass without a profiling
duration. Next qualify a complete Qwen default artifact, then add a local-only
Tune policy that starts from that artifact instead of population search. Existing
whole-program Tune remains the explicit comparison arm until the replacement
covers its useful cases; only then remove population/restart machinery.

Keep numerical contracts, required aliases, mutation ordering, physical-layout
checks, resource limits, aggregate bucket admission and strict artifact checks.
State planning continues to select one validated persistent realization. General
KV-layout search and persistent megakernels are outside this simplification.

Bucket construction now always splits the singleton request interval from larger
batches, including when the tuning profile omits explicit batch breakpoints. This
prevents the known `query_tokens=1, requests=[1,8]` upper geometry. General
relational interval constraints remain future work; the feasible correlated
representatives and provider/resource checks continue to fail closed.

The sections below describe the implementation that exists today.

## Stable snapshot order

The sampler sorts class IDs and each class's admitted node IDs before consuming
random numbers. Mutation pools and cycle-repair tie breaking use the same
ordering rule. Hash-map capacity, insertion order and alternative enumeration
therefore do not determine random choices. Pool construction retains custom-op
eligibility, operation-metadata checks and loop-marker constraints.

This applies to a particular serialized egraph. Fresh egglog saturation can
produce different IDs and different graph contents. It is not canonicalized by
the sampler. The `bucket_started` search event records the ordered snapshot's
SHA-256 and its scope so experiments can distinguish different search inputs.
Custom-op descriptors are included; binaries, weights and device state still
need separate provenance receipts.

## Bounded initial exploration

`CompileOptions::initial_population(n)` controls the initial genomes. OrbitKV
exposes the same policy as artifact-bound `initial_candidates` in the tuning
JSON. The default is one initial executable parent.

Each searchable class maintains a shuffled cycle of admitted alternatives.
Drawing a genome consumes one choice from every class. A class repeats an
alternative only after its cycle is exhausted. This improves initial spelling
coverage without enumerating the Cartesian product of all classes.

The first executable seed uses the existing finite retry and cooperative time
limits. That seed counts as one initial genome. Further generated genomes,
including ones that extract to duplicate programs or fail runtime preparation,
consume the remaining initial allowance. The allowance is clamped to the total
graph budget. After it is spent, genetic mutation and restarts use the remaining
measurement budget. Deduplication attempts are bounded too. A one-graph budget
still stops after the first successful seed.

Cycles describe draws before dependency repair. Unreachable choices, cycle
repair, duplicate programs and resource rejection mean spelling coverage does
not guarantee executable-provider coverage. The trace's measured `direct`
events, joined to their `program` operations, establish what was actually timed.
Their `sampling` field identifies initial coverage, hotspot exploration, mutation or restart.

The fixed seed does not by itself reproduce the complete search: snapshot,
options, RNG state, runtime outcomes and ranking feedback must also match.
Timing noise can change parents, finalists and cooperative stopping points.
The snapshot digest is deliberately separate from the selected program identity.

## Profile-directed local exploration

`CompileOptions::hotspot_candidates(n)` adds a bounded local phase after a
measured executable seed. OrbitKV exposes the same artifact-bound field in
`DecoderTuningProfile`. Zero is the default. All search policies participate in
the artifact-bound tuning digest, including their default values. The
[hotspot workload profile](../benchmarks/hotspot-search.json) requests 32 local
attempts, one initial seed and two deployment finalists, with eight measured
graphs per bucket supplied by the qualification runner.

The CUDA runtime first measures each complete candidate normally. It then runs
a separate direct execution with CUDA events around generated kernels, fused
regions and library islands. These diagnostic intervals can include launch gaps;
they order exploration and never replace the uninstrumented whole-program
fitness score. Deployment finalists are still remeasured as CUDA Graphs.

Extraction retains a sidecar mapping from each LLIR operation to its own IR
choice and the OpKind/IList bindings that decoded it. Operand IR producers have
their own provenance. Loop materialization copies that mapping alongside each
unrolled operation, and CUDA fusion retains all constituent LLIR nodes. One
fused region has one measured cost: it is shared across its distinct mutable
choices. Repeated uses of a shared choice accumulate cost. This metadata is
excluded from program identities and executable artifacts.

Starting from measured parents, search visits the highest-cost reachable choice
and enumerates its other admitted nodes in snapshot order.
`hotspot_max_changes` bounds the number of changed bindings in one proposal;
one is the default coordinate policy. Larger values also enumerate connected
producer dependencies, walking through immutable metadata and stopping at the
next mutable decision. A changed choice can expose a previously unreachable
dependency. Unchanged bindings always retain the parent's saved values.

Enumeration is lazy: an anchor proposal is followed by its dependency
combinations, then the next anchor alternative. It does not materialize a
Cartesian product. Cycles in dependency traversal terminate at visited classes;
the extractor and runtime still reject cyclic programs, invalid aliases, layouts
and resource requirements. No proposal repairs unrelated decisions. Shared
e-classes can affect multiple operations; a binding change is not necessarily
one kernel change. The [dependency profile](../benchmarks/dependency-search.json)
allows three connected changes within the same 32-attempt local allowance.

Search trace schema 3 records `targeted_mutation`: the measured parent, anchor
cost and every class/from/to binding. This replaces the single `targeted_choice`
record. The width is part of the decoder tuning digest; regenerate artifacts
whose saved tuning identity predates this field.

A measured improvement can become the next parent immediately. Rejected,
cyclic and duplicate neighbors consume the local attempt allowance, while
rejected candidates do not displace valid parents or consume the graph
measurement budget. Once local exploration is exhausted, ordinary coverage and
genetic exploration continue within the remaining budget. No implementation
family receives a preference in Rust.

This explores existing egraph alternatives. Connected changes can cross a
coordinate valley where each individual change loses, but the combination wins.
They do not create new kernel algorithms or prove a global optimum. Consumer
fanout groups outside the selected dependency paths, new region rewrites and
joint KV-layout alternatives remain separate work. Fresh-snapshot identity and
measurement noise still affect selection; the complete GPU measurement decides
fitness, including deployment-mode remeasurement of finalists.

## State constraints before preparation

The CUDA runtime validates required persistent-state aliases while checking the
extracted graph's topology and mutation ordering. It resolves each logical
output through `KernelOp::output_aliases_input()` to the required logical input.
Data ancestry through a copying operation does not establish storage identity.
Missing, wrong or ambiguous bindings fail closed.

Search and finalist preparation perform this check before fused CUDA source
generation, NVRTC and provider preparation. Direct and stitched artifact loads
run the same static check before compiling any replacement bucket; compiled
bucket metadata is checked again before installation. An invalid replacement
leaves the current executable intact.

This reuses existing operation contracts and does not rewrite LLIR, force a
provider, or remove alternatives from egglog. It saves preparation of rejected
graphs; local exploration preserves every binding outside its recorded connected
proposal. Optional aliases still permit materializing implementations, and
mutation-order checks remain mandatory.

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

## Verification and use

Host regressions under OrbitKV compiler's `tests/unit/egglog` and `tests/unit/search`
exercise map reallocation, alternative ordering, cycle repair, per-class
coverage and finite exploration when genomes produce duplicate programs.
The CUDA integration test `search_coverage` saturates each small BF16 matrix
graph once, searches that same snapshot twice, checks the actual evaluated
program sequence, and requires measured generated and cuBLASLt candidates.
Selected outputs are compared with independent CPU matrix products.

On a configured CUDA machine:

```sh
cargo test --release \
  -p orbitkv-cuda --test search_coverage -- --ignored
```

[search-coverage.json](../benchmarks/search-coverage.json) supplies a bounded
decoder workload with eight initial genomes and two finalists. Pass it through
`--tuning-profile` with `--search-graphs 8`, retain the search trace, and verify
independent logits and strict artifact replay before interpreting timings.
Broader initial sampling can cost more compilation time and need not improve
every workload. It is a foundation for measured region exploration; multi-choice
region search and fresh-saturation canonicalization remain separate work.

The `hotspot_search` CUDA regression combines a GEMM with a separate persistent
scatter branch. It requires a measured local transition between generated GEMM
and cuBLASLt, checks every measured region's LLIR provenance, and verifies CPU
matrix products plus exact updated and untouched state rows across two requests.
Core regressions cover fixed-feedback replay, unaffected bindings, rejected
parent feedback, finite neighbors, shared-region cost and loop provenance.

```sh
cargo test --release \
  -p orbitkv-cuda --test hotspot_search -- --ignored
```

## MoE query compilation

The legacy MXFP4 matcher in the `glumoe` ruleset now stages unpacking, routing
probabilities, index masks and the down projection through private relations.
Each stage retains the variables shared with later predicates or the final
action. Only the original final action adds an implementation. The existing
fixed-point schedule, search budgets, GLUMoE activation rules and provider
inventory are unchanged; there is no model-name filter or Rust graph rewrite.

This optimizes the cost of asking whether a pattern exists, including on a
dense/hybrid graph with no MoE match. It does not expand the old MXFP4 kernel's
narrow geometry contract or qualify released MoE checkpoints. Small routed
graphs can pay extra relation/setup costs, so both positive and no-match
workloads are measured.

The ignored `moe_rule_compilation_workloads` test accepts an explicitly recorded
baseline rule directory through `ORBITKV_RULE_BASELINE_DIR`. The same executable
runs both rule sets and reports input SHA-256, constructor counts and saturation
time. `ORBITKV_RULE_FIXTURE` supplies an exported decoder graph; the executor's
external-checkpoint structural test can write one with
`ORBITKV_RULE_FIXTURE_OUTPUT`. These settings exist only in tests. The fixture
uses metadata, normalization and decode intervals; it does not load weights,
profile GPU candidates or represent complete model compilation.

## Reusing compilation setup across buckets

`Graph::build_search_space` prepares operation declarations, backend facts,
late-pass definitions and the normalized model program once. A model-local
`PreparedEgglog` template is cloned before adding each bucket's interval facts.
Every bucket still runs all original main and late schedules. Its unions,
range proofs, aliases and rule execution state remain independent. No global
cache, cross-model reuse, search-budget reduction or runtime provider preference
is introduced. Single-run diagnostic helpers consume their setup directly.

Setup, template cloning and bucket facts have separate tracing spans. Shared
setup is outside the per-bucket saturation spans and must be counted once when
attributing complete compilation. Tests check conflicting intervals, exact-value
unions, alias isolation, late passes and fresh/prepared search-space agreement.
The saturation implementation lives in `egglog_utils/saturation.rs`; private
tests remain under the compiler crate's `tests/unit/egglog_utils/`.
