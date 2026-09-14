# Search coverage

OrbitKV compiler generates implementation alternatives in egglog. Core samples complete
genomes from those existing equivalence classes; the CUDA runtime checks their
resource contracts, prepares them and measures executable graphs on the GPU.
Deployment-mode measurement chooses among retained finalists. Neither sampling
nor the executor rewrites extracted LLIR to force a provider.

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
`DecoderTuningProfile`. Zero is the default. Adding this field changes the
artifact-bound tuning digest, including the zero/default policy; regenerate
older decoder artifacts before loading them with this version. The
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
and enumerates its other admitted nodes in snapshot order. A neighbor changes
exactly one binding. All other bindings, including choices in newly reachable
dependencies, retain the parent's values. The extractor does not repair an
invalid neighbor by changing unrelated decisions. Cycle, state-alias, layout and
resource checks still apply to the complete result. Shared e-classes can affect
multiple operations; a single binding change is not necessarily one kernel change.

A measured improvement can become the next parent immediately. Rejected,
cyclic and duplicate neighbors consume the local attempt allowance, while
rejected candidates do not displace valid parents or consume the graph
measurement budget. Once local exploration is exhausted, ordinary coverage and
genetic exploration continue within the remaining budget. No implementation
family receives a preference in Rust.

This is coordinate exploration over an existing egraph. It does not guarantee a
global optimum, jointly repair a dependency closure, add fusion rules, or create
a whole-model megakernel. Fresh-snapshot identity and measurement noise still
affect the search. Multi-choice regions and joint KV-layout alternatives remain
separate compiler work.

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
graphs; local exploration now preserves the remaining parent bindings, while
state-compatible dependency-closure generation remains further work. Optional aliases still permit
materializing implementations, and mutation-order checks remain mandatory.

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

The [27B H20 qualification](../results/search-coverage-20260914/README.md) passes
296 logit comparisons but records a mixed performance result. Every one of its
263 rejected graphs violates a required state alias. B8 decode measures eight
generic output projections while graphs with the cuBLASLt projection fail state
validation elsewhere. Earlier rejection addresses wasted preparation; constrained
exploration of expensive regions motivated the local phase above.

The [state-preflight qualification](../results/state-preflight-20260914/README.md)
passes another 296 logit comparisons and drains. Rejected-candidate evaluation
totals 8.00 seconds against the preceding 81.06-second observation, while warmed
decode stays close. Snapshot identities differ and caches were reused; this is
not a paired compiler-speedup or serving-throughput claim.

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

The [hotspot qualification](../results/hotspot-search-20260914/README.md) records
49 measured local neighbors with no state/resource rejection and two prefill
provider transitions inside valid parents. All 296 reference comparisons pass.
The B8 decode seed already uses cuBLASLt; improved historical runtime observations
are not proof of a hotspot provider transition or serving speedup.

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
