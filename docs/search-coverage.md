# Search coverage

Luminal generates implementation alternatives in egglog. Core samples complete
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
Their `sampling` field identifies initial coverage, mutation or restart.

The fixed seed does not by itself reproduce the complete search: snapshot,
options, RNG state, runtime outcomes and ranking feedback must also match.
Timing noise can change parents, finalists and cooperative stopping points.
The snapshot digest is deliberately separate from the selected program identity.

## Verification and use

Host regressions under Luminal's `tests/unit/egglog` and `tests/unit/search`
exercise map reallocation, alternative ordering, cycle repair, per-class
coverage and finite exploration when genomes produce duplicate programs.
The CUDA integration test `search_coverage` saturates each small BF16 matrix
graph once, searches that same snapshot twice, checks the actual evaluated
program sequence, and requires measured generated and cuBLASLt candidates.
Selected outputs are compared with independent CPU matrix products.

On a configured CUDA machine:

```sh
cargo test --release --manifest-path third_party/luminal/Cargo.toml \
  -p luminal_cuda_lite --test search_coverage -- --ignored
```

[search-coverage.json](../benchmarks/search-coverage.json) supplies a bounded
decoder workload with eight initial genomes and two finalists. Pass it through
`--tuning-profile` with `--search-graphs 8`, retain the search trace, and verify
independent logits and strict artifact replay before interpreting timings.
Broader initial sampling can cost more compilation time and need not improve
every workload. It is a foundation for measured region exploration; expensive
region prioritization, fresh-saturation canonicalization and `glumoe` join
optimization remain separate work.

The [27B H20 qualification](../results/search-coverage-20260914/README.md) passes
296 logit comparisons but records a mixed performance result. Every one of its
263 rejected graphs violates a required state alias. B8 decode measures eight
generic output projections while graphs with the cuBLASLt projection fail state
validation elsewhere. The next step is constrained exploration of expensive
regions inside a valid surrounding genome.
