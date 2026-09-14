# Search coverage and its limits

Correctness passes on NVIDIA H20. Broader initial exploration does **not**
establish a general performance improvement: B8 prefill improves against the
previous observation, while B8 decode regresses. The default initial population
remains one; the eight-candidate workload is an explicit experiment.

## Implementation

Luminal now samples existing egraph alternatives in sorted snapshot class/node
order. Mutation pools and cycle-repair tie breaking follow that order. Bounded
initial exploration uses shuffled per-class cycles before genetic mutation.
OrbitKV exposes its allowance as artifact-bound `initial_candidates`.
The trace records snapshot digests, sampling origins, generated-kernel names and
host-provider names. Graph rewriting and implementation generation stay in
egglog; no model-name or tensor-dimension provider override is added.

The guarantee concerns a fixed serialized snapshot. Fresh saturation remains
noncanonical, and runtime outcomes, timing noise and budgets can change later
generations. Per-class spelling coverage is not executable-region coverage.
See [the design](../../docs/search-coverage.md) for retry and budget semantics.

## Qualification

One frozen release binary runs the Qwen3.8-27B-FP8 hybrid text decoder through
seven buckets with seed 7, eight measured graphs per bucket, eight initial
genomes, two deployment finalists, three profiling trials and graph residency
capacity two. The [manifest](../../benchmarks/search-coverage.json) admits B1–B8
and 1–32 total query tokens. This does not qualify every shape in those ranges.

Nine separate processes cover B1 cold search, B1/B8 strict replay and event
profiles, then stage-disabled replay and profiles. All **296 full-vocabulary
teacher-forced comparisons** and final resource drains pass. Maximum absolute
logit error is **0.8671875**, below the unchanged **1.0** gate. Replay verifies
that the saved artifact stays unchanged. Model math, weights and oracle remain
unchanged.

Host checks pass: 247 root tests, 249 fork tests, 78 executor CUDA tests and 40
Python tests. The fixed-snapshot CUDA matrix regression measures generated and
cuBLASLt candidates, repeats the evaluated program sequence, and checks selected
outputs against independent CPU products. Three search-trace unit tests pass.
Counts describe overlapping suites, not unique cases. Formatting, Clippy,
source layout, frontend tests and website check/build are recorded in
[checks.json](checks.json).

## What search actually covered

The trace contains **319 direct evaluations: 56 measured and 263 rejected**,
followed by 14 deployment measurements and seven selected programs. Every direct
rejection reports a required persistent-state alias violation. Rejected graphs
cannot be treated as successful measurements of their matrix implementations.

| Representative requests / query tokens | Measured output projections | Selected projection | Rejected graphs with a cuBLASLt output projection |
| --- | --- | --- | ---: |
| 1 / 1 | 8 GEMV | GEMV | 4 |
| 1 / 4 | 7 cuBLASLt, 1 generic | cuBLASLt | 5 |
| 1 / 32 | 8 cuBLASLt | cuBLASLt | 55 |
| 2 / 4 | 8 generic | GenericMatmul | 29 |
| 2 / 32 | 8 generic | GenericMatmul | 18 |
| 8 / 8 | 8 generic | GenericMatmul | 46 |
| 8 / 32 | 7 cuBLASLt, 1 generic | cuBLASLt | 19 |

In the B8 decode bucket, all eight measured graphs retain the generic vocabulary
projection. Forty-six graphs containing the cuBLASLt projection fail state
alias validation elsewhere. Two-finalist reranking cannot recover a region
implementation that no valid evaluated graph retained. This is the concrete
limit of broad whole-genome sampling under correlated state constraints.

The table is a post-run diagnostic of this checkpoint's vocabulary matrix,
checked against each program descriptor; it is not a production dispatch rule.
[search.json](search.json) retains candidate outcomes, snapshot identities and
matrix descriptors. [selection.json](selection.json) retains selected egraph
templates and available alternatives.

## Compiler and runtime observations

Cold diagnostic process time is **753.18 s**; the decoder compile call takes
**750.97 s**. Enclosing egglog runs take **518.28 s**, including **337.09 s**
reported by the `glumoe` ruleset. These accounting views overlap. The enclosing
CUDA bucket-search spans total **185.30 s**, including preparation and finalist
work. Wider sampling has not solved the cold-compilation cost.

The comparison below uses stage-disabled, event-disabled strict replay. Decode
is the median of six subsequent diagnostic steps; prefill is one observation.
All steps request full-vocabulary logits. These are not serving throughput or
streaming TPOT measurements.

| Diagnostic wall time | Previous time-only record | This eight-graph search |
| --- | ---: | ---: |
| B1 prefill | 152.50 ms | 134.30 ms |
| B1 subsequent decode | 24.43 ms | 24.22 ms |
| B8 prefill | 247.36 ms | 77.32 ms |
| B8 subsequent decode | 36.95 ms | 78.96 ms |

Separate event profiles attribute the current B8 prefill output projection to
cuBLASLt (**0.718 ms**) and the final B8 decode projection to GenericMatmul
(**43.041 ms**). B1 uses cuBLASLt for prefill (**0.676 ms**) and GEMV for final
decode (**0.662 ms**).

The [previous record](../workload-attribution-20260914/README.md) used one graph
and one finalist. Binary, fresh search snapshots, budgets and selected programs
differ, and provider/JIT caches were reused. This is an observational comparison,
not an isolated sampling-policy ablation or evidence of a serving improvement.

## Next work

1. Apply provable persistent-state alias constraints before expensive candidate
   preparation. Preserve the operation alternatives and required state writes.
2. Explore measured expensive regions while retaining a valid surrounding
   genome, using graph semantics, contracts and costs. Cover prefill and decode
   independently; provider presence anywhere in a graph is insufficient.
3. Optimize `glumoe` joins in egglog, then repeat controlled compilation and
   uninstrumented serving workloads before changing performance defaults.

## Records and reproduction

Use the [qualification runner](../../tools/run_decoder_qualification.py) with a
prebuilt release `model_execution` binary, the manifest above,
`--search-graphs 8`, and independent model/reference directories. Keep the search
trace for the fresh run and use that run's artifact for B1/B8 replay. Stage and
event profiles belong in separate processes from uninstrumented timings.

- [summary.json](summary.json): all process results, policy, coverage and limits.
- [compiler.json](compiler.json), [runtime.json](runtime.json): CPU and device attribution.
- [environment.json](environment.json), [source.json](source.json): hardware,
  toolchain, frozen binary, code hashes, model/oracle and artifact identities.
- [checks.json](checks.json): local checks and resolved test-development failures.
- [files.json](files.json): hashes of retained local raw records; those ignored
  `.qualification/` paths are not downloadable repository artifacts.
- [SHA256SUMS](SHA256SUMS): integrity manifest for this curated record.

Base commits precede the final commits; source hashes identify the uncommitted
code actually built. Historical result records remain unchanged. Initial GPU
test-development failures concerned private imports and trace-label assertions;
the passing fixture uses structured fields and the original numerical gate.
