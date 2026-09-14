# Joint state and compute compilation

Status: target design. This document separates existing integration points from
proposed extensions; it does not claim that joint layout search or generated
persistent execution is implemented.

The goal is one native Rust inference process in which OrbitKV compiles and
owns persistent state, OrbitKV compiler compiles equivalent compute implementations, and
the executor selects a compatible state/compute deployment. The first acceptance
workload is the official Qwen3.8-27B-FP8 text decoder on H20. Its structural
requirements drive coverage; checkpoint names must never select an operator,
layout, fusion, or schedule in product code.

The deployment budget is one H20, with quantization and CPU offload allowed.
The [model targets](model-targets.md) extend this design toward Qwen, GLM, Kimi
and DeepSeek. Immutable weight residency and transfer scheduling join compute
and mutable state under a shared budget; host weight offload is not implemented
by the current checkpoint loader or external KV transport.

## Current foundation

- `RuntimeManifest` and backend-neutral `StateLayoutFacts` describe state classes,
  retention, address/retirement programs, and byte geometry. The executor joins
  these facts with stable arenas and lowers them into compiler facts.
- Token-KV updates require in-place aliases. Recurrent/convolution state uses
  typed shared arenas, generation-checked bindings, and event-backed completion.
  Search uses scratch state rather than live request state.
- Logical attention and a typed paged KV view have separate contracts.
  Capability rules admit FlashInfer CUDA-core/tensor-core algorithms and
  compatible FlashAttention-3 kernels in the same e-class. Block-FP8 linear semantics have DeepGEMM
  candidates. Existing candidates do not establish general layout search.
- An opt-in FP8 region alternative shares graph-visible activation preparation
  across compatible projections. Workload profiles now specify feasible joint
  batch/query/page representatives, scratch metadata and CUDA Graph finalist
  budgets. See [FP8 region tuning](fp8-region-tuning.md) for qualification and
  the private-page fixture boundary.
- `CustomOp`, `KernelOp`, and `HostOp` provide semantic facts, generated kernels,
  and host orchestration of GPU work. Host operations can declare workspace and
  CUDA Graph preparation requirements. DeepGEMM demonstrates native loading of
  a compiled provider library and passing device pointers plus a CUDA stream.
- Triton, TileLang, and FlashMLA are not integrated providers. General algorithm
  region search, joint state-layout search, and generated persistent megakernels
  remain planned. The existing bounded Qwen closure is not serving-scale or
  broad released-model qualification.

See [architecture.md](architecture.md), [implementation-status.md](implementation-status.md),
and [roadmap.md](roadmap.md) for the current implementation and qualification scope.

## Responsibilities

| Component | Owns | Boundary |
| --- | --- | --- |
| `orbitkv` | State semantics, legal state realizations, pages, generations, Prefix/COW, publication, retirement, acknowledgement, reuse | No OrbitKV compiler, egglog, CUDA, provider, or model-name dependency |
| `orbitkv-executor` | Proposed joint compilation coordinator, arena bindings, provider adapters, compiled deployment, streams and execution receipts | Can join state and compute contracts; cannot grant page ownership or bypass session transactions |
| OrbitKV compiler | Equivalent graphs, algorithm regions, kernel schedules, resource/alias validation and profiling | Consumes state constraints; never owns the runtime page lifecycle |
| Operator providers | Implementations and their numerical, launch, workspace and effect contracts | Cannot hide page allocation, state publication, or unaccounted device work |
| `orbitkv-engine` | Logical request/output contracts, optional client frontend, scheduling, admission, runtime plan selection, execution and completion coordination | Protocol/frontend modules contain no page identities or kernel policy; the coordinator invokes the executor and `RuntimeSession` without checkpoint-name dispatch |

The joint coordinator belongs in the executor. Core remains usable as a standalone
Rust crate by other engines. The coordinator supplies device and workload facts
to backend-neutral state compilation; backend-specific search stays outside core.

## Three compute compilation levels

These are alternative implementation granularities within one search space,
not mandatory stages through which every operation must pass.

1. **Operator providers.** A semantic operation admits equivalent library or
   generated implementations, each with explicit legality predicates. Examples
   include paged attention and block-scaled linear. Triton/TileLang integration
   would compile outside the request path and expose a native launch adapter;
   a Python frontend need not become the serving runtime. FlashMLA would require
   its own semantic and state-ABI admission before entering this mechanism.
2. **Multi-operation algorithm regions.** Egglog matches a complete equivalent
   subgraph and adds a fused or algorithmically different candidate. Useful
   regions include projection/quantization, normalization/gating, and recurrent
   update/readout where legality can be proved. A DSL provider receives the
   complete region before compilation. Loading an existing binary does not fuse
   its internals with neighboring operators. Preserve the semantic decomposition
   and mature-provider combinations as competing implementations.
3. **Generated persistent execution.** The compiler may schedule a larger region
   with persistent workers, shared intermediates, and explicit synchronization.
   The result may be one megakernel or several cooperating kernels, with
   host-orchestrated library calls between generated regions where appropriate.
   A full model in one kernel is not a requirement.
   Global dependencies, occupancy, progress, register/shared-memory pressure,
   and state effects must be validated before such a candidate is deployable.

Matching, fusion, and implementation alternatives belong in egglog rewrites,
not Rust graph-replacement postpasses or checkpoint dispatch. Compilation level
does not determine the winner: measured compatible deployments do.

## Shared state, layout, and effect contract

Joint compilation separates **semantic meaning**, **persistent-state realization**,
and **compute schedule**. Semantics are fixed and constrain the search over the
other two. Compute fusion may change intermediates but cannot silently change
retained history, mathematical state transitions, quantization rules, or the
persistent-state ABI.

The following contract fields are proposed extensions or consolidations of
existing facts, rather than a new public API already present in the repository:

| Contract area | Required information |
| --- | --- |
| Semantics | Operation/region identity, numerical rules and tolerances, visibility, state transition, class/layer identity |
| Layout | Logical shape, dtype, physical byte capacity, base allocation and offset, strides, alignment, page/component geometry |
| Effects | Read/write regions, output/input alias, mutation versus a pure view, exclusive-use and ordering requirements, externally visible outputs |
| Workspace | Size/alignment, ownership, lifetime, persistent scratch versus temporary storage, resource limits and allocation timing |
| Launch | Target/compiler/provider identity, module entry, argument ABI, grid/block/shared-memory requirements, supported dynamic dimensions |
| Replay and completion | Pointer and shape dependencies, metadata refresh, stream ordering, capture preparation, completion evidence tied to the bound state |

Logical shape alone is not a physical-layout proof. In particular, a view into
a wider projection cannot be flattened as dense rows without base/stride facts.
Region boundaries must also account for intermediate values used outside the
region and for all state writes, not only the final tensor output.

Required state aliases remain hard constraints. Ordinary `HostOp` wrapping does
not automatically inherit `KernelOp` alias/mutation proofs; a provider performing
state writes must carry equivalent validated effects through lowering. Opaque
binary adapters also need explicit ABI/resource metadata: the current generated
`KernelOp` interface derives some information from exact CUDA source, which must
not be bypassed with a placeholder source string.

Current Prefix ownership, page contiguity, leases, and in-flight accesses are
runtime facts. Static compilation may define guarded specializations, but the
executor must validate their guards against the current binding. Kernel-local
last use does not authorize page retirement: other requests, generations,
completion evidence, and cleanup acknowledgement still belong to OrbitKV.

## Outer state search and inner compute search

The proposed joint coordinator operates as follows:

1. Take model semantics, admitted workload buckets, hardware capabilities,
   numerical requirements, and hard memory/compilation budgets.
2. Ask state compilation for legal realizations. Initially this can be the single
   current realization. New page/component layouts require explicit compiler
   inputs, capability checks, and validation; they are not arbitrary edits to a
   generated manifest. Existing manifest validation reconstructs the expected
   plan from its source.
3. For each realization, generate a consistent manifest, state facts, arena ABI,
   and compute graph. OrbitKV compiler searches providers, algorithm regions, and tile
   schedules under that contract.
4. Reject numerical, layout, alias, synchronization, and resource violations
   before interpreting performance. Use independent references and scratch state
   for qualification. Profile complete candidates, including conversions,
   metadata preparation, workspace, copies, and launch/capture overhead.
5. Select compatible bucket implementations and package their shared state
   realization. Compare workload latency/throughput and peak memory under the
   declared objective; report compile cost separately rather than hiding it.

Inner provider autotuning must not multiply unchecked by graph candidates and
buckets. Generate a bounded shortlist, reuse compiled variants, and give compiler
subprocesses explicit time, concurrency, memory, and output-size budgets.
Compilation occurs before device timing. Cache identity includes semantic and
layout contracts, target, toolchain, resolved provider/dependency source
contents, wrapper/flags, variant, and launch ABI. Source revisions are provenance;
they cannot stand in for the actual contents of a modified local checkout.
Finalists must be measured on the actual deployment path, including CUDA Graph
execution where applicable.

Decode and prefill may use different providers and schedules while sharing one
persistent ABI. Different persistent layouts require an explicit, validated
transition with its own workspace, movement cost, and completion dependency;
the first milestone keeps one compatible state layout across phases. Runtime
scheduling chooses admitted plans using live facts. It does not recompile a model
for each request or turn static cost estimates into ownership authority.

## Deployment artifact

A proposed joint deployment bundle contains:

- Semantic/model-structure identity and the validated runtime manifest.
- State-realization identity, class/arena ABI, effect constraints, and their digest.
- Per-bucket selected regions/providers, schedules, modules, launch descriptors,
  resource plans, and any explicit phase transitions.
- Target/toolchain/provider identities, numerical qualification records, and the
  workload and measurement method used for selection.

Loading verifies compatibility before accepting executable work. A changed
persistent realization regenerates the manifest, facts, bindings, and affected
executable artifacts together. Runtime allocation identities and generation
leases are rebound and checked; cached raw pointers cannot supply fresh authority.
Execution returns evidence to `RuntimeSession`, which remains responsible for
publication, retirement, acknowledgement, and reuse.

## Staged acceptance

| Stage | Deliverable | Acceptance |
| --- | --- | --- |
| 1. Contract foundation | Explicit layout/effect and provider compilation contracts, preserving current providers | Reject wrong strides, invalid aliases, incompatible artifacts and undeclared resources; retain current state/lifecycle and model-parity gates |
| 2. Provider competition | Native adapters and bounded variants for the admitted semantics | Multiple candidates enter normal search; cold compile and strict replay agree; compiler budgets work; provider availability never triggers model-name dispatch |
| 3. Algorithm regions | At least one structurally matched multi-operation region motivated by the 27B profile | Independent region parity plus teacher-forced model logits; ragged prefill/decode and repeated state transitions; whole-workload benefit measured against the existing composition |
| 4. Joint realization | More than one validated state realization with compatible bucket deployments | Same semantics and lifecycle guarantees, explicit memory/performance tradeoff, correct artifact invalidation and phase compatibility |
| 5. Persistent execution | A generated region scheduled as one or more cooperating kernels | Progress/synchronization and effects validated; event-backed completion and final drain preserved; measured advantage over provider/region candidates on its admitted workload |

Qwen3.8 27B is the primary end-to-end witness for these stages. The structurally
equivalent small BF16 model remains a fast regression witness; dense Full and
Full+Sliding cases retain state-family coverage. Tests and benchmarks identify
checkpoints, while compiler rules identify semantics, geometry, dtype, effects,
and device capabilities. Near-tie token differences require logit-level analysis;
one selected token trace or microbenchmark does not qualify a deployment.

Stages do not promise a speedup from a new DSL, a particular provider winning, or
one full-model kernel. A candidate that loses remains evidence about the search
space. Broader model families, FlashMLA, external KV transport placement, and
multi-device execution require their own admitted semantics and acceptance gates.
