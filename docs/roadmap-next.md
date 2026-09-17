# OrbitKV Next: a compiled engine for stateful hybrid inference

Status: proposed reset charter. This document defines a new implementation line.
It does not extend the current seven-crate engine and does not treat existing
code volume as an asset that must be preserved.
The detailed three-model decomposition and component reuse decisions live in
[the next architecture](architecture-next.md).

## Decision

Build a small, model-specialized inference engine for new hybrid architectures,
using the imported `kern` source as its execution, state-management, and serving
layer. The first target is `Qwen3.8-27B-FP8` on one NVIDIA H20. The engine's
compiler turns a typed, stateful decode graph into a small number of GPU
execution islands and selects mature provider kernels where they remain the
best boundary.

The project is not a general replacement for vLLM or SGLang. It exists to test
and exploit one hypothesis:

> Hybrid decoders with recurrent state expose cross-operator scheduling,
> fusion, state-placement, and speculative-commit opportunities that
> kernel-per-operator engines do not capture cleanly.

The first release succeeds only if it produces correct end-to-end inference and
a material measured advantage over a pinned, tuned vLLM baseline. A novel IR, a
generated kernel, or a faster microbenchmark is not sufficient.

## Product boundary

### Own

- Model-specific topology, weight mapping, and numerical contracts.
- A small typed task IR with explicit tensor layout and persistent-state effects.
- Ahead-of-time schedule selection and artifact generation.
- Generated stateful kernels and adapters to mature GEMM/attention kernels.
- Lowering into the existing `kern` manifest and verified-program runtime.
- Deterministic compiler and kernel benchmark entrypoints.
- Layer, multi-step state, and full-model correctness oracles.

### Do not own initially

- A general tensor framework or arbitrary graph frontend.
- Another CUDA runtime, weight loader, state pool, or serving loop already
  supplied by `kern`.
- Algebraic saturation or unrestricted e-graph search.
- Training, autograd, distributed execution, or heterogeneous clusters.
- A general KV storage service, external tier, RDMA, NIXL, or Mooncake.
- A full OpenAI-compatible product surface, tokenizer ecosystem, chat templates,
  observability platform, or multi-tenant control plane.
- Arbitrary models, dtypes, GPUs, attention backends, or quantization formats.
- Online JIT compilation in the request path.

These capabilities may be supplied later by adapters or an existing serving
frontend. They must not enter the core until the compiled decode path wins.

## Design principles

1. **Model-specialized, family-general.** A model owns its topology and weight
   mapping. Execution primitives and compiler decisions are keyed by shape,
   dtype, layout, effects, and target hardware, never by checkpoint name.
2. **One hot path.** There is one production decode implementation per compiled
   artifact. Reference and diagnostic paths live outside release execution.
3. **Compile outside serving.** Kernel generation, schedule search, and
   qualification happen ahead of time. Startup loads and validates an artifact.
4. **Use the best primitive.** DeepGEMM, cuBLASLt, FlashInfer, FlashAttention,
   FlashQLA, CUTLASS/CuTe, or a generated kernel may be used when measurement
   justifies it. Reimplementing a mature primitive is not a goal.
5. **Fuse around expensive boundaries first.** Remove intermediate traffic,
   repeated quantization, launch bubbles, and fragmented state I/O before
   attempting an entire-model kernel.
6. **State effects are part of compilation.** Persistent KV, recurrent, and
   convolution reads/writes, aliases, versions, and completion requirements are
   explicit IR properties. They are not inferred from pointer coincidence.
7. **Performance claims are end to end.** Compare identical weights, inputs,
   output semantics, hardware, concurrency, and measurement protocol.
8. **Complexity has a budget.** No abstraction is admitted without eliminating
   duplicated model code, enabling another model geometry, or producing measured
   performance/correctness value.

## Initial target

The first vertical slice is deliberately narrow:

| Dimension | Initial contract |
| --- | --- |
| Model | `Qwen3.8-27B-FP8`, text decoder only |
| Device | One NVIDIA H20 / SM90 |
| Weights | Official block-scaled E4M3 FP8 checkpoint |
| Activations and state | The checkpoint's required BF16/FP32 boundaries |
| Batch | Decode batch 1, 2, 4, and 8 |
| Prefill | Correct provider-based fallback; not initially megakernelized |
| Attention | Existing qualified paged-attention provider |
| Sampling | Greedy only |
| Context | A declared bounded test range, expanded only after correctness |
| Serving | Benchmark CLI first; minimal HTTP only after the performance gate |

Qwen3.8 is a suitable first target because its 3:1 Gated DeltaNet to Full
Attention schedule stresses both persistent recurrent state and token KV while
repeating a compile-friendly three-GDN-layer pattern. It is not evidence of
generality by itself.

## Target model ladder

The first three target contracts are pinned and intentionally sequential:

| Order | Target | Architecture pressure | Deployment boundary |
| --- | --- | --- | --- |
| 1 | `Qwen3.8-27B-FP8` | 48 GDN plus 16 Full Attention layers, dense block-FP8, one MTP layer | Complete model on one H20 |
| 2 | `GLM-5.3-Flash` | 34 KDA plus 11 sparse-attention layers, 288-expert MoE, mHC, one MTP layer | Layer/pruned work on H20; complete model requires multi-GPU or bounded residency |
| 3 | `DeepSeek-V4.1-Flash` | CED, CSA2 shared compressed state, Engram, 384-expert FP4 MoE, three DSpark layers | Blackwell/multi-GPU primary performance target; H20 is not the native FP4 target |

The pinned GLM index is about 305.78 GiB and the DeepSeek index about 475.24
GiB. Neither is admitted as a single-H20 resident model. Work on them before
Qwen passes the target-only gate is limited to semantic fixtures and reusable
kernel contracts.

## Architecture

Keep one compiler crate until a kernel build boundary proves that a split is
necessary. Target the local `kern` manifest v5 and extend the imported runtime
only for measured cross-model requirements. A target
layout is:

```text
Cargo.toml
src/
  model/
    mod.rs
    qwen38.rs          # topology, weight names, exact numerical contract
  ir/
    task.rs            # typed dataflow and dependencies
    effect.rs          # persistent reads, writes, aliases, commit boundary
    layout.rs          # shape, dtype, stride, alignment and storage class
  compiler/
    partition.rs       # execution-island formation
    schedule.rs        # bounded target-specific schedule candidates
    memory.rs          # liveness and intermediate allocation
    artifact.rs        # immutable AOT deployment artifact
  lower/
    kern.rs            # verified manifest-v5 emission
  kernels/             # native wrappers and generated-kernel metadata
  bin/
    compile.rs
    bench.rs
kernels/
  gdn/
  fp8/
  attention/
tests/
  operator/
  layer/
  multistep/
  model/
tools/
  oracle/
  benchmark/
```

Do not create additional crates merely to express conceptual layers. Split only
when separate dependency closure, compilation, reuse, or FFI ownership requires
it. The first complete implementation should remain below roughly 30--40 KLOC
excluding generated bindings and vendored provider code.

## Stateful task IR

The IR is not a general-purpose tensor language. It represents the bounded
execution decisions needed by a hybrid decoder:

```text
Task
  operation and numerical contract
  tensor shape, dtype, layout and alignment
  input/output storage classes
  persistent-state read/write regions
  required aliases and mutation effects
  dependencies and completion scope
  supported implementation candidates
  static resource requirements
```

The initial task kinds are:

- provider GEMM;
- provider paged attention;
- generated elementwise/reduction region;
- causal convolution state transition;
- GDN recurrent state transition;
- state commit;
- sampling/output reduction.

The compiler performs only the following passes:

1. Validate shapes, dtypes, layouts, state effects, and aliases.
2. Bufferize intermediates with explicit liveness and capacity.
3. Partition the graph into provider calls, generated superkernels, and
   persistent islands.
4. Enumerate a small, explainable schedule set for each batch bucket.
5. Measure candidates on the deployment path.
6. Emit a content-addressed artifact containing the selected schedule, native
   images, provider identity, resource bounds, and numerical contract.

No genetic search is required. No Rust post-pass may silently change model
semantics. Search dimensions must correspond to an explicit hardware decision,
such as tile shape, pipeline depth, CTA count, SM allocation, fusion boundary,
or materialization choice.

## Execution hierarchy

The compiler deliberately emits several execution granularities. A larger
kernel wins only when measured.

### Level 0: provider baseline

Run the correct decoder using the best available GEMM, attention, and recurrent
providers, captured by CUDA Graph where applicable. This establishes the
performance floor and the numerical oracle for every later level.

### Level 1: GDN stateful superkernel

Keep large projections as provider calls. Fuse the stateful region between them:

```text
projected Q/K/V/Z/A/B
  -> split and convert
  -> causal convolution and history update
  -> decay/update gates
  -> delta recurrence and state update
  -> gated normalization
  -> output-projection input
```

The kernel consumes request-to-state-slot indices directly and commits
recurrent/convolution state in place under an explicit effect contract. It must
eliminate state Gather/Scatter and the large intermediate materializations, not
merely concatenate source files.

### Level 2: shared-input low-M FP8 projection

Compile the QKV/Z/A/B projections that consume the same hidden activation as a
coordinated task set:

- quantize the activation once;
- choose batch-specific GEMV/GEMM tiles;
- balance independent projections across SMs;
- fuse compatible epilogues;
- feed the GDN superkernel without unnecessary HBM materialization.

DeepGEMM or cuBLASLt remains the fallback. A custom projection path is retained
only if it wins end-to-end.

### Level 3: persistent GDN island

Represent one or more consecutive GDN layers as an SM-level task graph. Persistent
CTAs claim ready tiles, obey dependency counters, and overlap memory-bound and
tensor-core work. Full Attention layers form initial island boundaries and use
the mature attention provider.

The first target is one complete GDN layer. The next target is the repeated
three-GDN-layer region between Full Attention layers. An entire 64-layer decoder
kernel is not a milestone.

### Level 4: native speculative execution

After target-only decode is competitive, add the checkpoint's native MTP path.
One target-model weight pass verifies several candidate tokens. KV, recurrent,
and convolution state are versioned as tentative outputs and atomically committed
at the accepted boundary. Rejecting drafts must be equivalent to ordinary target
decode.

This is the path to multiplicative effective-token throughput. Megakernel fusion
alone cannot remove the need to read dense model weights for each target step.

## Model extensibility

The compiler is extensible by execution family, not by pretending every model is
the same graph. Adding a checkpoint in an implemented family should require only:

- a topology and geometry declaration;
- a weight-name/layout mapping;
- optional provider capability declarations;
- golden numerical fixtures and workload buckets.

Adding a new state family requires a semantic operation, reference implementation,
effect contract, and at least one optimized implementation. It must not require a
new scheduler, allocator, or artifact format.

| Model family | Shared compiled capability | Model-specific work |
| --- | --- | --- |
| Qwen3.5/Qwen3.8 dense hybrid | Full Attention + GDN + convolution | Geometry, weights, exact rounding |
| GLM-5.3 | KDA + sparse MLA + MoE + mHC | 45-layer topology, 288 experts, NoPE/indexer contract |
| DeepSeek-V4.1 | CED + CSA2 + Engram + FP4 MoE + DSpark | Cross-layer producers, compression modes and lookup tables |

The second Qwen checkpoint must reuse the same GDN task kinds and compiler passes
while changing geometry. This proves family generality. GLM-5.3 must introduce a
different state family while still targeting the same shared `kern` substrate.

Large MoE checkpoints, CPU weight offload, expert parallelism, and multi-node
execution are postponed until the single-GPU compiler hypothesis is validated.
GLM-5.3 then introduces grouped expert execution and explicit multi-GPU
residency. DeepSeek-V4.1 is admitted only after that substrate works.

## Reuse and archive policy

The current repository is evidence and a source mine, not the new architecture.
Before deletion, preserve the complete working tree on an archival branch or
tag, including currently uncommitted work. Do not copy modules wholesale.

Candidate code to extract selectively:

- Qwen3.8 configuration and weight mapping;
- DeepGEMM and FlashInfer/FlashAttention provider adapters;
- GDN and causal-convolution reference/optimized kernels;
- layer probes, teacher-forced logit oracle, and matched benchmark harness;
- persistent-state alias and multi-step correctness tests.

Do not carry into the new active implementation:

- the standalone KV-manager product and retention DSL;
- external KV tiers and distributed placement;
- the general Luminal-derived frontend, e-graph saturation, and genetic search;
- the current executor abstraction and seven-crate ownership graph;
- the HTTP engine and complete vLLM protocol surface;
- generic model import promises;
- GLM/DeepSeek/offload plans that are not needed by the first vertical slice;
- historical compatibility layers or prior artifact schemas.

PegaInfer is a design and implementation reference, not a dependency to copy in
bulk. Reuse its strongest lessons: model-local topology, AOT kernels,
pointer-stable buffers, CUDA Graph replay, strict accuracy gates, and matched
serving measurements. Its runtime lessons are already embodied by `kern`; do not
reimplement them. Avoid reproducing its later breadth before the first model
wins. Preserve upstream licenses and exact source provenance for any adapted
kernel.

`kern` is the imported execution baseline rather than only a design reference.
Preserve its upstream provenance, evolve its schema/runtime deliberately, and
emit its native manifest. Its existing Qwen3.8
and DeepSeek-V4.1 artifacts are baselines and executable oracles. The new work
must automate model-to-program lowering and execution-island optimization; a
handwritten manifest clone is not progress.

## Baselines and measurement rules

Pin the checkpoint, engine revisions, provider revisions, driver, CUDA toolkit,
clock policy, and request traces. Compare against tuned vLLM and SGLang on the
same H20. PegaInfer may be included when its exact checkpoint and precision are
compatible.

Required measurements:

- target-only TPOT and output throughput for batch 1, 2, 4, and 8;
- prefill TTFT for short and medium prompts;
- per-island and per-provider device time;
- achieved memory bandwidth and tensor-core utilization;
- launch gaps, active waves, occupancy, registers, shared memory, and spills;
- HBM allocated/resident bytes and graph/workspace overhead;
- cold compile, warm artifact load, and first-request latency.

Correctness gates precede performance interpretation:

- independent operator references;
- layer-boundary comparison;
- teacher-forced full-vocabulary logits;
- at least 64 sequential decode steps;
- batch invariance and request reordering;
- ragged prefill followed by decode;
- recurrent and convolution state drain;
- exact top-1 agreement outside declared near-tie cases.

Do not compare performance across engines until output semantics are equivalent.
Microbenchmark wins must be reported separately from model and serving wins.

## Milestones and gates

### M0 — Preserve and reset: 2 days

- Save the complete current repository and uncommitted work under an explicit
  archive reference.
- Create a clean implementation branch with one compiler package and the
  provenance-preserving `kern` execution crates, with no old OrbitKV workspace
  members in its build graph.
- Freeze the old roadmap and record the exact reusable source inventory.
- Pin the Qwen3.8 checkpoint, H20 environment, vLLM/SGLang baselines, inputs, and
  correctness oracle.
- Treat the imported `kern` Qwen3.8 manifest as a BF16 execution/topology oracle,
  and the official checkpoint config/index/headers as the separate FP8 weight
  oracle.

Gate: a clean checkout can reproduce the baseline and no historical code is lost.

### M1 — Minimal correct decoder: week 1--2

- Lower the Qwen model package into a `kern` manifest without handwritten
  per-layer calls.
- Validate all official text-tower FP8 matrices and their 128 by 128 inverse
  scale tensors before deriving packed execution buffers.
- Execute provider-based prefill and target-only decode through the shared
  `kern` substrate without a model-private runtime path.
- Use the substrate's fixed buffers, state allocations, and batch programs.
- Pass serial, changing-batch, ragged, and multi-step state checks.

Gate: correctness is closed before custom optimization. Compiler output matches
the existing `kern` Qwen topology and serving behavior while its FP8 weights are
qualified separately against the official checkpoint, and the new repository
contains no duplicate runtime.

### M2 — Stateful GDN superkernel: week 3--4

- Fuse convolution, recurrent update, normalization/gating, and state commit.
- Consume state-slot indices directly; remove Gather/Scatter intermediates.
- Support batch 1, 2, 4, and 8 through generated constants or bounded templates.
- Record Nsight Systems and Nsight Compute evidence.

Gate: at least `1.5x` faster for the covered GDN non-GEMM region and at least
`10%` faster for the complete target-only decoder, with unchanged correctness.
Otherwise stop persistent-kernel work and publish the negative result.

### M3 — Low-M FP8 projection scheduling: week 5--6

- Reuse activation quantization across compatible projections.
- Compare custom GEMV/GEMM tiles with pinned DeepGEMM and cuBLASLt.
- Fuse producer/consumer epilogues when it removes measured HBM traffic.
- Select schedules separately for B1/B2/B4/B8.

Gate: target-only B1 is within `10%` of the faster pinned mainstream baseline,
or beats it; B4/B8 do not regress by more than `5%`. If the custom GEMM cannot
approach the provider baseline, retain the provider and do not expand the GEMM
compiler.

### M4 — Persistent execution island: week 7--9

- Add a bounded device task queue and deadlock-safe dependency protocol.
- Compile one full GDN layer, then the repeated three-GDN-layer island.
- Search island boundaries, CTA count, SM allocation, and materialization.
- Keep Full Attention as a provider boundary.

Gate: at least `15%` lower B1 target-only TPOT than the faster pinned mainstream
baseline, with B8 throughput at parity or better. The gain must survive full
model serving, not only an isolated layer.

### M5 — Serving integration: week 10

- Run the artifact through `kern-serve` and its existing frontend bridge.
- Add only compiler/artifact metadata needed by serving and benchmarking.
- Do not add another scheduler, frontend, plugin system, or protocol surface.

Gate: benchmark-client results agree with direct decoder measurements and expose
no material host bottleneck.

### M6 — Same-family generalization: week 11--12

- Add a second Qwen hybrid checkpoint with different dimensions.
- Recompile from the same stateful IR and schedule machinery.
- Permit new tile parameters and weight mappings, but no copied scheduler or
  model-specific runtime.

Gate: at least `80%` of non-test compiler/kernel source is shared and both models
pass the same correctness and measurement pipeline.

### M7 — Native speculative decode: after target-only success

- Add the checkpoint's matching MTP/draft model.
- Represent base, tentative, and committed persistent-state versions.
- Compile verification and exact accepted-frontier commit/rollback.
- Measure acceptance, target passes, effective tokens/s, and state-copy cost.

Gate: at least `1.5x` effective output throughput over this compiler's target-only
path on a declared agentic workload, with forced-reject equivalence and no
batch-dependent state divergence.

### M8 — GLM-5.3 capability stack

- Add KDA, sparse MLA/indexer, mHC, grouped FP8 MoE, and expert communication
  as separately gated task families.
- Validate layers and pruned fixtures on H20 before complete-model deployment.
- Run the complete 305.78 GiB checkpoint only with an explicit multi-GPU or
  bounded-residency plan.
- Reuse the shared `kern` substrate plus the Qwen-stage task effects,
  compiler passes, artifact lowering, and test harness.

Gate: complete inference is numerically qualified; model-specific code is
limited to topology, weight mapping, numerical constants, and genuinely new
task implementations.

### M9 — DeepSeek-V4.1 capability stack

- Add CED execution, CSA2 producer/reuse dependencies, compressed FP4 KV,
  hierarchical sparse indexing, Engram residency, FP8xFP4 MoE, and DSpark.
- Use Blackwell or another hardware path that natively supports the selected
  low-precision contracts for performance claims.
- Keep H20 runs as semantic or fallback evidence unless a separately qualified
  Hopper implementation exists.

Gate: the complete 475.24 GiB checkpoint passes a pinned reference and matched
serving comparison on the same shared `kern` substrate and without
changing artifact semantics. This is the first credible claim that the compiler
supports multiple new-model architecture families.

## Stop conditions

Stop or narrow the project if any two conditions hold:

- M2 produces less than `10%` full-decoder improvement.
- M3 cannot approach the best provider for low-M projections.
- M4 wins only at one batch size or one synthetic shape.
- Correctness still changes with batch composition after M2.
- More than half of a second model's execution code must be copied or rewritten.
- Progress shifts back to HTTP, generic model support, offload, or documentation
  before the decode performance gate is met.
- The implementation exceeds the source budget without replacing an equivalent
  amount of code or demonstrating an additional model family.
- Eight weeks pass without a correct end-to-end result competitive with the
  pinned baseline.

## Claim ladder

Claims must be earned in this order:

1. Correct Qwen3.8 target-only decoder.
2. Faster GDN stateful region.
3. Faster complete target-only decode on H20.
4. Same compiler supports a second geometry.
5. Native speculative decode produces a multiplicative workload gain.
6. GLM-5.3 reuses the compiler and `kern` artifact path while adding KDA, sparse
   MLA, MoE, and mHC.
7. DeepSeek-V4.1 reuses that path while adding CED, CSA2, Engram, FP4, and
   DSpark.

Until level 4, describe the project as a Qwen3.8 specialized inference engine.
At level 4, its differentiator is a GDN-family compiler. Only after level 6
describe it as a new-generation hybrid-model inference engine; level 7
establishes that the architecture spans both recurrent and cross-layer
compressed-state families.
