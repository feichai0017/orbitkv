# OrbitKV Next architecture

Status: proposed architecture for the reset implementation line. The companion
[roadmap](roadmap-next.md) defines milestones, evidence gates, and stop conditions.

## Product definition

OrbitKV Next is a lightweight, source-integrated inference engine for recent
model families whose decode path mixes dense or sparse attention, recurrent
state, Mixture-of-Experts, residual-stream transforms, and native speculative
heads. Its central differentiator is a compiler that produces optimized,
verified inference programs for the imported `kern` execution layer.

It is deliberately not a broad framework. The repository owns model lowering,
execution-island selection, new kernels, runtime execution, KV/state management,
and a narrow serving path. Model-specific complexity stays in the compiler and
kernel layer; the imported runtime remains model-agnostic. The project does not
initially own a general model frontend, distributed control plane, remote cache,
training stack, or broad serving ecosystem.

The implementation starts with three pinned model contracts in increasing order
of systems difficulty:

1. `Qwen3.8-27B-FP8`: single-GPU dense hybrid execution.
2. `GLM-5.3-Flash`: KDA, sparse MLA, MoE, mHC, and multi-GPU residency.
3. `DeepSeek-V4.1-Flash`: CED, CSA2 shared state, Engram, FP4 experts, and
   native multi-token speculation.

These are sequential architecture witnesses, not three simultaneous bring-up
projects.

## Execution substrate decision

Fork [`pegainfer-project/kern`](https://github.com/pegainfer-project/kern) into
this workspace as the initial execution substrate. The imported revision is
`05df6d9cf8233b2438a7a584ce4ed7a0666abf53`; its current manifest schema is v5.

`kern` already provides the parts that should not distinguish this project:

- a typed and verified manifest;
- opaque cubin modules with content hashes;
- checkpoint tensor binding and load-time transformations;
- fixed, per-token, and per-sequence state allocation;
- CUDA Graph capture/replay;
- VMM-backed pooled state and peer mappings;
- multi-program composition for prefill, decode, and speculative rounds;
- differential kernel/program testing;
- a small CLI and optional serving frontend.

It already has Qwen3.8 target and DFlash2 manifests. Its recorded comparison is
about 81 versus 95 tokens/s for target decode and 178 versus 176 tokens/s for
speculative decode against its pinned vLLM setup. Those results establish a
usable substrate, not this compiler's performance claim.

Consequently, Qwen3.8 begins as a compiler-equivalence and optimization target,
not a fresh runtime bring-up. `kern`'s substantial DeepSeek-V4.1 generators and
kernel packages are an executable implementation oracle. GLM-5.3 is the main
missing model target after the Qwen compiler path is proven.

The missing layer, and the purpose of OrbitKV Next, is compilation:

```text
model/state semantics
        -> typed stateful task graph
        -> legal island and schedule alternatives
        -> generated or selected kernel package
        -> kern manifest v5
        -> kern verify / test / run / serve
```

The compiler consumes the local `kern-manifest` crate and emits artifacts
executed by the local `kern-runtime` and `kern-serve` crates. Source ownership
does not erase the abstraction boundary: runtime extensions are allowed only
when a measured serving or state-management requirement cannot be represented
as an opaque op, state, buffer, program, peer mapping, or program variable.
Compiler-only state effects remain in the higher-level IR and may disappear
after verification and lowering. Changes generally useful to `kern` should be
kept upstreamable.

## Pinned model contracts

### Qwen3.8-27B-FP8

- Repository: `Qwen/Qwen3.8-27B-FP8`.
- Revision: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- Text tower: 64 layers, hidden width 5,120, dense FFN width 17,408.
- Layer schedule: 16 repetitions of three Gated DeltaNet layers followed by
  one Full Attention layer.
- GDN: 16 key heads, 48 value heads, key/value width 128, convolution width 4,
  FP32 recurrent state.
- Full Attention: 24 query heads, four KV heads, head width 256, gated output.
- Quantization: dynamic E4M3 block FP8 weights with 128 by 128 scale blocks;
  BF16 activations.
- Native speculative surface: one MTP layer.

This is the first complete target because the released weights fit one H20 and
because its repeated `GDN, GDN, GDN, Full` structure creates a useful island
boundary. It proves stateful dense decode, not MoE or distributed execution.

### GLM-5.3-Flash

- Repository: `zai-org/GLM-5.3-Flash`.
- Revision: `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`.
- Text tower: 45 layers, hidden width 4,096.
- Layer schedule: 34 KDA layers and 11 sparse-attention layers in a repeating
  three-to-one pattern.
- Attention state: KDA recurrent/convolution state plus compressed sparse MLA;
  the indexer selects up to 2,048 entries and compresses its key pool by four.
- Feed-forward: first three layers are dense; the remaining 42 are MoE with 288
  routed experts, top eight, and one shared expert.
- Residual stream: mHC with multiplicity four.
- Quantization: dynamic E4M3 block FP8 with 128 by 128 scale blocks.
- Native speculative surface: one next-token prediction layer.
- Indexed checkpoint payload: 328,326,771,576 bytes, about 305.78 GiB.

This is the second target because it composes every major engine concern without
the additional CED and Engram semantics of DeepSeek-V4.1. It cannot be a
single-H20 resident target. Initial work on H20 is restricted to layer kernels,
pruned fixtures, and compiler validation; complete inference requires an
explicit multi-GPU or bounded weight-residency plan.

### DeepSeek-V4.1-Flash

- Repository: `deepseek-ai/DeepSeek-V4.1-Flash`.
- Revision: `dba1be0a40aa45a94ad051997016db3960a90277`.
- Text tower: 40 layers arranged as a 20-layer causal encoder followed by a
  20-layer decoder; maximum context is one million tokens.
- CSA2 state: Full, Reindex, and Reuse modes share main KV, indexer K, and
  top-k results across layers with static compression ratios.
- Sparse attention: top 512 over a candidate pool of up to 2,048 blocks; the
  main KV and index sources are cross-layer state producers.
- Feed-forward: 384 routed experts, top six, one shared expert, with 2,304
  intermediate width.
- Engram: conditional lookup state at layers 1 and 14 over two approximately
  384-million-entry tables.
- Quantization: dynamic FP8, 32 by 32 block scales in UE8M0, and FP4 expert
  weights.
- Native speculative surface: three DSpark layers with block size five.
- Indexed checkpoint payload: 510,286,023,000 bytes, about 475.24 GiB.

This is the third target and the strongest test of the IR. It requires explicit
cross-layer state sharing, immutable lookup tables, sparse candidates, MoE, and
speculative execution. Its native FP4 fast paths are principally a Blackwell
problem. H20 may run reference or converted paths, but must not be the primary
performance target for an SM100-specific implementation.

## Capability ladder

The models define a strict implementation order:

| Stage | Model | New capability proved | Infrastructure added |
| --- | --- | --- | --- |
| A | Qwen3.8 | Automatic lowering of Full Attention plus mutable GDN/conv state | Qwen model frontend, FP8 schedule selection, stateful-island generation |
| B | GLM-5.3 | KDA plus sparse MLA plus MoE plus mHC | New task/effect kinds, grouped-GEMM selection, expert placement, collectives, sparse-index carry |
| C | DeepSeek-V4.1 | Cross-layer compressed state plus Engram plus FP4 plus DSpark | Producer/consumer state lowering, immutable lookup placement, versioned speculative commit |

Every stage targets the same shared execution substrate and reuses the previous
artifact lowering, memory-planning, task-dependency, KV/state lifecycle, and
correctness machinery. A model may add task kinds and kernel families; it must
not add model-specific launch or allocation machinery.

## Ownership architecture

```text
checkpoint config + tensor index
              |
              v
       model package
  topology / weights / numerics
              |
              v
    typed static task graph
 tensors / state effects / collectives
              |
              v
       plan compiler
 partition / memory / schedule / lowering
              |
              v
       kern manifest v5
 verified calls / buffers / state / topology
              |
              v
        kern runtime
 CUDA Graph / VMM pools / peer mappings
              |
      +-------+--------+
      |                |
      v                v
 generated islands   provider kernels
```

### Execution substrate

`kern` owns mechanisms that are stable across the three model families:

- CUDA context, streams, events, modules, and error handling;
- typed device allocations and non-owning views;
- memory-mapped safetensors and bounded staged upload;
- fixed-capacity scratch and persistent arenas;
- CUDA Graph capture/replay and optional persistent-island launch;
- peer-address topology and opaque communication-kernel launch boundaries;
- batch-bucket dispatch;
- deterministic output collection and benchmark instrumentation.

It does not understand Qwen, GLM, DeepSeek, GDN, KDA, MLA, or MoE. OrbitKV Next
must preserve that boundary. NCCL, DeepEP, or a custom peer collective is a
kernel/provider decision represented by the generated program, not a second
runtime abstraction.

### Model package

Each model package owns facts that genuinely differ:

- exact checkpoint identity and configuration validation;
- tensor names, packing, sharding, and load-time transformations;
- ordered layer topology;
- numerical rounding and accumulation contracts;
- state schemas and cross-layer producer/consumer relationships;
- supported prefill, decode, and speculative modes;
- model-specific golden fixtures.

The model package constructs a task graph. It does not launch CUDA directly.
This is stricter than PegaInfer's current per-model execution ownership and is
the primary mechanism preventing scheduler/runtime duplication between models.

### Task graph

The graph is static and effect-aware rather than a general eager tensor graph.
Every task records:

```text
TaskId
Phase                 prefill | decode | verify
Inputs / Outputs      shape, dtype, stride, alignment, storage
StateEffects          read, tentative-write, commit, append, lookup
Dependencies          data, completion, collective, cross-layer state
Implementations       provider call or generated schedule family
Resources             scratch, registers, shared memory, resident allocations
DynamicBounds         batch, query rows, context pages, experts per rank
```

Initial task families are intentionally finite:

- dense and block-scaled projection;
- grouped/masked expert projection;
- Full or latent paged attention;
- sparse index and sparse MLA attention;
- GDN and KDA recurrent transitions;
- convolution history transition;
- mHC/residual-stream transform;
- Engram lookup;
- collective and peer-transfer task;
- MTP/DSpark verification and state commit;
- sampling.

### Plan compiler

The compiler is a bounded schedule compiler, not an algebraic superoptimizer. It
performs:

1. Model-plan validation.
2. State-effect and alias validation.
3. Lifetime-based bufferization.
4. Execution-island partitioning.
5. Provider capability matching.
6. Enumeration of a small schedule family.
7. Device measurement on declared buckets.
8. Artifact emission and strict replay.

The schedule space contains decisions with direct hardware meaning: tile shape,
CTA count, pipeline depth, SM partition, materialization, stream assignment,
fusion boundary, and provider versus generated implementation. It does not
contain arbitrary equivalent expression trees.

### Execution islands

An island is the largest useful unit whose dependencies and effects can be
scheduled together. It need not be one CUDA kernel. Its implementations may be:

- a CUDA Graph over provider and generated kernels;
- one stateful superkernel;
- a persistent SM-level task graph;
- a communication-aware collective pipeline.

The initial island boundaries follow real model structure:

- Qwen3.8: one GDN layer, then the repeated three-GDN region; Full Attention
  stays a provider boundary.
- GLM-5.3: one KDA layer; one sparse-MLA/indexer region; one MoE region with
  shared-expert overlap.
- DeepSeek-V4.1: one CSA2 producer/reuse group; one Engram lookup/merge region;
  one FP8xFP4 MoE region; one DSpark verification region.

## Compute, storage, and transfer planes

The three planes share one plan but remain separate mechanisms.

### Compute plane

- AOT-generated stateful and epilogue kernels.
- Mature GEMM/attention providers behind narrow C ABIs.
- CUDA Graph fallback.
- Persistent device task scheduling only after a fused island wins.
- Later, fixed collective schedules for expert and context parallelism.

### Storage plane

- Immutable weights, with explicit resident or staged status.
- Per-request recurrent/convolution state.
- Token- or compressed-page KV.
- Cross-layer shared KV/index state.
- Immutable Engram tables.
- Scratch and graph-private allocations.

Every allocation has a scope and lifetime. The first implementation uses static
budgets. Adaptive eviction or global search is deferred until a measured model
requires it.

### Transfer plane

The Qwen stage has only host-to-device initialization and optional local
multi-stream dependencies. The GLM stage adds NCCL/DeepEP collectives and, only
if required for the selected hardware, bounded host-to-device expert staging.
The DeepSeek stage may add peer/fabric access and cross-rank state movement.

No network KV store is part of the initial engine. Communication enters the task
graph as explicit tasks with buffer, stream, and completion dependencies; it is
never hidden inside a model callback.

## Reuse decision matrix

`Direct` means reuse a maintained library or copy a narrow Apache/MIT/BSD source
with its license and exact revision. `Adapt` means reuse the algorithm, ABI, or
code shape only after reconciling model geometry and numerical semantics.
`Reference` means use it as an oracle or baseline, not production code.

| Source | Candidate | Decision | Reason |
| --- | --- | --- | --- |
| `kern` | Manifest v5, verifier, runtime, test harness, pool, CLI/serve | Source fork | Imported at an exact revision so the complete engine can evolve in one workspace |
| `kern` | Qwen3.8 and DFlash2 manifests/kernel packages | Reference baseline | Proves runtime expressiveness and supplies a parity/performance target for compiler output |
| `kern` | DeepSeek-V4.1 generators, kernels, manifests, and EP4 evidence | Reference/Adapt | Rich implementation oracle; replace hand-authored generation with compiler lowering rather than copying it |
| Current OrbitKV | Qwen3.8 config and FP8 weight mapping | Direct extract | Already matches the pinned checkpoint and current H20 path |
| Current OrbitKV | DeepGEMM block-FP8 adapter | Direct extract | Correct 128 by 128 E4M3 contract and existing H20 validation |
| Current OrbitKV | FlashInfer/FA3 paged-attention adapters | Adapt | Preserve only the narrow algorithms used by the target; remove generic provider machinery |
| Current OrbitKV | state aliases, layer probes, logit oracle, matched benchmark tools | Direct extract | These are correctness infrastructure, not product breadth |
| Current OrbitKV | general tensor frontend, e-graph, genetic search | Do not reuse | Large search surface without demonstrated end-to-end value |
| Current OrbitKV | KV manager, external tiers, HTTP engine, CUDA runtime | Do not reuse initially | `kern` already owns the required substrate |
| PegaInfer | `DeviceContext`, typed buffers, CUDA Graph lifecycle | Reference | These mechanisms already exist behind `kern`; do not introduce a second implementation |
| PegaInfer | mmap/staged safetensors loader | Reference | Use its ownership lessons only; `kern` already binds original checkpoints |
| PegaInfer | fixed decode buckets and pointer-stable state slots | Reference | The shape is useful and already represented by `kern` programs/states |
| PegaInfer | Qwen3.5 GDN/conv CUDA kernels | Direct baseline | Qwen3.8 has compatible 128-wide GDN and convolution width four; benchmark before promotion |
| PegaInfer | Qwen3.5 model crate and scheduler | Reference | BF16-oriented and model-owned; copying it would also copy policy and feature debt |
| PegaInfer | GLM5.2 whole-step layout, sparse MLA, indexer, MoE overlap | Adapt | Strong structural template, but GLM-5.3 changes depth, hidden width, expert count, KDA and NoPE contracts |
| PegaInfer | K3 FlashKDA and FP8xFP4/MegaMoE work | Adapt | Useful kernel/communication patterns; K3 shapes differ and MegaMoE is SM100-specific |
| vLLM | Qwen3.5, GLM5Next, and DeepSeek-V4.1 model semantics | Reference | Most current executable specification and weight-loading oracle |
| vLLM | model-specific fused kernels with compatible license/ABI | Adapt | Use only after exact dtype, layout, SM, and numerical checks |
| SGLang | FlashKDA/GDN, sparse attention, mHC, MoE implementation choices | Reference/Adapt | Strong alternative implementation and benchmark oracle |
| FlashInfer | paged attention, sampling, selected GEMM/MLA paths | Direct dependency or narrow wrapper | Maintained optimized provider surface |
| FlashAttention/FlashMLA | attention implementations | Direct dependency or narrow wrapper | Do not rewrite mature attention unless the measured shape is deficient |
| DeepGEMM | dense/grouped low-precision GEMM | Direct dependency or pinned AOT instantiation | Central to FP8/FP4 model execution; preserve provider source identity |
| FlashKDA/FLA | KDA/GDN prefill and recurrent references | Direct dependency or adapted AOT kernel | Prefer upstream optimized recurrence over a generic decomposition |
| DeepEP/NCCL | expert communication | Direct dependency | Do not build a new collective transport |

PegaInfer revision `72cbbe8a72e06329b2b4d6fa1e8e906acf2acc85` was used
for this inventory. Its root code is Apache-2.0, but every copied or vendored
kernel must retain its own upstream attribution and license. Technical reuse is
per-contract: an SM100 cubin, a GLM5.2 geometry specialization, or a Kimi-specific
weight pack is not reusable merely because its wrapper compiles.

## Model-specific implementation plan

### Qwen3.8

Reuse the current exact FP8 loader and provider contracts. Use PegaInfer's
fixed-buffer/CUDA-Graph organization and GDN kernels as a simpler independent
baseline. Prefer the fastest qualified FLA/FlashQLA implementation where it
matches the exact numerical boundary. Optimize in this order:

1. Correct provider-composed decode.
2. Remove GDN Gather/Scatter and fuse convolution, recurrence, norm, gate, and
   state commit.
3. Share activation quantization and coordinate QKV/Z/A/B low-M projections.
4. Compile one- and three-GDN-layer persistent islands.
5. Add native MTP with exact state commit/rollback.

### GLM-5.3

Use current vLLM and SGLang `Glm5Next` implementations as the semantic oracle.
Use PegaInfer GLM5.2/K3 only as an implementation inventory. Build in this order:

1. KDA layer oracle and SM90 FlashKDA path.
2. Sparse MLA/indexer path with explicit shared top-k carry.
3. mHC residual stream.
4. Dense first-three-layer path.
5. Router plus grouped FP8 expert execution.
6. EP communication and complete-model residency.
7. Native MTP.

Do not reuse the GLM5.2 fixed constants or binaries. GLM-5.3 changes from 78 to
45 layers, 6,144 to 4,096 hidden width, 256 to 288 routed experts, and introduces
the new KDA/NoPE contract.

### DeepSeek-V4.1

Use current vLLM/SGLang implementations and the official minimal inference as
semantic references. Build no complete model until the target Blackwell or
equivalent multi-GPU environment is available. Implement in this order:

1. CSA2 cache record, compression modes, and producer/reuse dependency graph.
2. Sparse indexer and bounded candidate hierarchy.
3. FP4 KV and FP8xFP4 expert data contracts.
4. Engram lookup and residency.
5. Single-pass mHC.
6. Grouped MoE and communication.
7. Three-layer DSpark verification with versioned state commit.

PegaInfer K3's FP8xFP4 and MegaMoE path is a valuable design reference, but its
SM100 implementation, expert geometry, routing, and model topology are not a
drop-in DeepSeek-V4.1 implementation.

## Repository shape

Keep one compiler package above the imported, separately owned execution crates:

```text
src/
  ir/
  compiler/
  lower/kern.rs
  model/qwen38/
kernels/
  shared/
  qwen38/
tests/
tools/
crates/
  kern-manifest/
  kern-pool/
  kern-runtime/
  kern-test/
  kern-run/
  kern-serve/
```

After Qwen succeeds, add `model/glm53` and then `model/deepseek_v41`. Split a
kernel crate only when build dependencies or linkage make the split necessary.
Do not create one crate per conceptual noun. Modify an execution crate only when
the manifest boundary cannot express a measured requirement; do not move model
semantics or compiler policy into the runtime.

## Acceptance invariants

- The same shared `kern` substrate executes all admitted models.
- A model package cannot allocate device memory or launch a kernel directly.
- A kernel cannot mutate persistent state without a declared effect.
- Every dynamic dimension is bounded by an artifact bucket.
- Every provider and generated image is content-addressed in the artifact.
- Prefill, target decode, and speculative verify have distinct numerical gates.
- A faster kernel is rejected when it changes the model's declared rounding or
  state-transition semantics.
- Full-model performance, not source elegance or microbenchmarks, decides whether
  an optimization remains.

## What would make this architecture fail

- Qwen requires model-specific launch code outside its model package.
- GLM requires a private runtime path or cannot reuse task effects, artifact
  lowering, or the memory planner.
- DeepSeek cross-layer state requires bypassing the task dependency model.
- Provider integration dominates the codebase again.
- The compiler accumulates general tensor semantics unrelated to these models.
- Performance requires changing numerical behavior without an explicit contract.
- Complete-model gains disappear when measured through a maintained serving
  baseline.
