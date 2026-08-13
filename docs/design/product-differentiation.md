# Product differentiation and peer boundaries

**Decision date:** 2026-08-14 (Asia/Shanghai). This document describes intended
position, not current performance leadership.

Oxide Infer is a narrow, NVIDIA-focused Rust inference engine whose custom GPU
computation is supplied exclusively as qualified ahead-of-time TileLang
artifacts. Its distinguishing unit is not a new HTTP wrapper or a Rust rewrite
of every inference feature. It is the complete, auditable path from a model
contract through a typed execution plan to one immutable kernel artifact and
reproducible evidence.

## One-sentence position

> Oxide Infer is the fail-closed TileLang AOT inference engine: every supported
> GPU operation resolves to a manifest-verified artifact, or the model does not
> load.

This position is only valuable if the final engine is competitive. Artifact
discipline is not a substitute for kernel latency, scheduler goodput, memory
efficiency, correctness, or model coverage.

## Peer boundaries

| Project | Primary strength and scope | Oxide relationship | Oxide difference |
| --- | --- | --- | --- |
| Mistral.rs | Broad Rust inference product: many model, quantization, modality, hardware, SDK, and serving capabilities | Pinned behavioral oracle and source baseline for admitted non-compute modules | Oxide deliberately supports fewer profiles, owns a framework-free model/KV data plane, and optimizes one NVIDIA AOT path deeply |
| PegaInfer | Production-oriented pure Rust and CUDA engine assembled from reusable frontend, KV, cache, and kernel components | Closest Rust-native engine peer and an engine benchmark candidate when a model profile overlaps | Oxide uses one custom-kernel supply chain and a shared narrow model IR; PegaInfer currently composes multiple kernel technologies and model-specific crates |
| PegaFlow | Standalone Rust external KV cache service with host memory, SSD, RDMA, and cross-engine sharing | Future optional external KV backend, not an engine competitor | Oxide owns inference, local scheduling, model execution, and HBM page use; it does not initially reproduce PegaFlow's distributed storage plane |
| FlashInfer | High-performance GPU operator library | Matched kernel correctness and performance baseline | Oxide owns a complete engine and artifact lifecycle; it does not link FlashInfer as a production fallback |
| vLLM and SGLang | Mature high-throughput serving engines | Primary neutral serving baselines | Oxide targets a smaller Rust-native, AOT-only production footprint and must prove that trade against their goodput and latency |

The statements above are pinned to public project state on the decision date:

- [Mistral.rs repository](https://github.com/EricLBuehler/mistral.rs)
- [PegaInfer repository](https://github.com/pegainfer-project/pegainfer)
- [PegaInfer 0.1 architecture and measurements](https://openedinfer.com/blog/pegainfer-010/)
- [PegaFlow repository](https://github.com/novitalabs/pegaflow)
- [vLLM and PegaFlow architecture](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-05-18-pegaflow.md)

## Why this is not Mistral.rs with another kernel backend

Oxide may transplant complete implementations for protocol types, HTTP and
SSE handling, tokenizer and chat-template behavior, request state, tools,
grammar constraints, telemetry, and process lifecycle. It preserves the
source license and provenance and tests behavior against the external source
engine.

The boundary stops before framework tensors and GPU state. Model forward
implementations, device mapping, framework storage, physical KV cache,
attention implementations, quantized layers, and GPU-provider routing belong
to the data plane even when their filenames do not say `kernel`. Copying those
paths would leave the product dependent on the upstream execution architecture
and make `OxideTile` only another optional backend.

Oxide instead owns:

```text
ModelSpec -> narrow model IR -> immutable prefill/decode plans
          -> Oxide KV handles -> checked artifact requests
          -> verified TileLang bytes -> CommandScope -> Completion
```

Mistral.rs therefore supplies mature behavior and source experience, not the
runtime identity of the product.

## Difference from PegaInfer

PegaInfer demonstrates that a useful Rust engine should reuse stable
components. Its public architecture directly uses the vLLM Rust frontend,
Dynamo logical KV management, PegaFlow physical offload, and a heterogeneous
kernel portfolio: handwritten CUDA and cuBLAS, FlashInfer, Triton AOT, CuTe
DSL, and TileLang. Model-specific crates own the relevant execution and kernel
feature sets.

Oxide adopts the component lesson but chooses a different optimization:

- one manifest and launch ABI for every custom compute family;
- one build-time source language and qualification path;
- one checked Rust resource lifecycle from operands to completion;
- one shared narrow model IR, with model profiles rather than a new execution
  architecture per model;
- model load fails when exact artifact coverage is incomplete;
- kernel evidence and engine evidence ship as first-class release records.

This reduces provider interaction, runtime compilation, and artifact ambiguity.
It also creates a real cost: TileLang must reach competitive performance for
GEMM, attention, sampling, fusion, and future MoE shapes without falling back
to a stronger library. PegaInfer is currently much broader and closer to
production; Oxide has not yet earned an engine-level comparison claim.

## Difference from PegaFlow

PegaFlow treats KV cache as a long-lived service asset independent of an
engine process. Its daemon owns host pools, SSD, topology, RDMA, indexing, and
background work. That is a clean failure and lifecycle boundary for multi-node
serving, but it is not the scheduler, model executor, or GPU kernel engine.

Oxide first owns an in-process HBM KV pager because model plans, block tables,
CUDA Graph addresses, and attention artifacts must be proven together. A
future `ExternalKvBackend` boundary may export and import immutable logical KV
blocks. PegaFlow can implement that boundary without becoming part of the
kernel or scheduler dependency graph.

Oxide will not build an SSD/RDMA cache service merely to claim feature parity.
That work is admitted only if a measured deployment cannot use PegaFlow or
another stable external service through the boundary.

## Target architecture and code layout

```text
apps/oxide-infer
  process assembly and CLI
        |
crates/oxide-infer-server
  source-derived API, tokenizer, streaming, request state
        |
crates/oxide-infer-engine
  model IR, weights, scheduler, local KV pager, sampling, plans
        |
crates/oxide-infer-cuda        crates/oxide-infer
  checked resources,           semantic contracts,
  artifact registry, launch    CPU references
        |
kernels/tilelang
  offline sources, schedules, profiles, build and qualification
        |
artifacts/manifests + external immutable cubin/PTX

crates/oxide-infer-lab and benchmarks/
  correctness, FlashInfer kernel comparison, neutral engine comparison

future optional boundary:
oxide-infer-engine <-> external KV connector <-> PegaFlow or another service
```

Target repository ownership is:

| Path | Ownership |
| --- | --- |
| `apps/oxide-infer` | Release binary and process composition |
| `crates/oxide-infer-server` | Protocol, tokenizer, streaming, configuration, telemetry |
| `crates/oxide-infer-engine` | Model IR, weights, sequence scheduler, KV policy, execution plans, sampling |
| `crates/oxide-infer-cuda` | Memory, streams, modules, Graphs, manifest verification, checked launch and completion |
| `crates/oxide-infer` | Backend-independent operator contracts and references |
| `kernels/tilelang` | Offline kernel definitions, fixed schedules, target profiles, artifact build |
| `artifacts/manifests` | Release-visible artifact identities; generated binaries remain external |
| `crates/oxide-infer-lab` | Non-product device gates and evidence generation |
| `benchmarks/kernels` | Matched FlashInfer comparisons |
| `benchmarks/engines` | Neutral Mistral.rs, PegaInfer, vLLM, and SGLang drivers where profiles overlap |
| `docs/provenance` | Source-to-source mapping, upstream revisions, licenses, modifications |

Directories land only with their first working vertical slice. The target tree
is not created as empty scaffolding.

## What must be proven

Oxide's difference is accepted only after evidence answers four separate
questions:

1. **Artifact integrity:** can a release reproduce the source, compiler,
   target, ABI, hash, and exact shape domain of every loaded artifact?
2. **Kernel quality:** on matched inputs, how does each TileLang contract
   compare with FlashInfer or the appropriate diagnostic baseline?
3. **Engine quality:** on the same model and trace, how do TTFT, inter-token
   latency, throughput, goodput, memory, startup, and errors compare with the
   pinned peer engines?
4. **Operational quality:** do cancellation, overload, invalid artifacts,
   Graph misses, and teardown fail without leaks or hidden provider fallback?

Until those gates pass, the accurate claim is “Oxide is implementing a
standalone TileLang AOT Rust inference engine,” not “Oxide is faster” or
“production ready.”
