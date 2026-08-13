# TileLang engine architecture

**Decision:** accepted target on 2026-08-14.

Oxide Infer will become a Rust-first LLM inference engine. It will reuse the
Mistral.rs fork as its control-plane shell, own model execution and GPU
resource policy in Oxide crates, and use TileLang as the only source language
for product custom compute kernels.

This is a target architecture, not a statement about the current tree. The
current source is still a checked operator runtime with cuda-oxide and
cuBLASLt paths. Migration phases must remove those paths before the engine can
claim the target state.

## Product position

Oxide Infer is not a Rust wrapper around another serving engine and not a
general kernel DSL. Its intended position is:

- one deployable Rust inference engine and OpenAI-compatible server;
- Mistral.rs control-plane features without Candle CUDA execution in the
  supported fast path;
- Oxide-owned model plans, KV paging, memory, streams, Graphs, and sampling;
- ahead-of-time TileLang kernels loaded by a checked Rust CUDA runtime;
- evidence at both the operator and complete-serving levels.

The first product profile is deliberately narrow: one dense decoder-only
model family, BF16, one NVIDIA GPU, paged KV cache, continuous batching, and
greedy sampling. Quantization, MoE, multimodal models, speculative decoding,
and multi-GPU execution enter only after this profile is complete.

## Current and target boundaries

| Concern | Current source | Accepted target |
| --- | --- | --- |
| API, tokenizer, scheduler | External Mistral.rs fork | Embedded Mistral.rs shell |
| Model forward | Candle model implementations | Oxide execution plan |
| GPU tensors and weights | Candle storage | Oxide allocations and weight loader |
| KV allocation and paging | Consumer engine | Oxide engine runtime |
| Custom compute kernels | cuda-oxide | TileLang only |
| General dense GEMM | cuBLASLt | TileLang GEMM |
| Kernel compilation | Rust device build | Offline TileLang artifact build |
| Runtime process | Rust plus native/vendor providers | Rust plus CUDA driver and prebuilt artifacts |
| Engine evidence | Narrow decode adapter | Full model and serving workloads |

CUDA Driver APIs, CUDA Graph APIs, and NCCL are infrastructure interfaces, not
compute-kernel providers. Their use does not violate the TileLang-only kernel
rule. Product GPU computation does violate the rule if it executes through
FlashInfer, cuBLASLt, CUTLASS, Triton, handwritten CUDA, cuda-oxide, or Candle
CUDA kernels.

## System boundary

```text
OpenAI-compatible clients
          |
          v
Mistral.rs control plane
  HTTP, tokenizer, chat templates, request lifecycle,
  model registry, continuous-batch scheduling
          |
          | EngineBatch / EngineResult
          v
OxidePipeline
  the only Mistral-to-Oxide execution boundary
          |
          v
Oxide engine data plane
  weight loader -> model IR -> execution plan
  KV pager -> batch metadata -> executor -> sampler
  memory -> streams -> CUDA Graphs -> completions
          |
          v
Tile artifact registry
  contract + algorithm + ABI + target + artifact hash
          |
          v
prebuilt TileLang cubin/PTX -> CUDA Driver -> NVIDIA GPU
```

Mistral.rs currently exposes `Pipeline::forward_inputs` and
`NormalModel::forward`. The migration adds an `OxidePipeline` at the pipeline
boundary. A supported model's entire forward enters an Oxide-owned execution
plan there. It does not wrap every Candle layer or replace Candle operations
one call at a time.

This boundary keeps the fork patch small and makes provider coverage
auditable. If an admitted model cannot build a complete Oxide plan, startup
fails with a typed unsupported-model or unsupported-shape error. The product
profile never silently runs part of the model through Candle CUDA.

## Control-plane ownership

The embedded Mistral.rs fork keeps:

- OpenAI-compatible routes, streaming responses, and request validation;
- tokenizer, chat-template, and Hugging Face metadata handling;
- request lifecycle, cancellation, and continuous-batch scheduling;
- model selection and user-facing configuration;
- response assembly, telemetry hooks, and server process management.

The shell submits engine-neutral batches. It does not own GPU tensor classes,
kernel dispatch, model-layer execution, or product KV pages.

The first adapter should be one new pipeline implementation plus the minimum
loader and CLI registration needed to select it. Changes scattered through
Mistral.rs model files are an architecture failure because they expand the
fork diff and permit mixed Candle/Oxide execution.

## Oxide data-plane ownership

The Oxide engine owns:

- direct safetensors loading into Oxide-managed host and device regions;
- a small model IR and immutable prefill and decode execution plans;
- tensor layout, workspace, stream, event, and completion lifetimes;
- paged KV allocation, sharing, copy-on-write, eviction, and block tables;
- fused layer and model segments rather than eager operator-by-operator
  dispatch;
- logits processing, sampling state, and token outputs;
- CUDA Graph capture for admitted fixed-address decode plans;
- artifact selection, compatibility checks, and provider-hit telemetry.

The model IR is not a general framework. It represents only the admitted
model profiles and operations. Narrow IR coverage is a maintenance feature:
an unsupported architecture is rejected instead of partially interpreted.

## TileLang kernel rule

TileLang is a build-time kernel factory, not a runtime dependency.

```text
kernels/tilelang/src/*.py
        |
        v
pinned compiler + deterministic build profiles
        |
        v
cubin/PTX + artifact manifest + correctness report
        |
        v
release package
        |
        v
Rust artifact registry -> checked launch
```

Production servers must not import Python, compile a kernel, auto-tune, access
the network, or mutate the artifact registry. Development builds may tune
candidate schedules, but promotion converts the selected schedule into an
immutable artifact.

Each artifact manifest records at least:

- operator contract version and algorithm identity;
- TileLang source and compiler versions;
- CUDA toolkit, target compute capability, and required driver version;
- dtype, layout, shape domain, numerical limits, and determinism policy;
- parameter ABI, alignment, alias rules, launch geometry, and workspace;
- cubin or PTX SHA-256 and source-tree identity;
- correctness, sanitizer, Graph, and benchmark record identities.

The Rust runtime verifies the manifest, artifact hash, target device, ABI, and
shape domain before load. Enqueue cannot tune, switch algorithms, or fall back.

## Repository target

The Mistral.rs fork will be imported with `git subtree` under
`engine/mistralrs`. It remains a separate Cargo workspace excluded from the
root workspace, and CI builds it with an explicit manifest path. A subtree
gives users one clone and one release source tree while preserving a
repeatable upstream-sync command and an auditable fork patch.

```text
oxide-infer/
|-- crates/
|   |-- oxide-infer/          contracts and CPU references
|   |-- oxide-infer-cuda/     CUDA resources and artifact launcher
|   |-- oxide-infer-engine/   model IR, planner, KV pager, executor, sampling
|   `-- oxide-infer-lab/      correctness and benchmark programs
|-- engine/
|   `-- mistralrs/            subtree fork and OxidePipeline shell wiring
|-- kernels/
|   `-- tilelang/
|       |-- src/              kernel definitions and fixed schedules
|       |-- build/            compiler and manifest generator
|       `-- profiles/         admitted target and shape matrices
|-- benchmarks/
|   |-- kernels/              matched FlashInfer comparisons
|   `-- serving/              neutral OpenAI-compatible workload driver
|-- docs/results/             immutable evidence and artifact manifests
`-- tools/                    development-only analysis utilities
```

`oxide-infer-engine` is a justified fourth product crate: it has model and KV
state, a different dependency boundary, and a release surface that the
backend-independent contract crate must not acquire.

Subtree updates are reviewed separately from Oxide feature work. The sync
commit contains only the upstream import; a following commit reapplies or
updates the small shell adapter. A growing cross-tree patch is a stop signal.

## Supported-path policy

The server exposes two distinct build profiles:

- `oxide-production`: admitted models only, complete Oxide execution,
  TileLang artifacts only, and fail-closed startup;
- `mistral-reference`: test and comparison profile used to validate output
  behavior, never packaged as the product fast path.

There is no automatic runtime fallback between them. Benchmark records name
the selected profile and include provider-hit counters. A full request must
report zero Candle CUDA, FlashInfer, cuBLASLt, cuda-oxide, and unregistered
kernel hits before it can count as Oxide engine evidence.

## Maintenance consequences

This architecture reduces maintenance in four places:

- one kernel source language instead of Rust device code, C++/CUDA, and vendor
  provider wrappers;
- one model execution boundary instead of modifications across Candle layers;
- offline immutable artifacts instead of production JIT and tuning failures;
- one repository, pinned shell source, and reproducible engine evidence.

It does not make kernel work free. Oxide becomes responsible for every
admitted GEMM, attention, KV, normalization, activation, and sampling
algorithm and for their target-specific tuning. Maintenance remains tractable
only while model, dtype, layout, and hardware coverage stay explicit and
fail-closed.

## Initial implementation slice

The first complete slice is BF16 Qwen2.5-1.5B on one GPU because the repository
already has Mistral.rs traces and paged-attention evidence for that model
family. The slice includes:

1. safetensors loading and embeddings;
2. RMSNorm, RoPE, dense GEMM, SwiGLU, and residual operations;
3. paged prefill, paged decode, and KV append;
4. final norm, logits projection, and greedy token selection;
5. continuous batching through `OxidePipeline`;
6. one OpenAI-compatible streaming endpoint.

Completion requires exact greedy-token agreement with the reference profile,
complete provider-hit accounting, no device copies at the shell boundary, and
the benchmark gates in the engine evidence plan.

## Non-goals for the first release

- preserving all Mistral.rs model and modality coverage;
- exposing TileLang or Python as a user runtime API;
- accepting arbitrary eager graphs;
- claiming every TileLang kernel beats every FlashInfer kernel;
- claiming engine leadership from a single operator or single concurrency;
- multi-GPU, quantized, MoE, speculative, diffusion, vision, or audio serving.
