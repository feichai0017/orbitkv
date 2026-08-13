# Control-plane source map

This document defines how the standalone Oxide engine derives non-compute
capabilities from the pinned Mistral.rs source baseline without retaining it
as a product dependency or runtime shell.

No source has been imported by this design document. The import phase must add
`UPSTREAM.md`, `THIRD_PARTY_NOTICES.md`, per-module provenance, and original
license notices in the same commit as derived code.

## Import principle

Import functionality by ownership domain, not by filename. Code outside a
kernel directory can still own Candle tensors, KV buffers, device maps, or
model execution and therefore belongs on the replacement side of the
boundary.

Use four classifications:

| Class | Meaning |
| --- | --- |
| Import | Preserve the admitted implementation and tests, then rename its product boundary |
| Split | Preserve control-state logic but replace tensor, KV, or provider fields with Oxide handles |
| Reference | Reimplement the contract in Oxide and compare with the pinned upstream behavior |
| Exclude | Do not include in the first product profile |

## Initial mapping

| Upstream area | Oxide target | Class | Treatment |
| --- | --- | --- | --- |
| server routes and OpenAI request/response types | `oxide-infer-server::api` | Import | Preserve protocol behavior; rename product-specific types and metadata |
| HTTP streaming and response assembly | `oxide-infer-server::streaming` | Import | Preserve SSE ordering, termination, and error semantics |
| CLI and server configuration | `apps/oxide-infer` and `oxide-infer-server::config` | Import | Replace flags and environment variables with Oxide names |
| tokenizer, chat templates, model metadata paths | `oxide-infer-server::text` | Import | Keep generic library dependencies and source notices |
| request, response, cancellation, and finish state | `oxide-infer-engine::request` | Import | Remove server-framework types from engine-owned state |
| sequence lifecycle | `oxide-infer-engine::sequence` | Split | Replace framework tensor/cache references with `KvHandle` and Oxide plan state |
| default and paged scheduling policy | `oxide-infer-engine::scheduler` | Split | Preserve queue policy; replace upstream cache manager and pipeline capabilities |
| prefix hashing and logical cache policy | `oxide-infer-engine::kv::prefix` | Split | Preserve logical policy; move physical pages and copy-on-write into Oxide KV pager |
| grammar, JSON constraint, and token filtering state | `oxide-infer-engine::constraints` | Split | Preserve CPU state; execute admitted logits masks through Oxide sampling artifacts |
| tool-calling and agentic request flow | `oxide-infer-server::tools` | Import | Keep outside the GPU data plane and gate optional dependencies |
| metrics and request logging | `oxide-infer-server::telemetry` | Import | Add Oxide batch, KV, Graph, and artifact metrics |
| Hugging Face acquisition and local path resolution | `oxide-infer-server::repository` | Import | Keep downloads outside timed execution and direct weights to Oxide loader |
| model loaders and model registry | `oxide-infer-engine::model` | Reference | Parse supported configs into the narrow Oxide model IR |
| normal model `forward` implementations | none | Exclude | Replace with immutable Oxide prefill/decode plans |
| Candle tensors, storage, device mapping, and caches | none | Exclude | Replace with Oxide memory, placement, and KV ownership |
| paged-attention, FlashAttention, and custom CUDA crates | none | Exclude | Replace with registered TileLang artifacts |
| quantized layers and device kernels | future Oxide contracts | Exclude | Admit one format only with model-quality and kernel evidence |
| vision, audio, diffusion, and multimodal model code | future profiles | Exclude | Not part of the first dense text profile |
| tensor/pipeline parallel execution | future distributed runtime | Reference | Preserve API concepts only after communicator ownership is defined |

## Product naming

Imported modules are source-derived Oxide code. Product identifiers use:

- `OxideServer` for HTTP and process ownership;
- `OxideEngine` for scheduling and model execution;
- `EngineRequest`, `EngineBatch`, and `EngineEvent` for the control boundary;
- `ModelPlan`, `PrefillPlan`, and `DecodePlan` for model execution;
- `KvPager` and `KvHandle` for cache ownership;
- `OxideTile` for the custom compute provider.

Historical source filenames and external identifiers remain only in
provenance records and immutable evidence.

## Update procedure

An upstream refresh is not a whole-tree merge. For every admitted module:

1. compare the pinned source path with the new upstream revision;
2. classify each change as protocol fix, control-policy change, framework
   coupling, new feature, or refactor;
3. port only behavior relevant to the current Oxide capability matrix;
4. update the provenance source revision and patch notes;
5. run source-parity, Oxide integration, and serving regression tests;
6. benchmark scheduler changes when they alter batch composition or timing.

Security and protocol fixes have a fast path but still require attribution and
tests. Pure upstream refactors are not ported unless they reduce Oxide
maintenance or are required by a selected behavior.

## Admission gate

A derived module enters the product only when:

- its source path, commit, license, and local modifications are recorded;
- no public product identifier exposes the upstream engine name;
- it depends only on allowed Oxide layers and generic third-party libraries;
- tests cover preserved behavior and the new Oxide boundary;
- it cannot reach an unregistered GPU execution path;
- removing the external reference checkout does not break the release build.
