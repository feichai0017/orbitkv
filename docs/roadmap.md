# Oxide Infer roadmap

Oxide Infer is migrating from a checked Rust GPU operator runtime to a
standalone Rust inference engine. The target product uses Oxide names and owns
its server, scheduling, model execution, KV cache, sampling, GPU runtime, and
release artifacts.

The non-compute control plane starts from a pinned Mistral.rs source baseline.
Reusable implementations are transplanted and renamed into Oxide modules with
license and provenance records. Framework tensor, model-forward, physical KV,
and GPU provider paths are replaced by Oxide. TileLang is the only source
language for product custom compute kernels.

The current source still contains cuda-oxide and cuBLASLt execution paths and
does not yet own a full model or server. Historical evidence remains valid
only for its recorded source and provider.

The [standalone architecture](design/standalone-oxide-engine.md) defines
ownership, the [control-plane source map](design/control-plane-source-map.md)
defines reuse, and the
[engine benchmark plan](development/engine-benchmark-plan.md) defines claims.

## Global rules

### Product identity

Product crates, APIs, features, configuration, telemetry, and binaries use
Oxide names. External engine names occur only in license notices, provenance,
historical records, and benchmark baselines.

### Source reuse

Reuse mature non-compute code instead of rewriting it without benefit. Every
derived module records its source commit, original path, license, local path,
and modifications. Source reuse stops where a module owns or depends on
framework tensors, model forward, physical KV storage, device kernels, or a
non-Oxide provider.

### Supported fast path

An admitted model executes all GPU computation through registered TileLang
artifacts. It cannot silently use Candle CUDA, FlashInfer, cuBLASLt, CUTLASS,
Triton, handwritten CUDA, cuda-oxide, or another framework kernel.

CUDA Driver, CUDA Graph, and NCCL APIs are infrastructure, not custom compute
providers. Every request records provider and artifact hits.

### Artifact gate

Every promoted kernel has:

- one versioned contract and named algorithm;
- pinned TileLang, CUDA, and target identities;
- a fixed ABI, launch geometry, workspace, and shape domain;
- a cubin or PTX hash and generated manifest;
- correctness, sentinel, lifetime, and applicable sanitizer evidence;
- CUDA Graph proof or an explicit exclusion;
- matched performance evidence with balanced provider order.

Runtime enqueue cannot compile, tune, allocate hidden workspace, switch
algorithms, or fall back.

### Model and serving gate

A supported model needs direct Oxide weight loading, immutable prefill and
decode plans, an Oxide KV pager, complete artifact coverage, deterministic
output comparison, error recovery, memory records, and serving evidence.

Release artifacts contain no Python runtime, compiler, tuner, model weights,
generated cache, upstream reference engine, or unregistered kernel path.

## F0: Preserve the operator foundation

**State:** complete historical foundation.

Completed work includes:

- the Oxide Infer rename and checked contract lifecycle;
- memory, stream, command, Graph, and completion ownership;
- current BF16 attention, KV append, RMSNorm, RoPE, and GEMM paths;
- current-source correctness, Graph, and sanitizer qualification;
- matched FlashInfer attention records;
- a stopped native M=1 GEMV experiment;
- a historical external-engine paged-decode proof;
- a fixed-shape TileLang paged-prefill admission spike.

The TileLang spike measured 31.83 microseconds against 47.01 microseconds for
the current Oxide path and 25.90 microseconds for FlashInfer on its fixed
shape. It admits the toolchain for continued work, not a production or
full-engine performance claim.

## E0: Freeze the standalone engine and source boundary

**State:** complete.

Work:

- replace the embedded-shell proposal with a standalone Oxide product;
- define `oxide-infer-server`, `oxide-infer-engine`, `oxide-infer-cuda`, and
  offline TileLang ownership;
- define the pinned upstream source transplant, naming, license, provenance,
  and update policy;
- classify upstream areas as import, split, reference, or exclude;
- keep the external source engine as an independent behavioral and performance
  baseline;
- define FlashInfer, vLLM, and SGLang comparison protocols;
- choose BF16 Qwen2.5-1.5B on one NVIDIA GPU as the first complete profile.

Exit gate:

- architecture, code layout, source map, roadmap, and benchmark plan agree;
- current docs do not claim that the target is implemented;
- no whole upstream tree or branded shell is a planned product dependency;
- implementation can proceed without another ownership decision.

## E1: Build the TileLang artifact boundary

**State:** in progress.

Current source defines strict manifest JSON, independent schema and launch ABI
versions, structural validation, target and driver checks, SHA-256 and size
verification, owned verified bytes, and exact fail-closed registry selection.
CUDA module loading, checked launch, packaging, and device evidence remain
open.

Work:

- move the current spike into `kernels/tilelang` build tooling;
- define a versioned artifact manifest and launch ABI;
- compile fixed schedules into cubin or PTX outside the Rust server;
- validate hash, target, driver, shape, dtype, layout, alignment, and workspace;
- load and launch the first artifact through `oxide-infer-cuda`;
- retain current memory, stream, status, Graph, and completion safety;
- package and load the artifact with no Python installation present.

First slice: BF16 paged prefill for the existing batch-2 long GQA4 contract,
with the same tensors, stream, and timed boundary as the FlashInfer record.

Exit gate:

- one immutable artifact passes correctness, lifecycle, sanitizer, Graph, and
  matched-performance gates;
- incompatible hash, ABI, device, or shape fails before launch;
- the serving dependency graph contains no TileLang or Python package;
- enqueue performs no JIT, tuning, hidden allocation, or fallback.

## E2: Complete one Oxide model data plane

**State:** planned.

Work:

- add `oxide-infer-engine` with model IR, plans, KV pager, and sampling state;
- load safetensors directly into Oxide-owned storage;
- add TileLang artifacts for embedding, RMSNorm, RoPE, dense GEMM, SwiGLU,
  residuals, paged attention, KV append, logits, and greedy sampling;
- derive GEMM coverage from the recorded model shape census;
- build immutable prefill and decode plans;
- reject incomplete artifact coverage at model load;
- add cancellation, invalid-page, out-of-memory, and teardown tests.

Exit gate:

- Qwen2.5-1.5B BF16 produces the same greedy token IDs as the pinned external
  reference over the declared prompt matrix;
- every GPU operation reports a registered `OxideTile` artifact;
- no framework CUDA tensor, old provider, or hidden fallback executes;
- prefill, decode, Graph hit rate, and peak memory are recorded.

## E3: Transplant the non-compute control plane

**State:** planned after the engine request boundary exists.

Work:

- pin one upstream source commit and create `UPSTREAM.md` and
  `THIRD_PARTY_NOTICES.md`;
- import all admitted server, protocol, tokenizer, streaming, configuration,
  request-state, scheduler, constraint, tool, and telemetry modules;
- preserve original copyright and license notices;
- mechanically rename product crates, modules, types, features, environment
  variables, and log fields to Oxide identities;
- split sequence, scheduler, prefix-cache, and constraint code at Oxide-owned
  handles rather than framework tensors;
- exclude upstream model forward, physical cache, paged-attention, quantized,
  custom CUDA, and unsupported modality modules;
- add source-parity tests before changing behavior;
- record the exact source-to-target mapping under `docs/provenance`.

Exit gate:

- `oxide-infer-server` serves the E2 model through an OpenAI-compatible
  streaming endpoint;
- request scheduling uses `EngineRequest`, `EngineBatch`, and `EngineEvent`;
- the product dependency graph contains no upstream engine or Candle GPU
  execution crate;
- removing the external reference checkout does not change the release build;
- protocol and scheduler parity tests pass;
- public product output contains no upstream engine branding.

## E4: Qualify the complete scheduler and server

**State:** planned.

Work:

- connect continuous batches to Oxide prefill and decode plans;
- qualify KV allocation, sharing, copy-on-write, eviction, and compaction;
- qualify mixed prefill/decode batches and dynamic sequence completion;
- add fixed-address decode Graph pools with explicit registered eager misses;
- add admission limits, memory watermarks, backpressure, and overload errors;
- validate streaming, cancellation, malformed requests, and clean teardown;
- expose request, batch, KV, Graph, artifact, and latency telemetry.

Exit gate:

- concurrency sweeps preserve token correctness and isolation;
- cancellation and overload leak no pages, events, graphs, or buffers;
- a Graph miss selects a registered TileLang plan, never another provider;
- p50, p95, p99, throughput, goodput, memory, and errors are recorded.

## E5: Make the first competitive claim

**State:** planned.

Kernel work compares admitted attention, KV, sampling, and applicable GEMM
contracts with pinned FlashInfer using identical inputs, metadata, streams,
outputs, and timed boundaries.

Engine work runs one neutral OpenAI-compatible request trace against Oxide
Infer and pinned Mistral.rs, vLLM, and SGLang baselines. It fixes model and
tokenizer hashes, dtype, KV dtype, context, sampling, caching, speculation,
warmup, arrival schedule, and run order.

Exit gate:

- all correctness and artifact-coverage gates pass;
- no primary p50 latency, peak-memory, or saturated-throughput metric regresses
  by more than 10% against either primary throughput baseline;
- at least one useful metric wins beyond the 5% noise band;
- p99, goodput, and errors do not invalidate the average;
- a clean release artifact reproduces the result.

The first acceptable wording is "competitive standalone Rust inference
engine" for the named profile. Broader leadership claims require broader
models, workloads, and hardware.

## E6: Remove transitional providers

**State:** planned after E2 and E3 evidence.

Work:

- remove cuda-oxide from Cargo manifests, lockfiles, build scripts, and CI;
- remove product cuBLASLt and old native-provider implementations;
- remove the historical paired-adapter feature paths;
- replace the transitional environment guides with the TileLang supply chain;
- retain old JSON evidence unchanged and clearly historical.

Exit gate:

- repository searches find no live product dependency or build command for
  cuda-oxide, cuBLASLt, or an external-engine runtime adapter;
- the clean release builds and serves without those toolchains;
- product GPU computation has exactly one custom source: TileLang.

## E7: Expand from measured demand

**State:** planned after the first competitive release.

Candidate order:

1. larger dense Llama, Qwen, and Mistral model profiles using the same IR;
2. prefix caching, grammar constraints, logprobs, and tool-facing behavior;
3. FP8 or weight-only quantization with explicit quality gates;
4. speculative decoding;
5. MLA and one measured MoE profile with grouped GEMM;
6. tensor parallelism and NCCL lifecycle qualification;
7. additional NVIDIA targets with independent artifacts and evidence;
8. multimodal profiles only after their GPU contracts fit the same boundary.

For control-plane features, consult the pinned upstream source map and port
the complete admitted behavior. Feature breadth does not automatically become
Oxide support merely because the source baseline implements it.

## Evidence rule

Every result states its contract, source, provider, algorithm, artifact,
hardware, timed region, accepted claims, and excluded claims. Reviewed records
remain immutable. A faster kernel never proves a faster model or server, and
source lineage never changes which runtime owns execution.
