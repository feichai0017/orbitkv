# Oxide Infer roadmap

Oxide Infer is migrating from a checked Rust GPU operator runtime to a complete
Rust inference engine. The accepted target embeds a Mistral.rs control-plane
shell, moves model execution and KV policy into Oxide, and uses TileLang as the
only source language for product custom compute kernels.

The current source still contains cuda-oxide and cuBLASLt execution paths and
does not yet own a full model or server. Historical evidence remains valid only
for its recorded source and provider. It does not transfer to the new engine.

The [target architecture](design/tilelang-engine-architecture.md) defines
ownership. The [engine benchmark plan](development/engine-benchmark-plan.md)
defines how kernel and serving claims are admitted.

## Global rules

Every phase produces one usable vertical slice. Planned names do not justify
empty modules, manifests, or public APIs.

### Supported fast path

An admitted product model must execute all GPU computation through registered
TileLang artifacts. It cannot silently use Candle CUDA, FlashInfer, cuBLASLt,
CUTLASS, Triton, handwritten CUDA, or cuda-oxide.

CUDA Driver APIs, CUDA Graph APIs, and NCCL are infrastructure interfaces, not
compute-kernel providers. Every request records provider and artifact hits so
that the rule is testable.

### Contract gate

Before kernel work starts, record:

- model call site and tensor roles;
- shape, dtype, layout, masking, and post-operations;
- numerical and determinism limits;
- alias, workspace, page-table, and stream ownership;
- an independent CPU or framework reference;
- unsupported cases and typed errors;
- the model-census frequency and expected engine impact.

### Artifact gate

Every promoted kernel has:

- one named contract and algorithm;
- pinned TileLang, CUDA, and target identities;
- a fixed ABI, launch geometry, workspace, and shape domain;
- a cubin or PTX hash and generated manifest;
- correctness, sentinel, lifetime, and applicable sanitizer evidence;
- fixed-address CUDA Graph proof or an explicit exclusion;
- a matched performance record with balanced provider order.

Runtime enqueue cannot compile, tune, allocate hidden workspace, switch
algorithms, or fall back.

### Model gate

A supported model needs:

- direct weight loading into Oxide-owned storage;
- one immutable prefill plan and one immutable decode plan;
- paged KV allocation and mutation owned by Oxide;
- deterministic output comparison with a pinned reference;
- provider-hit coverage for every GPU operation;
- cancellation, invalid-input, out-of-memory, and teardown recovery;
- peak-memory, offline throughput, and serving evidence.

### Release gate

A release is built from pinned kernel artifacts. It contains no Python runtime,
compiler, tuner, model weights, generated caches, or unregistered kernel path.
All current claims link to source-bound evidence from that release candidate.

## F0: Preserve the operator foundation

**State:** complete historical foundation.

Completed work includes:

- the Oxide Infer rename and contract normalization;
- checked resources, command scopes, completion lifetimes, and stream handoff;
- BF16 attention, KV append, RMSNorm, RoPE, and GEMM source paths;
- current-source H20 correctness, Graph, and sanitizer qualification;
- matched FlashInfer attention records;
- a stopped native M=1 GEMV experiment with immutable evidence;
- a Mistral.rs paged-decode proof of concept;
- a fixed-shape TileLang paged-prefill admission spike.

The first TileLang spike measured 31.83 microseconds against 47.01
microseconds for the current Oxide path and 25.90 microseconds for FlashInfer
on its exact fixed shape. It admits the toolchain for further work, not a
production provider or full migration performance claim.

Historical cuda-oxide, cuBLASLt, and adapter results stay readable. They are
baselines and design input, not target-engine qualification.

## E0: Freeze the engine decision

**State:** complete.

Work:

- accept the Mistral.rs shell, Oxide data plane, and TileLang-only kernel rule;
- separate current source facts from target statements;
- freeze the target repository and ownership boundaries;
- define the FlashInfer, vLLM, and SGLang evidence protocol;
- remove obsolete Loom and cuda-oxide installations and global Cargo redirects
  from the development host without deleting historical evidence;
- choose the first complete model profile and non-goals.

Exit gate:

- design, repository layout, benchmark plan, and roadmap agree;
- current docs do not claim the target is implemented;
- the first profile is BF16 Qwen2.5-1.5B on one NVIDIA GPU;
- implementation work can proceed without another provider-boundary decision.

## E1: Build the TileLang artifact boundary

**State:** planned.

This phase establishes the product kernel supply chain before translating the
whole operator catalog.

Work:

- move the current spike into `kernels/tilelang` build tooling;
- define a versioned artifact manifest and parameter ABI;
- compile fixed schedules into cubin or PTX outside the Rust server;
- add hash, target, driver, shape, dtype, layout, and alignment validation;
- load and launch the first artifact through the checked Rust CUDA runtime;
- retain stream, memory, Graph, status, and completion safety from the current
  runtime without retaining cuda-oxide device compilation;
- package an artifact into a clean release and load it with no Python present.

First vertical slice:

- BF16 paged prefill for the existing batch-2 long GQA4 contract;
- same inputs, outputs, stream, and timed interval as the FlashInfer record;
- eager and fixed-address Graph execution;
- typed rejection for incompatible artifacts and unsupported shapes.

Exit gate:

- one immutable artifact passes correctness, sanitizer, lifecycle, Graph, and
  matched performance gates;
- the release loader detects a changed hash, ABI, or target before launch;
- the serving-process dependency graph contains no TileLang or Python package;
- enqueue performs no JIT, tuning, hidden allocation, or fallback.

## E2: Complete one Oxide model data plane

**State:** planned.

Work:

- add `oxide-infer-engine` with model IR, execution plans, KV pager, and
  sampling state;
- load safetensors directly into Oxide-owned storage;
- implement TileLang artifacts for embedding, RMSNorm, RoPE, dense GEMM,
  SwiGLU, residuals, paged attention, KV append, logits, and greedy sampling;
- derive the admitted GEMM matrix from the recorded model shape census;
- build immutable prefill and decode plans;
- record every artifact hit and reject incomplete coverage at startup;
- add cancellation, invalid page, out-of-memory, and clean-teardown tests.

Exit gate:

- Qwen2.5-1.5B BF16 generates the same greedy token IDs as the pinned reference
  over the declared prompt matrix;
- provider telemetry reports only registered TileLang artifacts for GPU
  computation;
- the path uses no Candle CUDA tensor or kernel and no old provider fallback;
- peak memory, prefill throughput, decode throughput, and Graph hit rate are
  recorded;
- all critical kernel rows meet the first-release promotion policy or have a
  measured engine-level admission reason.

## E3: Embed the Mistral.rs shell

**State:** planned.

Work:

- import the pinned fork under `engine/mistralrs` with `git subtree`;
- exclude its Cargo workspace from the Oxide root workspace;
- add one `OxidePipeline` implementation at the full-model forward boundary;
- reuse HTTP routes, streaming, tokenization, chat templates, request
  lifecycle, and continuous-batch scheduling;
- translate scheduler batches into `EngineBatch` and results into
  `EngineResult` without exposing Candle GPU storage;
- add an `oxide-production` profile that fails closed and a separate
  `mistral-reference` test profile;
- document and automate subtree upstream sync.

Exit gate:

- the production profile serves the E2 model through the OpenAI-compatible
  endpoint;
- a complete request has zero Candle CUDA, FlashInfer, cuBLASLt, cuda-oxide,
  and unregistered provider hits;
- the shell boundary issues no device-to-device copy;
- streaming, cancellation, malformed requests, and engine failures settle all
  resources and keep the server reusable;
- the maintained fork diff is localized to pipeline registration, adapter
  wiring, and build configuration.

Stop gate:

- if support requires edits distributed through Mistral.rs model layers, move
  the missing behavior into the Oxide pipeline or data plane before proceeding;
- if the subtree cannot be updated independently of feature changes, repair
  the import process before adding model coverage.

## E4: Qualify scheduling and serving behavior

**State:** planned.

Work:

- connect continuous batches to Oxide prefill and decode plans;
- implement paged KV allocation, sharing, copy-on-write, eviction, and
  compaction in the Oxide engine;
- qualify mixed prefill/decode batches and dynamic sequence completion;
- add fixed-address decode Graph pools with explicit miss behavior;
- measure prefix caching in a separate, named cohort;
- add request admission, memory watermarks, backpressure, and overload errors;
- expose provider, artifact, batch, KV, Graph, and latency telemetry.

Exit gate:

- concurrency sweeps preserve token correctness and request isolation;
- cancellation and overload do not leak pages, events, graphs, or buffers;
- Graph misses and unsupported batch shapes return to a registered eager
  TileLang plan, not another provider;
- p50, p95, and p99 latency, throughput, goodput, memory, and errors are
  recorded over the frozen workload matrix.

## E5: Compare and make the first competitive claim

**State:** planned.

Kernel work:

- compare admitted attention, KV, sampling, and applicable GEMM contracts with
  pinned FlashInfer;
- keep the same tensors, metadata, stream, outputs, and timed boundary;
- publish raw balanced-order samples and artifact hashes.

Engine work:

- run one neutral OpenAI-compatible trace against Oxide Infer, vLLM, and
  SGLang;
- pin identical local model and tokenizer files, dtype, KV dtype, maximum
  context, sampling, cache, and speculation settings;
- cover interactive, balanced, prefill-heavy, long-context, decode-heavy, and
  mixed-arrival workloads;
- publish TTFT, inter-token latency, end-to-end latency, token throughput,
  SLO goodput, peak memory, CPU usage, and error rates;
- reproduce secondary runs with the official vLLM and SGLang benchmark clients.

Exit gate:

- all correctness and provider-coverage gates pass;
- no primary p50 latency, peak-memory, or saturated-throughput metric regresses
  by more than 10% against either pinned engine;
- at least one useful metric wins beyond the 5% noise band;
- p99, goodput, and error rates do not invalidate the average result;
- a clean release artifact reproduces the record.

The first acceptable wording is "competitive Rust inference engine" for the
named profile. A general "faster than vLLM and SGLang" claim requires broader
workloads and hardware and is not implied by this phase.

## E6: Remove transitional providers

**State:** planned after E2 and E3 evidence.

Work:

- remove cuda-oxide from Cargo manifests, lockfiles, build scripts, and CI;
- remove product cuBLASLt and native-provider implementations;
- move any still-useful current provider code to immutable historical tags,
  not a disabled production fallback;
- replace current environment and CUDA validation documents with the TileLang
  artifact toolchain;
- remove legacy Mistral operator-adapter feature paths;
- retain old JSON evidence unchanged and label its source as historical.

Exit gate:

- repository searches find no live product dependency or build command for
  cuda-oxide, cuBLASLt, or the paired operator adapter;
- the clean release builds, tests, and serves without those toolchains present;
- historical records and links remain understandable;
- the product has exactly one registered custom compute-kernel source:
  TileLang.

## E7: Expand only from measured demand

**State:** planned after the first competitive release.

Candidate order:

1. larger dense Llama, Qwen, and Mistral profiles using the same IR;
2. FP8 or weight-only quantization with explicit quality gates;
3. prefix caching and speculative decoding;
4. MLA and one measured MoE model with grouped GEMM;
5. tensor parallelism and NCCL lifecycle qualification;
6. additional NVIDIA architecture artifacts with independent evidence;
7. multimodal models only after their preprocessing and GPU contracts fit the
   same fail-closed boundary.

Each addition repeats contract, artifact, model, serving, and release gates.
Mistral.rs feature breadth does not automatically become Oxide product
coverage.

## Evidence rule

Every result states its contract, source, provider, algorithm, artifact,
hardware, timed region, accepted claims, and excluded claims. Reviewed records
remain immutable. A faster kernel never proves a faster model or server by
itself, and a target design never changes the state of current source.
