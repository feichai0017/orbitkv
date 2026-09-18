# AletheiaRT product contract

## Objective

Build a production inference system that can improve its physical execution
without changing model semantics or deploying unqualified code. Given a model,
hardware profile, workload trace, and service-level objective, the system emits
and serves a qualified physical plan.

The product is AletheiaRT: a proof-carrying, self-optimizing inference platform.
Its long-term optimization target is constrained goodput, not an isolated
kernel minimum:

```text
maximize goodput(plan, workload, hardware)
subject to correctness, p99 latency, memory, availability, and cost bounds
```

## What this repository owns

1. Stable model, hardware, workload, physical-plan, and evidence schemas.
2. Candidate qualification and reproducible benchmark orchestration.
3. Plan storage, compatibility checks, selection, fallback, and rollback.
4. The provider ABI used by production executors.
5. Trace ingestion and offline replay contracts.
6. A small reference executor and, later, a high-performance native executor.

## What this repository reuses

- SGLang for a production serving surface and as a scheduler implementation to
  study or integrate through versioned extension points.
- FlashInfer and vendor libraries for qualified attention, GEMM, MoE, and
  sampling candidates.
- PyTorch export initially; compiler IRs such as TVM Relax or Mirage only after
  a measured need.
- CUDA, communication, HTTP, tokenization, and serialization libraries rather
  than local substitutes.

## Safety boundary

Search may change kernels, fusion, layouts, buffering, graph capture, batching,
and placement. It may not redefine model semantics, numerical acceptance, state
ownership, or request guarantees. An optimizer proposes candidates; a separate
qualification path decides whether they are deployable. Production only loads
content-addressed artifacts whose certificate covers the current model,
hardware, software stack, shape domain, and resource limits.

## Initial scope

- One NVIDIA GPU and one dense decoder family after the CPU control-plane slice.
- BF16 first, prefill and decode, continuous batching, streaming, cancellation,
  and bounded paged KV.
- Offline search and deployment-time tuning. Online code generation is out of
  scope; runtime selection is restricted to already-qualified plans.
- SGLang is an integration target, not part of the trusted schema layer.

## Definition of production-grade

A configuration is production-grade only when it has differential numerical
tests, resource-bound checks, cancellation and OOM recovery, repeatable latency
and goodput measurements, low-overhead telemetry, soak and fault tests, and an
exercised fallback/rollback path. Model count alone is not a release criterion.
