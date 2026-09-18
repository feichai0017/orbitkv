# Why AletheiaRT exists

## The thesis

Inference performance is a moving target. The best implementation depends on a
cross product of model semantics, hardware, driver and library versions, tensor
shapes, batch and context distributions, memory pressure, and latency goals. A
human-maintained engine can choose good defaults, but it cannot permanently
encode the best physical plan for every point in that space.

At the same time, unconstrained automatic optimization is unsafe for production.
A generated kernel can win one microbenchmark while changing numerics, exceeding
memory limits, regressing a tail shape, or failing after a dependency upgrade.

AletheiaRT's thesis is:

> Treat inference optimization as the compilation and qualification of
> evidence-carrying physical plans, then let production choose only among plans
> whose evidence covers the current execution point.

## Why this is not another SGLang or vLLM

SGLang already provides the difficult production mechanisms: request lifecycle,
continuous batching, KV ownership, model runners, sampling, distributed workers,
and a large model ecosystem. Rebuilding all of those would spend most of the
project on compatibility and maintenance while adding little differentiation.

AletheiaRT initially runs *with* source-pinned SGLang. It observes real batches,
qualifies candidate implementations, and feeds selected physical choices back
through the narrowest available backend or compiler interface. If a required
interface is missing, the project contributes or maintains a small, explicit
patch rather than forking the entire serving architecture.

The standalone Rust executor exists to define and test the trusted plan boundary,
not to claim that the production server has already been replaced.

## Why this is not only a kernel library

A faster kernel is not necessarily a faster service. Layout conversion, graph
capture, workspace, batching, KV traffic, launch order, and SLO violations can
erase a microbenchmark win. FlashInfer, DeepGEMM, sglang-kernel, vendor libraries,
and generated kernels are candidate providers. AletheiaRT compares their effects
under an end-to-end workload and keeps their exact artifact and tactic identity.

The durable product boundary is therefore above any one kernel implementation:

```text
logical semantics
    -> candidate kernel/fusion/memory/schedule plans
    -> numerical and systems qualification
    -> immutable plan bundle
    -> runtime eligibility and fallback
```

## What self-optimization means

Self-optimization is a constrained feedback loop, not an agent editing production
CUDA in place.

1. SGLang records representative workload points and end-to-end outcomes.
2. AutoDeploy, provider autotuners, and future agents generate bounded candidates.
3. Qualification runs reference comparisons, raw device timings, memory checks,
   and fault tests on declared hardware and shapes.
4. The staging gate verifies every content hash and constructs a certificate.
5. The registry admits the plan only over the evidenced domain.
6. Production selects among admitted plans and retains a qualified fallback.
7. Telemetry detects drift and starts the next offline iteration.

Exploration happens outside the request path. Online adaptation is selection
among already-qualified choices.

## The industrial asset

The valuable output is not one benchmark or one model implementation. It is the
ability to shorten and de-risk the path from a new model, GPU, or traffic pattern
to a production-quality execution plan. The reusable assets are:

- the semantic and physical-plan contracts;
- the qualification harness and evidence corpus;
- source-to-artifact provenance;
- workload replay and cost models;
- safe selection, fallback, revocation, and rollback;
- integration adapters for established serving engines.

Even when AletheiaRT does not own the entire server, these components can improve
SGLang, TensorRT-LLM, embedded runtimes, or future accelerator backends.

## The learning value

The project deliberately exposes the full inference path. A contributor can trace
one request from SGLang scheduling through model execution, FlashInfer or
DeepGEMM dispatch, generated native artifacts, CUDA timing, qualification, and
registry admission. We reuse mature source rather than treating wheels as black
boxes, and we write new code where the cross-system contract is the contribution.

This gives a concrete way to learn serving, compilers, kernels, memory systems,
measurement, and production safety as one system instead of isolated tutorials.
