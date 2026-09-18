# AletheiaRT

> Proof-carrying, self-optimizing inference.

AletheiaRT is a production inference optimization and execution platform. It
continuously searches for better kernels, graph transforms, memory layouts, and
runtime policies, but deploys a candidate only when the candidate carries
reproducible evidence for semantics, numerical accuracy, hardware scope,
workload coverage, resource bounds, and rollback.

The name combines *aletheia*—the Greek idea of truth as disclosure—with RT,
for runtime. An optimization is not accepted because it claims to be fast. It must
disclose what it executes, where it is valid, how it was measured, and what
happens when it fails.

## The problem

Production inference is not one stable program. Its best physical implementation
changes with the model, GPU, driver, tensor shape, batch distribution, context
length, latency objective, and kernel ecosystem. Mature engines therefore
accumulate hard-coded thresholds and backend flags, while automated optimizers
often stop at a microbenchmark and cannot safely deploy their result.

AletheiaRT closes that loop:

```text
model semantics + hardware + workload trace + SLO
                         |
                         v
             bounded candidate generation
        AutoDeploy / FlashInfer / DeepGEMM / custom
                         |
                         v
                 qualification harness
        correctness / raw timings / memory / faults
                         |
                         v
              proof-carrying physical plans
                         |
                         v
               deterministic plan registry
                         |
                         v
               source-pinned SGLang runtime
                         |
                         +---- telemetry and replay ----> next search
```

## What AletheiaRT owns

- portable physical-plan, evidence, workload, SLO, and trace contracts;
- qualification and content-addressed artifact publication;
- deterministic plan selection, fallback, revocation, and rollback;
- the provider boundary used by kernel and compiler backends;
- the closed loop from a real SGLang workload trace to the next qualified plan.

## What AletheiaRT reuses

- **SGLang** owns production requests, batching, KV pools, sampling, and workers;
- **TensorRT-LLM AutoDeploy** supplies export and graph-transform machinery plus
  an independent NVIDIA baseline;
- **FlashInfer**, **DeepGEMM**, and SGLang's in-tree **sglang-kernel** supply
  high-performance operations and candidate tactics;
- PyTorch supplies the first numerical oracle.

All four upstream source trees are real Git submodules under `third-party/`,
including their nested CUTLASS, CCCL, NIXL, spdlog, and fmt revisions. They are
readable and patchable; AletheiaRT does not hide them behind opaque wheels.

## Current state

The trusted Rust control plane is implemented and tested. The first real GPU
candidate has also been qualified on an NVIDIA H20: source-pinned FlashInfer
RMSNorm, BF16, batch 1, hidden size 4096. Its plan, raw CUDA-event samples,
PyTorch-oracle results, software fingerprint, and JIT `norm.so` are produced by
`make qualify-rmsnorm`. The candidate remains unpublished until a matching real
SGLang trace satisfies the admission gate.

The project is not yet a complete model-serving release. M1 proves the
operator-level optimization loop; M2 adds a complete dense decoder; M3 makes
the supported single-GPU configuration production-grade.

## Workspace

```text
crates/aletheia-contracts   portable plans, certificates, workloads and SLOs
crates/aletheia-control     registry, eligibility and deterministic selection
crates/aletheia-executor    provider preparation, execution and fallback
crates/aletheia-cli         validation and local orchestration (`aletheia-rt`)
integrations/sglang         in-process SGLang workload trace hook
integrations/autodeploy     registered AutoDeploy transforms
integrations/providers      source provenance and GPU qualification runners
third-party/                pinned upstream source submodules
```

## Start here

```bash
make source-init source-verify
make doctor
make test

# Inspect source-build plans before executing them.
make bootstrap-kernels-dry-run
make bootstrap-sglang-dry-run
make bootstrap-autodeploy-dry-run
make bootstrap-tensorrt-source-dry-run

# Exercise the complete CPU publication gate.
make stage-reference
target/debug/aletheia-rt validate-registry .aletheia/registry

# Build the first real GPU candidate after bootstrapping kernels.
make qualify-rmsnorm

# Numerically smoke-test the source-built sglang-kernel on the local GPU.
make smoke-sglang-kernel
```

Read [the product contract](docs/product-contract.md),
[architecture](docs/architecture.md), [qualification pipeline](docs/qualification.md),
[project purpose](docs/purpose.md), [roadmap](docs/roadmap.md), and
[upstream source map](docs/upstream-source-map.md).
