<div align="center">
  <h1>Loom Infer</h1>
  <p>Rust-native GPU kernels for LLM inference.</p>
  <p>
    <a href="docs/README.md">Docs</a> ·
    <a href="docs/operator-catalog.md">Operators</a> ·
    <a href="docs/flashinfer-parity.md">Parity</a> ·
    <a href="docs/roadmap.md">Roadmap</a> ·
    <a href="docs/results/README.md">Evidence</a>
  </p>
</div>

Loom Infer is an inference GPU operator library written in Rust. It
provides checked operator contracts, CPU references, and Rust CUDA kernels
compiled with [cuda-oxide](https://github.com/NVlabs/cuda-oxide).

The project is not a model server. An inference engine owns requests, models,
scheduling, and distributed execution. Loom Infer owns operator contracts,
launch planning, GPU execution, and reproducible evidence.

## Project direction

Loom Infer targets functional parity with the operator contracts required from
FlashInfer while keeping Loom-owned product code in Rust. Parity is measured by
behavior, not by copying FlashInfer's Python API, source layout, or
implementation choices.

- FlashInfer defines the pinned comparison surface for operator functionality.
- Each admitted contract fixes shapes, dtypes, layouts, numerical behavior,
  stream semantics, workspace policy, and CUDA Graph behavior.
- Loom-owned custom GPU kernels are Rust compiled with cuda-oxide.
- Qualified vendor libraries remain explicit providers for operations such as
  GEMM and collectives where a custom kernel has no measured advantage.
- Correctness, performance, Graph, engine, and serving claims pass independent
  evidence gates.

The current implementation is a partial foundation toward that target, not a
claim of complete FlashInfer parity. See the
[parity matrix](docs/flashinfer-parity.md) for the exact admitted surface.

## Current scope

The repository contains three crates:

| Crate | Responsibility |
| --- | --- |
| `loom-infer` | Safe operator contracts and CPU reference implementations |
| `loom-infer-cuda` | Rust host code and Rust CUDA kernels built with cuda-oxide |
| `loom-infer-validation` | Non-published H20 runners and shared validation support |

The current device paths are:

| Operator | Provider | H20 state |
| --- | --- | --- |
| Contiguous RMSNorm F32, FP16, BF16 | Rust device kernels compiled with cuda-oxide | Owned-binding revision device-correct and sanitizer-clean |
| Contiguous BF16 GEMM with F32 accumulation | One fixed cuBLASLt algorithm | Device-correct with fixed-address Graph replay and sanitizer coverage |
| BF16 single-request decode attention | Rust device kernel compiled with cuda-oxide | Narrow NHD D128 contract device-correct and sanitizer-clean |

All providers use the same public flow:

```text
validated spec
  -> provider::load
  -> immutable plan
  -> CommandQueue::bindings
  -> CommandQueue::begin
  -> plan::enqueue_into
  -> CommandScope::finish
  -> CommandCompletion::wait
  -> CheckedBindings::take_read_write
```

The Graph path consumes a one-shot `GraphQueue` with a private stream and
returns the same bindings through `GraphExec::into_bindings`.

The GEMM contract is `D[M,N] = A[M,K] * W[N,K]^T` over contiguous row-major
BF16 tensors. Algorithm selection happens during planning. Enqueue does not
tune or fall back.

Paged attention, sampling, KV-cache, MoE, quantization, and performance
qualification remain roadmap work.

The first single-decode slice covers BF16 MHA, MQA, and GQA with NHD caches and
head dimension 128. It does not establish FlashInfer performance parity. See the
[single-decode result](docs/results/h20-bf16-single-decode-correctness-20260803.json).

### Execution

The command scope chains Rust kernels and vendor calls on one caller-owned
stream. It retains typed buffers, loaded kernel functions, and external plans
until one completion event settles the scope. The cuda-oxide launcher still
allocates its argument vector during Rust-kernel enqueue.

## Source boundary

- Loom-owned product code is Rust.
- Custom device kernels are Rust compiled with cuda-oxide.
- The repository has no Python product API, CUDA C++, compatibility layer, or
  silent fallback.
- CUDA drivers and cuBLASLt are current vendor dependencies. Other established
  GEMM and collective libraries may enter through explicit providers.
- The caller owns streams. Checked bindings share read-only buffers through
  `Arc` and take exclusive ownership of writable buffers until completion.

## Local checks

Install the pinned toolchains described in the
[development environment](docs/development/environment.md). The common
CPU-only and website gate does not require CUDA:

```bash
make install-website
make check
```

CUDA host compilation and release tests use the pinned nightly and cuda-oxide
revision:

```bash
make cuda-doctor
make cuda-check
make cuda-test
make h20
```

See the [H20 validation contract](docs/development/h20-validation.md) before
reporting a device result.

## Evidence

Operator correctness, kernel latency, graph execution, engine integration, and
serving performance are separate claims. A microbenchmark does not establish
an engine or serving speedup.

The current attention and shared-command regression records cover correctness,
Graph replay, and Compute Sanitizer results. They make no performance, engine,
or serving claim.

See the [architecture](docs/design/loom-infer-architecture.md),
[repository layout](docs/design/repository-layout.md),
[operator catalog](docs/operator-catalog.md), [roadmap](docs/roadmap.md), and
[evidence index](docs/results/README.md).

## License

[MIT](LICENSE)
