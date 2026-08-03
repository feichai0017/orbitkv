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

## Current scope

The repository contains two crates:

| Crate | Responsibility |
| --- | --- |
| `loom-infer` | Safe operator contracts and CPU reference implementations |
| `loom-infer-cuda` | Rust host code and Rust CUDA kernels built with cuda-oxide |

The current device paths are:

| Operator | Provider | H20 state |
| --- | --- | --- |
| Contiguous RMSNorm F32, FP16, BF16 | Rust device kernels compiled with cuda-oxide | Owned-binding revision device-correct and sanitizer-clean |
| Contiguous BF16 GEMM with F32 accumulation | One fixed cuBLASLt algorithm | Device-correct with fixed-address Graph replay and sanitizer coverage |

Both use the same public flow:

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

Attention, sampling, KV-cache, MoE, quantization, and performance qualification
remain roadmap work. The fixed-address Graph path passed its declared H20
correctness and sanitizer gates. See the
[2026-08-03 result](docs/results/h20-owned-bindings-cuda-graph-correctness-20260803.json).

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

The default workspace build does not require CUDA:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo check --workspace --release
cargo package -p loom-infer
```

H20 device validation uses the pinned nightly and cuda-oxide revision:

```bash
cd crates/loom-infer-cuda
cargo oxide doctor
cargo oxide run rms_norm_h20 --bin rms_norm_h20 --features cuda --arch sm_90
cargo oxide run bf16_gemm_h20 --bin bf16_gemm_h20 --features cuda --arch sm_90
cargo oxide test --arch sm_90 -- --workspace --features cuda --release
```

See the [H20 validation contract](docs/development/h20-validation.md) before
reporting a device result.

## Evidence

Operator correctness, kernel latency, graph execution, engine integration, and
serving performance are separate claims. A microbenchmark does not establish
an engine or serving speedup.

The current Graph record covers correctness and Compute Sanitizer results. It
makes no performance, engine, or serving claim.

See the [architecture](docs/design/loom-infer-architecture.md),
[operator catalog](docs/operator-catalog.md), [roadmap](docs/roadmap.md), and
[owned-binding Graph result](docs/results/h20-owned-bindings-cuda-graph-correctness-20260803.json).

## License

[MIT](LICENSE)
