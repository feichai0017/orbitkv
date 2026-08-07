<div align="center">
  <h1>Loom Infer</h1>
  <p><strong>High-performance CUDA operators for Rust inference engines.</strong></p>
  <p>Rust host code. Rust device kernels via <a href="https://github.com/NVlabs/cuda-oxide">cuda-oxide</a>. FlashInfer-class contracts and evidence.</p>
  <p>
    <a href="https://github.com/feichai0017/loom-infer/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/feichai0017/loom-infer/actions/workflows/ci.yml/badge.svg"></a>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-c8ff4d"></a>
    <img alt="Rust 2024" src="https://img.shields.io/badge/Rust-2024-ff8b52">
    <img alt="CUDA SM90" src="https://img.shields.io/badge/CUDA-SM90-74e7db">
  </p>
  <p>
    <a href="docs/README.md">Docs</a> ·
    <a href="docs/operator-catalog.md">Operators</a> ·
    <a href="docs/flashinfer-parity.md">Parity</a> ·
    <a href="docs/results/README.md">Evidence</a> ·
    <a href="docs/roadmap.md">Roadmap</a>
  </p>
</div>

Loom Infer is a Rust-native inference operator layer: checked contracts,
immutable plans, owned asynchronous execution, and custom Rust kernels compiled
for NVIDIA GPUs with cuda-oxide.

It is **not** a model server. Engines keep requests, scheduling, models, and
distributed policy; Loom owns the operator boundary.

## Why Loom

- **Rust end to end** — public API, planning, resource ownership, host
  execution, and Loom-owned device kernels.
- **Checked asynchronous execution** — typed bindings retain reads and
  exclusively own writable CUDA buffers until completion.
- **Evidence before claims** — correctness, sanitizer, eager performance,
  Graph, engine, and serving results are separate gates.

There is no CUDA C or CUDA C++ product source. CUDA remains the execution
platform; cuda-oxide is the Rust compiler and artifact toolchain.

## Operator Surface

| Family | Current admitted CUDA path |
| --- | --- |
| Normalization | F32, FP16, and BF16 RMSNorm |
| Matrix | Fixed-algorithm BF16 cuBLASLt GEMM |
| Decode | BF16 single-request direct and split-K attention |
| Paged decode | BF16 NHD D128, page size 16 |
| Prefill | BF16 ragged bottom-right causal attention |
| Position | BF16 D128 NeoX RoPE |
| KV mutation | Fused RoPE + explicit 1–64-token paged append |
| Graphs | Fixed-address capture and replay for admitted paths |

The surface is intentionally narrow and hardware-qualified. See the
[operator catalog](docs/operator-catalog.md) and
[FlashInfer parity matrix](docs/flashinfer-parity.md) for exact contracts and
open domains.

## Execution Model

```text
validated spec
  → immutable provider plan
  → checked typed bindings
  → one caller-owned CUDA stream
  → Rust kernel | qualified vendor provider
  → one completion fence
  → returned writable buffers
```

Plans fix the provider, algorithm, launch configuration, workspace, and Graph
policy before enqueue. There is no silent fallback.

## Workspace

| Crate | Responsibility |
| --- | --- |
| `loom-infer` | Backend-independent contracts and CPU references |
| `loom-infer-cuda` | Rust CUDA execution, cuda-oxide kernels, Graphs, and vendor providers |
| `loom-infer-validation` | H20 correctness, sanitizer, and matched performance runners |

## Validation

```bash
# CPU contracts, lint, package, website, and dependency audit
make install-website
make check

# Pinned cuda-oxide host/device toolchain
make cuda-doctor
make cuda-check
make cuda-test
make h20
```

Current records cover H20 correctness, all four Compute Sanitizer tools,
fixed-address Graph replay, and shape-specific comparisons with FlashInfer
`v0.6.16.post1`. Representative admitted results include:

- fused RoPE + paged append: **2.942× lower eager latency**;
- explicit multi-token append Graph: **1.656× lower replay latency**;
- paged batch-1 MHA: **4.41× lower eager latency**.

These are contract- and shape-specific operator results, not engine or serving
claims. Raw records and exclusions live in the
[evidence index](docs/results/README.md).

## Status

Loom Infer is alpha software. Near-term work focuses on broader attention
contracts, Graph coverage, and real Rust engine integration.

Read the [architecture](docs/design/loom-infer-architecture.md),
[development environment](docs/development/environment.md), and
[contribution guide](CONTRIBUTING.md) before adding a provider.

## License

[MIT](LICENSE)
