<p align="center">
  <img src="website/public/readme-banner.svg" alt="OrbitKV — Compile the lifetime of state" width="100%" />
</p>

<p align="center">
  <a href="https://feichai0017.github.io/orbitkv/">Website</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/roadmap.md">Roadmap</a> ·
  <a href="results/README.md">Results</a>
</p>

**A state-aware, compiled inference engine in Rust.**

OrbitKV turns attention lifetimes into memory plans and owns KV pages and
persistent model state. Its integrated compiler builds model graphs, profiles
legal implementations on CUDA, and saves selected programs for replay. State
management, compilation and serving share one workspace and release.

- **State ownership:** prefix sharing, copy-on-write, cancellation, and safe reuse.
- **Measured compilation:** generated CUDA alongside cuBLASLt, DeepGEMM,
  FlashInfer, and optional FlashAttention-3.
- **Native serving:** batching and an optional OpenAI-compatible HTTP frontend
  in one Rust process.

## Architecture

| Crate | Responsibility |
| --- | --- |
| [`orbitkv`](crates/orbitkv) | Backend-independent state compiler and KV manager; usable on its own |
| [`orbitkv-compiler`](crates/orbitkv-compiler) | Symbolic tensor graphs, equivalence rules, and search infrastructure |
| [`orbitkv-ops`](crates/orbitkv-ops) | Portable inference semantics and graph builders |
| [`orbitkv-cuda`](crates/orbitkv-cuda) | CUDA implementations, device measurement, and execution |
| [`orbitkv-tracing`](crates/orbitkv-tracing) | Compiler and runtime diagnostics |
| [`orbitkv-executor`](crates/orbitkv-executor) | Model import, state bindings, compilation, and artifacts |
| [`orbitkv-engine`](crates/orbitkv-engine) | Request scheduling, batching, streaming, and HTTP serving |

Model semantics, state contracts, shapes, and device capabilities determine
kernel candidates. Joint search across state layouts and execution strategies
is the [next compiler direction](docs/joint-compilation.md).

## Quickstart

Host tests need current stable Rust and Python 3.

```sh
git clone https://github.com/feichai0017/orbitkv.git
cd orbitkv
cargo test --locked --all-targets
```

Compile an example state plan:

```sh
cargo run --locked -p orbitkv --bin orbitkv -- \
  compile-runtime-manifest crates/orbitkv/examples/hybrid-attention-state-plan.json
```

For CUDA inference, follow the [engine setup](crates/orbitkv-engine/README.md)
and [provider setup](docs/attention-providers.md). Provider sources are prepared
explicitly before model compilation.

## Models and performance

**Qwen3.8-27B-FP8**: verified text inference on one NVIDIA H20, with Full
attention and Gated DeltaNet state. Local weights, configuration and tokenizer
match the official checkpoint revision. Vision and MTP are outside this scope.

See [models and measured performance](https://feichai0017.github.io/orbitkv/models/)
and [reproducible model reports](results/README.md). Measurements state the
checkpoint, precision, request lengths, concurrency and source revision.
The [support contract](docs/capability-matrix.md) records the exact checkpoint
and tested bounds. Engine comparisons use the same `vllm bench serve` suite for
OrbitKV, vLLM and SGLang; see the [benchmark method](docs/benchmarking.md).

## Documentation

- [OrbitKV compiler and CUDA compilation](docs/compiler.md)
- [State ownership and lifecycle](docs/runtime-session.md)
- [Execution artifacts](docs/module-artifacts.md)
- [External KV tiers](docs/external-kv.md)
- [Code and test layout](docs/code-layout.md)
- [Benchmarking](docs/benchmarking.md)

[Core MIT licensed](LICENSE). The compiler crates derive from Luminal and retain
their MIT/Apache-2.0 licenses. The optional frontend reuses vLLM's Rust components. See
[upstream components and licenses](docs/components.md).
