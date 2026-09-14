<p align="center">
  <img src="website/public/readme-banner.svg" alt="OrbitKV — Compile the lifetime of state" width="100%" />
</p>

<p align="center">
  <a href="https://feichai0017.github.io/orbitkv/">Website</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/roadmap.md">Roadmap</a> ·
  <a href="results/README.md">Results</a>
</p>

**An attention-state compiler and native Rust inference stack.**

OrbitKV turns attention lifetimes into memory plans and owns KV pages and
persistent model state. The inference-only [Luminal fork](third_party/luminal)
compiles model graphs, profiles legal implementations on CUDA, and saves the
selected programs for replay.

- **State ownership:** prefix sharing, copy-on-write, cancellation, and safe reuse.
- **Measured compilation:** generated CUDA alongside cuBLASLt, DeepGEMM,
  FlashInfer, and optional FlashAttention-3.
- **Native serving:** batching and an optional OpenAI-compatible HTTP frontend
  in one Rust process.

## Architecture

| Crate | Responsibility |
| --- | --- |
| [`orbitkv`](crates/orbitkv) | Backend-independent state compiler and KV manager; usable on its own |
| [`orbitkv-executor`](crates/orbitkv-executor) | Model import, Luminal execution, state bindings, and artifacts |
| [`orbitkv-engine`](crates/orbitkv-engine) | Request scheduling, batching, streaming, and HTTP serving |

Model semantics, state contracts, shapes, and device capabilities determine
kernel candidates. Joint search across state layouts and execution strategies
is the [next compiler direction](docs/joint-compilation.md).

## Quickstart

Host tests need current stable Rust and Python 3.

```sh
git clone --recurse-submodules https://github.com/feichai0017/orbitkv.git
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

## Current status

Active development. Bounded NVIDIA H20 qualification covers dense Full,
Full + Sliding, and the Qwen3.8-27B-FP8 hybrid text decoder with Gated DeltaNet
and convolution state. The [latest report](results/search-coverage-20260914/README.md)
covers independent logits, artifact replay and compiler search. Broader sampling
still misses expensive kernel choices; serving performance remains an optimization target.

See the [capability matrix](docs/capability-matrix.md) for supported contracts
and qualification limits.

## Documentation

- [Luminal and CUDA compilation](docs/luminal-design.md)
- [State ownership and lifecycle](docs/runtime-session.md)
- [Execution artifacts](docs/module-artifacts.md)
- [External KV tiers](docs/external-kv.md)
- [Code and test layout](docs/code-layout.md)
- [Benchmarking](docs/benchmarking.md)

[MIT licensed](LICENSE). Built on Luminal and the CUDA provider libraries;
the optional frontend reuses vLLM's Rust components. See
[upstream components and licenses](docs/components.md).
