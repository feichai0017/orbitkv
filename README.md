# OrbitKV Next

OrbitKV Next is a source-integrated, high-performance inference engine for
stateful hybrid models. Its compiler lowers pinned model semantics into
optimized execution islands and manifest-v5 programs; its imported `kern`
substrate verifies and executes those programs, manages model state, and serves
requests.

The implementation order is intentionally narrow:

1. `Qwen3.8-27B-FP8` on one NVIDIA H20;
2. `GLM-5.3-Flash` with an explicit multi-GPU or bounded-residency plan;
3. `DeepSeek-V4.1-Flash` on a native FP4, multi-GPU target.

The repository contains the Qwen3.8 compiler skeleton and a source import of
`kern` 0.2.3. Runtime, KV/state pool, CLI, and serving code are therefore
available for direct modification, while model semantics remain above the
manifest boundary.

```sh
cargo test --locked --all-targets
# Serving is intentionally outside the default host build:
cargo build --locked --release -p kern-serve
```

See [the engine design](docs/engine-design.md),
[the architecture](docs/architecture-next.md), [the roadmap](docs/roadmap-next.md),
and [the pinned baselines](docs/baselines.md).
The complete pre-reset source remains recoverable from branch
`archive-orbitkv-pre-next-20260917`.
