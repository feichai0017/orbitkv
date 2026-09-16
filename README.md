# OrbitKV Next

OrbitKV Next is a compiler and kernel toolchain for stateful hybrid inference.
It lowers pinned model semantics into optimized execution islands and verified
[`kern`](https://github.com/pegainfer-project/kern) manifest-v5 programs. It is
not another CUDA runtime, serving frontend, weight loader, or KV service.

The implementation order is intentionally narrow:

1. `Qwen3.8-27B-FP8` on one NVIDIA H20;
2. `GLM-5.3-Flash` with an explicit multi-GPU or bounded-residency plan;
3. `DeepSeek-V4.1-Flash` on a native FP4, multi-GPU target.

The repository currently contains the M0 compiler skeleton: the Qwen3.8 model
contract, an effect-aware task graph, deterministic baseline island partitioning,
and the pinned `kern-manifest` verification boundary. No runtime is implemented
here.

```sh
cargo test --locked --all-targets
```

See [the architecture](docs/architecture-next.md),
[the roadmap](docs/roadmap-next.md), and [the pinned baselines](docs/baselines.md).
The complete pre-reset source remains recoverable from branch
`archive-orbitkv-pre-next-20260917`.
