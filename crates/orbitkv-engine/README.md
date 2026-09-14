# OrbitKV engine

This crate owns request scheduling, execution coordination, logical request/event
contracts, and the optional OpenAI-compatible HTTP frontend. It joins one
`RuntimeSession` and one compiled decoder while core retains all KV lifecycle
authority.

| Module | Responsibility |
| --- | --- |
| `protocol` | `Engine`, `BatchIntent`, request IDs, sampling intent, and output events |
| `model_engine` | Bounded admission, batching, cancellation, backpressure, executor dispatch, release, and drain |
| `frontend` | Pinned vLLM Rust HTTP/tokenizer/chat/SSE adaptation through the logical `Engine` contract |
| `bin/serve` | Typed configuration and single-process composition |

The modules are internal; public types and functions are re-exported from
`orbitkv_engine`. Existing consumers of `orbitkv_server` should import those
same protocol/frontend types from `orbitkv_engine`. The separate server package
has been removed. The executable remains `orbitkv-serve`.

| Feature | Enables |
| --- | --- |
| default (empty) | Logical contracts, without CUDA or HTTP dependencies |
| `cuda` | The model-backed coordinator through `orbitkv-executor` |
| `vllm-frontend` | HTTP/tokenizer/transport integration, usable with a host-only `Engine` implementation |
| `server` | CUDA coordinator, HTTP frontend, and the `orbitkv-serve` executable |

```bash
cargo test --locked -p orbitkv-engine --features vllm-frontend
cargo build --locked --release -p orbitkv-engine --features server --bin orbitkv-serve
cargo run --locked --release -p orbitkv-engine --features server --bin orbitkv-serve -- --help
```

`--graph-cache-capacity` bounds materialized decoder buckets and defaults to one.
Retaining decode and prefill together can avoid full graph reconstruction on
phase switches. Choose the capacity with the deployment memory budget;
`ModelEngineConfig.graph_cache_capacity` is a required `NonZeroUsize` field.
The selected artifact stays independent of this runtime policy. See
[graph residency](../../docs/graph-residency.md) for resource ownership and checks.

`--prepare-execution true` (the default) prepares artifact representatives before
the engine becomes ready. `false` keeps preparation on demand for controlled
comparisons. The policy preserves existing materializations first and fills
unused slots in artifact order, bounded by the graph cache capacity. It does
not predict request frequency. Unprepared buckets and other admitted dynamic
shapes remain usable. Preparation does not
execute the model or acquire requests, and real request metadata is always
uploaded before execution. `ModelEngine::startup_report()` returns the elapsed
startup time, prepared dimensions and graph counts; `orbitkv-serve` emits it as
`ORBITKV_ENGINE_STARTUP` before starting the frontend. Preparation failures
propagate through `ModelEngine::start_with_tuning()` without publishing readiness.

`ModelEngine::shutdown()` closes admission for all handle clones, cancels
outstanding requests, joins the worker and returns an `EngineShutdownReport`.
It is blocking; async callers should use a blocking task. The report verifies
token-KV and fixed-state drain and records graph builds before decoder teardown.
Execution failure, worker panic or incomplete retirement returns an error.
The `completed_requests` counter includes cancelled requests that reached a
terminal state; `cancelled_requests` is a subset of that count.
`orbitkv-serve` emits this report on graceful shutdown so serving qualification
can verify the lifecycle without adding a public diagnostic endpoint.

The frontend currently admits greedy text generation. The local transport adapts
the upstream `EngineCoreClient` protocol; scheduling, model execution, and KV
management stay in this process. Its logical modules cannot import physical
execution types; the source-boundary checker enforces that distinction after
the crate merge.

All test source lives in `tests/`. Private suites mirror the owning module under
`tests/unit/`; model and HTTP lifecycle suites are public-API integration tests.
Released-checkpoint tests remain explicit opt-ins. See
[code layout](../../docs/code-layout.md), [architecture](../../docs/architecture.md),
and [benchmarking](../../docs/benchmarking.md) for the complete contracts.
