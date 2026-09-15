# Architecture

OrbitKV is a state-aware inference engine in Rust. It combines a standalone
state manager, an integrated tensor compiler, a CUDA backend and an optional
OpenAI-compatible server. [Qwen3.8-27B-FP8](capability-matrix.md) is the current
validated model on one H20.

## Workspace boundaries

| Crate | Owns | Must not own |
| --- | --- | --- |
| `orbitkv` | State semantics, manifests, pages, generations, Prefix/COW, retirement and reuse | CUDA, kernel selection or request transport |
| `orbitkv-compiler` | Symbolic graphs, egglog equivalences and search infrastructure | GPU execution or page lifecycle |
| `orbitkv-ops` | Portable operation semantics and inference graph builders | Provider selection |
| `orbitkv-cuda` | Generated/library implementations, resource validation, GPU profiling and execution | Logical state ownership |
| `orbitkv-tracing` | Compiler/search/runtime diagnostic records | Execution policy |
| `orbitkv-executor` | Checkpoint import, model graph, arena bindings, compilation and artifacts | Page reuse authority or HTTP |
| `orbitkv-engine` | Request admission, scheduling, token streaming and optional frontend | Provider rewrites or a second KV allocator |

All seven crates share the root workspace and lockfile. The state manager remains
usable independently. Compiler source and its original licenses are maintained
in this repository; see [maintenance](compiler-maintenance.md) and
[code layout](code-layout.md).

## Model initialization

1. **Import semantics.** The executor normalizes explicit checkpoint architecture,
   quantization and tensor metadata. Unsupported or contradictory inputs fail
   before weight loading. Topology assigns each layer to token KV or a matching
   recurrent/convolution state pair.
2. **Compile state.** The manager derives a `RuntimeManifest`. Stable arena
   registrations join backend-neutral `StateLayoutFacts` to produce compiler
   facts and typed bindings. Persistent writes must alias their registered state.
3. **Build the graph.** Portable attention, block-scaled linear and recurrent
   semantics are composed with normalization, projections and sampling.
   Model names do not select CUDA implementations.
4. **Search or replay.** `Graph::build_search_space` saturates egglog for feasible
   workload buckets. The CUDA runtime extracts candidates, rejects invalid
   aliases/resources, prepares native code and measures complete programs.
   Retained finalists are compared on the CUDA Graph deployment path.
   A compatible artifact instead restores the selected schedules and images.
5. **Prepare serving.** Weights upload into owned device buffers; profiling
   scratch is replaced with manager-bound arenas. Configured bucket residency
   controls preparation before readiness. A saved artifact removes graph search;
   provider plans and dynamic specialization can still require preparation.

The [compiler](compiler.md), [search policy](search-coverage.md),
[weight loader](weight-loading.md) and [artifact contract](module-artifacts.md)
describe these boundaries in detail.

## Request execution

The HTTP frontend tokenizes and submits logical generation requests. One model
worker owns the compiled decoder, active requests and `RuntimeSession`. It
admits work within configured request, sequence, query-token and state budgets,
and may combine a new prefill with existing decode requests.

For each batch, the session prepares state transitions. The executor lowers
manager-issued bindings to page CSR metadata, positions and fixed-state slots
inside stable-capacity input allocations. It selects an already compiled bucket
whose guards admit the actual batch. Provider metadata and captures are refreshed
when their dependency contracts require it.

CUDA executes generated regions and library calls on the owning stream. Full
attention reads a typed paged view; recurrent layers read/update their own state
arenas. Required alias checks preserve storage identity. The serving graph
produces logits and greedy token IDs for all query rows; the worker selects
the final token ID of each request. `DecoderCompileConfig.output_rows` binds
output geometry to the artifact. `LastTokenPerRequest` selects hidden rows
before final normalization/projection and produces one token ID per request.
Both modes execute all layer and state updates. The reduced-row mode remains
an explicit executor option while full-model equivalence is under qualification;
serving keeps `AllTokens`. Logit readback is explicit. Tokens return to the
frontend for ordered streaming.

The row-selection policy follows the work elimination used by
[vLLM's model runner](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/worker/gpu_model_runner.py)
and [SGLang's logits processor](https://github.com/sgl-project/sglang/blob/095ec6c997bfdd25d3864cb0ce77a6562a934b96/python/sglang/srt/layers/logits_processor.py).
OrbitKV expresses it with the existing gather operation before final
normalization/projection, preserving explicit dtype boundaries and symbolic CSR
request geometry. It does not require their Python runtime or model dispatch.

Event-backed completion receipts are tied to the exact state bindings. The
session commits completed transitions, publishes state and applies compiled
retirement. Cancellation and output termination release ownership. Generations
become reusable only after execution and acknowledgement make reuse safe.
Provider-local last use never grants that authority.

See [runtime sessions](runtime-session.md), [state lifecycle](state-lifecycle.md)
and [graph residency](graph-residency.md) for transition and capture lifetimes.

## Compute implementations

| Implementation | Role |
| --- | --- |
| Generated CUDA | Elementwise/reduction/index operations and admitted fused regions, including recurrent state operations |
| cuBLASLt | Dense and batched matrix products |
| DeepGEMM | SM90 block-scaled FP8 linear candidates; optional shared activation preparation |
| FlashInfer | Explicit paged decode and packed-prefill attention algorithms |
| FlashAttention-3 | Optional SM90 F16/BF16 paged attention candidates |

Logical attention and its KV view are separate contracts. Provider rules check
semantics, dtype, geometry, device and workspace before adding equivalent
implementations. Loading a library does not fuse its internals with surrounding
operators. General algorithm-region compilation and persistent megakernels are
future compiler work.

Provider revisions and dependencies share one lockfile and native build/cache
policy. The [CUDA map](cuda-backend.md), [attention contracts](attention-providers.md)
and [compiler extension points](compiler-boundaries.md) define the interfaces.

## Joint compilation

Today, state facts constrain compute search and all deployment buckets share one
validated persistent realization. Search uses private scratch, so measurement
cannot mutate live request state. General competition between physical KV layouts,
resident executable policies and external restore/recomputation is still planned.

The executor will coordinate that outer search under a shared memory and compile
budget. The manager will continue validating manifests and owning runtime state.
[Joint compilation](joint-compilation.md) defines the proposed design;
[external KV](external-kv.md) defines the existing byte-transport boundary.

## Verification

`tools/verify_active_source.py` checks dependency edges, source/test layout and
removed compatibility surfaces. Host tests cover transaction and failure
invariants. CUDA tests add independent operator references, alias/resource checks
and capture lifetime regressions. Model tests add teacher-forced logits and
state drain; HTTP tests add completion, cancellation and serving behavior.

Only reviewed model measurements enter [results](../results/README.md).
Compiler timings and kernel diagnostics remain distinct from serving performance.
The [roadmap](roadmap.md) tracks remaining qualification and optimization work.
