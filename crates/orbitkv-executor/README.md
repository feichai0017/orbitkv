# OrbitKV executor

The executor connects the state manager with OrbitKV's integrated model compiler
and CUDA backend. It imports checkpoint semantics, builds a symbolic decoder,
binds manager-authored state arenas, and executes scheduled batches. It does
not allocate logical pages or authorize their reuse.

`ExecutorPlan` carries attention classes, fixed-state geometry, and physical
arena facts. `CompilerFacts` lowers those contracts into each search bucket.
Persistent KV updates must alias their registered inputs; recurrent and
convolution state updates produce event-backed completion evidence.

`CompiledDecoder` searches legal implementations during initialization or loads
a compatible schema-11 artifact. Token IDs, positions and attention metadata
update stable-capacity device allocations. The normal path reads device-selected
greedy token IDs; full logits are available through an explicit diagnostic API.
Optional outer decode capture retains selected child graphs under a guarded
signature. Provider replanning and bucket materialization have separate costs.

The executor also owns `ExternalKvTransport`, the asynchronous byte-movement
boundary over manager-authored spans. Its host-memory implementation is a
reference transport. Production remote storage and CPU weight offload remain
separate qualification work.

The Qwen3.8 H20 execution scope and remaining gaps are recorded in the
[model support contract](../../docs/capability-matrix.md). See
[compiler architecture](../../docs/compiler.md),
[maintenance](../../docs/compiler-maintenance.md), and
[state lifecycle](../../docs/runtime-session.md) for the execution contracts.
