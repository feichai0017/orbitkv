# Oxide Infer architecture

This document describes the current transitional operator source. The
[accepted target architecture](tilelang-engine-architecture.md) turns Oxide
Infer into a complete engine with a Mistral.rs control-plane shell and
TileLang-only product custom kernels. Current source and historical evidence
remain accurately documented here until each migration gate is complete.

The current source provides checked GPU operator contracts and CUDA providers
for LLM inference engines. A consumer engine retains its model graph,
scheduler, continuous batching, KV allocation policy, distributed control,
and serving API.

## Status terms

This document uses three implementation states.

| State | Meaning |
| --- | --- |
| Current | Source exists in a product crate. Current-source evidence can still be pending. |
| Experimental | Source exists, but one or more promotion gates remain open. |
| Planned | The roadmap admits the work. No empty module or public placeholder exists. |

Each evidence claim has a separate level: host, device correctness, lifecycle,
sanitizer, Graph, performance, engine, and serving. Passing one level does not
pass the next level.

## Product boundary

| Concern | Owner |
| --- | --- |
| Operator semantics and CPU references | `oxide-infer` |
| CUDA planning, checked execution, Graphs, and provider calls | `oxide-infer-cuda` |
| Native NVIDIA kernels | Oxide provider in `oxide-infer-cuda` |
| Hardware gates and benchmark generation | `oxide-infer-lab` |
| Model graph and tensor routing | Consumer engine or adapter |
| Request scheduling and continuous batching | Consumer engine |
| KV allocation, sharing, eviction, and copy-on-write | Consumer engine or KV pager |
| Tokenizer, serving API, and distributed control | Consumer engine |

Engine adapters translate engine tensors, metadata, and stream authority into
Oxide Infer operands. Core crates do not depend on engine types.

## Crate dependency boundary

```text
consumer engine or adapter
  -> oxide-infer-cuda
       -> oxide-infer
       -> native Oxide provider or explicit vendor provider
       -> caller-selected CUDA stream

oxide-infer-lab
  -> oxide-infer-cuda
  -> oxide-infer
```

`oxide-infer` builds without CUDA. `oxide-infer-cuda` owns every CUDA type,
unsafe device boundary, plan artifact, stream rule, and asynchronous lease.
`oxide-infer-lab` is not a product dependency.

A fourth crate needs a distinct dependency, build artifact, safety boundary,
or release cycle. Namespace size alone does not justify another crate.

## Operator families

Public namespaces follow operator semantics. Fusion belongs to an algorithm,
not to a top-level `fused_ops` family.

| Family | Current | Experimental | Planned |
| --- | --- | --- | --- |
| `attention` | Single decode, paged decode, ragged prefill, paged prefill; declared R1 runner paths are device-qualified | None | Sliding window, mixed-batch attention, MLA, broader head dimensions |
| `gemm` | Contiguous BF16 dense GEMM through cuBLASLt | Native SM90a M=1 BF16 GEMV | FP8, FP4, grouped, quantized, and broader small-M algorithms |
| `kv_cache` | Fused RoPE plus exclusive-page paged append; declared R1 runner path is device-qualified | None | Gather, scatter, compaction, remapping, FP8, and INT8 storage |
| `normalization` | RMSNorm for F32, FP16, and BF16; declared R1 runner paths are device-qualified | None | Additional normalization contracts from engine demand |
| `position` | BF16 NeoX RoPE with explicit positions; declared R1 runner path is device-qualified | None | Other layouts, dimensions, and position transforms |
| `activation` | None | None | SwiGLU and other measured activation contracts |
| `sampling` | None | None | Logits transforms, penalties, Top-K, Top-P, Min-P, logprobs, and RNG |
| `speculation` | None | None | Draft verification and token compaction |
| `quantization` | None | None | Scales, packing, conversion, and dequantization |
| `moe` | None | None | Routing, permutation, grouped-GEMM inputs, and expert combine |
| `communication` | None | None | Qualified tensor-parallel and expert-parallel collectives |

The project creates a family only with its first admitted contract. It does not
pre-create empty directories for planned work.

## One operator lifecycle

Every operator converges on one public lifecycle:

```text
Spec
  -> Provider
  -> Algorithm
  -> Plan
  -> Operands
  -> CommandScope
  -> Completion
```

| Name | Meaning |
| --- | --- |
| `Spec` | Backend-independent shape, dtype, layout, numerical, and alias contract |
| `Provider` | Implementation owner, such as `Oxide` or `CublasLt` |
| `Algorithm` | Stable named strategy within one provider |
| `Plan` | Immutable provider, algorithm, launch, workspace, artifact, and Graph decision |
| `Operands` | Typed resources and invocation metadata |
| `CommandScope` | Checked admission, alias resolution, retention, and submission boundary |
| `Completion` | Fence and status result that retains resources until quiescence |

Planning fixes the provider and algorithm. Enqueue does not tune, switch
providers, or fall back. Unsupported combinations return a planning error.

The current dense GEMM path already uses this lifecycle. Framework migration
renames remaining `*Args` values to `*Operands` without compatibility aliases.

## Planning and execution

Planning and command execution have different ownership.

```text
Spec + device capability + explicit policy
                  |
                  v
      Provider -> Algorithm -> Plan
                               |
Operands + stream authority ---+
                  |
                  v
       CommandScope -> Completion
```

The `Plan` owns or retains:

- provider and provider version
- algorithm identity
- launch configuration and architecture artifact
- exact workspace byte and alignment requirements
- Graph capture policy
- provider-private immutable state

The caller owns workspace storage. The plan declares its required size and
alignment. The provider cannot allocate hidden workspace inside enqueue.

The `Operands` own invocation-specific tensor regions and metadata. A paged
operator receives its page table through operands. The engine or KV pager owns
the page allocation policy and the page-table contents.

The command runtime owns binding checks, stream admission, resource retention,
device status, Graph capture, and completion settlement. It does not choose an
operator algorithm.

## Provider paths

Oxide Infer has two explicit CUDA provider paths.

```text
                         Plan
                          |
            +-------------+-------------+
            |                           |
            v                           v
      Oxide provider              vendor provider
      Rust device code            audited Rust FFI
            |                           |
        cuda-oxide                   cuBLASLt
            |                           |
            +-------------+-------------+
                          |
                     CUDA driver
                          |
                       NVIDIA GPU
```

cuda-oxide compiles native Rust kernels. It does not select providers, manage
engine tensors, or compile vendor calls. The cuBLASLt path does not pass
through native Oxide kernels.

Provider identities remain explicit in plans and result records. The current
native provider identity is `Oxide`. The current vendor identity is
`CublasLt`.

## Architecture and instruction boundaries

Architecture directories belong inside a native provider. They do not form a
runtime dispatcher. Each plan selects one exact target artifact before enqueue.

| Target | State | Boundary |
| --- | --- | --- |
| `sm_90a` | Current first target | Current kernels use SIMT, warp operations, WMMA, and selected `cp.async` paths. The published qualification row records an NVIDIA H20. |
| `sm_100a` | Planned | Separate Blackwell contracts, artifacts, numerical gates, SASS review, and benchmarks are required. |
| `sm_120` | Planned | Separate consumer Blackwell contracts and evidence are required. No forward-compatibility claim exists. |

TMA, WGMMA, and tcgen05 matrix operations are algorithm tools. They are not
provider identities or proof of hardware support. A helper imported from an
instruction namespace does not qualify the corresponding architecture.

The project adds an architecture module only when one admitted algorithm needs
it. Each new target records exact PTX or cubin hashes and rejects incompatible
devices before load.

## Device memory and stream ownership

The binding layer accepts complete owned buffers and typed subregions.
Read-only regions retain shared-read authority. Writable regions retain
exclusive authority until completion settles.

An external region binds these values as one capability:

- typed pointer and exact element span
- CUDA context
- access mode
- lifetime lease

Construction checks null pointers, range arithmetic, alignment, and overflow.
Binding checks the stream context and writable overlap.

`ExternalCudaStream` borrows an engine stream. It never adopts or destroys the
stream. The interop queue orders engine and Oxide work with CUDA events. The
completion retains every device lease until the runtime proves quiescence.

## Page-table and KV ownership

Paged attention reads physical pages. Different requests can share read-only
pages when their logical page tables are valid.

Paged append writes physical pages. The engine or KV pager must make every
target page private before submission. It then supplies one stable page-table
and reference-count snapshot through completion.

The operator validates the snapshot. It does not allocate pages, copy a shared
tail, update reference counts, or remap requests.

## Dynamic metadata

Host references validate host-resident metadata before execution. CUDA plans
also validate device-resident page tables and index arrays on the selected
stream.

A semantic rejection returns a typed completion error and preserves checked
bindings. It does not poison the queue or a fixed-address Graph. CUDA failure
or malformed status data poisons the affected execution scope.

## CUDA Graph contract

The current Graph path uses fixed device addresses and a private non-default
capture stream. Capture transfers bindings, functions, plans, and leases into
the graph owner.

One replay takes unique mutable graph access and returns one completion. Safe
code cannot replay concurrently or release retained resources before
settlement.

The current contract rejects rebinding, graph updates, cross-stream launch,
concurrent replay, and default-stream capture. One passing operator graph does
not qualify another plan or mutable metadata policy.

## Consumer engine and adapters

A consumer engine supplies:

- model and layer selection
- tensor allocations and views
- request and batch metadata
- page allocation and copy-on-write policy
- CUDA stream authority
- sampling, serving, and distributed control

An adapter converts these values into typed operands without an intermediate
device copy. It records provider hits, algorithm identities, pointer spans,
stream order, and output comparison.

The current Mistral.rs work is a narrow historical proof of concept. It does
not qualify the renamed source, general model coverage, production recovery,
or performance. A future vLLM adapter requires a stable external boundary and
its own versioned integration evidence.

## Evidence policy

Every admitted contract records its source, call site, tensors, numerical
limit, provider, algorithm, hardware, artifact, metric, and stop condition.

Qualification proceeds in this order when the boundary applies:

1. Host contract and independent reference.
2. Device correctness and negative cases.
3. Asynchronous ownership and completion settlement.
4. Compute Sanitizer.
5. Fixed-address Graph behavior.
6. Matched operator performance in both provider orders.
7. Real-engine output and no-copy evidence.
8. TTFT, TPOT, throughput, and memory measurements.
9. Serving and distributed behavior.

A faster microbenchmark does not establish a faster model or server.
