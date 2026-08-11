# Loom Infer architecture

Loom Infer provides GPU operator contracts and providers for Rust LLM
inference engines.
Engines retain model graphs, request scheduling, KV-cache policy, distributed
execution, and serving APIs.

## Product boundary

| Concern | Owner |
| --- | --- |
| Operator contract, planning, ownership, and host execution | Loom Rust crates |
| Custom NVIDIA kernels | Loom Rust device code compiled with cuda-oxide |
| GEMM and communication implementations | Explicit Loom or vendor providers selected before execution |
| Request scheduling and continuous batching | Consumer engine |
| KV allocation, sharing, eviction, and copy-on-write | Consumer engine or KV pager |
| Model graph and serving API | Consumer engine |

Loom-owned product code has no Python API or CUDA C++ source. cuda-oxide still
targets the CUDA platform. The runtime must obey CUDA context, stream, event,
Graph, and asynchronous lifetime rules.

FlashInfer defines the pinned comparison surface. Loom evaluates parity by
operator contract, not by file or symbol count. A matching contract includes
shape, dtype, layout, masking, numerical behavior, aliasing, workspace,
stream, and Graph semantics.

## Crates and dependencies

```text
consumer engine
  -> loom-infer-cuda
       -> loom-infer
       -> Rust device kernel or explicit vendor provider
       -> caller-selected CUDA stream

loom-infer-validation
  -> loom-infer-cuda
  -> loom-infer
```

| Crate | Responsibility |
| --- | --- |
| `loom-infer` | Public `Spec` types, errors, capabilities, and CPU references |
| `loom-infer-cuda` | Immutable plans, checked execution, Rust device kernels, Graphs, and vendor calls |
| `loom-infer-validation` | Non-published H20 gates, fixtures, comparisons, benchmarks, and reports |

`loom-infer` has no CUDA dependency. Product crates do not depend on the
validation crate. A new crate requires a separate dependency, ownership, or
safety boundary. GEMM and the CUDA runtime remain modules until such a boundary
exists.

## Operator families

The complete architecture uses semantic families across both product crates:

| Family | Scope |
| --- | --- |
| `attention` | Decode, prefill, masking, and attention-state merge |
| `kv_cache` | Paged append, gather, scatter, compaction, and remapping |
| `gemm` | Dense, grouped, and quantized matrix operations |
| `normalization` | RMSNorm and later normalization operations |
| `position` | RoPE and other position transforms |
| `activation` | Activation and gated-activation operations |
| `sampling` | Logits transforms, RNG, and token selection |
| `speculation` | Draft verification and token compaction |
| `quantization` | Scale, packing, conversion, and dequantization |
| `moe` | Routing, permutation, expert inputs, and combine |
| `communication` | Qualified tensor-parallel and expert-parallel collectives |

Only attention, dense GEMM, RMSNorm, RoPE, and paged append have current
implementations. The [repository layout](repository-layout.md) records their
present paths and target families. Planned families do not imply source or
device support.

## Operator lifecycle

The target operator framework uses one public lifecycle. The dense GEMM slice
already follows it. K0.0 migrates the remaining APIs.

```text
Spec
  -> Provider
  -> Algorithm
  -> Plan
  -> Operands
  -> CommandScope
  -> Completion
```

These names have one meaning throughout the project:

| Name | Meaning |
| --- | --- |
| `Spec` | Backend-independent tensor semantics, shape, dtype, layout, and numerical contract |
| `Provider` | Implementation owner, such as `Loom` or `CublasLt` |
| `Algorithm` | One named execution strategy within a provider |
| `Plan` | Immutable provider, algorithm, launch, workspace, artifact, and Graph decision |
| `Operands` | Typed resources for one invocation |
| `CommandScope` | Checked queue admission, alias resolution, retention, and submission boundary |
| `Completion` | Fence and status result that retains resources until quiescence |

Planning selects the `Provider` and `Algorithm`. Enqueue does not tune, switch
providers, or fall back. Unsupported combinations return a planning error.
Names such as `Fast`, `Optimized`, `Auto`, and `V2` do not identify a stable
algorithm.

The current CUDA API still exposes several `*Args` types. The framework
migration will rename them to `*Operands` rather than add aliases. Internal
`CheckedBindings` values remain the resolved resource set owned by a
`CommandScope` or `Completion`.

`finish()` records one queue event. The completion retains bindings, loaded
functions, and vendor plans until `wait()` or destruction proves quiescence.
Contract errors remain recoverable. CUDA or vendor submission failures poison
the scope and queue.

Cleanup first synchronizes the stream and then the context. If neither proves
quiescence, the process aborts before Rust releases retained GPU resources.

## Device memory ownership

The binding layer accepts complete `DeviceBuffer<T>` allocations and typed
subranges. `ReadDeviceRegion<T>` retains shared-read authority.
`ReadWriteDeviceRegion<T>` transfers exclusive write authority until command
completion. Both owned and external regions use the same resolver path.

An external region binds a typed pointer, element span, CUDA context, access
mode, and retained lifetime lease. Construction checks range arithmetic,
alignment, null pointers, and pointer overflow.

Binding checks the stream context. It rejects overlapping spans when either
span is writable, but accepts overlapping read-only spans.

The external constructors are unsafe because Loom cannot inspect allocation
extent, global aliases, or stream ordering in another runtime. Host tests do
not qualify this boundary on H20.

`ExternalCudaStream` retains an ordinary engine stream without adopting or destroying it.
`EngineInteropQueue` uses pre- and post-events to order a Loom stream against that external stream.
The current adapter covers direct single decode and NHD or HND paged decode.
After Loom enqueues the post-event wait, it returns the engine's stream-ordered authority without waiting on the host.
Each owned completion keeps its checked bindings, allocation leases, status storage, and event slot private until settlement.
Backpressure returns the original coupled authority and bindings before any bridge work starts.

The H20 interop gate uses a simulated engine.
It covers two in-flight commands, slot recovery, cross-thread completion, HND GQA6 paged decode, typed metadata rejection, unchanged pointers, and lease behavior.
It is not a controlled negative test for a missing post-event wait.
A real engine must supply its own stream and allocations before Loom can claim engine interoperability.

## Paged KV read and write semantics

Paged decode and prefill treat physical pages as read-only. Different requests
may reference the same physical page when their logical mappings are valid.

Fused append mutates K and V pages. Its write contract requires:

- one caller-supplied reference count for every physical page.
- reference count one for each target page.
- distinct physical slots for distinct tokens.
- the same validated metadata snapshot on host and device.

The engine or pager makes a shared tail private before append:

1. Detect a shared target tail page.
2. Allocate or copy a private page.
3. Update the request page table and page reference counts.
4. Order those writes before the append launch.
5. Keep the metadata snapshot stable until completion.

The operator does not allocate pages, copy shared tails, update reference
counts, or remap requests. Shared prefix pages remain valid read-only state.

The exclusive-page input changes the append contract. H20 append records from
2026-08-06 predate it and require replacement.

## Attention plan selection

The current plans use source-defined selection rules.

| Operator | Current rule | Published Graph boundary |
| --- | --- | --- |
| Single decode | Direct plan for the admitted MHA, MQA, and GQA contract. Split-K is a separate explicit plan | None |
| Paged batch decode | Direct for MHA. Eight-warp token parallelism for MQA and GQA | None |
| Ragged prefill | Direct below average KV length 64. Sixteen warps for long MQA. Eight warps for other long shapes. Tiled split-eight for GQA group size four at average KV length 256 or greater | Tiled long GQA4 only |
| Paged prefill | The caller selects direct, eight-warp, or sixteen-warp execution. Contract checks reject unsupported combinations | One historical direct GQA4 fixture |
| Fused append | One validator builds a compact `AppendMap`; one fused kernel consumes it. A scope may reuse the map only with the same K/V cache binding | Requalification after ownership and status changes |

Evidence limits are narrower than the source surface:

- The single-decode split-K H20 matrix covers MQA and GQA, not MHA.
- Ragged Graph evidence covers only the tiled long-GQA plan.
- Paged-prefill Graph evidence covers one recorded direct GQA4 fixture.
- The 2026-08-07 token-parallel records cover long MQA and long GQA4 at source `8478ee9` only. They predate the DeviceRegion path.
- Ragged selection uses the batch average KV length.
- Planning has no length grouping or persistent tuning database.

## Dynamic metadata

Host references validate page tables and `indptr` arrays before execution.
CUDA plans bind device-resident metadata, whose values can change between
invocations.

Fused append uses a checked status path:

- One validator checks the page table, token mapping, duplicate slots, and
  target-page ownership.
- The validator fully overwrites one status packet and the compact append map.
- One workspace may produce only one map in a command scope. This prevents a
  later validator from replacing an earlier map or status packet.
- The map records the exact writable K/V binding. A mapped append rejects a
  different cache before CUDA submission.
- Mapped append kernels read the compact map and skip all writes after a
  rejection.
- Completion returns a typed `ContractError` and the checked bindings.
- Semantic rejection does not poison the queue or Graph. CUDA failure and a
  malformed status packet do.

Paged decode and paged prefill use the same completion protocol. Each plan
launches one metadata validator, one attention kernel, and one status copy.
The validator gates all metadata-dependent pointer arithmetic in the attention
kernel.

## CUDA Graph contract

The current Graph implementation uses fixed addresses and host management.
`GraphQueue` owns a private non-default capture stream. Capture consumes the
queue and transfers bindings, functions, and vendor plans into
`CapturedGraph`. Instantiation transfers them into one non-`Send`, non-`Sync`
`GraphExec`.

Each replay takes unique mutable access and records one completion event
outside the graph. `GraphExec` retains in-flight state, so safe code cannot
launch again or release resources before settlement.

The current contract rejects rebinding, graph updates, cross-stream launch,
concurrent replay, and default-stream capture. Passing this contract does not
qualify mutable metadata or a different operator plan.

Historical published Graph evidence covers:

- one RMSNorm-to-BF16-GEMM chain.
- one tiled long-GQA ragged-prefill plan.
- one direct paged-prefill GQA4 fixture.

All three records predate the DeviceRegion submission path. Fused append also
predates the exclusive-page contract.

Current-source Graph evidence covers the experimental Loom SM90a M=1 GEMV on
five census shapes. It does not qualify performance, mutable bindings, other
GEMM contracts, or other providers. No other current-source Graph record is
published yet.

Benchmark `graph_nodes` fields are source declarations. The tools do not query
the CUDA driver for instantiated node counts. Use command count for lifecycle
checks, and treat node count as unverified until the tool enumerates the Graph.

## Model-runner boundary

Mistral.rs is the first experimental paired-repository integration. Historical
H20 records show one Qwen path with completed Loom provider calls, matching
selected token strings, and no adapter-issued device-to-device copy.

Those records bind older Loom and Mistral.rs commits. They do not qualify the
current source, full-model zero-copy execution, production safety, or
performance. The [integration document](../integrations/mistralrs.md) owns the
exact source pairs and excluded claims.

## Device code

Rust device modules keep unsafe memory access and CUDA intrinsics inside the
kernel boundary. Host validation must establish every static span and alignment
used by unchecked access.

Attention kernels use BF16 storage and F32 score, softmax, partial-state, and
merge arithmetic. Direct plans use online softmax. Split plans write
unnormalized F32 `[max_log2, normalizer, weighted_value]` states and merge them
with stable base-two weights.

The first admitted attention contracts fix NHD layout and head dimension 128.
Paged contracts also fix page size 16. Ragged and paged prefill use
bottom-right causal alignment.

The paged-prefill source includes block-local token partitioning for long
contexts. Long MQA uses sixteen warps. The admitted long GQA4 shape uses eight
warps.

Each warp computes an F32 online-softmax partial. Warp zero then merges the
shared states without caller-owned workspace.

The 2026-08-07 correctness and performance records describe source `8478ee9`.
They do not qualify the merged DeviceRegion submission path.

The first RoPE contract fixes BF16 NHD D128 NeoX split-half rotation, scale
one, theta 10,000, and explicit I32 positions. Loom full-math and FlashInfer
fast-math paths use independent references and a shared error limit. Their
bits need not match.

## GEMM provider architecture

Dense GEMM has one contract and one execution lifecycle. Providers do not
expose parallel public APIs.

Target provider set:

```text
Bf16DenseGemmSpec
  -> Bf16DenseGemmSelection
  -> GemmPlanner::plan_bf16_dense
  -> Bf16DenseGemmPlan
  -> Bf16DenseGemmOperands
  -> CommandScope
  -> Completion
```

The selection fixes the `Provider` and requested `Algorithm`. Plan information
reports their stable identities after planning.

The current `GemmPlanner` accepts explicit `CublasLt` and `Loom` selections.
Plan information reports the selected provider and frozen algorithm. The
private providers compute contiguous row-major BF16
`D[M,N] = A[M,K] * W[N,K]^T` with F32 accumulation. Both use the same checked
operands, command scope, completion, and Graph lifecycle.

`CublasLt` selects `CublasLtHeuristic`. `Loom` selects the experimental
`LoomSm90SimtGemvM1N16K64` only for its exact H20 `sm_90a` contract. The Loom
module loads lazily on the first explicit Loom plan. Unsupported contracts fail
during planning, and enqueue never changes providers.

The cuBLASLt provider remains the explicit general baseline. `GemmPlanner`
still creates the cuBLASLt provider when it loads, even for a later Loom-only
request. Provider-specific planner construction remains future work. This
eager construction does not create an enqueue-time fallback.

Dense BF16, dense FP8, grouped BF16, and grouped FP8 use separate `Spec` and
`Plan` types to preserve scale, grouping, layout, and epilogue semantics.

## Evidence model

Loom separates four kinds of facts:

| Layer | What it proves | What it does not prove |
| --- | --- | --- |
| Host contract | Rust validation, reference semantics, and recoverable errors | CUDA behavior |
| Device correctness | One GPU provider matches its oracle on declared cases | Graph replay or speed |
| Graph correctness | One captured plan replays with its declared binding policy | Mutable graphs or lower latency |
| Performance | One timed boundary on recorded hardware and inputs | Engine or serving improvement |

Lifecycle and Compute Sanitizer gates attach to the relevant device or Graph
layer. Engine evidence requires a real invocation, provider hit count, no-copy
proof, and model output. Serving evidence requires a workload plus TTFT, TPOT,
throughput, and memory.

The [evidence index](../results/README.md) records current limitations and
historical results. Passing one layer never implies that a later layer passes.

## State sources

Each repository surface answers one question:

| Source | Authoritative fact |
| --- | --- |
| Rust source and manifests | Which crates, contracts, providers, algorithms, and checks exist |
| `docs/operator-catalog.md` | Human-readable projection of the admitted source surface |
| `docs/results` | Immutable qualification for the exact recorded source and environment |
| Integration documents | External source pairs, adapter boundary, and excluded claims |
| `docs/roadmap.md` | Planned order and exit conditions only |
| Design documents | Target boundaries and naming rules only |

Source existence does not create evidence. Historical evidence does not
qualify changed source. A roadmap item does not upgrade the operator catalog.
If two surfaces disagree, correct the projection and preserve the immutable
record.
