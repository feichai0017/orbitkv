# Loom Infer architecture

Loom Infer provides Rust-native GPU operators for LLM inference. Model engines
keep model graphs, request scheduling, KV-cache policy, and serving APIs.

## Parity target

FlashInfer defines the pinned functional comparison surface. Loom implements
matching operator contracts where they fit the project boundary, but it does
not copy FlashInfer's Python API, source tree, or implementation language.

Parity is evaluated contract by contract. A matching contract includes tensor
shapes, dtypes, layouts, numerical behavior, aliasing, streams, workspaces, and
CUDA Graph semantics. A domain is not complete while Loom supports only a
narrower contract or has not passed its declared evidence gates.

## Source boundary

Loom-owned product code is Rust. Custom GPU kernels compile with
`cuda-oxide`. Vendor libraries remain binary dependencies called through
audited Rust FFI.

The core product has no Python API, CUDA C++ source, framework dispatcher, or
silent fallback path.

## Crates

```text
consumer engine
  -> loom-infer-cuda
       -> loom-infer
       -> Rust device kernel | explicit vendor provider
       -> caller-owned CUDA stream

loom-infer-validation
  -> loom-infer-cuda
  -> loom-infer
```

| Crate | Responsibility |
| --- | --- |
| `loom-infer` | Public specifications, errors, capabilities, and CPU references |
| `loom-infer-cuda` | Plans, CUDA execution, Rust device kernels, and explicit vendor calls |
| `loom-infer-validation` | Non-published hardware runners, fixtures, comparisons, and reporting |

Consumers may use `loom-infer` alone for contracts and references.
`loom-infer-cuda` depends on those contracts and exposes the CUDA execution
path. The validation crate depends on both product crates, but neither product
crate depends on validation. The contract crate never depends on a device
backend.

The workspace adds another crate only when a working vertical slice needs a
separate ownership or safety boundary.

## Module strategy

Loom follows Rust visibility and ownership boundaries rather than FlashInfer's
Python package and wrapper hierarchy. Public operator modules are facades;
private files follow complete operator domains:

```text
loom-infer::attention
  -> single_decode     contiguous decode and split-K reference state
  -> paged_decode      page-table contract and paged CPU reference
  -> ragged_prefill    indptr contract and ragged causal CPU reference

loom-infer-cuda::attention
  -> decode            one decode provider domain and cuda-oxide artifact bundle
  -> prefill           one prefill provider domain and cuda-oxide artifact bundle
```

FlashInfer still defines the semantic domains and acceptance surface. Loom maps
`BatchDecodeWithPagedKVCacheWrapper` planning to an immutable Rust plan, page
utilities to a validated borrowed page-table view, and execution resources to
the shared command/graph ownership layer. Python wrapper state is not copied
into product architecture.

Future MLA and KV-cache mutation become sibling private modules while the
public `attention::*` paths remain stable. A new crate requires a real
dependency or ownership boundary, not merely another attention algorithm.

## Operator lifecycle

The current RMSNorm, BF16 GEMM, decode, and ragged prefill attention device
slices implement this lifecycle:

```text
validated specification
  -> immutable plan
  -> checked bindings
  -> command scope
  -> enqueue one or more plans
  -> completion
```

The specification defines shapes, dtypes, layouts, aliasing, and numerical
behavior. The plan fixes the provider, algorithm, launch configuration,
and device artifact. Checked bindings validate the CUDA context, and the launch
contract validates buffer spans before enqueue.

### Command resources

`CommandQueue` owns one exact caller stream, one preallocated event, and fixed
resource-retention storage. One heterogeneous binding arena holds F32, FP16,
BF16, and byte buffers. Read-only bindings retain `Arc<DeviceBuffer<T>>`.
Read-write bindings own `DeviceBuffer<T>` until completion returns or destroys
the buffer after quiescence.

Typed handles preserve each element type and cannot move between binding sets.

One command scope submits several plans to the same stream. CUDA stream order
makes each output available to the next launch without a host wait.

`finish()` records the queue event once. The completion retains bindings, loaded kernel
functions, and external provider plans until `wait()` or destruction confirms
completion.

### Graph resources

Loom implements the graph path on the host. The first fixed-address contract
passed H20 correctness and sanitizer gates on 2026-08-03. `GraphQueue` owns a
private non-default stream and synchronizes the context once to establish input
readiness. It then begins thread-local capture.

Capture consumes the one-shot queue. The private handle and consuming API stop
safe caller code from injecting untracked nodes. They also prevent recapture of
a stream retained by an earlier graph.

Capture transfers fixed bindings, loaded CUDA functions, and vendor plans into
`CapturedGraph`.
Instantiation transfers them again into one non-`Send`, non-`Sync` `GraphExec`.

Each replay takes unique mutable access to `GraphExec` and records one event
outside the graph. `GraphExec` also stores persistent in-flight state, so a
forgotten completion cannot permit another launch or release allocations.

Leaking the graph also leaks retained `Arc` reads and owned writes. The GPU does
not lose its resources or expose writable access while work is in flight.

Drop settles any replay before it destroys graph handles, provider resources,
and bindings. The first contract rejects rebinding, graph updates, cross-stream
launch, and default-stream capture.

### Failure handling

Contract errors are recoverable. CUDA driver failures and external provider
submission failures poison the scope and queue.

Cleanup tries stream
synchronization, then context synchronization. If neither can prove
quiescence, the process aborts before Rust releases any retained GPU resource.

### Enqueue admission

The pinned cuda-oxide launcher creates a kernel-argument `Vec` during enqueue.
The queue prepares its event and function storage before launch, but full
allocation-free admission remains open.

## Device code

Production builds use pinned PTX or cubin artifacts for each supported GPU
architecture. Artifact preparation occurs before the execution path. The
runtime records the source revision, toolchain, target architecture, and
artifact hash.

Rust device modules keep unsafe code local to audited memory access and CUDA
intrinsics. Host-side validation must establish each bound used by unchecked
device access.

The FP16 and BF16 RMSNorm plans select scalar access for odd widths. Even widths
use two-element loads and stores after a four-byte alignment check. Both paths
use F32 arithmetic and round the stored result to nearest-even.

The first attention plan fixes BF16, NHD caches, head dimension 128, and one
warp per query head. It computes scores and online softmax state in F32. The
kernel writes BF16 output and F32 log2-LSE.

The split-K plan divides KV into balanced, non-empty ranges. Its partial kernel
writes one F32 `[max_log2, normalizer, weighted_value[128]]` state per query
head and partition. A second kernel merges those states and writes the same
BF16 output and F32 log2-LSE contract. Both kernels enter one checked command
scope with caller-owned workspace.

The merge kernel uses eight warps to merge contiguous partition ranges into
block-local shared states, then warp zero merges those eight states. CUPTI
activity timing records a KV-length-4096 merge reduction from `20.192` to
`5.056` microseconds.

The backend-independent paged batch-decode contract separates immutable tensor
shape from per-invocation page metadata. The specification fixes BF16, NHD,
D128, page size 16, batch size, head mapping, and page-pool capacity. A
validated view checks `i32` `indptr`, physical page indices, and last-page
lengths before exposing logical-token mapping. This lets immutable CUDA plans
retain fixed page-pool addresses while each decode step updates only
the page-table buffers.

The first paged CUDA plan is now admitted on H20. One warp handles one request
and query-head pair with packed BF16x2 access and F32 online softmax. Host
preflight checks every fixed buffer span. Device-resident page-table content
cannot be synchronously inspected without a copy, so valid metadata is the
numerical precondition and request-local device guards prevent invalid dynamic
values from producing out-of-bounds K/V access. The provider does not yet
report asynchronous metadata-content errors.

The grouped-head plan now partitions each request-head KV range across eight
warps. Every warp writes one F32 `[max_log2, normalizer, weighted_value]`
state to 4,192 bytes of block-local shared memory, then warp zero performs the
stable merge. MHA retains the lower-overhead direct warp.

The backend-independent ragged prefill contract fixes BF16, NHD, D128,
separate query/KV `indptr` arrays, and bottom-right causal alignment. Its first
CUDA plan assigns one warp to each query-row and query-head pair. The warp
scans query `indptr` to identify the request, then scans the causal KV prefix
with packed BF16x2 access and F32 online softmax. This direct schedule is
admitted for correctness and memory safety only; row-to-request preprocessing,
tiling, and matched performance remain open.

CUPTI activity records Loom kernel medians of `2.176`, `4.864`, and `4.928`
microseconds for MHA, MQA, and GQA, versus direct medians of `2.176`,
`20.928`, and `18.304` microseconds. The current stable-shape eager result puts
Loom 4.41x lower-latency for MHA and 2.35x lower-latency for MQA than the
pinned FlashInfer path.

Relative to the recorded direct baseline, the current matched result lowers
Loom latency by 5.39x at GQA KV length 127 and 38.19x at KV length 4096.
FlashInfer remains 1.17x and 2.09x lower-latency. Hardware-counter metrics and
Graph performance remain separate open gates.

## Vendor providers

Loom Infer includes GEMM and communication in its planning surface. Vendor
providers use qualified library implementations unless a measured Loom
implementation wins on the same contract.

A vendor plan fixes the library, algorithm, layouts, packed weights, scales,
epilogue, workspace, and graph policy. Provider selection never changes during
enqueue.

The first provider fixes one cuBLASLt algorithm for contiguous row-major BF16
`D[M,N] = A[M,K] * W[N,K]^T` with F32 accumulation. It checks exact spans,
16-byte tensor alignment, 256-byte workspace alignment, CUDA context, and the
algorithm's actual workspace requirement. Planning allows up to 32 MiB.
Enqueue has no tuning or fallback.

## Evidence boundary

An operator advances through independent gates:

1. Contract and CPU reference.
2. Device correctness.
3. Command lifecycle and non-default-stream behavior.
4. Compute Sanitizer.
5. Matched kernel performance in both provider orders.
6. CUDA Graph capture and replay.
7. Real engine invocation and model output.
8. Serving latency, throughput, and memory.

Passing one gate does not imply that a later gate passes.
