# Loom Infer architecture

Loom Infer provides Rust-native GPU operators for LLM inference. Model engines
keep model graphs, request scheduling, KV-cache policy, and serving APIs.

## Source boundary

Loom-owned product code is Rust. Custom GPU kernels compile with
`cuda-oxide`. Vendor libraries remain binary dependencies called through
audited Rust FFI.

The core product has no Python API, CUDA C++ source, framework dispatcher, or
silent fallback path.

## Crates

```text
consumer engine
  -> loom-infer
  -> loom-infer-cuda
  -> Rust device kernel | explicit vendor provider
  -> caller-owned CUDA stream
```

| Crate | Responsibility |
| --- | --- |
| `loom-infer` | Public specifications, errors, capabilities, and CPU references |
| `loom-infer-cuda` | Plans, CUDA execution, Rust device kernels, and explicit vendor calls |

The workspace adds another crate only when a working vertical slice needs a
separate ownership or safety boundary.

## Operator lifecycle

The current RMSNorm and BF16 GEMM slices implement this lifecycle:

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
