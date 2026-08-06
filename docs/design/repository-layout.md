# Repository layout

Loom Infer keeps the repository small until a complete operator slice requires
a new ownership or safety boundary. Directory growth follows functionality,
not a one-module-per-concept template.

## Top level

```text
loom-infer/
|-- crates/
|   |-- loom-infer/             backend-independent contracts and references
|   |-- loom-infer-cuda/        CUDA runtime, plans, providers, and Rust kernels
|   `-- loom-infer-validation/
|                               non-published hardware validation programs
|-- docs/
|   |-- design/              architecture and repository decisions
|   |-- development/         hardware validation procedures
|   `-- results/             immutable machine-readable evidence
|-- website/                 documentation site; never a product runtime
`-- .github/                 CPU-only CI, policy checks, and Pages deployment
```

The product dependency direction is:

```text
consumer engine
  -> loom-infer-cuda
       -> loom-infer
       -> cuda-oxide runtime and Rust device artifacts
       -> explicit vendor providers

loom-infer-validation
  -> loom-infer-cuda
  -> loom-infer
```

`loom-infer` remains usable without CUDA. Product crates do not depend on the
validation crate, website, or result records. The validation crate is a
workspace member but not a default member.

## Contract crate

`crates/loom-infer` owns behavior that every backend must agree on:

| Path | Responsibility |
| --- | --- |
| `src/lib.rs` | Stable re-exports for admitted contracts |
| `src/dtype.rs` | Backend-independent storage types |
| `src/error.rs` | Recoverable contract and host-buffer errors |
| `src/rms_norm.rs` | RMSNorm specification and CPU references |
| `src/gemm.rs` | Contiguous BF16 GEMM specification and CPU reference |
| `src/attention.rs` | Stable attention facade and public re-exports |
| `src/attention/single_decode.rs` | Contiguous decode, split-K state, and CPU references |
| `src/attention/paged_decode.rs` | Paged decode, page-table validation, and CPU reference |
| `src/*/tests.rs` | Contract, exact-span, mapping, and numerical tests |

This crate contains no CUDA types, FFI, provider selection, launch
configuration, or engine policy. A new GPU provider starts with a contract and
reference here unless an established independent oracle is explicitly recorded.

Attention is the first family converted to a facade plus private domain
modules. Future prefill, MLA, and KV mutation contracts extend
`attention/{prefill,mla,kv}.rs`; they do not create another crate or expose the
private directory structure as API.

## CUDA crate

`crates/loom-infer-cuda` owns the device boundary:

| Path | Responsibility |
| --- | --- |
| `src/command/mod.rs` | Stable command API facade and shared errors |
| `src/command/binding.rs` | Typed ownership, leases, and opaque handles |
| `src/command/resolve.rs` | Disjoint operand resolution and alias rejection |
| `src/command/submission.rs` | Queue admission, command retention, and capture transfer |
| `src/command/completion.rs` | Completion fencing, settlement, and quiescence fallback |
| `src/graph.rs` | Fixed-address CUDA Graph capture, replay, and retained resources |
| `src/driver.rs` | Small raw-driver cleanup helpers |
| `src/rms_norm.rs` | Rust device kernels plus immutable RMSNorm plans |
| `src/attention.rs` | Stable CUDA attention facade |
| `src/attention/decode.rs` | Single and paged decode vertical slice and cuda-oxide artifact bundle |
| `src/gemm.rs` | Explicit fixed-algorithm cuBLASLt provider |

The cuda-oxide `#[cuda_module]` macro must discover one inline kernel bundle.
Do not split one bundle across file-backed modules merely to shorten a file.
Split CUDA attention by complete vertical domains instead:
`attention/{decode,prefill,mla,kv}.rs`. Each domain may own kernels, typed
plans, arguments, and errors when those pieces share one artifact and safety
proof. The facade preserves `loom_infer_cuda::attention::*`.

The command facade preserves `loom_infer_cuda::command::*` while implementation
details stay in private modules. Add operand resolution patterns in
`resolve.rs`; keep queue poisoning and resource retention in `submission.rs`;
keep synchronization fallback and completion settlement in `completion.rs`.
Do not create a generic runtime abstraction before a second device backend
exists.

## Validation and evidence

`crates/loom-infer-validation` owns hardware correctness executables, shared
finite comparisons, stable digests, and machine-readable gate prefixes:

| Path | Responsibility |
| --- | --- |
| `src/bin/*_h20.rs` | Operator-specific H20 cases and acceptance limits |
| `src/comparison.rs` | Shared finite comparisons, bit mismatches, and digests |
| `src/reporting.rs` | Stable `gate/case/status` output prefixes |

Validation binaries exercise permanent providers against CPU references and
error contracts on real hardware. They are not benchmark or serving APIs.

`docs/results` contains immutable evidence tied to a source projection,
toolchain, artifact, device, and exact command matrix. Correctness records do
not imply performance. Performance records do not imply engine or serving
improvements.

Reusable benchmark harnesses may gain a dedicated non-product directory after
the first matched provider benchmark. Generated PTX, cubins, profiler captures,
model weights, and build output remain outside version control.

## Adding a vertical slice

Add functionality in this order:

1. Pin the matching FlashInfer contract and declare admitted differences.
2. Add the backend-independent specification, errors, and reference behavior.
3. Add an immutable provider plan and checked enqueue path.
4. Extend typed binding resolution only for the required access pattern.
5. Add CPU tests and a hardware validation program.
6. Record correctness, sanitizer, performance, Graph, and engine evidence as
   separate gates.
7. Update the parity matrix, operator catalog, roadmap, and website only for
   behavior that the source and evidence support.

Add a crate only for a real dependency, ownership, or safety boundary. A larger
operator family by itself is a module-layout concern, not a crate boundary.

## FlashInfer mapping

FlashInfer defines the feature domains, not Loom's language-level layout:

| FlashInfer concept | Loom Rust location |
| --- | --- |
| Single decode and state merge | `loom-infer/attention/single_decode.rs` |
| Paged decode and page-table semantics | `loom-infer/attention/paged_decode.rs` |
| CUDA decode providers and dispatch | `loom-infer-cuda/attention/decode.rs` |
| Wrapper planning lifecycle | Immutable Rust plan types, not Python wrapper classes |
| Workspace and stream ownership | Shared `command` and `graph` modules |
| Hardware tests and benchmarks | `loom-infer-validation`, never product modules |

This mapping preserves functional parity while keeping Rust ownership,
visibility, error handling, and dependency direction idiomatic.
