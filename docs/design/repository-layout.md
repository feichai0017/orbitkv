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
| `src/attention.rs` | Admitted attention specifications and CPU references |
| `src/*/tests.rs` | Contract, exact-span, mapping, and numerical tests |

This crate contains no CUDA types, FFI, provider selection, launch
configuration, or engine policy. A new GPU provider starts with a contract and
reference here unless an established independent oracle is explicitly recorded.

When one operator family gains several independent contracts, convert its
single file into a directory while preserving the public module path. For
example, paged decode, prefill, and state merge can become
`attention/{decode,prefill,state}.rs` without creating another crate.

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
| `src/attention.rs` | Rust device kernel plus immutable single-decode plan |
| `src/gemm.rs` | Explicit fixed-algorithm cuBLASLt provider |

Operator files currently keep their kernel and host plan together because each
implemented slice is narrow. Split an operator into private `kernels`, `plan`,
and `args` modules when it has multiple independently planned algorithms or the
single file obscures the host/device safety proof. Preserve the existing public
provider and plan paths unless a deliberate API revision requires otherwise.

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
