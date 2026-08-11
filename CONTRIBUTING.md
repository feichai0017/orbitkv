# Contributing to Oxide Infer

Each change must implement or improve one measurable inference path.

Oxide Infer's product target is a high-performance, FlashInfer-class operator
layer for Rust inference engines. CUDA is the execution platform. cuda-oxide is
the Rust compiler and artifact toolchain for Oxide-owned device kernels.

## Before implementation

Record:

- the model or engine call site
- shapes, dtypes, layouts, alignment, aliasing, and stream behavior
- the exact baseline
- the numerical contract
- the acceptance and stop conditions.

Vendor libraries should own plain GEMM and collectives. Oxide Infer may own
their Rust plans, layouts, epilogues, and measured fusion boundaries.

## Source rules

- Product source is Rust.
- Do not add Python product APIs, CUDA C or CUDA C++ product source,
  compatibility aliases, duplicate execution paths, or silent fallback.
- Keep `oxide-infer` independent from CUDA and FFI.
- Keep unsafe code inside the device or vendor boundary that requires it.
- Add a crate only when a complete vertical slice needs that boundary.
- Keep measurement tools outside library binaries.

Every operator follows one execution pattern:

```text
validated spec -> immutable plan -> checked buffers -> enqueue
```

## Required checks

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo check --workspace --release
cargo package -p oxide-infer
```

A device change must also pass its documented GPU correctness matrix on a
non-default stream. Add graph, lifetime, and matched performance gates when the
operator exposes those capabilities. Report unavailable tools instead of
claiming their coverage.

## Pull requests

Keep a pull request to one operator or runtime slice. Include exact commands,
tolerances, failures, and machine-readable evidence for performance claims.
State whether each result applies to an operator, graph, engine, or serving
workload.

Do not commit toolchains, virtual environments, build output, model weights,
or profiler captures.
