# Code and test layout

All test source belongs in the owning crate's `tests/` directory. `src/`
contains production code and only the declarations needed to attach private unit
tests. Checkpoint names, GPU names, artifact versions, and experiment dates
belong in fixtures, benchmark manifests, or evidence records rather than
production module names.

```text
crates/<crate>/
  Cargo.toml
  src/
    lib.rs                         public entry and module declarations
    component.rs                   one responsibility
    component/child.rs             a cohesive part of that responsibility
    bin/<command>/main.rs          multi-file executable entry
  tests/
    unit/
      mod.rs                       crate-root private tests, if any
      support.rs                   shared unit-test helpers, if any
      component/
        mod.rs                     private tests of component
        lifecycle.rs               additional scenario groups
        fixtures.rs                helpers shared by those groups
        child/mod.rs               private tests of component::child
    contract.rs                    single-file public-API integration test
    execution/
      main.rs                      multi-file integration-test entry
      scenarios.rs                 scenario modules, not separate Cargo targets
    fixtures/                      checked-in test inputs
  examples/                        runnable usage examples and example plans
```

Production modules use `component.rs` with `component/` for children. A private
unit-test owner connects its suite with a test-only path declaration, for example
in `src/component.rs`:

```rust
#[cfg(test)]
#[path = "../tests/unit/component/mod.rs"]
mod tests;
```

This preserves the original `component::tests` namespace and access to private
implementation details. It does not create a public testing API or a separate
Cargo test target. Within `tests/unit/`, ordinary `mod` declarations organize
scenarios and helpers. Do not flatten Rust source with `include!`.
`include_str!` and `include_bytes!` remain appropriate for data fixtures and
generated-kernel source.

Keep helpers at the narrowest shared test ancestor. Crate-wide helpers belong in
`tests/unit/support.rs`. Small `#[cfg(test)]` observation methods may remain next
to the private implementation they inspect; test functions, suites, and helper
files live under `tests/`.

Integration tests use the crate's public API. Single-file suites use
`tests/<suite>.rs`; multi-file suites use `tests/<suite>/main.rs`, preserving the
Cargo test target `<suite>`. `tests/unit/` uses `mod.rs`, so Cargo does not discover
it as an independent integration binary. Model/device tests retain explicit
hardware requirements and opt-in execution; moving files does not widen support.

Shared checked-in test data lives in `tests/fixtures/`. Example plans stay in
`examples/`. Runtime downloads, model weights, generated schedules, logs, and
caches are not checked-in test fixtures.

| Directory | Contents |
| --- | --- |
| `crates/` | Seven owned crates in one Cargo workspace, each with a `tests/` tree |
| `tools/` and `tools/tests/` | Qualification/invariant scripts and Python tests |
| `benchmarks/` | Workload and tuning manifests |
| `docs/` | Architecture, contracts, and qualification boundaries |
| `website/` | Documentation site with its own toolchain layout |
| `.qualification/` | Local runs, frozen inputs, and large raw evidence |
| `results/` | Published model inference performance, with workload and source identities |

Compiler, operation, tracing and CUDA tests follow the same layout as the state
manager and engine. Inherited inline suites and `src/tests/` trees have moved
under their owning crate's `tests/`, preserving private test namespaces.

`python tools/verify_active_source.py` enforces all seven owned crates' layout,
checks that explicit unit-test bridges stay under `cfg(test)` and resolve within
`tests/unit/`, and rejects test files/functions in `src/`. It also checks
repository dependency direction, generic filenames, and file-size limits.
Inherited large files are listed in `tools/source-size-baseline.json`; their
limits may only decrease. Extract cohesive responsibilities and remove entries
when they reach the normal limit. Generated protobuf code is identified by its
generator declaration. Compilation and behavioral tests establish correctness.

The default Cargo members include all host crates. CUDA device tests are
explicit: `cargo test -p orbitkv-cuda` requires a CUDA toolchain and GPU.
`cargo fmt --all` checks every workspace member. There are no nested workspaces,
crate-local lockfiles, or compiler submodules.

CUDA provider integration follows the same ownership layout: Rust adapters,
`.egg` rules and `.cu`/`.cuh` sources sit under their owning component. Templates
requiring Rust interpolation use `.egg.in`/`.cu.in`. Central source provenance is
in `crates/orbitkv-cuda/providers.lock.json`; native build/cache code is shared.
See [CUDA backend](cuda-backend.md) for the directory map.
