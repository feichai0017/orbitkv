# OrbitKV contributor guide

All seven crates belong to the root Cargo workspace. Internal dependencies use
workspace bindings. The state manager remains independent of the compiler,
CUDA, and request engine. Compiler/CUDA crates consume execution contracts and
never acquire page-lifecycle authority.

## Compiler boundary

The compiler core ends at egglog saturation: `Graph::build_search_space`
produces a `SearchSpace`. Backend `Runtime::compile` implementations own
candidate selection, profiling, and loading. Core search utilities may be
composed by a backend; they do not impose its strategy.

All graph pattern matching and implementation selection must be expressed in
egglog rewrites. Do not add Rust LLIR post-passes that search for patterns,
fuse kernels, select providers, or rewrite extracted graphs. Lowering emits
and executes the selected program.

## Correctness and tests

Keep all test source under the owning crate's `tests/` directory. Private unit
suites attach through `#[cfg(test)]` path declarations into `tests/unit/`.
Use sibling `component.rs` entries for production modules. See
`../docs/code-layout.md` for the layout and inherited size-debt policy.

Fix incorrect results at the violated graph, shape, dtype, layout, alias,
provider or runtime contract. Do not change model semantics, reference inputs,
or tolerance gates to conceal a compiler bug. Every admitted candidate must
preserve its operation semantics; numerical disagreement is an equivalence
bug, not a reason to search around it. Add independent reference regressions at
the affected boundary and use reproducible seeds for randomized cases.

Run formatting, Clippy, host tests and the relevant CUDA/device qualification.
Host checks must remain usable without CUDA. Keep source provenance and the
original licenses of the Luminal-derived compiler crates; see
`../docs/compiler-maintenance.md`.
