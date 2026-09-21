# Rust Quality Gates

The workspace Clippy configuration is intentionally opt-in: enable lints that
catch production risks or API-boundary drift without forcing broad mechanical
rewrites.

Run the same command used by the pre-commit hook before merging Rust changes:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

On CUDA 13 development machines, use
`--no-default-features --features cuda-13,mooncake`, as configured in `prek.toml`.

## Test Layout

Keep each crate's private unit tests in `tests/unit/`, mirroring the source
module tree. For example, `src/storage/read_cache.rs` loads
`tests/unit/storage/read_cache.rs` with:

```rust
#[cfg(test)]
#[path = "../../tests/unit/storage/read_cache.rs"]
mod tests;
```

These remain unit tests inside the owning module, so `use super::*` preserves
private access without expanding the production API. Test-only fixtures and
helper implementations belong in the same test tree. Public integration tests
stay in `tests/`, with shared fixtures in `tests/common/`. Cargo discovers the
existing integration targets and runs the unit tests through their library or
binary target; no additional test crate or runner is needed.

## Enabled Strict Lints

The first strict pass enables rules that are low-noise on the current codebase
and guard against common operational failures:

- `allow_attributes_without_reason`: any future lint suppression must explain
  why it is correct.
- `await_holding_lock` and `await_holding_refcell_ref`: async code must not hold
  synchronous locks or `RefCell` borrows across `.await`.
- `let_underscore_future`: spawned or constructed futures must be explicit.
- `mem_forget`: leaking `Drop` types must not be used as a lifetime shortcut.
- `unimplemented`: placeholder code must not enter the main branch.
- `unused_result_ok`: do not hide errors by converting them to `Option`.

## Deferred Inventory

The following candidates were audited but left out of the hard gate because
they require design judgement or generated-code handling:

| lint | unique warnings | decision |
| --- | ---: | --- |
| `indexing_slicing` | 137 | Needs invariant-by-invariant review in cache, RDMA, and tests. |
| `unwrap_used` | 86 | Production and test policy should be separated before enabling. |
| `undocumented_unsafe_blocks` | 51 | Worth enabling after a dedicated unsafe documentation pass. |
| `let_underscore_must_use` | 28 | Useful, but several intentional fire-and-forget sends need clear helpers or logging policy. |
| `map_err_ignore` | 25 | Useful once error messages are upgraded without obscuring call-site context. |
| `panic` | 13 | Tests and binary entrypoints need their own policy before production code. |
| `string_slice` | 6 | Should be reviewed with `indexing_slicing`. |
