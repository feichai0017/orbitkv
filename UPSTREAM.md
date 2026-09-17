# Upstream source inventory

## `pegainfer-project/kern`

The following directories are imported without modification from
[`pegainfer-project/kern`](https://github.com/pegainfer-project/kern) commit
`05df6d9cf8233b2438a7a584ce4ed7a0666abf53` (version `0.2.3`):

- `crates/kern-manifest`
- `crates/kern-pool`
- `crates/kern-runtime`
- `crates/kern-test`
- `crates/kern-run`
- `crates/kern-serve`
- selected `examples` fixtures used by the imported Qwen/manifest tests
- `schema`
- `docs/kern-upstream` (selected architecture and operation documents)
- `clippy.toml` and `rustfmt.toml`

The imported work is licensed under Apache-2.0. A verbatim copy is retained at
`licenses/kern-Apache-2.0.txt`. Subsequent local modifications to imported files
are part of OrbitKV Next and remain traceable from this baseline commit.

Large model-specific generation tools, captured binaries, website assets, and
historical experiment output were intentionally not bulk-imported. Individual
kernels or generators may be imported later with their exact upstream path,
revision, and applicable third-party notices.
