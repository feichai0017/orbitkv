# Development environment

Oxide Infer uses separate toolchains for ordinary Rust checks and CUDA device
compilation. Keep their versions explicit so a successful local run means the
same thing as CI or an H20 validation run.

## Version contract

| Tool | Version | Source of truth |
| --- | --- | --- |
| Rust host toolchain | `1.97.1` | `/rust-toolchain.toml` |
| Rust CUDA toolchain | `nightly-2026-04-03` | CUDA and validation crate toolchain files |
| cuda-oxide | revision `868f8ec4ef900bae7e67e7f9508b2da66eee5472` | `Cargo.toml` and `Cargo.lock` |
| cargo-oxide | `0.2.1` from the same cuda-oxide revision | installation command below |
| Node.js | `24.19.0` | `/mise.toml` and `/website/package.json` |
| npm | `11.17.x` | `/website/package.json` |
| CUDA toolkit | `13.1` for the qualified H20 row | H20 evidence records |
| Clang/libclang | `21` | cuda-oxide host binding requirement |
| Device target | `CUDA_ARCH` for generic CUDA tests; fixed `sm_90a` for every H20 correctness and benchmark target | Makefile and H20 validation contract |

The root Rust toolchain applies to CPU-only workspace commands. Entering
`crates/oxide-infer-cuda` or `crates/oxide-infer-lab` selects the pinned
nightly for cargo-oxide. Do not replace one with the other globally.

## Host setup

Install both Rust toolchains:

```bash
rustup toolchain install 1.97.1 --profile minimal \
  --component rustfmt --component clippy
rustup toolchain install nightly-2026-04-03 --profile minimal \
  --component rust-src --component rustc-dev --component rust-analyzer \
  --component clippy --component llvm-tools --component rustfmt
```

Install cargo-oxide from the revision used by the workspace:

```bash
cargo +nightly-2026-04-03 install --locked \
  --git https://github.com/NVlabs/cuda-oxide.git \
  --rev 868f8ec4ef900bae7e67e7f9508b2da66eee5472 \
  cargo-oxide
```

Install Node through mise, then install every website dependency even if the
calling shell sets `NODE_ENV=production`:

```bash
mise trust
mise install
USE_MISE=1 make install-website
```

Review `mise.toml` before trusting it. The committed configuration selects
Node 24.19.0 and prepends the Clang 21 binaries and libclang directory.

Make uses the current shell by default. It does not invoke an installed mise
automatically. Set `USE_MISE=1` only after you trust the repository config:

```bash
USE_MISE=1 make install-website
USE_MISE=1 make check
```

This opt-in prevents an untrusted mise config from blocking unrelated Make
targets. An activated mise shell also works without `USE_MISE=1`.

The website permits only the version-pinned esbuild script and the optional
macOS fsevents script declared in `package.json`. An unreviewed dependency
install script fails under npm's strict allow-scripts policy.

## CUDA host setup

The CUDA path requires:

- a compatible NVIDIA driver and CUDA toolkit.
- `clang-21`, `libclang-21-dev`, and Clang 21 resource headers.
- the pinned Rust nightly components above.
- Compute Sanitizer for sanitizer evidence.

On Debian or Ubuntu, install LLVM and Clang 21 from the LLVM package repository
when the distribution does not provide them. Make sure `/usr/lib/llvm-21/bin`
precedes older Clang installations and set:

```bash
export LIBCLANG_PATH=/usr/lib/llvm-21/lib
```

`mise activate` applies both settings from the repository's `mise.toml`.
cuda-oxide normally selects the `llc` shipped by the pinned Rust `llvm-tools`
component, which is LLVM 22.1.2 for this nightly.

Before a device run:

```bash
make cuda-doctor
```

The doctor output must report the pinned nightly, all required components,
Clang resource directory 21, LLVM `llc` 21 or newer, the CUDA toolkit, and the
target GPU.

## Validation entry points

```bash
make check          # CPU-only Rust checks, package verification, website, audit
make cuda-check     # CUDA-feature host compilation and Clippy
make cuda-test      # generic release tests; CUDA_ARCH selects the device target
make h20            # H20 correctness programs fixed to sm_90a
```

`make h20` runs device programs sequentially because they generate the same
workspace artifact and share one validation GPU. It is a correctness gate, not
a Compute Sanitizer or performance gate. Follow
[H20 validation](h20-validation.md) before recording evidence.

`H20_ARCH` is fixed to `sm_90a` and ignores command-line overrides. Use
`CUDA_ARCH` only for generic `cuda-test` runs.

## Dependency caching

`Cargo.lock` and `website/package-lock.json` remain the dependency identities.
A validation host may use a Cargo vendor directory or an external cache when
Git transport is unreliable, but machine-specific source replacement paths do
not belong in the repository. The cached dependency set must resolve with
`--locked --offline` before it is accepted as equivalent.

Point the common commands at such an external Cargo home without changing the
repository:

```bash
OXIDE_CARGO_HOME=/path/to/validated/cargo-home make check
OXIDE_CARGO_HOME=/path/to/validated/cargo-home make cuda-test
```
