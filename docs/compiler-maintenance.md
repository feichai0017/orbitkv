# Compiler ownership and maintenance

OrbitKV owns its model compiler and CUDA runtime in the root Cargo workspace.
There is one repository revision, one lockfile, and one CI pipeline for state
contracts, graph compilation, providers and serving. The compiler is linked
into the inference process through local workspace dependencies.

## Source origin

The compiler was imported from
[feichai0017/orbitkv-luminal](https://github.com/feichai0017/orbitkv-luminal)
at commit `c15b50a2daad961461e533a2ac44c25e1700119d`, derived from
[luminal-ai/luminal](https://github.com/luminal-ai/luminal).
The previous parent revision was `d96486066bf0cbafa72fe864c1b4d6d225e58a8f`.
Upstream history remains accessible in those repositories and the parent's
historical submodule pointer. Current development does not require a second
compiler repository or a submodule update.

| Former package | Owned package |
| --- | --- |
| `luminal` | `orbitkv-compiler` |
| `luminal_nn` | `orbitkv-ops` |
| `luminal_cuda_lite` | `orbitkv-cuda` |
| `luminal_tracing` | `orbitkv-tracing` |

All four imported crates retain the original `LICENSE-MIT` and
`LICENSE-APACHE` files. Renaming packages does not replace their attribution or
license terms. The root state-manager implementation retains its MIT license.

Pinned egglog, cudarc, and CUDA provider sources remain external dependencies;
their versions and source identities are independent of this repository move.

## Changes and upgrades

Make contract, compiler, executor and regression changes in one reviewable
commit series. Useful upstream changes may still be ported selectively, with
the originating repository/commit recorded in the change description. Validate
them as compiler changes; upstream claims do not establish OrbitKV correctness
or performance.

The compiler core builds search spaces; backend runtimes own candidate
selection, profiling and loading. Graph pattern matching, implementation
selection and fusion remain egglog rewrites. Rust lowering implements the
selected program. Providers declare applicability, resource ownership, layouts
and state effects. Model names and fixed checkpoint dimensions cannot select
implementations.

`orbitkv` retains sole lifecycle authority and has no dependency on the model
compiler or CUDA. `orbitkv-executor` binds state contracts to the model program;
compiler and CUDA crates do not depend on the request engine or KV manager.
The [layout checker](../tools/verify_active_source.py) enforces these edges.

## Migration compatibility

Package/import names, `ORBITKV_*` diagnostic/provider environment variables,
trace namespaces and default provider caches use the OrbitKV namespace.
Provider sources default to `~/.cache/orbitkv/providers`; explicit provider
checkout variables remain the preferred way to reuse prepared sources.
Source fetching is explicit and never occurs during model compilation.

Decoder schema 11 rejects earlier decoder artifacts before weight loading.
Recompile schedules and CUDA modules with the new workspace. The source and
provider cache identities also change; cached libraries are rebuilt as needed.
Historical `results/` records retain their original source, names and hashes.
They are evidence for their recorded revisions, not fresh qualification of the
renamed workspace.

Host checks run all default members; the CUDA crate is an explicit target.
The standard CI compiles CUDA integration and checks the backend library.
The manually dispatched device workflow needs a self-hosted CUDA runner.
Model qualification additionally requires a local checkpoint and independent
reference data; a successful compile check does not substitute for that gate.
