# Upstream provenance

OrbitKV's storage and transfer data plane was imported from
[Novita AI PegaFlow](https://github.com/novitalabs/pegaflow), version `0.24.5`,
commit `3bc9d7fd609695e34168c226bb72a907b4ca2eec` (2026-09-17).

The imported implementation was copied into the OrbitKV repository rather than
linked as a Git fork or vendored submodule. It was then renamed and modified:

- Rust crates and module paths use the `orbitkv-*` / `orbitkv_*` namespace.
- The Python distribution, package, entry points, and connector names use the
  `orbitkv` namespace.
- Protocol packages, configuration keys, metrics, scripts, examples, and
  documentation use the OrbitKV namespace.
- OrbitKV's existing planning, qualification, provider, and SGLang integration
  code remains in this repository and will be connected to the imported data
  plane incrementally.

PegaFlow is licensed under Apache License 2.0. The imported and modified code is
distributed under the same license; see [LICENSE](LICENSE). PegaFlow and Novita
AI are upstream sources and do not endorse this derivative project. The
pre-import OrbitKV planning components remain MIT licensed; their license text
is preserved in [LICENSES/OrbitKV-legacy-MIT.txt](LICENSES/OrbitKV-legacy-MIT.txt).
