# Python releases

OrbitKV 0.1.0 is being prepared for its first Python release. Build from source
until a release is published. This page describes the artifact contract and
how to validate a candidate without uploading it to PyPI.

## Packages

| Distribution | Runtime | Release targets |
| --- | --- | --- |
| `orbitkv-llm` | CUDA 12 | Linux x86_64, CPython 3.10–3.14 |
| `orbitkv-llm-cu13` | CUDA 13 | Linux x86_64 and aarch64, CPython 3.10–3.14 |

Install one CUDA variant per environment; both provide the `orbitkv` import and
`orbitkv-cache-manager` command. Engine extras pin vLLM 0.29.0 or SGLang 0.5.20.
Use separate engine environments. Build-matrix coverage does not establish
that every engine/GPU combination works on every Python version; serving
qualification currently uses Python 3.11 on H20.

The wheel contains the PyO3 extension, Manager executable, Mooncake libraries,
engine plugins, type stubs, Apache-2.0 license and third-party attribution in
`NOTICE`. The host supplies its NVIDIA
driver and compatible PyTorch/CUDA environment. Tests, benchmarks and build
caches are excluded from the package.

GPU ANS encoding additionally needs the separately installed nvCOMP 5.3 runtime;
it is not bundled in the wheel. FP8 and TurboQuant kernels are compiled by NVRTC
in the Manager. All storage codecs are opt-in; see [storage formats](storage-formats.md)
for dependencies, model-quality results and GPU workspace limits.

## Build a candidate

From the repository root, select the Python interpreter and CUDA variant:

```bash
git submodule update --init --recursive third-party/mooncake
PYO3_PYTHON=/absolute/path/to/python3.11 \
  ./scripts/build-wheel.sh --release --no-default-features --features cuda-13,mooncake
```

Omit the feature arguments for CUDA 12. The script builds and stages the
Manager and Mooncake libraries, builds the extension for the selected Python,
checks the package version/content, and imports the installed wheel in a fresh
non-editable environment. Wheels are written to `target/wheels/`.
Do not run native builds while a source-built Manager is using the staged
Mooncake libraries. `maturin build` alone does not stage the complete runtime.

## Validate before publishing

1. Run `python3 scripts/check-versions.py --tag v0.1.0`. Rust, Python,
   Commitizen and the proposed tag must agree.
2. Trigger the **Release** workflow manually on the candidate branch. It builds
   all 15 wheel targets and validates their versions and contents. Manual runs
   upload Actions artifacts only; they do not create a release or publish to PyPI.
3. Install a candidate wheel into each engine environment. From outside the
   source checkout, check the installed import path, Manager help and health,
   then verify a completion and an external restore after engine restart.
   Run the [release smoke and correctness gates](../python/tests/README.md#release-smoke).
4. Review the final benchmark summaries and known limits. Publish only after
   these checks pass and the release is approved.

Pushing a matching `v*` tag runs the same build/validation jobs and then publishes
GitHub release assets and both PyPI distributions. The workflow requires the
repository's `PYPI_API_TOKEN` to authorize both names. Preparing a candidate or
running the manual workflow does not test that credential or reserve the names.

## Scope of 0.1.0

The initial release targets single-node DRAM/SSD cache recovery with vLLM and
SGLang. Supported hybrid layouts use compiled range selection and leased-state
validation. Cancellation, lost notifications and engine/Manager restart have
native and model-serving gates.

Request preparation and read cutoffs remain opt-in. The repeated Qwen3-8B
controls show a throughput/latency tradeoff on SGLang; see
[the policy results](request-preparation.md#measured-results).
Distributed serving, catalog HA, automatic hybrid lookahead and cross-host TP
remain outside the qualified single-node release scope. See
[deployment support](deployment.md) and [the roadmap](roadmap.md).
