<div align="center">
  <h1>Loom Infer</h1>
  <p><strong>Rust-native CUDA operators for LLM inference.</strong></p>
  <p>Checked execution, cuda-oxide kernels, vendor GEMM, and bounded evidence.</p>
  <p>
    <a href="https://github.com/feichai0017/loom-infer/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/feichai0017/loom-infer/actions/workflows/ci.yml/badge.svg"></a>
    <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-d6ff63"></a>
    <img alt="Rust 2024" src="https://img.shields.io/badge/Rust-2024-ff9a68">
    <img alt="CUDA SM90" src="https://img.shields.io/badge/CUDA-SM90-82adff">
  </p>
  <p>
    <a href="docs/README.md">Docs</a> ·
    <a href="docs/operator-catalog.md">Operators</a> ·
    <a href="docs/flashinfer-parity.md">FlashInfer parity</a> ·
    <a href="docs/results/README.md">Evidence</a> ·
    <a href="docs/roadmap.md">Roadmap</a>
  </p>
</div>

Loom Infer is a FlashInfer-class operator layer for Rust inference engines.
The public contracts, plans, resource bindings, and Loom-owned kernels use
Rust. [cuda-oxide](https://github.com/NVlabs/cuda-oxide) compiles the device
code for NVIDIA GPUs. Qualified vendor libraries provide matrix operations.

Loom Infer does not implement a model server. Engines retain model execution,
request scheduling, batching, KV allocation policy, and distributed control.

## Execution model

Every provider uses one lifecycle:

```text
contract
  → immutable plan
  → checked bindings
  → caller-owned CUDA stream
  → Rust kernel | qualified vendor call
  → completion fence
  → returned write authority
```

A plan fixes the provider, algorithm, workspace, launch configuration, and
Graph policy before submission. Unsupported contracts return an error. Loom
does not select a silent fallback.

## Current operator surface

| Family | Admitted boundary | Current state |
| --- | --- | --- |
| RMSNorm | F32, FP16, and BF16 scalar and packed paths | Requalification |
| GEMM | Fixed contiguous BF16 `D = A × Wᵀ` | Requalification |
| Single decode | BF16 NHD D128 direct and split-K | Requalification |
| Paged decode | BF16 NHD D128, page size 16 | Requalification |
| Ragged prefill | BF16 bottom-right causal MHA, MQA, and GQA | Requalification |
| Paged prefill | BF16 NHD D128, page size 16 | Requalification |
| RoPE | BF16 D128 NeoX with explicit I32 positions | Requalification |
| Fused KV append | RoPE plus explicit paged append with exclusive write pages | Requalification |
| CUDA Graph | Fixed-address capture for named paths | Path-specific |

The surface stays narrow until each combination passes its declared gates.
The [operator catalog](docs/operator-catalog.md) lists exact combinations. The
[FlashInfer parity matrix](docs/flashinfer-parity.md) records open domains.

## Evidence

Loom keeps host correctness, H20 correctness, sanitizer, performance, Graph,
engine, and serving evidence separate.

The current comparison tools add a common F32 attention oracle, strict
contract grouping, and validated provider package versions. The FlashInfer
Graph scripts also poison replay outputs before validation.
Existing matched attention timings predate these checks and remain historical
until they are rerun.

Existing fused-append records also predate the exclusive-page ownership
contract.

The DeviceRegion refactor changed every CUDA submission path. All published
device and Graph records predate the current source and require replacement.

The current append path validates device metadata once, emits a scope-bound
compact map, and reports a typed error at completion. Its eager and Graph H20
gates pass in an isolated working-tree run, but the project has not published
a new immutable record yet.

Performance claims are contract-specific. Loom does not claim that every
operator or shape is faster than FlashInfer.

The [evidence index](docs/results/README.md) identifies the records that still
qualify current source and the records that require replacement. No published
record proves model or serving speedup.

## Workspace

| Crate | Responsibility |
| --- | --- |
| `loom-infer` | Backend-independent contracts and CPU references |
| `loom-infer-cuda` | CUDA providers, command runtime, Graphs, and vendor calls |
| `loom-infer-validation` | H20 gates, matched benchmarks, and evidence generation |

Product code has no Python API, CUDA C++, or Triton implementation. Python in
`tools/flashinfer` runs the pinned comparison provider and evidence tooling.

## Validate a checkout

Install `mise`, then review `mise.toml` before you trust it.

```bash
git clone https://github.com/feichai0017/loom-infer.git
cd loom-infer

mise trust
mise install
make install-website
make check
```

Run device gates inside the pinned CUDA environment:

```bash
make cuda-doctor
make cuda-check
make cuda-test
make h20
```

The [environment guide](docs/development/environment.md) lists the pinned Rust,
Node.js, CUDA, and cuda-oxide versions. The [H20 guide](docs/development/h20-validation.md)
defines device qualification.

## Project status

Loom Infer is alpha software. The merged source adds sixteen-warp long-MQA and
eight-warp long-GQA4 paged-prefill providers whose H20 records cover the source
tree before the DeviceRegion merge. Both providers require current-source
requalification.

Current work publishes new shared-KV and external-region evidence, extends
typed device errors to the remaining paged operators, and proves one real
engine invocation.

Read the [architecture](docs/design/loom-infer-architecture.md) and
[contribution guide](CONTRIBUTING.md) before adding a provider.

## License

[MIT](LICENSE)
