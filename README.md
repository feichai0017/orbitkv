<div align="center">
  <h1>Loom Infer</h1>
  <p>Rust-native GPU kernels for LLM inference.</p>
  <p>
    <a href="docs/README.md">Docs</a> ·
    <a href="docs/operator-catalog.md">Operators</a> ·
    <a href="docs/roadmap.md">Roadmap</a> ·
    <a href="docs/results/README.md">Evidence</a>
  </p>
</div>

Loom Infer is a high-performance inference kernel library written in Rust. It
provides checked operator contracts, CPU references, and Rust CUDA kernels
compiled with [cuda-oxide](https://github.com/NVlabs/cuda-oxide).

The project is not a model server. An inference engine owns requests, models,
scheduling, and distributed execution. Loom Infer owns operator contracts,
launch planning, GPU execution, and reproducible evidence.

## Current scope

The repository contains two crates:

| Crate | Responsibility |
| --- | --- |
| `loom-infer` | Safe operator contracts and CPU reference implementations |
| `loom-infer-cuda` | Rust host code and Rust CUDA kernels built with cuda-oxide |

The first device path is contiguous RMSNorm for F32, FP16, and BF16. Its public
flow is:

```text
RmsNormSpec::new
  -> RmsNormProvider::load
  -> RmsNormProvider::plan_{f32,f16,bf16}
  -> CommandQueue::bindings
  -> CommandQueue::begin
  -> prepared plan::enqueue_into
  -> CommandScope::finish
  -> CommandCompletion::wait
```

All three dtypes have passed permanent-provider H20 correctness gates. FP16 and
BF16 use scalar access for odd widths and packed 32-bit access for even widths.
Attention, sampling, KV-cache, MoE, quantization, and vendor GEMM remain roadmap
work.

### Execution

The command scope chains operators on one caller-owned stream and holds mixed
F32, FP16, BF16, and byte buffers. Typed handles prevent dtype confusion, and
one preallocated event completes the scope. The cuda-oxide launcher still
allocates its argument vector during enqueue.

## Source boundary

- Loom-owned product code is Rust.
- Custom device kernels are Rust compiled with cuda-oxide.
- The repository has no Python product API, CUDA C++, compatibility layer, or
  silent fallback.
- CUDA drivers and established GEMM or collective libraries remain vendor
  dependencies when Loom adds those paths.
- The caller owns streams and device buffers. Checked bindings retain borrowed
  resources while the GPU uses them.

## Local checks

The default workspace build does not require CUDA:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo check --workspace --release
cargo package -p loom-infer
```

H20 device validation uses the pinned nightly and cuda-oxide revision:

```bash
cd crates/loom-infer-cuda
cargo oxide doctor
cargo oxide run rms_norm_h20 --bin rms_norm_h20 --features cuda --arch sm_90
```

See the [H20 validation contract](docs/development/h20-validation.md) before
reporting a device result.

## Evidence

Operator correctness, kernel latency, graph execution, engine integration, and
serving performance are separate claims. A microbenchmark does not establish
an engine or serving speedup.

See the [architecture](docs/design/loom-infer-architecture.md),
[operator catalog](docs/operator-catalog.md), [roadmap](docs/roadmap.md), and
[low-precision RMSNorm result](docs/results/h20-rms-norm-low-precision-20260802.json).

## License

[MIT](LICENSE)
