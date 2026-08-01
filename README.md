<div align="center">
  <h1>Loom Kernels</h1>
  <p>Rust-first CUDA operators for LLM inference.</p>
  <p>
    <a href="docs/README.md">Docs</a> ·
    <a href="docs/operator-catalog.md">Operators</a> ·
    <a href="docs/compatibility.md">Compatibility</a> ·
    <a href="docs/results/README.md">Evidence</a> ·
    <a href="https://feichai0017.github.io/loom-kernels/">Website</a>
  </p>
  <p>
    <a href="https://github.com/feichai0017/loom-kernels/actions/workflows/ci.yml"><img alt="CI status" src="https://github.com/feichai0017/loom-kernels/actions/workflows/ci.yml/badge.svg"></a>
  </p>
</div>

Loom implements memory-bound fusion, layout conversion, quantization support,
sampling, and data movement around the matrix core. It does not implement an
inference engine, tensor framework, or GEMM library.

> [!IMPORTANT]
> Loom is alpha software. Engine routes are explicit opt-ins. Unsupported
> contracts stay on the engine's native path.

## Scope

Loom owns five parts of an operator path:

| Part | Responsibility |
| --- | --- |
| Contract | Dtypes, shapes, layouts, aliasing, and invalid-input behavior |
| Reference | Deterministic CPU oracles for correctness |
| Execution | Safe Rust dispatch and handwritten CUDA |
| Integration | Current-stream PyTorch operators and narrow vLLM routes |
| Evidence | Correctness, CUDA Graph, engine, and clean-install results |

cuBLASLt, CUTLASS, FlashInfer, or the engine keeps every dense, quantized,
sparse, and grouped GEMM. Loom only owns measured work around those kernels.

## Operator surface

Bridge ABI12 exposes 21 semantic operators through one checked execution path.

| Family | Public paths | Boundary |
| --- | --- | --- |
| Normalization | RMSNorm, Add+RMSNorm, dynamic FP8 and INT8 output | INT8 remains explicit while quality and stable engine benefit are open |
| MLP | SiLU-and-Mul, block FP8 output, dynamic INT8 output | GEMM remains vendor-owned. The INT8 route is profile-gated |
| Position and KV | RoPE with paged-KV write, static FP8 E4M3 cache write | The engine owns cache storage, page tables, and attention |
| Decode tail | Logits preprocessing, penalties, top-k, top-p, Min-P, logprobs, categorical sampling | Each vLLM route has an exact shape and semantic gate |
| Speculative decode | Greedy draft verification and token compaction | Tree, stochastic, and KV extensions need a new engine profile |
| MoE movement | Stable expert-major permutation and weighted combine | Grouped GEMM and routing remain engine-owned |
| Attention | Paged MQA/GQA decode and local split-K/LSE merge | Only short qualified shapes enter the vLLM route. FA3 handles the rest |

The [operator catalog](docs/operator-catalog.md) records each status and
admission rule.

## Architecture

```text
engine adapter
  -> LibTorch Stable ABI dispatcher
  -> checked Rust bridge
  -> safe Rust CUDA backend
  -> internal launch ABI
  -> handwritten CUDA
```

| Path | Owns |
| --- | --- |
| `crates/loom-kernels` | Public contracts, capabilities, and CPU oracles |
| `crates/loom-cuda` | Safe streams, memory views, layouts, dispatch, and benchmarks |
| `crates/loom-cuda-bridge` | Checked C entrypoints, spans, aliasing, and panic containment |
| `crates/loom-cuda-sys` | Internal launch declarations, CUDA build logic, and kernels |
| `python` | Stable ABI PyTorch operators and vLLM admission policy |

Framework adapters translate tensors and streams. They do not launch CUDA or
duplicate Rust validation. See the [code layout](docs/design/code-layout.md).

## Install

Use the backend-independent contracts from Rust:

```bash
cargo add loom-kernels@1.0.0-alpha.1
```

Enable the CUDA backend on a CUDA build host:

```bash
cargo add loom-cuda@1.0.0-alpha.1 --features cuda
```

Validate a source checkout without CUDA:

```bash
git clone https://github.com/feichai0017/loom-kernels.git
cd loom-kernels
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
```

A qualified native Python wheel exists, but no package index publishes it.
Build and install it from a clean Linux x86_64 checkout by following the
[Python guide](python/README.md).

## Qualified boundary

The current artifact is
`loom_kernels-1.0.0a1-12cu131torch210sm90-py3-none-linux_x86_64.whl`.
It contains exactly `libloom_cuda_bridge.so` and
`libloom_kernels_torch.so`.

| Component | Qualified boundary |
| --- | --- |
| GPU | NVIDIA H20, SM90, CUDA 13.1 |
| Python | 3.11 |
| PyTorch | 2.10 and 2.11 |
| vLLM | 0.24 and 0.25 |
| Clean install | 359 tests on each vLLM row and 245 applicable tests on PyTorch 2.10 |

The [compatibility matrix](docs/compatibility.md) records the exact revisions,
hashes, and revalidation rules.

## Evidence

These results show the range of current claims. Each result links to its raw
H20 record.

| Path | Result | Limit |
| --- | --- | --- |
| [RMSNorm to dynamic FP8](docs/results/h20-rms-norm-dynamic-fp8-residual-20260727.json) | `1.0066-1.0506x` Qwen prefill batch-latency ratio | Decode-heavy runs cross parity |
| [Sparse token penalties](docs/results/h20-token-penalties-20260725.json) | `5.82-34.30x` operator ratio and exact output | Serving-scale goodput remains open |
| [Deterministic categorical sampling](docs/results/h20-categorical-sample-20260727.json) | One kernel and `1.15-5.40x` at 4-32 rows | Batch 1-4 pays an engine cost |
| [Short paged decode](docs/results/h20-vllm-paged-decode-backend-20260722.json) | `1.154-2.374x` across 24 admitted cases | Other shapes use FA3 |
| [MoE engine admission](docs/results/h20-vllm-engine-moe-movement-20260801.json) | Exact tokens and `48/48` movement hits | Synthetic model. No production speedup claim |
| [FP8 KV system candidate](docs/results/h20-fp8-kv-system-rejected-20260727.json) | `1.99879x` cache capacity | Rejected because perplexity regressed by about `3.07x` |

A fast kernel does not imply a faster model. Loom records operator, graph,
engine, and serving results as separate gates. The
[evidence index](docs/results/README.md) includes accepted, parity, fallback,
and rejected experiments.

## Next work

1. Run the production MoE movement gate on a pinned pretrained workload.
2. Add quantization plumbing only for a measured vendor-kernel consumer.
3. Build one zero-copy Rust decode step over borrowed tensors and streams.

KV movement and speculative extensions remain profile-gated. The
[roadmap](docs/roadmap.md) defines their entry and exit conditions.

## Documentation

- [Documentation index](docs/README.md)
- [Operator catalog](docs/operator-catalog.md)
- [vLLM integration guide](docs/guides/vllm-ir-provider.md)
- [Implementation status](docs/status.md)
- [Compatibility and distribution](docs/compatibility.md)
- [Contributing](CONTRIBUTING.md)

## License

[MIT](LICENSE)
