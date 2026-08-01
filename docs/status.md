# Implementation status

Current source uses bridge ABI12. The latest qualified native wheel contains
21 semantic operators and two native libraries. It is not publicly released.

## Execution path

Every framework operator follows one path:

```text
PyTorch or vLLM
  -> LibTorch Stable ABI dispatcher
  -> checked Rust bridge
  -> safe Rust dispatch
  -> internal CUDA launch ABI
  -> handwritten CUDA
```

The public Rust contracts and CPU oracles define the semantics for every layer.
The repository contains no ctypes path, unchecked dispatcher twin, direct
C++-to-CUDA path, or bridge compatibility shim.

## Operator families

| Family | Current state | Open boundary |
| --- | --- | --- |
| RMSNorm and Add+RMSNorm | Supported for F32, FP16, and BF16 | Broader model-level benefit remains shape-specific |
| RMSNorm to dynamic FP8 | Supported, including optional residual | Decode-heavy engine latency crosses parity |
| RMSNorm to dynamic INT8 | Implemented and distributed | Quality, stable engine benefit, and default admission remain open |
| SiLU-and-Mul | Supported for F32, FP16, and BF16 | None for the standalone contract |
| SiLU-and-Mul to block FP8 | Supported | Model-level benefit remains workload-specific |
| SiLU-and-Mul to dynamic INT8 | Implemented and distributed | Measured performance rejects default admission |
| RoPE and paged-KV write | Supported for native and static FP8 cache storage | The first FP8 system candidate failed quality |
| Logits and sampling tail | Supported across preprocessing, penalties, filters, logprobs, and explicit-state categorical sampling | Each adapter keeps its measured shape gate |
| Greedy speculative verification | Supported | Further tree, stochastic, and KV work is profile-gated |
| MoE permutation and combine | Supported around unchanged vendor grouped GEMM | A pretrained production workload has not closed the system gate |
| Paged MQA/GQA decode | Supported through the Rust and PyTorch layers | vLLM admits only short H20-qualified shapes. FA3 owns the rest |

The [operator catalog](operator-catalog.md) lists every operator and its exact
status.

## Distribution

| Gate | Result |
| --- | --- |
| Artifact | ABI12 Linux x86_64 wheel for CUDA 13.1 and SM90 |
| Contents | `libloom_cuda_bridge.so`, `libloom_kernels_torch.so`, and `native.json` |
| PyTorch 2.10 | 245 applicable tests pass |
| PyTorch 2.11 with vLLM 0.24 | 359 tests pass |
| PyTorch 2.11 with vLLM 0.25 | 359 tests pass |
| Installation | Fresh environments load only package-local libraries |
| Publication | Not published |

See [compatibility and distribution](compatibility.md) for the artifact hash
and ABI rules.

## Representative H20 results

| Boundary | Result | Interpretation |
| --- | --- | --- |
| [RMSNorm to FP8](results/h20-rms-norm-dynamic-fp8-residual-20260727.json) | Exact output and `1.0066-1.0506x` Qwen prefill ratio | Qualified for the measured prefill path |
| [Sparse penalties](results/h20-token-penalties-20260725.json) | Exact output and `5.82-34.30x` operator ratio | Order-stable pinned Qwen benefit. Serving goodput remains open |
| [Categorical sampling](results/h20-categorical-sample-20260727.json) | One kernel and `1.15-5.40x` at 4-32 rows | Persistent request-state engine gate also passes |
| [Short paged decode](results/h20-vllm-paged-decode-backend-20260722.json) | `1.154-2.374x` over 24 admitted cases | Context 64 and unqualified shapes use FA3 |
| [MoE movement](results/h20-moe-movement-20260801.json) | Exact metadata and up to `1.163x` all-local graph ratio | Direct movement claim only |
| [MoE engine admission](results/h20-vllm-engine-moe-movement-20260801.json) | Exact tokens and `48/48` movement hits | Synthetic model. Production benefit remains open |

The [evidence index](results/README.md) contains every accepted, parity,
fallback, and rejected result.

## Open gates

- Run the production MoE benchmark with a pinned pretrained model and prompts.
- Admit new quantization pack, scale, or layout work from a measured engine
  consumer.
- Find a model or cache format whose FP8 KV path passes the quality gate.
- Add physical KV movement only for a measured offload, beam, or compaction path.
- Build one engine-neutral zero-copy Rust decode step.
- Validate serving-scale concurrency, goodput, and memory on larger workloads.

SGLang, other Rust-native engines, non-SM90 GPUs, and Python versions other
than 3.11 do not have qualified runtime evidence.
