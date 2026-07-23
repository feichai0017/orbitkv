# Compatibility and distribution

Loom separates source compatibility, GPU validation, engine compatibility, and
binary portability. A green row below applies only to the stated boundary.

## Qualified matrix

| Component | Qualified version | Boundary | Evidence |
| --- | --- | --- | --- |
| Rust | current stable toolchain | format, Clippy, tests, release checks, source crate archives | GitHub CI |
| CUDA | 13.1, `sm_90` | `loom-cuda`, `loom-cuda-sys`, and `loom-cuda-bridge` build and execute | NVIDIA H20 gate |
| PyTorch | 2.10.0+cu128 | the exact dispatcher binary built on 2.11 loads without recompilation; 123 applicable Loom tests pass | [Stable ABI gate](results/h20-libtorch-stable-abi-20260723.json) |
| PyTorch | 2.11.0+cu130 | Stable ABI dispatcher, current stream, `torch.compile`, FakeTensor/opcheck, CUDA Graph replay | [Stable ABI gate](results/h20-libtorch-stable-abi-20260723.json) |
| vLLM | 0.24.0 | all registered adapters plus the existing operator and real-engine evidence | [evidence index](results/README.md) |
| vLLM | 0.25.1 | official wheel import, registered adapters, dispatcher behavior, and the complete 192-test H20 suite | [Stable ABI gate](results/h20-libtorch-stable-abi-20260723.json) |

The 0.25.1 gate proves that the current adapters and CUDA paths execute against
the official vLLM wheel. It does not retroactively transfer the 0.24
model-level latency results to 0.25.1. A new engine benchmark is required before
making a 0.25.1 performance claim.

Python package metadata therefore accepts:

```text
vllm>=0.24,<0.26
```

Versions outside that interval are not supported. Loom's optional registration
functions also check the installed vLLM series before patching engine classes
or compiler tables.

## Current binary boundary

The published Rust crates are self-contained source distributions. The Python
wheel currently contains Python adapters only; users build these native
libraries against their local CUDA and PyTorch installations:

- `libloom_cuda_bridge.so` — Rust contracts, borrowed safe dispatch, and the
  internal handwritten CUDA launch layer;
- `libloom_kernels_torch.so` — boxed LibTorch Stable ABI dispatcher.

The production dispatcher targets PyTorch 2.10 with
`TORCH_TARGET_VERSION`, registers through `STABLE_TORCH_LIBRARY`, and uses
`torch::stable::Tensor`. The exact binary built with PyTorch 2.11.0+cu130
passed its applicable H20 GPU gates without recompilation on PyTorch
2.10.0+cu128. This proves the recorded two-minor binary boundary; it is not a
claim for untested PyTorch releases or a published native wheel.

## Current Stable ABI boundary

PyTorch documents a [LibTorch Stable ABI](https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html)
and stable registration APIs for PyTorch 2.10 and newer. Loom's single
production dispatcher now uses that boundary:

- all schemas use boxed Stable ABI registration;
- tensor metadata, allocations, pointers, device guards, and the current CUDA
  stream use stable headers or AOTI C shims;
- all ten semantic operators continue into `loom-cuda-bridge`; the dispatcher
  has no ATen/c10 C++ symbol dependency and consumes no raw CUDA launch symbol;
- the public Python APIs and vLLM admission predicates reject tensors requiring
  gradients. No autograd kernel is advertised;
- the temporary Add+RMSNorm probe and the previous ATen dispatcher were deleted
  after the production migration passed.

The remaining distribution task is to package the already-qualified boundary
as automated Python/PyTorch/CUDA matrix wheels and prove clean installs. Source
builds remain the supported path until those native wheels are published.

## What must be revalidated

| Change | Minimum gate |
| --- | --- |
| Rust contract or aliasing rule | CPU oracle, invalid-input tests, safe CUDA wrapper |
| CUDA kernel | edge shapes, representative shapes, external stream, CUDA Graph |
| PyTorch dispatcher | opcheck/FakeTensor, mutation schema, `torch.compile`, full GPU suite |
| vLLM minor release | official wheel import, all adapter tests, explicit fallback tests |
| Performance claim | named baseline, warmed samples, correctness first, provider-order reversal for engines |
| Binary wheel claim | clean install on every published Python/PyTorch/CUDA matrix entry |
