# Compatibility and distribution

Loom qualifies source, GPU, framework, and binary boundaries separately.

## Qualified matrix

| Component | Version | Qualified boundary |
| --- | --- | --- |
| Rust | Current stable | Format, Clippy, tests, release checks, and crate archives |
| CUDA | 13.1 on SM90 | Source build and execution on NVIDIA H20 |
| Python | 3.11.2 | Fresh native-wheel install. The wheel does not use the CPython C API |
| PyTorch | 2.10.0+cu128 | The wheel built with 2.11 loads without recompilation. 245 applicable tests pass |
| PyTorch | 2.11.0+cu130 | Current stream, FakeTensor/opcheck, `torch.compile`, and CUDA Graph replay |
| vLLM | 0.24.0 | Fresh install and 359 operator and adapter tests |
| vLLM | 0.25.1 | Fresh install and 359 operator and adapter tests |

The [ABI12 clean-install result](results/h20-native-wheel-clean-install-abi12-20260801.json)
records the complete matrix.

Python package metadata accepts:

```text
torch>=2.10,<2.12
vllm>=0.24,<0.26
```

Loom rejects versions outside these ranges. vLLM registration also checks
the installed minor series before it changes engine classes or compiler tables.

## Native wheel

The current qualified artifact is:

```text
loom_kernels-1.0.0a1-12cu131torch210sm90-py3-none-linux_x86_64.whl
```

Revision `d4c13e2` and SHA256
`f13445d8a286b2a1afb931d284ccaa40ddec241a4e228d673d8bc0d5b11a0107`
identify it. Loom qualified the artifact but has not published it to PyPI or a
GitHub release.

The wheel contains exactly two native libraries:

- `libloom_cuda_bridge.so` contains checked Rust dispatch and the internal
  CUDA launch layer.
- `libloom_kernels_torch.so` contains the boxed LibTorch Stable ABI dispatcher.

`native.json` binds their hashes, Git revision, CUDA toolkit, SM targets,
bridge ABI, and PyTorch range. Installed packages load only this packaged pair.
The clean-install gates use no repository checkout or library-path override.

The `py3-none` tag is valid because neither library uses the CPython C API.
The platform tag remains `linux_x86_64`. H20 runtime validation covers Python
3.11 only.

## ABI policy

Current source and the wheel use bridge ABI12 and expose 21 semantic operators.
ABI12 added MoE permutation and combine without keeping ABI11 entrypoints.

Loom does not carry bridge compatibility shims. A dispatcher rejects a bridge
with a different ABI. Historical wheel results remain under
[`docs/results`](results/), and [CHANGELOG.md](../CHANGELOG.md) records each
breaking boundary.

## Stable ABI boundary

The PyTorch dispatcher targets the LibTorch Stable ABI introduced in PyTorch
2.10. It uses boxed registration, stable headers, and AOTI C shims for tensor
metadata, allocations, device guards, and the current CUDA stream.

Neither native library depends on ATen or c10 C++ symbols. Public Python and
vLLM entrypoints reject tensors that require gradients. Loom does not advertise
autograd support.

## Revalidation rules

| Change | Minimum gate |
| --- | --- |
| Rust contract or aliasing | CPU oracle, invalid-input tests, and safe CUDA wrapper |
| CUDA kernel | Edge shapes, representative shapes, external stream, and CUDA Graph |
| PyTorch dispatcher | Mutation schema, FakeTensor/opcheck, `torch.compile`, and GPU suite |
| vLLM minor | Official wheel import, adapter suite, and fallback tests |
| Performance claim | Named baseline, warmed samples, correctness, and reversed provider order for engines |
| Binary claim | Fresh install on every stated Python, PyTorch, CUDA, and GPU row |

Compatibility does not transfer performance claims between framework versions,
models, shapes, or GPUs.
