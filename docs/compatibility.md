# Compatibility and distribution

Loom separates source compatibility, GPU validation, engine compatibility, and
binary portability. A green row below applies only to the stated boundary.

## Qualified matrix

| Component | Qualified version | Boundary | Evidence |
| --- | --- | --- | --- |
| Rust | current stable toolchain | format, Clippy, tests, release checks, source crate archives | GitHub CI |
| CUDA | 13.1, `sm_90` | `loom-cuda`, `loom-cuda-sys`, and `loom-cuda-bridge` build and execute | NVIDIA H20 gate |
| Python | 3.11.2 | clean native-wheel install; the `py3-none` artifact does not use the CPython C API | [ABI9 native-wheel gate](results/h20-native-wheel-clean-install-abi9-20260727.json) |
| PyTorch | 2.10.0+cu128 | the exact wheel built on 2.11 loads without recompilation; 201 applicable Loom tests pass | [ABI9 native-wheel gate](results/h20-native-wheel-clean-install-abi9-20260727.json) |
| PyTorch | 2.11.0+cu130 | clean wheel install, current stream, `torch.compile`, FakeTensor/opcheck, and CUDA Graph replay | [ABI9 native-wheel gate](results/h20-native-wheel-clean-install-abi9-20260727.json) |
| vLLM | 0.24.0 | repository-free install and all 305 registered-adapter/operator tests | [ABI9 native-wheel gate](results/h20-native-wheel-clean-install-abi9-20260727.json) |
| vLLM | 0.25.1 | repository-free install from the official wheel and all 305 registered-adapter/operator tests | [ABI9 native-wheel gate](results/h20-native-wheel-clean-install-abi9-20260727.json) |

The current rows qualify the immutable `7df4133` ABI9 artifact, including
optional-residual RMSNorm-to-dynamic-FP8,
deterministic categorical sampling, persistent request-owned RNG state, fused
mixed-sampling logits preprocessing, sparse token penalties, sampled-token
plus top-k logprobs, exact in-place top-k filtering, and fused top-p filtering
and renormalization.

Bridge ABI9 replaces the former
RMSNorm-to-FP8 schema with vLLM's exact optional-residual mutation schema and
retains no ABI8 overload or bridge shim. Its exact two-library wheel passes
every repository-free matrix row.

The preceding `e2c2982` ABI8 wheel is immutable historical evidence for the
pre-residual normalization schema. It passed 293 tests on each vLLM minor and
199 applicable PyTorch 2.10 tests.

The historical `f98a931` ABI7 refresh closed the earlier source-overlay
packaging gap: its wheel contains the final FP8 KV adapter, loads only its
package-local libraries, and passed the full 286-test vLLM 0.24 GPU suite plus
a focused 22-test FP8 KV/adapter gate. Its first run against a shared Inductor
cache retained a non-Loom missing-`cubin` failure; fresh isolated
`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` values passed both the failing
test and the full suite. The current ABI9 artifact supersedes that refresh and
qualifies the full vLLM 0.24/0.25 plus PyTorch 2.10/2.11 matrix.

Bridge ABI9 is a deliberate breaking boundary. An ABI8 dispatcher cannot load
the ABI9 bridge, and no compatibility shim exists.

The ABI7 wheel includes greedy speculative verification and static FP8 E4M3
KV quantize-on-write through bridge ABI 7. Both vLLM minors pass the same
expanded 286-test suite. The separate
[FP8 KV evidence](results/h20-fp8-kv-cache-write-20260724.json) closes the
exact-byte, framework, operator, clean-wheel, and real-engine invocation gates;
the [first pinned Qwen2.5-7B system candidate](results/h20-fp8-kv-system-rejected-20260727.json)
also proves operational capacity and native-vLLM/Loom FP8 equivalence, but an
8-sequence early-stop slice rejects its FP8 representation before TTFT/TPOT
measurement. That candidate used the cross-matrix wheel's native libraries
plus a SHA-pinned Python adapter overlay. The later `f98a931` refresh packages
that adapter and passes vLLM 0.24 clean-install tests, but does not change the
candidate's quality rejection or complete the remaining refresh matrix rows.
An accepted pretrained native-versus-FP8 quality, capacity, and serving
artifact remains an open family-level system-value gate.

The process-isolated Qwen2.5 draft/target engine benchmark is qualified on
vLLM 0.24 only; its [native-first](results/h20-vllm-qwen25-speculative-native-first-20260723.json)
and [Loom-first](results/h20-vllm-qwen25-speculative-loom-first-20260723.json)
reports prove invocation and provider equivalence, not acceleration. The
sparse-penalty feature first landed after ABI2 and has a separate vLLM 0.24
order-reversed Qwen gate:
[baseline first](results/h20-vllm-qwen25-token-penalties-baseline-first-20260725.json)
and [Loom first](results/h20-vllm-qwen25-token-penalties-loom-first-20260725.json).
The top-k logprob adapter also has exact order-reversed Qwen gates:
[baseline first](results/h20-vllm-qwen25-topk-logprobs-baseline-first-20260725.json)
and [Loom first](results/h20-vllm-qwen25-topk-logprobs-loom-first-20260725.json).
Their latency ratios cross parity, so they qualify compatibility and invocation,
not a stable engine speedup.
The fused logits-preprocessing adapter has its own vLLM 0.24
[baseline-first](results/h20-vllm-logits-preprocess-baseline-first-20260727.json)
and [Loom-first](results/h20-vllm-logits-preprocess-loom-first-20260727.json)
Qwen gates. They preserve exact tokens and order-stable TPOT, while batch
latency crosses parity at batch 32.
The 0.25.1 compatibility gate does not retroactively transfer any 0.24
performance result to 0.25.1. A new engine benchmark is required
before making a 0.25.1 performance claim.

Python package metadata therefore requires or accepts:

```text
torch>=2.10,<2.12
vllm>=0.24,<0.26
```

Versions outside that interval are not supported. Loom's optional registration
functions also check the installed vLLM series before patching engine classes
or compiler tables.

## Current native-wheel boundary

The published Rust crates remain self-contained source distributions. The
current ABI9 Python artifact name is
`loom_kernels-1.0.0a1-9cu131torch210sm90-py3-none-linux_x86_64.whl`.
Because the wheel is unpublished, its immutable manifest and SHA identify the
build:
`7df4133`/`c47f482eb088d69c3286791dc131ff6f770c1ef609271f444c139ed64852bf29`.
The `e2c2982` ABI8 artifact, `d58ebf8` ABI7 cross-matrix artifact, and
`f98a931` vLLM 0.24 refresh are historical evidence. Every native artifact is
built only through
`python/build_wheel.py` from a clean Git revision and contains exactly:

- `libloom_cuda_bridge.so` — Rust contracts, borrowed safe dispatch, and the
  internal handwritten CUDA launch layer;
- `libloom_kernels_torch.so` — boxed LibTorch Stable ABI dispatcher.

`native.json` records their hashes, Git revision, CUDA 13.1 toolkit, SM90
target, bridge ABI, and PyTorch runtime range. Installed wheels load only this
package-local pair. `PYTHONPATH`, `LD_LIBRARY_PATH`, and an external dispatcher
override were absent from every clean gate.

The earlier `8cu131torch210sm90` ABI-8, `6cu131torch210sm90` ABI-6,
`5cu131torch210sm90` ABI-5,
`4cu131torch210sm90` ABI-4, `2cu131torch210sm90` ABI-2, and
`1cu131torch210sm90` ABI-1 artifacts remain historical evidence.
The ABI-specific build tag prevents incompatible bridge signatures from
colliding; ABI9 is the current qualified artifact boundary.

The wheel is Python-ABI-independent (`py3-none`) because neither native library
uses the CPython C API. Its platform tag remains the conservative
`linux_x86_64`; auditwheel 6.7 found a `manylinux_2_34_x86_64` symbol floor,
but Loom does not claim an earlier manylinux baseline. H20 runtime validation
currently covers Python 3.11 only.

The artifact is not published to PyPI or a GitHub release. This is a qualified
build/install boundary, not a claim that `pip install loom-kernels` can fetch a
native wheel from a public index.

## Current Stable ABI boundary

PyTorch documents a [LibTorch Stable ABI](https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html)
and stable registration APIs for PyTorch 2.10 and newer. Loom's single
production dispatcher now uses that boundary:

- all schemas use boxed Stable ABI registration;
- tensor metadata, allocations, pointers, device guards, and the current CUDA
  stream use stable headers or AOTI C shims;
- all seventeen ABI9 semantic operators continue into `loom-cuda-bridge`.
  The dispatcher has no ATen/c10 C++ symbol dependency and consumes no raw
  CUDA launch symbol;
- the public Python APIs and vLLM admission predicates reject tensors requiring
  gradients. No autograd kernel is advertised;
- the temporary Add+RMSNorm probe and the previous ATen dispatcher were deleted
  after the production migration passed.

`python/build_wheel.py` now automates the first CUDA/PyTorch/Python matrix row,
audits its ELF boundary, and rejects accidental source-only wheels. One exact
artifact passed fresh-venv gates on PyTorch 2.10/2.11 and vLLM 0.24/0.25.
Publishing that artifact remains a separate, explicit release action.

## What must be revalidated

| Change | Minimum gate |
| --- | --- |
| Rust contract or aliasing rule | CPU oracle, invalid-input tests, safe CUDA wrapper |
| CUDA kernel | edge shapes, representative shapes, external stream, CUDA Graph |
| PyTorch dispatcher | opcheck/FakeTensor, mutation schema, `torch.compile`, full GPU suite |
| vLLM minor release | official wheel import, all adapter tests, explicit fallback tests |
| Performance claim | named baseline, warmed samples, correctness first, provider-order reversal for engines |
| Binary wheel claim | clean install on every published Python/PyTorch/CUDA matrix entry |
