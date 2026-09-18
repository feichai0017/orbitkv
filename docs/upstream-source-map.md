# Upstream source map

This map is the code-reading route for the first vertical slice. Paths are
relative to the repository root and are pinned by Git submodule commit.

## One decode request

```text
SGLang Scheduler
  third-party/sglang/python/sglang/srt/managers/scheduler.py
       | ScheduleBatch
       v
TpModelWorker.forward_batch_generation
  third-party/sglang/python/sglang/srt/managers/tp_worker.py
       | ForwardBatch
       v
ModelRunner.forward
  third-party/sglang/python/sglang/srt/model_executor/model_runner.py
       | model ops and selected backends
       +--------------------+----------------------+
       v                    v                      v
FlashInfer attention   SGLang GEMM wrapper    torch.compile / graph
layers/attention/      layers/quantization/   srt/compilation/
       |                    |                      |
       v                    v                      v
FlashInfer source      DeepGEMM source        captured execution
third-party/flashinfer third-party/deepgemm
```

Our SGLang plugin wraps the `TpModelWorker` method above. That location sees the
real batch after scheduling and before model execution without taking ownership
away from SGLang. The first hook is diagnostic only and records host dispatch
time; qualification adds CUDA-event and profiler evidence separately.

## SGLang reading order

1. `srt/managers/scheduler.py`: waiting/running batches and scheduling policy.
2. `srt/managers/schedule_batch.py`: the request-to-batch representation.
3. `srt/managers/tp_worker.py`: worker boundary used by our hook.
4. `srt/model_executor/model_runner.py`: model, CUDA Graph, pools and forward.
5. `srt/layers/attention/*_backend.py`: attention implementation selection.
6. `srt/layers/quantization/fp8_utils.py`: FP8 GEMM backend selection.
7. `srt/layers/deep_gemm_wrapper/`: SGLang-to-DeepGEMM ABI and JIT policy.
8. `srt/compilation/backend.py`: piecewise `torch.compile` backend.
9. `srt/plugins/hook_registry.py`: current external integration contract.
10. `python/sglang/kernels/aot/`: the `sglang-kernel` CUDA/C++ source, build,
    Python bindings, tests, and benchmarks. It belongs to the SGLang submodule;
    there is no separate NVIDIA `sglang-kernel` repository to add.

We initially configure and observe these paths. We only patch SGLang after a
trace proves that an external plan cannot be expressed by existing flags, hook,
attention backend, fused-op, compilation backend, or platform contracts.

## TensorRT-LLM AutoDeploy reading order

1. `_torch/auto_deploy/models/factory.py`: model construction and dynamic inputs.
2. `_torch/auto_deploy/export/export.py`: `torch.export` to GraphModule.
3. `_torch/auto_deploy/transform/interface.py`: `BaseTransform` and registry.
4. `_torch/auto_deploy/transform/optimizer.py`: ordered transform pipeline.
5. `_torch/auto_deploy/transform/library/`: attention, KV, fusion and sharding.
6. `_torch/auto_deploy/compile/compiler.py`: compile-backend registry.
7. `_torch/auto_deploy/compile/backends/`: eager, compile and CUDA Graph paths.
8. `_torch/auto_deploy/shim/ad_executor.py`: connection to serving execution.

`integrations/autodeploy` explicitly registers a non-mutating graph inventory
transform. The next additions are candidate export and compilation-evidence
transforms; we do not modify AutoDeploy's core registry until an upstreamable
interface change is necessary.

## FlashInfer reading order

1. `flashinfer/autotuner/autotuner.py`: `AutoTuner`, `TunableRunner`, caches.
2. `flashinfer/gemm/gemm_base.py`: backend and tactic families.
3. `flashinfer/jit/`: source specialization, build and artifact cache.
4. `flashinfer/decode.py` and `prefill.py`: attention APIs.
5. `include/flashinfer/`: native kernel and runner contracts.
6. `3rdparty/`: the exact CUTLASS, CCCL, NIXL and spdlog revisions it builds.

Our qualification layer must preserve FlashInfer's runner and tactic identity,
raw timing samples, JIT source hash, cubin hash and its complete dependency
fingerprint. Selecting only the final function name is insufficient.

## DeepGEMM reading order

1. `build_sgl_deep_gemm.sh`: SGLang-compatible ABI and wheel construction.
2. `sgl_deep_gemm/`: packaging contract imported by SGLang as `deep_gemm`.
3. `deep_gemm/__init__.py`: public operations and extension calls.
4. `deep_gemm/include/deep_gemm/impls/`: architecture-specific implementations.
5. `deep_gemm/include/deep_gemm/scheduler/`: tile and persistent scheduling.
6. `csrc/tvm_ffi_api.cpp`: Python/TVM-FFI ABI.
7. `third-party/cutlass` and `fmt`: exact nested dependencies.

Use the SGLang fork for production compatibility. The DeepSeek upstream remains
useful for design comparison, but substituting its HEAD would invalidate the
SGLang package ABI and qualification fingerprint.

## Artifact flow into the Rust core

```text
AutoDeploy graph inventory and candidates
          + workload trace from SGLang
          + FlashInfer/DeepGEMM measurements
                         |
                         v
              PhysicalPlan JSON
              QualificationCertificate JSON
                         |
                         v
             Aletheia control registry
                         |
                         v
          selected plan ID + exact artifacts
                         |
                         v
       SGLang in-process backend configuration
```

The JSON boundary is intentional during M1: it lets the compiler, kernel
qualification process, and production server use incompatible Python stacks
without linking them into one process. It may later become a compact binary
format, but only after the contract stabilizes.
