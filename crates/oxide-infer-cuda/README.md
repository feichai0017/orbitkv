# oxide-infer-cuda

`oxide-infer-cuda` contains Oxide Infer's Rust CUDA host and device code.
`cuda-oxide` compiles each Oxide-owned kernel. Audited Rust FFI calls qualified
vendor libraries. The crate does not use CUDA C++, Python, or framework
bindings.

CUDA names the target platform and execution contract. cuda-oxide is the Rust
toolchain used for Oxide-owned kernels. It does not replace CUDA contexts,
streams, events, Graphs, or vendor libraries.

The default feature set keeps CPU-only workspace checks platform independent.
Enable `cuda` only inside the pinned CUDA build environment.

Hardware runners live in the sibling `oxide-infer-lab` crate. Use
`make cuda-test` and `make h20` from the repository root.

## Module layout

```text
src/
|-- attention/{mod.rs,decode.rs,prefill.rs}
|-- command/{mod.rs,binding.rs,resolve.rs,status.rs,submission.rs,completion.rs}
|-- device_status.rs
|-- gemm/{mod.rs,plan.rs,planner.rs}
|   `-- provider/{mod.rs,cublaslt.rs,oxide/{mod.rs,sm90/mod.rs}}
|-- graph/mod.rs
|-- memory.rs
|-- rms_norm/mod.rs
|-- rope/mod.rs
|-- driver.rs
`-- lib.rs
```

Public operator modules are stable directory facades. Attention's decode and
prefill domains keep one file per complete cuda-oxide artifact bundle because
the macro must discover each bundle in one token tree. Future MLA and KV
domains become sibling private modules without changing
`oxide_infer_cuda::attention::*`.

## Current providers

The crate exposes these provider families:

- F32, FP16, and BF16 contiguous RMSNorm. Low-precision plans select scalar or
  packed-pair kernels from the hidden size.
- BF16 single decode with direct and explicit split-K plans.
- BF16 NHD or HND paged batch decode. MHA uses direct attention. MQA and GQA
  use an eight-warp token-parallel plan. A device validator reports typed
  page-table errors at completion.
- BF16 ragged causal prefill with direct, eight-warp, sixteen-warp, and tiled
  GQA4 plans.
- BF16 paged causal prefill with direct, eight-warp, and sixteen-warp plans.
  The caller selects the algorithm. A device validator reports typed query and
  page metadata errors at completion.
- BF16 D128 NeoX RoPE with explicit positions.
- Fused RoPE and paged KV append for one token per request or 1 through 64
  explicit tokens.
- One contiguous BF16 dense GEMM path through `GemmPlanner`. The caller selects
  `CublasLt` or the experimental `OxideSm90SimtGemvM1N16K64` algorithm. The
  native candidate is currently admitted only for the recorded H20 `sm_90a`
  target. The common immutable plan and checked enqueue path does not change
  providers at runtime.

## Append execution

Fused append uses one checked three-command sequence. A validator writes a
cache-bound append map after it checks page metadata. The mapped kernel rotates
Q and K and writes private K/V pages before a status copy reports rejection.

Each target page must have reference count one. The pager must make a shared
tail private before enqueue. One append map remains bound to its command scope,
workspace, and exact writable K/V regions.

The 2026-08-06 append records measured an older single-kernel path. They did
not include exclusive-page ownership, the append map, or status transfer. They
do not qualify the current eager or Graph path.

## Qualification status

The current R1 record qualifies the declared DeviceRegion paths for
correctness, lifecycle, fixed-address Graph behavior, and Compute Sanitizer on
its recorded target. The experimental M=1 GEMV remains limited to its exact
`sm_90a` contract and still needs matched performance and engine evidence.

The current R2 record adds matched eager-provider timing for paged decode,
ragged prefill, and paged prefill against FlashInfer 0.6.17. It exposes strong
short paths and large long-context GQA prefill gaps; it is not Graph, model, or
serving evidence.

The fixed-address Graph runtime retains bindings and provider resources until
replay completes. Operator-specific Graph evidence remains path-specific.
Mutable bindings, Graph updates, concurrent replay, and cross-stream launch
remain outside the current contract.

## Engine interop boundary

`ExternalCudaStream` retains an engine-owned stream without adopting it.
`EngineInteropQueue` orders direct single decode or NHD or HND paged decode
through pre- and post-events on a Oxide-owned stream. Checked external regions
retain allocation leases and preserve their device addresses.

After Oxide enqueues the post-event wait, the submission returns the engine's
stream-ordered authority without a host wait. An owned completion keeps the
checked bindings, allocation leases, device status storage, and event slot
private until settlement.

The `engine_interop_h20` gate uses an in-repository simulated engine. It is not
evidence for Mistral.rs, vLLM, or another model runner. The separate Mistral.rs
integration record covers one narrow model-owned runtime path.
