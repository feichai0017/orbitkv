# Loom Infer roadmap

Loom Infer is a Rust operator layer for production LLM inference engines.
Custom NVIDIA kernels use cuda-oxide. Qualified GEMM and communication
libraries remain explicit providers.

The complete target stays within `loom-infer`, `loom-infer-cuda`, and
`loom-infer-validation`. The roadmap fixes the framework before it expands the
operator surface.

## K0.0: Normalize the operator framework

**State:** active. Design fixed. Source migration pending.

All current operators will use the same lifecycle:

```text
Spec -> Provider -> Algorithm -> Plan -> Operands -> CommandScope -> Completion
```

- Keep the three-crate dependency boundary.
- Move implemented domains to the final attention, KV-cache, GEMM,
  normalization, and position namespaces.
- Add a planned family only with its first contract or provider.
- Rename current `*Args` types to `*Operands` without compatibility aliases.
- Select providers and algorithms before immutable plan creation.
- Keep tuning and fallback outside enqueue.
- Split large source files only at an operator or provider boundary.
- Preserve one public execution path for each operator.
- Update the operator catalog from source and keep old evidence immutable.

The source migration must preserve current contract behavior. It does not
inherit old H20 qualification under a new source commit.

Exit: every implemented operator follows the common lifecycle and final family
namespace. Host tests pass, and no duplicate API or empty provider remains.

## K0.1: Requalify paged KV writes

**State:** active.

The fused RoPE plus KV append contract now accepts a physical-page reference
count array. Every target page must have reference count one.

- Reject shared target pages in host validation and device guards.
- Preserve Q, K pages, and V pages when the write contract fails.
- Cover one-token and explicit one-through-64-token forms.
- Test private tails that follow shared read-only prefix pages.
- Keep the page table and reference-count snapshot stable through completion.
- Run correctness and all declared Compute Sanitizer tools on H20.
- Create new eager and fixed-address Graph records after correctness passes.

The engine or KV pager makes shared tails private. The operator does not
allocate, copy, or remap pages.

Exit: new H20 records qualify the exclusive-page contract. The old 2026-08-06
append records remain historical.

## K0.2: Accept external device regions

**State:** source implemented. Model-owned paired-repository POC recorded on one H20.

The command layer accepts typed owned and external device regions.
`ExternalCudaStream` retains an engine stream without taking ownership.
`EngineInteropQueue` orders direct single decode and NHD or HND paged decode through CUDA events.

- Keep pointer, element span, CUDA context, access mode, and lifetime lease in
  one region value.
- Reject invalid ranges, alignment, context, and binding-set overlap before
  enqueue.
- Keep writable access exclusive until completion settles.
- Test non-zero offsets and lifetime retention on H20.
- Qualify fixed-address Graph retention for external leases.
- Preserve the simulated-engine H20 gate for external pointers, stream order,
  bounded in-flight completion, typed rejection, and zero adapter device-to-device copies.
- Extend the model-owned Mistral.rs qualification beyond one model and one ordinary stream.

Exit: a real model runner passes its own allocation and stream without an
adapter copy. Completion retains every lease through asynchronous execution.

## K0.3: Report device metadata errors

**State:** source implemented. Immutable evidence pending.

Paged decode, paged prefill, and fused append report device metadata failures
through typed completion errors. Rejection preserves outputs and bindings. It
does not poison the queue or fixed-address Graph.

- Keep each validator responsible for fully overwriting its status packet.
- Read status only after the completion fence.
- Keep semantic rejection recoverable. Poison only on CUDA failure or a
  malformed status packet.
- Cover eager and fixed-address Graph rejection, output preservation, and
  binding recovery.
- Reuse one append map only with the same K/V cache binding. Add a pager-issued
  cache epoch before allowing one map to address several layer buffers.

Exit: every admitted dynamic metadata failure reaches the host as a typed
error. No operator silently returns success after a device guard rejects work.

## K0.4: Prove one engine invocation

**State:** paired-repository POC and model-owned runtime evidence recorded.

The first adapter stays in the Mistral.rs fork.
Loom does not vendor the engine or copy its raw evidence.
The historical POC ran Mistral.rs sources `9f6acf2a` and `805dc8f1` against Loom
`d27b6e5`. The Mistral.rs fork publishes the records.

On one H20 Qwen path, the adapter recorded Loom provider hits and no adapter-issued device-to-device copy.
Loom and the standard provider selected the same eight token strings.
A separate gate returned a typed metadata error in FIFO order and then reused the same runtime.

Mistral.rs source `84602212` owns runtime state and pending completions in each model pipeline.
Its H20 record covers 196 completed Qwen paged-decode calls, typed rejection and same-runtime reuse, and concurrent drain serialization.
It does not qualify production safety, full-model zero-copy execution, general model coverage, or performance.

- Use the HND paged-decode path now present in Loom source.
- Bind engine-owned device regions and the engine's non-default stream.
- Adapt the engine's linear stream and storage authority to Loom's
  stream-ordered handoff without exposing a second writable capability.
- Record provider hit counts and selected algorithms.
- Prove that Q, KV, output, workspace, and metadata cross no host copy.
- Compare model output or generated tokens with the engine baseline.
- Carry a typed, linear runner authority through the model path.
- Define fail-closed behavior for a panic or abandoned model forward.
- Replace the sibling path dependency with an immutable published source.
- Measure the complete engine interval before making a latency claim.
- Document the fallback policy at the engine boundary. The Loom provider
  itself does not silently fall back.

Exit: a real model invokes Loom, preserves declared numerical behavior, and
produces an auditable no-copy trace.

## K0.5: Make algorithm policy explicit

**State:** partial.

Paged decode still chooses direct MHA and eight-warp MQA or GQA.

Ragged prefill still uses the batch average KV length and head mapping. Paged
prefill now requires an explicit direct, eight-warp, or sixteen-warp algorithm.

The long MQA and GQA4 records apply to source `8478ee9`, not the merged
DeviceRegion path.

- Separate the remaining automatic selection from immutable launch plans.
- Expose the selected algorithm in traces and result records.
- Add a caller override with contract checks.
- Replace average-only ragged decisions with measured shape classes or request
  grouping when evidence supports it.
- Keep planning and tuning outside enqueue.
- Reject unsupported algorithm and workspace combinations.

Exit: the same shape and policy produce the same plan. Selection uses recorded
evidence rather than an undocumented enqueue-time choice.

## K0.6: Complete fixed-address Graph coverage

**State:** partial.

The current Graph contract fixes addresses and uses one private capture stream.
Evidence exists for RMSNorm-to-GEMM, tiled long-GQA ragged prefill, and one
direct paged-prefill GQA4 fixture.

- Requalify fused append after K0.1.
- Add paged-decode Graph correctness.
- Add the single-decode split-K path.
- Keep mutable bindings and graph updates outside the fixed-address claim.
- Record capture, replay, completion, and owner-retention boundaries.

Exit: every performance-relevant admitted plan has a fixed-address Graph
correctness record or an explicit exclusion.

## K0.7: Establish dual-provider dense GEMM

**State:** active. Provider-neutral cuBLASLt source path implemented. Loom
provider planned after the current-source baseline.

The current source admits one contiguous BF16 cuBLASLt provider through
`GemmPlanner`, explicit selection, and one provider-neutral plan type. The next
slice adds an explicit Loom provider without replacing the vendor baseline.

- Keep one `Bf16DenseGemmSpec`, plan surface, operands type, and enqueue path.
- Add explicit `CublasLt` and `Loom` provider identities.
- Keep provider and algorithm selection outside enqueue.
- Record `(M, N, K, dtype, layout, frequency)` from a real model path. The
  repository now contains an independent schema and aggregation tool. The
  current-source Qwen census remains pending.
- Implement the first Loom SM90 BF16 small-M algorithm with cuda-oxide.
- Use separate algorithms for GEMV-like and WGMMA-suitable M ranges when the
  profile requires them.
- Compare both providers in both benchmark orders on current source.
- Inspect SASS, local memory, spills, Tensor Core use, memory traffic, and
  scheduler stalls.
- Run host, H20 correctness, lifecycle, sanitizer, Graph, and matched
  performance gates.
- Measure TTFT, TPOT, throughput, and peak memory in the model runner.
- Return a planning error for unsupported Loom shapes. Do not switch providers
  during enqueue.

Record the admission threshold before tuning. The initial target is at least
10% lower median operator latency on declared high-frequency shapes and a
positive engine change outside measured noise. Otherwise the Loom algorithm
remains experimental and cuBLASLt remains the selected plan.

Exit: the same dense GEMM contract can produce either explicit provider plan.
Each published result names its provider, algorithm, source, and timed region.

## K1: Broaden attention contracts

**State:** planned after K0.

- Add sliding-window decode and prefill.
- Add broader head dimensions and page sizes from real model demand.
- Extend ragged tiling beyond the admitted GQA4 shape.
- Add paged tensor-core tiling and asynchronous K/V staging beyond the current
  token-parallel paged-prefill paths.
- Add mixed-batch attention after the engine defines its scheduler interface.
- Scope MLA from a real engine call site.

Exit: each added slice passes host, device, lifecycle, sanitizer, performance,
Graph, and engine gates that apply to its declared contract.

## K2: KV cache, sampling, and speculation

**State:** planned.

- Add KV gather, scatter, compaction, and remapping.
- Add FP8 and INT8 KV storage with explicit quality limits.
- Add logits processing, penalties, Top-K, Top-P, Min-P, and logprobs.
- Add deterministic sampling and RNG-state contracts.
- Add speculative verification and token compaction.

Exit: the operators reduce measured engine work without changing token output
or the declared stochastic distribution.

## K3: Quantization, MoE, and GEMM expansion

**State:** planned.

- Add scale, pack, unpack, dequantize, and layout conversion.
- Add activation and gated-activation contracts from measured model paths.
- Add expert routing, permutation, grouped-GEMM inputs, and weighted combine.
- Add separate dense, grouped, and quantized GEMM contracts.
- Extend explicit Loom and vendor providers through fixed plans.
- Keep FP8 and FP4 out of Loom device code until cuda-oxide supports the
  required types and instructions or Loom contributes them upstream.
- Fuse adjacent work only when the matched complete path improves.

Exit: dense and MoE workloads have separate operator and engine evidence.

## K4: Hardware and distribution

**State:** planned.

- Keep Hopper as the first qualified architecture.
- Add Blackwell as a separate evidence row.
- Publish hashed device artifacts for supported targets.
- Stabilize the Rust API after the first engine integration.
- Add a checked C ABI only when an external engine requires it.
- Add collectives only for a measured distributed workload.

Exit: every supported hardware and API row has reproducible correctness and
integration evidence.

## Admission rule

A faster microbenchmark does not prove a faster model or server. Every result
states its contract, source, hardware, timed region, and excluded claims.
