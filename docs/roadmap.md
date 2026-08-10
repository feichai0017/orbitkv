# Loom Infer roadmap

Loom Infer is a Rust operator layer for production LLM inference engines.
Custom NVIDIA kernels use cuda-oxide. Qualified GEMM and communication
libraries remain explicit providers.

The roadmap closes ownership and engine boundaries before it adds more
operators.

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

**State:** partial.

The command layer now has typed read and read-write regions for complete
`DeviceBuffer<T>` allocations, owned subranges, and leased external
allocations. Owned buffers and external regions use one resolver path.

- Keep pointer, element span, CUDA context, access mode, and lifetime lease in
  one region value.
- Reject invalid ranges, alignment, context, and binding-set overlap before
  enqueue.
- Keep writable access exclusive until completion settles.
- Test non-zero offsets and lifetime retention on H20.
- Qualify fixed-address Graph retention for external leases.
- Prove that an engine-owned region crosses no tensor copy.

Exit: an engine-owned allocation enters an operator without a tensor copy, and
H20 evidence proves that completion keeps its lease alive through asynchronous
execution.

## K0.3: Report device metadata errors

**State:** partial.

Fused append now runs one validator, writes a compact scope-bound `AppendMap`,
and reports semantic metadata failures through completion. A rejected command
returns its checked bindings and does not poison the queue or Graph. Paged
decode and prefill still use silent in-kernel guards.

- Extend the status protocol to paged decode and paged prefill.
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

**State:** planned.

The first integration target should exercise the same Rust plan and command
path used by hardware validation.

- Select one real decode or prefill call site.
- Bind engine-owned device regions and the engine's non-default stream.
- Record provider hit counts and selected algorithms.
- Prove that Q, KV, output, workspace, and metadata cross no host copy.
- Compare model output or generated tokens with the engine baseline.
- Measure the complete engine interval before making a latency claim.
- Document the fallback policy at the engine boundary. The Loom provider
  itself does not silently fall back.

Exit: a real model invokes Loom, preserves declared numerical behavior, and
produces an auditable no-copy trace.

## K0.5: Make algorithm policy explicit

**State:** planned.

Current attention selection uses fixed source heuristics. Paged decode chooses
direct MHA and eight-warp MQA or GQA. Ragged prefill uses the batch average KV
length and head mapping.

- Separate algorithm selection from immutable launch plans.
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

## K1: Broaden attention contracts

**State:** planned after K0.

- Add sliding-window decode and prefill.
- Add broader head dimensions and page sizes from real model demand.
- Extend ragged tiling beyond the admitted GQA4 shape.
- Add optimized paged-prefill long-context paths.
- Add mixed-batch attention after the engine defines its scheduler interface.
- Scope MLA from a real engine call site.

Exit: each added slice passes host, device, lifecycle, sanitizer, performance,
Graph, and engine gates that apply to its declared contract.

## K2: KV cache and decode tail

**State:** planned.

- Add KV gather, scatter, compaction, and remapping.
- Add FP8 and INT8 KV storage with explicit quality limits.
- Add logits processing, penalties, Top-K, Top-P, Min-P, and logprobs.
- Add deterministic sampling and RNG-state contracts.
- Add speculative verification and token compaction.

Exit: the operators reduce measured engine work without changing token output
or the declared stochastic distribution.

## K3: Quantization, MoE, and matrix providers

**State:** planned.

- Add scale, pack, unpack, dequantize, and layout conversion.
- Add expert routing, permutation, grouped-GEMM inputs, and weighted combine.
- Call qualified dense, quantized, and grouped GEMM libraries through fixed
  plans.
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
