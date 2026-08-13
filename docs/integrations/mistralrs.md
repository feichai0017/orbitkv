# Mistral.rs integration

The current Mistral.rs adapter is an experimental paired-repository proof of
concept. The accepted target imports the fork under `engine/mistralrs` with
`git subtree` and replaces the paired operator adapter with one
`OxidePipeline` full-model boundary. See the
[target engine architecture](../design/tilelang-engine-architecture.md).

Until that migration lands, the adapter stays in the separate Mistral.rs fork
because the current engine owns model execution, storage, streams, and forward
lifecycle. The current Oxide Infer tree does not yet vendor, snapshot,
submodule, or subtree the full engine.

## Repository boundary

Oxide Infer contains engine-neutral contracts, checked bindings, provider plans, stream handoff, and qualification rules.
The Mistral.rs fork contains Candle storage adaptation, feature wiring, model-runner calls, completion drain, and raw integration evidence.

This split keeps the current operator API independent from one engine release
cycle. The target replaces the split with a pinned subtree while keeping the
fork diff localized to shell registration and `OxidePipeline` wiring.

The historical records use these sources:

| Record | Mistral.rs run source | Historical operator source | Record commit |
| --- | --- | --- | --- |
| Model smoke | [`9f6acf2a`](https://github.com/feichai0017/mistral.rs/commit/9f6acf2ac3fd65cf04d9044a3c91939c8fa5d793) | [`d27b6e5`](https://github.com/feichai0017/loom-infer/commit/d27b6e5825755a811cc413be3fa05109cf10abd8) | [`91787004`](https://github.com/feichai0017/mistral.rs/commit/91787004c010afeb8e723fdd8752cb545f84d94d) |
| Recovery gate | [`805dc8f1`](https://github.com/feichai0017/mistral.rs/commit/805dc8f1f5aa80c2d37460d18779290071889a88) | [`d27b6e5`](https://github.com/feichai0017/loom-infer/commit/d27b6e5825755a811cc413be3fa05109cf10abd8) | [`4f096d7c`](https://github.com/feichai0017/mistral.rs/commit/4f096d7ca4a4b87e1de07d903331f1aa744b3811) |

Later source does not inherit this qualification.
Each adapter lifecycle or binding change requires a new pinned source pair and new evidence.

## Historical POC boundary

The 2026-08-11 H20 model smoke used Qwen2.5-1.5B-Instruct with BF16 HND paged decode.
The former provider completed 196 operator submissions across 28 layers and seven decode steps.
The former and standard Mistral.rs providers selected the same eight token strings.

The adapter recorded nine external regions and no adapter-issued device-to-device copy.
This proves the adapter submission path for that fixture, not full-model zero-copy execution.

The recovery gate queued valid commands around one invalid page index.
The first drain returned a typed `PageIndexOutOfRange` error at FIFO position two.
Each valid output matched the CPU reference, and the runtime accepted another command after the rejection.

The Mistral.rs fork owns the [model smoke record](https://github.com/feichai0017/mistral.rs/blob/91787004c010afeb8e723fdd8752cb545f84d94d/mistralrs/examples/advanced/loom_paged_attn/h20-smoke-20260811.json) and [recovery record](https://github.com/feichai0017/mistral.rs/blob/4f096d7ca4a4b87e1de07d903331f1aa744b3811/mistralrs/examples/advanced/loom_paged_attn/h20-adapter-recovery-20260811.json).
Oxide Infer links to those records and does not copy them into its evidence directory.

The paired POC feature resolves Oxide Infer through `../oxide-infer`. A
standalone Mistral.rs checkout cannot build that feature.

## Model-owned runtime evidence

Mistral.rs source [`84602212`](https://github.com/feichai0017/mistral.rs/commit/846022129a43550bb5383b2d2faa33ac380dc4ca)
moves the runtime, pending completions, and provider statistics into each
`NormalPipeline`. The validated source manifest used the former source
`d27b6e5`.

The H20 adapter gate completed seven commands. It returned one typed
`PageIndexOutOfRange` rejection at FIFO position two and then reused the same
runtime. Two concurrent drain callers settled one two-command FIFO without
splitting it. All six valid gate outputs matched the CPU oracle.

The Qwen model path completed 196 of 196 paged-decode operator calls with no
provider error. The former and standard providers selected the same eight token
strings and decoded text. The adapter recorded nine external regions and no
adapter-issued device-to-device copy.

The Mistral.rs fork owns the immutable
[model-owned runtime record](https://github.com/feichai0017/mistral.rs/blob/470a54ab25768ca9d271df8e86569c1ff253ea53/mistralrs/examples/advanced/loom_paged_attn/h20-model-owned-runtime-84602212-20260811.json).
Commit [`470a54ab`](https://github.com/feichai0017/mistral.rs/commit/470a54ab25768ca9d271df8e86569c1ff253ea53)
adds that record without changing the validated runtime source.

## Not qualified

The published POC does not qualify:

- performance or serving speedup.
- general production safety.
- bitwise numerical equivalence.
- full-model zero-copy execution.
- failed, panicking, or abandoned model-forward recovery.
- bridge, execution, or CUDA-context failure recovery.
- concurrent enqueue and drain.
- CUDA Graph execution.
- speculative decode or tensor parallelism.
- multiple GPUs, CUDA streams, or concurrent models.
- general batching or model coverage.

The historical run sources use process-global runtime and completion state.
The model-owned record qualifies one single-H20, single-model, single-ordinary-stream path.
It does not qualify a general engine provider.

## Admission rule

An engine adapter remains downstream until its engine needs a stable Oxide
Infer boundary. A separate adapter package can then use that boundary. The
three core crates do not absorb engine runtime types.

New integration evidence must name both commits, the hardware, the model contract, the provider trace, and excluded claims.
The adapter repository keeps engine-specific commands and raw outputs.
