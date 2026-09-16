# Roadmap

The immediate acceptance model is **Qwen3.8-27B-FP8 on one H20**. The delivery
goal is correct, repeatable model execution followed by a measured serving
advantage over both vLLM and SGLang on declared workloads. An architecture,
generated kernel or successful search does not establish that advantage.

The engine, compiler and state manager live in one Rust workspace. Model-specific
facts belong in import contracts, workload manifests and tests; implementation
selection belongs in compiler rules. Keep one execution path for the default
plan and optional tuning. Wider search, joint layout optimization and persistent
execution must earn their complexity through model measurements.

[Model support](capability-matrix.md) defines the current scope.
[Results](../results/README.md) record measured inference performance.

## Current evidence

- The [three-engine baseline](../results/qwen3.8-27b-fp8-h20-20260915/README.md)
  uses official `vllm bench serve` and fixed token IDs. It is a diagnostic
  comparison; output equivalence, changing-batch correctness and tail-latency
  qualification remain open.
- Prepared provider execution, strict artifact replay, retained-plan CSR checks
  and gather/cast regions are implemented. The
  [region rerun](../results/qwen3.8-27b-fp8-h20-20260915-regions/README.md) did not
  pass the performance-promotion gate. Removing preparation work from a measured
  request path does not by itself establish a throughput improvement.
- Egglog definition/query-plan reuse and bounded connected-choice exploration
  are implemented. They reduce repeated preparation or expand existing choices;
  they do not synthesize a new algorithm. Complete candidate and deployment GPU
  measurements remain authoritative.
- The [FP8 serving rerun](../results/qwen3.8-27b-fp8-h20-20260915-fp8/README.md)
  records corrected quantizer rounding, exact measured first-layer QKV/Z and
  passing independent kernel checks. Full-model numerical qualification still
  fails, model throughput is similar, and C8 P95 TTFT regresses. The isolated
  kernel improvement is not a model win.
- Request-final projection is an artifact-bound executor option, but the
  published serving run retains all-token projection. Shared FP8 preparation is
  opt-in. Independent local checks do not replace full-model qualification.
- A SGLang-derived vector SiLU × up candidate now matches the existing MLP's
  F32 arithmetic and BF16 activation rounding in egglog. It reuses upstream
  vector memory access without adding a second runtime. This is a kernel source
  port, not evidence that the complete model has passed its numerical or serving
  gates; see [kernel reuse](components.md#ported-vector-activation).
- GDN can absorb the input-state Gather into the register scan while retaining
  the original index/data strides, packed output and explicit state commit.
  This removes selected-state materialization where no other reader needs it;
  compact slot descriptors and direct state writes remain separate work.
  Kernel equivalence and manager lifecycle checks pass. The full-model rerun
  did not retain this candidate and still fails its numerical gate, so this
  addition has no demonstrated model-serving benefit yet.
- The first compilation simplification is implemented. Artifact-bound decoder
  profiles default to deterministic `Default` lowering: stable bounded
  extraction, ordinary alias/provider/resource checks and aggregate bucket
  validation, with no candidate GPU timing. Existing benchmark profiles are
  explicitly `Tune`; the whole-program genetic path remains until a complete
  Qwen Default artifact passes numerical and performance comparison. An H20
  two-bucket smoke generated and replayed a Default schedule without recording a
  profiling duration. Singleton decode batches are also split before resource
  planning to reject the previously observed impossible upper geometry early.

### Refactor log — 2026-09-16

The first simplification slice is complete in the working tree. It adds an
artifact-bound `Default`/`Tune` policy, stable extraction and shared deterministic
cycle repair, aggregate bucket validation, and early singleton-batch splitting.
Both policies still produce the same schedule/module artifact and execute through
the same CUDA runtime. No provider, state-lifecycle or model-name dispatch was
added.

The slice passed formatting and diff checks, the source-layout verifier, all host
workspace tests excluding the CUDA crate, 58 Python tool tests, strict compiler
and CUDA Clippy, and the H20 two-bucket Default compile/execute/replay regression.
The device regression observed no candidate profiling. Six existing non-test
executor warnings expose that layer-boundary diagnostics are still intertwined
with production graph construction; moving those diagnostics behind a clean test
boundary is retained as explicit simplification debt.

This checkpoint does **not** include a newly built complete Qwen Default artifact,
does not close the serial/C8/changing-batch numerical gate, and does not establish
a serving-speed improvement. The published serving profile remains explicitly
`Tune` until those checks pass.

The second cleanup slice separates production model construction from layer
diagnostics. Layer traversal now has a cohesive owner, compiler preparation is
split into graph construction, fact admission, capacity validation and schedule
selection/replay, and release builds no longer contain the diagnostic boundary
enum or observed intermediate outputs. The shared attention/MLP formulas remain
single-source so this is a structural cleanup, not a second execution path.
`model.rs` decreased from 1,575 to 1,265 lines and the source-layout gate passes.

An explicit reference audit found that the remaining random extraction helpers
still support Tune and compiler/CUDA equivalence tests. They are retained until
the complete Qwen Default-versus-Tune comparison allows the old population and
restart machinery to be removed as one coherent unit. The locked DeepGEMM source
override remains required for provider-rewrite tests; without it the provider
correctly contributes no rewrite rather than pretending an implementation exists.

## Next milestones

These are delivery priorities, not claims that the proposed modes or models are
implemented. Preserve the existing numerical oracle and error gates throughout.

| Priority | Deliverable | Acceptance |
| --- | --- | --- |
| 1. Correct default execution | Resolve remaining convolution, recurrence, normalization and MLP boundaries; finish HTTP tokenizer parity; establish bounded deterministic lowering and extraction for a reproducible baseline | Independent same-input probes and unchanged full-model gates pass for serial, changing batches, state reuse and tokenize/detokenize parity; no model-name kernel dispatch or Rust graph-replacement pass |
| 2. Useful kernels and regions | Qualify vector SiLU × up and GDN indexed-state reads; then direct state writes and FLA chunk prefill; qualify request-final projection, shared FP8 preparation and a normalization/gating region | Independent references, complete-model logits, ragged prefill/decode and drain pass; complete workloads improve over the existing composition |
| 3. Selected model bring-up | Add GLM-5.3-Flash and DeepSeek-V4.1-Flash on one H20 with CPU weight offload; build shared MoE and bounded weight-residency/transfer support, then admit each model's actual attention, quantization and residual semantics | Pinned checkpoint, measured host/device/storage budget, supported packed-weight format, independent numerical checks and complete request/state drain for each model; small fixtures qualify components, not the complete checkpoint |
| 4. Demonstrated serving advantage | Compare the default plan, local tuning and existing whole-graph search; extend contexts, concurrency, memory pressure, prefix and cancellation traces | Use matched official `vllm bench serve` against both tuned baselines; adequate repeated samples and confidence bounds, no hidden tail-latency or correctness regression; report the exact winning and losing workloads |

The [simplified compilation target](search-coverage.md#simplification-target)
now has an initial Default implementation and the existing Tune comparison,
sharing one runtime and artifact. Next build and validate a complete Qwen
Default artifact, implement hotspot-only Tune from a saved parent, and then
remove redundant population/restart machinery. Numerical disagreement remains
an equivalence bug; resource rejection remains normal admission control.

## Complete kernel code generation

The backend already emits complete CUDA functions for selected elementwise
regions and generates dedicated operations from CUDA templates. It does not yet
synthesize arbitrary reduction, attention or stateful algorithms. An opaque
library call does not become fusible merely by entering the e-graph.

Start with a pure row-normalization/gating region using the existing RMSNorm
emitter. Preserve every dtype boundary, reduction order and external consumer;
state updates additionally require explicit effects and alias validation.
Egglog admits alternatives; lowering emits the selected program without further
graph rewriting. Reuse compilation/cache/capture machinery and qualify full-model
correctness before measuring serving benefit. Mature GEMM/attention providers
remain boundaries. A general tensor-core compiler, multiple new DSLs and
whole-model megakernels are not prerequisites; see [joint compilation](joint-compilation.md).

## Model admission

The selected product targets are **Qwen3.8-27B-FP8, GLM-5.3-Flash and
DeepSeek-V4.1-Flash**, with CPU offload allowed. The public supported-model list
remains the qualified scope in [model support](capability-matrix.md). The two
Flash checkpoints are bring-up targets, not implemented model support. Qwen is
the existing correctness/performance anchor; finish its numerical qualification
before using it to claim a performance advantage.

Official model cards inspected on **16 September 2026**:

| Target checkpoint | Published structure | Required capabilities beyond the current model |
| --- | --- | --- |
| [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | Dense FFN, Full attention and Gated DeltaNet | Close numerical/tokenizer gates; qualify efficient recurrence, normalization/gating and FP8 preparation |
| [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 320B total / 18B active; KDA + sparse attention, dense/MoE FFNs, mHC | Independent KDA transition and sparse-index semantics, expert routing/grouped GEMM, residual mixing, CPU weight offload |
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 552B backbone plus 196B Engram; causal encoder-decoder, CSA2, single-pass mHC | Engram lookup, shared KV/indexer state and top-k reuse, separate FP4 expert-weight and KV contracts, block-32 FP8 with UE8M0 scales, MoE, CPU weight offload |

Pinned checkpoint contracts:

- Qwen: [config at `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/config.json);
  existing checkpoint verification is recorded in [model support](capability-matrix.md).
- GLM: [config at `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json);
  [indexed tensor payload](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/model.safetensors.index.json)
  is 328,326,771,576 bytes (305.78 GiB).
- DeepSeek: [config at `dba1be0a40aa45a94ad051997016db3960a90277`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/config.json);
  [indexed tensor payload](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/model.safetensors.index.json)
  is 510,286,023,000 bytes (475.24 GiB).

Payload uses the pinned indices' `metadata.total_size` and includes auxiliary
checkpoint tensors; it is not text-only runtime residency. Qwen's index has no
`total_size` field, so it is not assigned a value by the same method.

Flash does not mean the weights fit one H20. Use total stored tensors and the
actual encoding for capacity; active MoE parameters describe compute, not weight
residency. Record official revisions, config and tensor-index identities before
downloading weights. A card's parameter count is not a measured runtime budget.

The inspected host has approximately 1.88 TiB RAM, 1.51 TiB available at the time
of inspection, 95.58 GiB H20 memory and 1,991.7 GiB free workspace storage. These
are observations, not reserved resources or a deployment qualification. Plan one
checkpoint at a time and avoid full-weight copies, eager whole-model
dequantization or unbounded pinned-memory allocations. GPU inspection reports a
maximum PCIe Gen5 x16 link, not measured transfer throughput.

Account for host RAM, disk/conversion staging, device KV, recurrent state,
workspace and resident CUDA Graphs. FP4 storage requires H20-compatible execution;
the current E4M3 loader does not establish it.

Build shared capabilities with small, independent fixtures before applying them
to the selected checkpoint: MoE models need expert routing and grouped GEMM;
MLA needs a latent-state ABI and compatible attention; a different recurrent
architecture needs its own transition semantics. Sparse attention, residual
mixing and shared state across layers need their actual contracts as well.
A distilled Qwen model does not qualify native DeepSeek architecture. A linear
attention label does not establish equivalence to the current Gated DeltaNet
transition, and DeepSeek V4 support does not establish V4.1/CSA2 support.

CPU weight offload is required work for this target, not a capability of the
existing loader. The first implementation needs immutable host weight storage,
a bounded pinned staging pool and GPU weight/expert cache, explicit asynchronous
transfer dependencies and completion-safe reuse. Start with an explicit residency
policy, not a global placement search. Immutable weight residency must not create
a second owner of mutable KV/recurrent state.

Resident CUDA Graphs must retain valid weight allocation owners and addresses.
Cache eviction or slot replacement waits for all consuming executions, not just
the upload event. Reusing a stable address requires the correct weight contents
before its next launch; an address/descriptor change requires the corresponding
binding update or recapture before execution.

Use the [offload benchmark contract](benchmarking.md#shared-workload), including
transferred bytes, cache conditions, host resources and measured H2D bandwidth.
CPU expert execution is another qualified implementation, not a consequence of
host weight storage. Loading weights alone does not qualify a model deployment.

## Focused implementation changes

Keep the seven-crate workspace and one executable runtime. Extend the existing
import, operation, provider and binding boundaries as each feature becomes
executable; do not introduce an independent model framework or placeholder
backends for the new names.

1. **Layer semantics.** Evolve the normalized decoder description from the current
   global dense FFN and `Full/Sliding/Linear` classification into explicit per-layer
   attention, FFN and residual contracts. Resolve checkpoint conventions at import;
   algorithm identity, visibility and storage are separate facts. Add only the
   variants with implemented lowering and reference fixtures. GLM uses sigmoid
   routing; DeepSeek uses sqrtsoftplus and different weight/KV quantization.
   Router arithmetic and per-tensor encoding must remain explicit contracts.
2. **State bindings.** Reuse the manager's existing latent-storage and lifetime
   contracts. The executor currently executes ordinary token KV and gated-delta
   state; latent attention and shared producer/consumer bindings still need work.
   Cross-layer sharing must declare readers, writers and completion dependencies.
3. **MoE execution.** Add router/dispatch, grouped expert GEMM and combine semantics.
   The current generic MoE builder gathers selected expert weights; it is not the
   weight-resident/offloaded serving implementation required here. Extend the
   DeepGEMM provider only after checking the pinned grouped API and Hopper support.
4. **Weight residency.** Add immutable host storage, bounded staging and stable GPU
   expert slots in the executor. Keep weight transfers separate from mutable
   KV ownership, with one deployment resource budget and explicit dependencies.
5. **Kernel reuse.** Prioritize dtype-correct normalization/gating and packed
   recurrence, then MoE kernels and model-specific attention. Reuse existing
   providers for GEMM/attention and preserve upstream attribution. The concrete
   [upstream inventory](components.md#kernel-and-scheduling-reuse) distinguishes
   source ports, adapters and scheduling policies.

GDN now offers input state Gather absorption as an indexed pool-read candidate,
retaining the explicit packed next-state output and state commit. Qualify its
complete-model cost and correctness before extending the state ABI. Removing the
commit requires an operation contract for the written input and a state version
tied to the actual writer; an untracked pointer write would bypass existing
alias and dependency validation. A decode-only candidate must establish exactly
one token per request, not merely equal total token and request counts.

The delivery sequence is Qwen correctness and FLA-derived GDN, shared MoE/offload,
GLM's KDA plus compatible FlashMLA sparse/latent attention, then DeepSeek-V4.1-Flash.
The [backend integration plan](attention-providers.md#next-backend-integrations)
records reviewed source pins and prerequisites. FlashMLA's reviewed V4.1 decode
requires SM100, so DeepSeek on H20 needs a separately qualified Hopper path;
CPU offload does not supply that missing kernel.

Keep one validated state realization while proving these changes. General joint
KV-layout search and persistent execution remain deferred. Cold-compilation
improvements must preserve source/target/toolchain identities, bounded caches
and strict replay; compare search policies at declared budgets. See
[compiler boundaries](compiler-boundaries.md) for current extension points.

## Performance and release gates

Use the [benchmark method](benchmarking.md) for every published model comparison.
Keep compile/startup cost separate from HTTP serving and compare representative
workloads rather than one favorable row. Small fixed traces are diagnostic;
a performance-win claim needs adequate repeated samples and confidence bounds
against both baselines, with no P95 TTFT/TPOT regression.

Further models enter the public support list only after checkpoint identity,
independent numerical checks, lifecycle closure and an H20 serving report.
Quantization and CPU offload are permitted by the deployment budget, but each
requires explicit storage, transfer and execution contracts before admission.

Engineering gates remain formatting, Clippy, host tests, relevant CUDA/model
checks and website validation. Keep tests under their owning `tests/` directory,
retain upstream licenses, and remove superseded notes instead of maintaining a
second implementation history in documentation.
