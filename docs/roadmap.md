# Roadmap

The model compiler is now integrated as four owned OrbitKV crates in the root
workspace. The [migration qualification](../results/workspace-integration-20260914/README.md)
records source identities, B1/B8 replay, numeric parity and state drains.
See [compiler maintenance](compiler-maintenance.md) for the package/namespace
change and artifact regeneration requirements.

The checkpoint/semantic boundary and inference-only fork reduction are implemented;
see [checkpoint import](checkpoint-import.md) and its
[qualification](../results/semantic-boundaries-20260914/README.md). Logical attention,
explicit paged KV views and declarative provider admission are now separated;
see [attention providers](attention-providers.md) and the
[H20 provider qualification](../results/provider-kernels-20260914/README.md).
FlashInfer algorithms and optional FlashAttention-3 replace the handwritten
native attention candidate. [B1/B8 workload attribution](../results/workload-attribution-20260914/README.md)
now connects full rule identities, bucket dimensions and GPU execution steps.
Fixed-snapshot sampling and bounded initial exploration are implemented; the
[27B coverage follow-up](../results/search-coverage-20260914/README.md) passes
correctness but records mixed performance. Required state-alias checks now run
before fusion source generation and CUDA compilation, with final-load checks
retained. [Profile-directed local exploration](search-coverage.md) is now
implemented behind an artifact-bound attempt budget: exact execution provenance
orders single-choice neighbors of measured valid parents. Whole-graph and final
CUDA Graph scores still select programs. Multi-choice dependency closures and
the costly `glumoe` ruleset remain the next compiler priorities.
The [preflight qualification](../results/state-preflight-20260914/README.md)
passes 296 reference comparisons; candidate rejection is cheaper in the recorded
run, while warmed decode remains close to the preceding observation.
The [hotspot qualification](../results/hotspot-search-20260914/README.md)
passes 296 comparisons: 49 local candidates are measured, including two prefill
provider transitions inside valid parents. B8 diagnostic decode is 37.29 ms,
but its initial seed already contains cuBLASLt; this is not a same-snapshot
search ablation or a serving claim. The `glumoe` counters still total 358.31 s.
Additional KV representations and joint state/layout competition remain open.

OrbitKV targets one native Rust inference process. `orbitkv` compiles and owns
attention-state lifetimes, the OrbitKV compiler compiles and executes model graphs,
`orbitkv-engine` combines request/device coordination and the optional client
frontend behind separate internal modules. Planned work is not a current capability.

The compiler direction is specified in [Joint compilation](joint-compilation.md).
Qwen3.8 27B block-FP8 is the first acceptance workload. Optimizations must be
selected from mathematical semantics, state/layout contracts, dtype, shape,
and target capabilities; checkpoint names and fixed layer numbers must never
select implementations.

The hardware budget is **one H20, with quantization and CPU offload allowed**.
Qwen, GLM, Kimi and DeepSeek are the primary families. The dated
[model target matrix](model-targets.md) records their latest release targets,
smaller architecture witnesses, hardware blockers and acceptance gates.

## Current baseline

- `RuntimeManifest` is the shared source of truth for lifecycle and execution.
- OrbitKV is the sole KV authority: it owns pages, generations, snapshots,
  Prefix/COW, retirement, acknowledgement, and reuse.
- Full, Sliding, Full+Sliding, and exact Chunked lifetimes compile and pass host
  lifecycle tests. Token-level relocation and live-token compaction are not part
  of the product.
- Token-KV and stateful hybrid decoders search one symbolic graph into decode
  and packed-prefill buckets. Dynamic inputs use stable addresses and each
  state class has one persistent arena.
- Persistent K/V updates are required aliases during search and artifact load.
  Candidates that materialize incompatible state fail closed.
- Recurrent and convolution classes now have stable per-class CUDA arenas,
  generation-checked byte-range lowering, typed OrbitKV compiler state bindings, and
  runtime-identity- and event-gated completion evidence. Dynamic slot metadata
  selects manager-authored destinations while the arena address stays fixed;
  graph search uses a private scratch arena and cannot mutate live OrbitKV
  state. The production decoder now owns these arenas and returns event-backed
  state evidence for decode and packed prefill. The stable-arena two-step gate
  and ragged packed-kernel parity gate pass on H20.
- The decoder emits logical attention and an explicit paged KV view. Egglog
  admits FlashInfer CUDA-core decode, FlashInfer tensor-core attention, and the
  optional SM90 FlashAttention-3 adapter. The handwritten native attention
  implementation and experimental attention flag have been removed.
- DeepGEMM, FlashInfer and FlashAttention use one pinned provider-source policy:
  explicit local checkout or explicit prefetch into OrbitKV compiler's provider cache.
  Model compilation performs no network fetch. Decoder schema 11 binds explicit
  request geometry through provider lowering; older artifacts require fresh search.
- Released H20 closures exist for a dense Full checkpoint, an interleaved
  Full+Sliding checkpoint, and bounded text-only Qwen3.8-27B-FP8 execution with
  recurrent/convolution state and block-FP8 linear operators. Exact Chunked,
  MLA, MoE, multimodal execution, and multi-device execution are not
  released-model-qualified.
- The single-process Rust engine and OpenAI-compatible server pass bounded
  batching, cancellation, streaming, shutdown, and final-drain tests.
  They now share `orbitkv-engine`: logical protocol, optional frontend, and
  model scheduling remain separate modules, with all test source under `tests/`.
- The primary 27B block-FP8 checkpoint now has a bounded single-process serving
  path. After schema 5 removed token-KV copy-back, the current 4-input/8-output/C1
  diagnostic reaches 0.620x SGLang and 0.539x vLLM output throughput. An
  eight-token fixed-prompt trace contains
  a near-tied step; the current teacher-forced oracle gate keeps all steps on the
  same input sequence and observes maximum absolute logit error 0.625. This is a
  negative diagnostic rather than a competitive result. The best older
  released-model comparison remains 0.598x SGLang on its recorded C2 trace.
- External export, restore, deletion, and failure semantics pass through the
  host-memory reference transport. Mooncake, NIXL, remote leases, and network
  benefit remain open.

## Product model strategy

The primary release target is the local official Qwen3.8-27B-FP8 checkpoint.
Its Hugging Face architecture class remains `Qwen3_5ForConditionalGeneration`
because Qwen3.8 is built on that architectural foundation; the model card and
`base_model` identify the release as Qwen3.8. Its text decoder is the forcing
function for the product architecture: 64 layers with a
3:1 Gated DeltaNet/Full-attention schedule, persistent recurrent and causal
convolution state, partial rotary dimensions, and dynamic block-FP8 linear
operators. OrbitKV already compiles the checkpoint into 16 Full token-KV layers
plus 48 recurrent and convolution layers. The executor now parses the nested
  text configuration and carries fixed-state geometry into its compiler contract.
  Its production GDN graph covers split projections, minimal
  convolution history, grouped-head recurrence, gating, normalization, output
  projection, stable manager-owned arenas, and completion evidence. Packed
  prefill now uses typed causal-convolution and delta-scan custom ops selected
  through OrbitKV compiler's normal rewrite/search path. Checkpoint FP8 execution,
  bounded serving, and eight-step independent logit parity now pass; robust
  near-tie output equivalence and serving-scale qualification remain explicit
  gaps.

Keep the structurally equivalent small BF16 checkpoint as a fast regression
witness for the same 3:1 layer schedule and state transitions. The Qwen3.8 27B
FP8 checkpoint is now the active correctness and performance target. Existing
dense Full and Full+Sliding checkpoints remain lifecycle witnesses. Model
support stays structural, so no checkpoint name
may select an operator or physical layout in product code.

The next shared foundations are MoE routing/dispatch and latent attention,
qualified first with GLM-4.7-Flash and DeepSeek-V2-Lite-Chat. Kimi Linear supplies
a smaller KDA/MLA witness. They then support advancement toward
Qwen3.8-Flash-Next, GLM-5.3-Flash, Kimi K3 and DeepSeek-V4.1-Flash; newer sparse
attention, residual and state-sharing semantics require their own contracts.
Single-H20 expansion also requires validated quantization and bounded host
weight residency. Memory fit and compatible kernels are independent gates;
the latest targets are not currently executable in OrbitKV. Follow
[model targets](model-targets.md) for the sequence and exact claim boundaries.

## Joint compiler milestones

The target compilation space has three complementary implementation levels.
A compiled model may mix them, with profiling choosing among legal candidates
at each boundary:

| Level | Intended compilation effect | Current scope |
| --- | --- | --- |
| Provider selection | Choose a complete optimized implementation with explicit layout, resource, and state contracts | Logical attention + paged KV admits explicit FlashInfer algorithms and compatible FlashAttention-3; DeepGEMM variants implement block-FP8 linear. FlashMLA requires an adapter and the semantic/KV contract of the specific kernel, rather than selection by library or checkpoint name. |
| Algorithm regions and fusion | Combine primitive operations, direct state access, and finite tiled algorithm templates into generated kernels | Elementwise and several dedicated rewrites exist. An opt-in egglog alternative shares FP8 activation preparation across projections while retaining separate quantizer/GEMM launches. General attention-internal scheduling, multiple packed GDN scan algorithms, and wider layout-aware fusion remain open. CUDA, Triton, and TileLang can supply implementations; DSL adapters are not integrated today. |
| Generated persistent execution / megakernels | Derive bounded device schedules from model regions and reduce dispatch and intermediate materialization where measurements justify it | Planned. Requires explicit synchronization, resource-residency, state-access, and completion contracts. Whole-model single-kernel execution is not a release requirement. |

Advance the following compiler milestones while retaining the primary model's
correctness and serving qualification gates:

1. **Reproducible contracts.** Bind provider caches to resolved source and
   dependency contents, compiler identity, target, flags, and wrapper source.
   Extend reusable layout/access facts for base storage, strides, aliasing, and
   state mutations; preserve required state writes on both search and replay.
   Record code, provider, artifact, oracle, and workload identities together.
2. **Measured region optimization.** Supply valid batch/ragged/context profile
   fixtures and compare multiple finalists on the deployment path under an
   explicit compile budget. Prioritize quantized-linear preparation and
   GDN/view/norm/gate regions using the primary model's attribution profile.
   Each new implementation must pass independent numeric and next-state parity,
   strict artifact replay, and a complete warm-path comparison.
3. **Joint state-realization search.** Let the executor coordinate a small set
   of explicitly supported OrbitKV physical realizations with OrbitKV compiler algorithm,
   provider, and kernel choices. Start with the admitted page-16 contract.
   Changing a persistent layout requires a newly validated manifest and matching
   bindings/artifact; prefill and decode must share a compatible state ABI or
   pay for an explicit, validated conversion. Evaluate total state/workspace
   memory, throughput, tail latency, and compilation cost on fixed workloads.
4. **Bounded persistent-kernel candidates.** Introduce a generated schedule for
   one legal region, including cross-layer regions when dependencies permit,
   then widen its coverage only after correctness and measured
   benefit. Preserve a deployable multi-kernel alternative, account for register
   and shared-memory pressure, and express all cross-block synchronization.
   External host-launched library calls remain explicit execution boundaries.
   OrbitKV continues to authorize state ownership and publication; a device
   schedule cannot infer page-reuse permission from its last local read.

The first provider reproducibility slice is implemented: source/dependency and
wrapper identities are checked during compiler-selected schedule replay, and
shared-library keys include the resolved `nvcc` and declared compilation inputs.
The first region slice adds artifact-bound workload profiles, valid private-page
batch/ragged profiling inputs, multiple deployment finalists, and an opt-in
shared-FP8 preparation alternative. Its acceptance evidence and limitations are
recorded in [FP8 region tuning](fp8-region-tuning.md). Reusable layout/effect
consolidation, a complete deployment/toolchain identity, broader regions and
joint state-layout search remain open; see
[implementation status](implementation-status.md) for the exact cache scope.

The next work inside the measured-region milestone is search quality and compile
cost. The current two-bucket attention-contract qualification spends 240.30 s in
fresh `compile_or_load`, including 192.63 s building/saturating the search space;
strict prepared replay takes 16.81 s. This reinforces the priority of reusable
interval-invariant work. These are CPU wall intervals, not GPU kernel timings.
The earlier seven-bucket 27B acceptance run spent about 953 seconds inside
`compile_or_load` for fresh schedule creation and 42.5 seconds for strict replay;
the corresponding test processes took 955.6 and 44.4 seconds. These intervals
include more than compiler search and exclude the earlier Rust build. Its B8
decode still selected a
generic BF16 output projection that accounts for about 43 ms in the instrumented
trace. Three deployment finalists can correct an ordering error among retained
candidates, but cannot recover an implementation that exploration never retained.
Stage-level measurements are now available; next isolate reusable interval-invariant work and
direct finite exploration toward expensive operations and coherent multi-consumer
regions using measured costs and semantic contracts. Keep this independent of
checkpoint names, and repeat complete-workload comparisons before enabling a
candidate by default.

The first final-server off/on diagnostic reinforces this ordering: paired
throughput rises only 1.82%, TPOT falls 5.82%, TTFT rises 8.26%, and two of eight
generated texts differ. The on-mode decode changes its LM-head provider but
does not select a quantizer with multiple consumers. Keep the option off by
default. The subsequent frozen-v3 teacher-forced probe localizes both first
divergences to tied OFF logits and OrbitKV compiler's existing highest-index argmax rule;
ON has unique maxima at those steps and matches the independent reference.
This explains the observed sampling decisions, not the numerical error of each
internal operator. Keep numerical tolerance and sampling tie semantics explicit;
see [independent logits diagnosis](logit-diagnosis.md).

[Structured candidate tracing](compiler-boundaries.md) now preserves program
identity, complete operation manifests, rejection reasons, and direct/deployment
scores together. A [compiler-boundary H20 check](../results/compiler-boundaries-20260912/README.md)
verifies both bucket identities through saved-artifact replay, alongside eight
reference steps and final drain. Use that evidence to explore coherent shared regions and
expensive operations deliberately. Also measure first-use and
bucket-transition costs. Budget retained prefill/decode executables together
with KV storage and workspace instead of increasing graph residency without
accounting for memory. These followups precede broader kernel families or a
full-model megakernel.

The [first stage-attribution run](../results/engine-stage-attribution-20260913/README.md)
now separates those costs on a newly selected B1/two-bucket artifact. The 299.7 s
search process includes 200.6 s building the search space (180.8 s executing
egglog schedules), 70.9 s in CUDA search, and 26.7 s preparing graph/weights.
These enclosing intervals explain the observed startup; inner NVRTC/provider
timers overlap them and must not be added again. Strict replay takes 37.3 s,
including 20.9 s weight loading and 11.1 s schedule loading. It still invokes
NVRTC 425 times. A stage-off replay of the same artifact confirms the numerical
result and roughly 24.2 ms warm diagnostic decode, while first decode is about
138 ms. The traced first decode spends 89 ms materializing its CUDA Graph.
These are bounded diagnostic timings with logits, not serving TPOT or an
improvement over the earlier, differently selected artifact.

The [B1/B8 attribution follow-up](../results/workload-attribution-20260914/README.md)
also fixes request geometry: FlashInfer carries an explicit request expression
through lowering and retained-bucket resource planning. Runtime CSR lengths are
validated against that contract. Seven buckets now compile and replay with
schema 10; two frozen builds pass 592 reference comparisons and final drain.

Ordinary compilation collects egglog timings without materializing verbose
query-plan reports. One full-plan/time-only process pair takes 744.69/657.04 s,
with `glumoe` reporting 433.61/354.19 s. These are observations, not an isolated
speedup measurement: all seven selected program fingerprints differ despite the
same seed and one-graph search budget. B8 decode improves while prefill regresses.
The final prefill's generic BF16 output projection takes 171.15 ms in the event
profile, although its saved equivalence class contains a `cublaslt` candidate.
Both modes pass the unchanged logit gate. Search coverage is the immediate
runtime problem; this result does not establish a serving improvement.

[Snapshot sampling and bounded initial coverage](search-coverage.md) now remove
hash-table iteration from candidate draws, mutation pools and cycle repair.
Caller-selected `initial_candidates` adds broad initial exploration before
mutation; trace records identify snapshots, sampling origins and measured kernel
implementations. Fixed-snapshot GPU regression covers generated matrix products
and cuBLASLt with independent numerical checks. Fresh saturation is still not
canonicalized, and broad sampling does not guarantee that expensive regions
receive all useful provider alternatives.

The [eight-graph qualification](../results/search-coverage-20260914/README.md)
records 319 evaluations, 56 measured graphs and 263 state-alias rejections. All
296 logit comparisons and final drains pass, but B8 decode's eight measured
graphs all retain the slow generic vocabulary projection. Forty-six graphs with
the cuBLASLt projection fail state validation elsewhere. The next coverage slice
motivated preserving a valid surrounding genome while exploring expensive
regions and applying provable state constraints before costly preparation.
The optional hotspot phase now implements single-choice exploration with those
checks; its measured coverage must be distinguished from broad random draws.

The next implementation slices are:

1. **Candidate coverage and cold compilation:** build on the bounded hotspot
   qualification. Add a same-snapshot search ablation and a matched serving run
   before changing defaults, retaining exact parent/choice traces and independent
   full-model outputs.
   Extend beyond single-choice neighbors only where measured misses require a
   dependency closure, preserving unrelated bindings. Measure legal provider alternatives
   for expensive regions under a shared budget, covering both prefill and decode.
   Do not force a provider by model name or tensor dimensions. Restructure the
   `glumoe` joins in egglog using semantic anchors, then separate reusable setup
   and bucket-independent transformations from interval-dependent rewrites.
   Reuse requires a semantic key and must preserve the bucket's alias, range,
   and state constraints. In the earlier stage-only run, candidate
   generation costs 3.0 s here; the 118 rejected candidates account for 37.3 s
   of evaluation. Move provable legality checks ahead of expensive preparation
   while preserving accepted implementations. Keep candidate order and program
   identities in comparisons: a fixed RNG seed alone is not evidence that two
   runs explored the same programs.
2. **Artifact startup:** generated module capture/replay is now connected through
   [decoder schema 6](module-artifacts.md), with target/NVRTC/options checks,
   source-keyed images, integrity checks and retained strict runtime lookup.
   Older decoder formats now require regeneration; the schedule-only
   compatibility and image-omission APIs are removed. The recorded fixed-program
   image experiment qualifies replay startup and measures the extra capture
   pass separately. [Weight loading](weight-loading.md) now borrows mapped bytes
   for storage-compatible inputs and uses one typed buffer for conversions;
   file, encoding and CUDA errors propagate through a single fallible API.
3. **Runtime transitions and regions:** budget retained decode/prefill graphs
   with KV, fixed state, and workspace; test prefill/decode alternation and
   eviction before changing residency policy. Use the selected artifact to
   target quantized projections and gather/cast/fused regions. The instrumented
   profile does not identify attention as the dominant cost on this short
   context. Preserve numerical and next-state gates, then repeat an
   uninstrumented complete serving workload before claiming a benefit.

The [module-image follow-up](../results/module-image-artifact-20260913/README.md)
closes the generated-module replay slice. Two fixed-artifact ABBA timing pairs
on H20 reduce schedule loading from 11.40 s to 4.96 s and complete diagnostic
process time from 38.69 s to 33.28 s (medians). Each cached replay obtains all
428 images and invokes NVRTC zero times. All nine processes preserve eight
teacher-forced reference steps and drain, with maximum absolute error 0.5 under
the unchanged 1.0 gate. Fresh compilation adds a 7.43 s selected-module capture
pass; its newly selected programs differ from earlier records, so cold-process
times must not be compared as a compiler speedup. Warm diagnostic decode stays
about 24.5 ms.

The [weight-loading follow-up](../results/weight-loading-20260913/README.md)
finds 17.36–18.52 s of explicit host copies in the old loader. Two fixed-artifact
timing pairs reduce weight loading from 23.12 s to 6.60 s median and complete
diagnostic process time from 33.36 s to 17.04 s. All eight processes load the
same 1,251 tensor bindings, hit all 428 module images without NVRTC, pass the
same eight reference steps and drain. Warm diagnostic decode remains about
24.6 ms. The new loader also validates a shard's encodings before binding,
returns errors and keeps test source outside production modules.

The [bucket ownership slice](graph-residency.md) now gives each retained FlashInfer
plan private integer metadata, serializes pinned staging reuse, and accounts for
the temporary coexistence of old and replacement plans. The executor and merged
engine/server expose a finite bucket cache capacity, retaining the default of
one. H20 provider and repeated 27B request regressions cover phase alternation,
reference logits, state drain and eviction. Automatic residency selection from
a joint budget and broader long-context/ragged model transitions remain open.
Two [fixed-artifact timing pairs](../results/bucket-resources-20260913/README.md)
compare capacities one and two without stage tracing: repeated prefill falls
from 127.91 ms to 29.41 ms and first decode from 128.68 ms to 25.93 ms. Warm
diagnostic decode stays about 24.4 ms. Five model processes pass 160 reference
rows and 20 complete drains; these measurements qualify phase-switch savings.

The [serving follow-up](../results/bucket-serving-20260913/README.md) now runs
complete HTTP requests through one fixed serving artifact at capacities one and
two. Eight C1 timing processes complete 128 requests and 4,608 tokens with exact
paired text and final KV/fixed-state drain. Observed throughput rises 86.8% for
eight-token output and 13.0% for 64-token output; steady inter-token latency stays
about 23.6 ms. The short-output P99 TPOT increases 16.8%, including a slow first
decode interval. Keep the default capacity of one and retain this tail result.
The engine now returns a checked shutdown report; an active SSE request is
cancelled and drained on SIGTERM.

Explicit startup preparation now uses the selected artifact's valid bucket
representatives and the configured residency capacity. The engine performs it
before publishing readiness and exposes a startup report. Preparation preserves
dynamic execution, does not launch the model or write persistent state, and
applies the normal eviction policy. Existing materializations take priority;
unused slots are filled in artifact order. Runtime residency, executor
representative inputs and engine startup each have a separate module owner,
with their tests under the corresponding `tests/` trees.

The [startup preparation qualification](../results/startup-preparation-20260913/README.md)
holds one binary/artifact fixed and checks preparation at capacities two and one.
All 24 H20/C1 timing processes complete 352 requests and 13,824 tokens with
paired output and final state drain. At capacity two, the short-output first
stream interval is 138.9→29.9 ms and P99 TPOT 37.7→24.5 ms; approximately
126 ms of preparation occurs before readiness. Steady ITL remains around
23.6 ms. Context growth through 128 generated tokens passes; that profile's
P99 TPOT is 25.1→24.8 ms. These are narrow paired observations, without a
statistical-significance or production-tail claim. Four model processes compare
128 full-vocabulary output rows against an independent reference and verify
state release through 16 request lifecycles.

The default capacity remains one. Its preparation preserves the already loaded
bucket without an extra full graph build, while ordinary phase switches still
require eviction. Its observed P99 TPOT is 42.4→43.9 ms and throughput
is 18.34→18.12 token/s; no default-capacity performance benefit is established.
A first implementation's unnecessary default-capacity eviction
was caught, corrected and retained as a regression record under its earlier
source identity. All final timing profiles were rerun on the corrected binary.
An earlier 128-token prefill exceeded the fixed four-token admission limit;
its empty responses were rejected by the output-length gate and remain outside
the performance samples. Multi-request and long-prefill qualification require
appropriate capacity configurations and artifacts. Automatic residency selection
and workload-aware preparation ordering remain open.

Extend the search objective to measured request costs
before attempting automatic residency selection. For the selected program's
dominant regions, add legal quantized-linear and gather/cast/norm/gate
alternatives with independent parity and complete-request timing. The
remaining replay graph construction (about 4.29 s) and schedule loading (4.97 s)
are separate startup work. Repeat a complete uninstrumented serving workload
after runtime changes; the current loading improvement does not raise token
throughput.

The output is a versioned execution program containing the state realization,
compatible bucket schedules, provider/generated-kernel identities, resource
plans, and runtime applicability guards. The search finds the best validated
candidate within its supported space and budget; it does not claim global
optimality or guarantee that a megakernel beats a mixed execution plan.

## Primary model closure

1. Keep the H20-passing stable fixed-state arena, shared-alias, stream-ordered
   completion, and ragged packed-kernel parity gates as regression prerequisites
   for all recurrent/convolution kernels.
2. The backend-neutral gated-delta recurrence now has an independent f32 oracle,
   grouped key/value heads, and a pure OrbitKV compiler single-token graph. Projection,
   minimal `K-1` causal-convolution history, gates, recurrent update, gated
   RMSNorm, and output projection compose without model-name dispatch. The
   first OrbitKV compiler-native in-place state-update candidate exists and is
   introduced only by an exact rank-four egglog match. A generic graph arena
   gathers and commits manager-selected request slots in manifest layer order,
   while initialization copies the prior published slot before execution.
   Packed convolution and delta scan are typed custom ops whose state commits
   are selected through egglog and must alias manager-owned arenas. Their CUDA
   sources compile through the production renderer for static and symbolic
   geometry, and a two-request 2/3-token H20 test matches independent
   convolution and recurrence references.
   The production decoder now consumes the joint topology, binds both arenas
   after search, executes decode or packed prefill with manager-authored slot
   plans, and returns event-backed evidence to the engine.
3. Qualify the complete recurrent decode and packed-prefill model paths,
   cancellation, Prefix boundaries, and state-slot reuse on the small BF16
   checkpoint; the operator-level H20 gates do not substitute for this
   released-checkpoint closure.
4. Preserve the implemented partial RoPE and nested text-checkpoint loading
   contracts, and continue rejecting unimplemented image/video inputs.
5. Landed the first block-FP8 execution slice: the decoder now declares FP8
   projection weights and their 128x128 inverse scales, while OrbitKV compiler exposes a
   provider-neutral `BlockScaledLinear` semantic op. An independent CUDA
   reference implementation and four pinned DeepGEMM schedules join
   the same e-class and are selected by device profiling. Full-graph search now
   fits within the compiler memory budget. A general egglog rule unifies
   equal-valued `LoopInput` streams, allowing required in-place state contracts
   to eliminate multi-gigabyte recurrent/convolution copy-back while preserving
   the independent eight-step logit gate.
6. The multi-token gate now teacher-forces the independent reference token
   after each step and requires `max_abs <= 1.0`. Exact top-1 is required when
   the reference margin exceeds that measured error envelope; near ties may
   reorder only within the envelope. This avoids turning a one-step BF16/FP8
   tie into a different downstream request while still rejecting material
   semantic divergence. Next widen continuous-batching, long-context, and
   pressure qualification.

## Searchable attention execution

Logical attention and KV storage have separate contracts. The production backend
uses mature upstream kernels: FlashInfer CUDA-core decode, FlashInfer tensor-core
attention, and optional FlashAttention-3 on SM90. The former handwritten native
attention provider is removed. Algorithm identity belongs in serialized schedule,
plan and capture keys; the selected family is not changed at request launch.

Maintain these regression gates when expanding coverage:

1. Keep attention mathematics in `AttentionSpec`, with state class, traversal
   metadata and physical geometry in an explicit `KvView`.
2. Admit complete dtype, head-dimension, layout, phase, page-size and target
   combinations through provider capabilities.
3. Express matching and selection through egglog and measured OrbitKV compiler search.
   Model-name, release-name and GPU-product-name dispatch do not belong here.
4. Preserve required K/V aliases and account for metadata conversion, scheduler
   launches, workspace and retained graph owners. Persist provider identity.
5. Independently verify each candidate, then verify saved schedules, complete
   models, changing CSR data, phase transitions, cancellation and final drain.

The [provider contract](attention-providers.md) documents the implemented FA3
non-TMA, packed-GQA, unsplit algorithm. Next, use measured workload evidence to
prioritize split-KV/TMA candidates, narrower metadata tables and joint physical
layout choices. A provider registry does not yet supply those alternatives.
Historical native-provider evidence remains in its original result directories.
That closure also found and removed an unsound 3-D RMSNorm candidate: logical
`(tokens, heads, dim)` shape did not prove dense rows for Q/K slices with a
wider projection pitch. Searchable fused RMSNorm is now limited to proven-dense
2-D layouts; 3-D views remain on the semantic decomposition until base-layout
facts are represented explicitly.

## Additional attention-state families

Coverage advances by state family rather than checkpoint-name branches:

1. Preserve the implemented recurrent/convolution arena bindings and atomic
   engine step; extend their bounded primary-model closure to longer sequences,
   mixed requests, cancellation, and state-slot reuse.
2. Independently qualify exact Chunked token KV on device.
3. Add sparse retrieval/index state and low-rank attention components for the
   second architecture target.
4. Add MoE routing, mixed low-precision experts, and multi-device placement only
   after the dense primary target closes.
5. Defer tree, cross-attention, speculative decoding, and vision execution until
   their visibility and completion contracts are explicit.

Each family must pass plan compilation, randomized lifecycle checks, executor
lowering, operator parity, released-checkpoint end-to-end correctness, and then
a matched benefit experiment.

## Serving performance

Attention is now a compiler choice for the admitted geometry. The active
performance milestone optimizes the complete warm path:

The first bounded 27B serving diagnostic now establishes the optimization
baseline. After schema 5 makes every token-KV write in-place, a two-epoch
4-input/8-output, eight-request, C1 trace reaches about 19.98 output token/s
versus 32.21 for SGLang (`0.620x`) and 19.63 versus 36.44 for vLLM (`0.539x`).
Median OrbitKV TPOT is about 38.68/39.40 ms versus 19.02/18.15 ms. This is a
substantial improvement over the schema-4 14 token/s result, but it remains a
negative diagnostic; generated-text digests also differ at a known near tie.

A schema-4 CUDA-graph step profile first exposed a 12.1 ms decode copy-back from
32 token-KV tensors. Schema 5 now makes those cache aliases mandatory rather
than merely measuring them: cold search selected 32/32 in-place updates in both
buckets, eliminated the 12.1 ms copy, and reduced the profiled decode graph from
about 53.0 to 41.7 ms. The remaining decode cost is led by DeepGEMM variants
(about 18.5 ms in aggregate), fused elementwise regions, and gathers; all 16
FlashInfer attention calls together take only about 0.16 ms. These are
instrumented attribution times; per-node events substantially perturb this
graph. The first FP8 region experiment demonstrates preparation reuse, and also
shows that changing the chosen BF16 LM-head kernel can dominate a whole-model
comparison. Continue with deployment-path finalist selection, projection/GDN
regions, and uninstrumented serving measurements.

1. Attribute TTFT and TPOT to attention, graph dispatch, scheduler, metadata
   upload, sampling, and frontend overhead.
2. Remove exact-shape CUDA Graph recapture cliffs with stable capacity
   signatures and measured capture policy.
3. Add on-device temperature, top-k, and top-p sampling.
4. Run fixed-model, fixed-weight, fixed-dtype, fixed-memory, and fixed-trace
   comparisons with SGLang and vLLM through the same benchmark client.
5. Promote a claim only when output, completion, final drain, p95/p99 latency,
   throughput, and admission-capacity gates all pass.

The product goal is to beat both current vLLM and SGLang on the primary 27B FP8
checkpoint. A win requires the lower confidence bound of output throughput to
exceed both baselines while p95 TTFT and p95 TPOT are no worse, under the same
weights, quantization, device budget, request trace, scheduler limits, output
semantics, and benchmark client. Long-prompt, decode-heavy, concurrency, and
memory-pressure suites are reported separately; winning one selected point is
not an overall claim. No date or speedup is promised before the attribution
profile shows which layer owns the current gap.

## External KV transports

Implement Mooncake behind `ExternalKvTransport` and reuse the host adapter's
conformance suite. Add remote lease epochs, renewal, eviction intent, active
restore pins, exact deletion acknowledgement, timeout reconciliation, shared
Prefix restore, and node-failure recovery. Add NIXL only after Mooncake
semantics are stable, then compare cold prefill, local retention, and external
restore with identical workloads.

Dynamo may provide routing, topology, discovery, events, and telemetry. Do not
import `kvbm-logical`, Dynamo `KvBlockManager`, lifecycle pins, or another page
allocator: OrbitKV remains the sole KV authority.

## Formal and production closure

State the Minimum Persistent State Realization objective and constraints
formally. Prove optimality for bounded-window and exact-chunked subclasses and
compare generated plans with a small exact oracle for randomized instances.

Then add authenticated completion envelopes, metrics, tracing, crash recovery,
long soak tests, multi-device placement, release artifacts, and a supported
combination matrix. Production claims require every applicable correctness,
pressure, cancellation, and long-running gate.
