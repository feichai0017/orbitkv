# Implementation status

Support is reported separately for compiler representation, manager lifecycle,
executor lowering, device execution, complete-model execution, and measured
benefit. A model is supported end to end only when every required state and
operator passes all applicable layers.

## Current implementation

Logical attention, paged KV views and algorithm admission have separate owners.
The backend now offers FlashInfer CUDA-core/tensor-core algorithms and an optional
native C ABI adapter for upstream FlashAttention-3 on SM90. The handwritten
native attention implementation and experimental policy flag are removed.
See [attention providers](attention-providers.md) for exact geometry and ABI
limits and [provider qualification](validation/provider-kernels-20260914/README.md)
for the preceding schema-9 evidence. Additional KV representations remain open.

The engine and HTTP frontend now live in one `orbitkv-engine` crate. Its
protocol/frontend modules retain logical-only contracts and host-only feature
tests; the model coordinator joins core and executor. The
[engine/stage validation](validation/engine-stage-attribution-20260913/README.md)
adds buffered CPU stage attribution, complete-trace validation, and fixed-artifact
stage-on/off correctness on H20. It identifies compilation, weight-loading, and
first-decode costs without qualifying a serving-performance improvement.

[Generated-module artifact replay](validation/module-image-artifact-20260913/README.md)
now skips all 428 NVRTC compilations on a fixed 27B schedule. Two H20 timing
pairs reduce schedule load by 56.5% and complete diagnostic process startup by
14.0%. The recorded run used decoder schema 6 with validated images. Current schema 11
also binds logical attention/KV-view semantics, explicit provider algorithms and request geometry; older formats are
rejected and must be regenerated.
Fresh compilation adds a selected-program capture pass and warm diagnostic
decode remains about 24.5 ms, so this qualifies startup only. Model-specific
dispatch, sampling policy and the shared-FP8 default are unchanged.

The [weight-loader refactor](validation/weight-loading-20260913/README.md) removes
intermediate host byte copies and returns contextual loading errors. Two
fixed-artifact H20 timing pairs reduce median weight loading from 23.12 s to
6.60 s and complete diagnostic process time from 33.36 s to 17.04 s. Eight
processes retain identical per-step logit errors and pass final drain. Warm
diagnostic decode stays about 24.6 ms; this qualifies startup, not serving
throughput. The [loader contract](weight-loading.md) documents ownership,
supported encoding conversions, error behavior and stage timing boundaries.

| State or attention family | Compiler and manager | Executor | Complete model | Benefit |
| --- | --- | --- | --- | --- |
| Full MHA/GQA token KV | Implemented and host-tested | Logical attention + explicit paged KV with FlashInfer algorithms and optional SM90 FA3 F16/BF16/HD64/128/256 candidates, Prefix/COW, and stable-arena writes | Narrow released-checkpoint H20 closure; see current provider qualification for per-kernel parity | No whole-model speedup established by provider admission; no independent Full-lifetime L5 result |
| Sliding Window token KV | Periodic placement, retirement, ACK, and generation reuse host-tested; request-lifetime residence provides a same-semantics baseline | CSR/window lowering implemented; compiled/baseline CSR geometry is host-matched | Sliding layers cross a 512-token window in the released hybrid H20 closure | Released-hybrid matched run: Sliding residency 48 to 32 pages; no serving-throughput claim |
| Full + Sliding interleaving | Independent class lifetimes and joint transactions host-tested | Manifest-driven per-layer graph construction, independent arenas, write slots, CSR metadata, and capture signatures pass host tests | Released 18-layer 3-Full/15-Sliding checkpoint passes independent token parity, retirement/reuse, cancellation, and final drain on H20 | Narrow same-executor L5: 27.8% less resident payload, 6.1% longer fixed-budget boundary, 0.77% lower median test-path time |
| Exact Chunked attention | Resettable epoch arena host-tested; one whole-domain class only | Metadata lowering implemented | Not independently device-qualified | Unproven |
| MLA/latent KV | Component-aware latent/RoPE lifecycle compiles | Matching OrbitKV compiler attention kernel contract missing | Unsupported | Unproven |
| Mamba/GDN/KDA/linear attention | Recurrent checkpoint geometry compiles and checkpoint pool is host-tested | GDN has distinct key/value-head semantics, checkpoint-shaped split projections, gates, packed delta scan, gated RMSNorm/readout, dynamic arena addressing, and required in-place writes to manager-owned state arenas | Ragged packed operator parity and bounded full-checkpoint prefill/decode/drain pass on H20; a fresh 16-candidate artifact passes eight teacher-forced reference steps with 0.625 maximum absolute logit error and one within-envelope top-1 tie | Negative bounded serving diagnostic |
| Convolution state | Generation-checked checkpoint lifecycle host-tested | Minimal `K-1` BF16 history, typed packed causal convolution, dynamic arena addressing, and required in-place writes to manager-owned history arenas are part of the production graph | Ragged packed operator parity, bounded full-checkpoint drain, and eight-step logit parity pass on H20 | Negative bounded serving diagnostic |
| Block-FP8 linear compiler | Not a state owner | Provider-neutral BF16 x E4M3/128x128-scale semantics; four DeepGEMM tile variants plus opt-in graph-visible shared activation preparation compete per bucket | H20 independent quantizer/combined-path parity, semantic search/replay and eight-step complete-checkpoint reference parity pass | Isolated two-consumer preparation saves 1.45–4.19%; complete-graph selection also changes other kernels, so that is not a serving speedup claim |
| Sparse, tree, speculative, cross-attention | No complete general contract | Missing | Unsupported | Unproven |

The native dense decoder accepts multiple token-KV classes with exact,
non-overlapping layer coverage. Each class owns independent dynamic page
metadata and arena geometry. Its configuration-driven block vocabulary now
covers both pre-norm/SwiGLU and sandwich-norm/GeGLU dense decoders, direct or
unit-offset RMSNorm weights, optional QKV bias and QK norm, non-hidden query
widths, and per-layer local/global RoPE. A safetensors-header contract verifies
every required tensor, shape, dtype, and optional family before graph search.
Unknown or incomplete semantics fail closed instead of silently becoming Full
attention.

The released end-to-end checkpoint set is currently Qwen2.5-0.5B-Instruct for
uniform Full attention and Gemma 3 270M text for interleaved Full+Sliding
attention. This names tested artifacts, not model-specific dispatch: admission
is derived from config and tensor structure. Other dense checkpoints matching
the same vocabulary are structurally admissible but are not claimed as
released-model-qualified until they run the same H20 gates.

The primary product target is a 27B block-FP8 hybrid checkpoint with 16 Full
attention and 48 GDN layers. Its canonical state manifest compiles from the real
checkpoint configuration. The executor now parses nested text-decoder geometry,
the `linear_attention` schedule, partial rotary dimensions, the language-model
tensor namespace, and the block-FP8 format; it also carries recurrent and
convolution state geometry into compiler facts. Token KV, recurrent, and
convolution state now share one host-qualified RuntimeSession lifecycle and one
completion frontier. The executor allocates stable per-class CUDA arenas, maps
generation-checked state slots to byte ranges, copies prior published state to
the selected destination, and feeds only dynamic slot ids into a fixed-address
OrbitKV compiler graph. Search profiles a runtime-owned scratch arena; the real OrbitKV
allocation is bound only after schedule selection. Success evidence requires an
opaque receipt tied to the exact runtime alias and an event recorded after the
model execution. The stable-arena two-step CUDA gate and the packed
causal-convolution/delta-scan parity gate pass on H20. The production decoder
now builds mixed token-KV/GDN layers, owns fixed-state device arenas, binds them
after search, and returns event-backed evidence that the engine submits
atomically with token KV. Packed prefill is admitted through shared request
segmentation. Block-FP8 projections now load the checkpoint's E4M3 tensors and
128x128 inverse scales into a provider-neutral OrbitKV compiler op. Its independent CUDA
reference and four pinned DeepGEMM schedules share one e-class and are
selected by device profiling. Operator parity passes on H20, and a bounded
full-checkpoint run now completes search, prefill, seven decode steps, manager
publication, release, and token/fixed-state drain. An independent Transformers
5.12.1 oracle drives the teacher-forced sequence; the fresh schema-5 artifact's
largest measured absolute logit error is `0.74609375`. The compiler now proves that
equal-valued loop-input streams address the same slots and requires both fixed
state classes to update in place. This removes the former multi-gigabyte
copy-back path without changing the pre-rolling model graph. One near-tied
position may reorder top-1 within that measured error envelope; exact
greedy-text equivalence remains a separate serving criterion.
The gated-delta recurrence
has an independent f32 sequence oracle, a pure OrbitKV compiler single-token expression,
and a typed packed CUDA scan. Token values and next state match the oracle,
including grouped key/value heads. A checkpoint-shaped graph composes
split projections, minimal-history causal convolution, gates, recurrent update,
gated RMSNorm, and output projection. The local small BF16 and 27B FP8
checkpoint headers pass structural config, tensor, shape, dtype, and block-scale
validation. The OrbitKV compiler can derive in-place CUDA candidates for both
recurrent-state and convolution-history commits. Its static alias validator
accepts an ordered old-state read before mutation and rejects competing reads.
The operator and arena paths are H20-qualified, while the complete checkpoint
has only a bounded eight-step logit-parity closure and a negative short-trace
serving diagnostic.

The latest open DeepSeek V4 Flash Vision checkpoint is tracked as a second
architecture target, not a current capability. Its sparse index, low-rank
projection, MoE, mixed low-precision, vision, and multi-device requirements make
it downstream of the primary model's fixed-state and quantized-linear work.

The released hybrid qualification uses the checkpoint's unmodified native
attention schedule. Four short probes and the complete 34-token greedy sequence
match a separate Transformers run. The 512-token prefill plus 33 decodes cross
the Sliding window, observe retirement and generation reuse, then execute and
cancel a second request using recycled storage; both releases drain all manager
state and arenas. This is an L4 correctness/lifecycle result, not a performance
or model-family-wide claim.

The manager exposes `PhysicalResidencePolicy::RequestLifetime` only through an
explicit constructor. It preserves compiler-authored Sliding execution
visibility while retaining physical pages through request release. Host tests
prove identical attention geometry, different physical residency, generation
reuse in compiled mode, and a fixed-capacity admission difference. Chunked
reset, Prefix publication, and external export reject this diagnostic baseline.

## External tiers

Backend-neutral export, restore, replica admission, deletion acknowledgement,
completion ordering, and ambiguity quarantine are implemented. A host-memory
reference adapter moves real bytes and verifies a partial-tail round trip across
sessions. Mooncake/NIXL, remote lease renewal, eviction races, shared Prefix
restore, and distributed recovery remain open.

## Serving

The async local `Engine` contract and optional vLLM Rust frontend adapter are
implemented and host-tested. The `orbitkv-engine` composition root has bounded
admission and output queues and combines independently submitted fresh prompts
into decode-first token-budgeted batches over one `RuntimeSession` and compiled
OrbitKV compiler decoder. On H20, two concurrent 512-token hybrid requests execute with
B=2 prefill/decode and match the existing reference prefix; a late prefill also
joins an active decode request. Length, stop-token, cancellation, backpressure,
and complete drain are covered. The same released checkpoint also passes a
direct B=1/B=8 teacher-forced parity gate: all B=8 rows are bit-identical,
maximum absolute cross-batch logit difference is 0.4296875, and 16 tested
argmax decisions match. Sampling remains
greedy. Chunked prefill, fairness, and production soak remain open.

The `orbitkv-serve` binary provides the complete single-process HTTP product
path. A released hybrid checkpoint passes real H20 OpenAI completion tests with
pre-tokenized prompts, matching tokenizer assets, non-streaming output, ordered
SSE token IDs plus `[DONE]`, concurrent requests, client-disconnect auto-abort,
graceful shutdown, and manager final drain. This is a correctness closure, not a
general performance result. A fixed 16-request trace completes at C1/C2/C4/C8
with full 256-token outputs and no errors. Throughput rises from 184.24 to
518.88 output token/s, but median TTFT rises from 163.06 to 1127.81 ms and TPOT
from 4.80 to 11.06 ms. This qualifies a narrow internal scaling frontier, not a
win over SGLang.

A strict schedule artifact now removes cross-process search variation. It stores
no model weights, device pointers, or KV bytes; loading rebinds current custom
ops and verifies LLIR fingerprints plus the manifest/model/arena/bucket identity.
Four artifact-loaded H20 restarts produced identical candidate output digests
and 0.74% output-throughput coefficient of variation.

Persistent K/V state is now a compiler constraint rather than a post-search
observation. OrbitKV compiler rejects candidates and stored artifacts unless every K/V
output aliases its registered input arena in every retained bucket. A deeper
16-candidate search produced 36/36 in-place tensors and zero copy-back bytes.
Four alternating C2 epochs improved the same engine's throughput by 14.5%, TTFT
by 32.2%, TPOT by 10.2%, and E2E by 12.8% versus the previous artifact. The new
artifact passes the existing independent B2 reference-token probe.
An exact-LLIR audit also found that the former 3-D fused RMSNorm rewrite could
select a logically shaped Q/K view whose physical row pitch was wider than the
kernel contract. The rewrite now admits only proven-dense 2-D rows; per-head
views remain decomposed. A fresh 16-candidate 27B artifact then passed prefill
plus seven teacher-forced decode comparisons with maximum absolute logit error
0.625.

The joint-compiler seam is implemented structurally. A validated manifest now
derives backend-neutral facts for storage components, retention, addressing, and
retirement. The executor binds token classes to stable arenas, lowers
deterministic facts into every OrbitKV compiler search bucket, and tags each
attention KV view with its manager class. The facts digest is part of
decoder artifact identity. Logical attention and physical KV representation now
have separate compiler facts. Declarative capability rules admit explicit
FlashInfer algorithms and compatible FlashAttention-3 kernels. FA3 converts
CSR metadata on-device while retaining the original K/V payload allocations.
Independent CPU softmax tests are shared across providers; captured allocations
belong to the graph that uses them.
Unmasked/HND/unequal-dimension semantics do not
acquire an executable provider merely because the graph vocabulary can express
them. Broader geometry and joint physical-layout competition remain open; see
[attention providers](attention-providers.md).
DeepGEMM, FlashInfer and FlashAttention sources are resolved by the same pinned provider-source
manager. They can come from explicit local directories or an explicit prefetch
into the OrbitKV compiler cache; normal model compilation is offline and no recursive
DeepGEMM source submodule is required.

Compiler-generated DeepGEMM, FlashInfer and FlashAttention nodes now record an identity of the
resolved provider/dependency headers and embedded wrapper contents. Strict
schedule extraction rejects missing or changed provider identities. Their
shared-library cache keys additionally include the resolved `nvcc` executable
and version, target, compilation arguments, and selected compiler environment
inputs. The resolver discovers explicitly prefetched sources without network
access. Sources and toolchains must remain unchanged for the process lifetime;
editing them requires a restart because their digests are memoized.

This is the first reproducibility slice of the
[joint compilation design](joint-compilation.md), not a hermetic toolchain or
complete deployment identity. Host compiler binaries, CUDA system headers, and
supporting compiler tools are not fingerprinted. Changing `nvcc`, target, or
flags rebuilds the library but does not by itself invalidate the selected
schedule. The historical H20 model and performance results above predate this
source-identity change; they do not qualify a newly compiled deployment.

The next implemented slice is [workload and FP8 region tuning](fp8-region-tuning.md).
An artifact-bound profile supplies preferred batch/query/context representatives,
valid private-page scratch metadata, a compile budget, and a CUDA Graph finalist
count. Compiler rules can expose one packed FP8 activation/scale allocation to
multiple prequantized GEMMs. Original combined implementations remain eligible;
the new alternative is opt-in. Its packed ABI, buffer capacities and provider
identity are validated, and ordinary graph edges govern intermediate lifetime.
This is preparation reuse across separate kernels.

The v3 runtime passes B4/B8 aligned and ragged full-checkpoint tests using one
unchanged seven-bucket artifact: eight reference steps per request and complete
drain in both strict replay and separate profiling processes. All requests share
one fixed prompt and teacher-forced continuation. Ragged validation uncovered
a DeepGEMM scratch-lifetime bug; captured child graphs now retain exact allocation
owners through graph retirement and resident shape switches. The isolated
regression also verifies old allocations survive growth and are eventually freed.

Search now excludes known non-deployable custom-op spellings before sampling,
including parents without a finite executable term. It preserves the e-graph
and all eligible provider alternatives. Initial retries are bounded, and the
metadata parser accepts integer ID classes containing legitimate derived
aliases. Host regressions include six complete saved model egraphs.

The [compiler boundary refactor](compiler-boundaries.md) separates serialized
policy, correlated bucket planning and profiling input ownership. It removes
the arbitrary 256-bucket and two-candidate minimum restrictions, and skips
context combinations that exceed a smaller attention class's arena. FP8 ABI
definitions now drive both Rust layout checks and CUDA quantizer source;
scratch lifetime, tile policy and JIT have separate modules. The provider source
identity changes, so the historical v3 schedules require their frozen source.
New CUDA search traces connect complete program/operation identity with direct
and deployment scores, timeout/rejection decisions and final installation.
The [current-source validation](validation/compiler-boundaries-20260912/README.md)
passes eight reference steps and drain in fresh search, strict replay, and
instrumented replay on H20, with maximum absolute error 0.5423088. Both bucket
identities agree from measurement through the stored artifact. Runtime caches
are excluded from semantic host-operation fingerprints; a focused real cuBLASLt
regression also checks preparation, capture, and replay. This is a B1 wiring and
correctness check, not a serving-performance result.

The two serving output differences have now been independently localized using
the frozen v3 implementation. Both first divergences have equal OFF maximum
logits; its highest-index tie rule chooses a different token from the reference.
ON has unique maxima at those steps. The rule is consistent between frontend
and fused CUDA implementation and is preserved, with a generic regression.
This diagnosis does not turn the previous serving experiment into a qualified
performance improvement; [logit diagnostics](logit-diagnosis.md) distinguish
actual selected tokens from a canonical lowest-index argmax comparison.

## Evidence interpretation

The 27B block-FP8 path has bounded L4 correctness but is not yet production or
performance qualified. On the
recorded schema-5 4-input/8-output/C1 trace, OrbitKV provides 0.620x SGLang and
0.539x vLLM output throughput, with 2.034x and 2.170x median TPOT respectively.
All requests completed, but output digests differ, so these remain diagnostic
rather than promoted matched-performance results. The fixed four-token prompt
has a near-tied reference step where exact greedy text can diverge even though
logits stay inside the numeric envelope. The teacher-forced gate keeps every
subsequent input identical to the independent reference and requires
`max_abs <= 1.0`. The recorded B1 eight-step trace observed a maximum of 0.625;
the later workload-tuned B8 aligned artifact observed 0.90625. These are separate
artifacts and batch shapes. Those measurements identify a performance deficit and a
strict-output-equivalence boundary; they do not qualify a cross-engine benefit.
The first schema-4 CUDA-graph attribution found a 12.1 ms decode copy-back from
32 token-KV tensors. Schema 5 makes those aliases mandatory; a fresh cold search
selected 32/32 in-place updates in both buckets and reduced the profiled `B=1`,
`s=1` graph from about 53.0 to 41.7 ms. The remaining decode cost is led by
DeepGEMM variants (about 18.5 ms in aggregate), fused regions, and gathers, while
all 16 FlashInfer attention calls total only about 0.16 ms. Per-node timing
events substantially perturb this graph: a later frozen baseline measured about
24.3 ms for an uninstrumented diagnostic decode and about 40.4 ms with those
events. Attribution identifies candidates to investigate; whole-graph timing
and uninstrumented serving determine benefit. The new FP8 ablation also selected
different BF16 LM-head implementations. That change explains most of its
diagnostic latency difference and prevents attributing the total to shared
quantization.

The final v3 same-server off/on diagnostic completes four short C1 runs with
repeatable output within each arm. Median paired throughput is 1.82% higher and
TPOT 5.82% lower with the option enabled, while TTFT is 8.26% higher and generated
text differs for two of eight prompts. This is not a qualified benefit. Decode
changes its output-projection provider and has no selected multi-consumer
quantizer; prefill does share preparation. Shared FP8 remains opt-in. Independent
logit diagnosis and candidate-level scoring records are now available. Measuring
compilation stages and fixed-artifact startup optimizations have separate
qualification records. [Graph residency](graph-residency.md) now has explicit
provider ownership, replacement-peak accounting and a finite engine/CLI capacity;
the 27B transition fixture verifies repeated requests and eviction. Guiding
region exploration with measured costs, automatic joint residency budgeting,
and wider serving measurements remain open. The
[bucket-serving follow-up](../results/bucket-serving-20260913/README.md) closes
two narrow C1 HTTP workloads with identical output and final resource drain;
its short-output P99 TPOT regression remains visible. The engine also exposes
blocking, checked shutdown and the executable reports final state ownership.

[B1/B8 workload attribution](validation/workload-attribution-20260914/README.md)
now retains full egglog rule identities and complete GPU step descriptions,
joined to workload dimensions and selected programs. Explicit FlashInfer request
geometry fixes retained-bucket planning. The integrated compiler uses schema 11 and rejects earlier decoder artifacts.
Two frozen builds pass 592 reference comparisons across fresh search, strict
replay and profiling, with final drain and maximum absolute error 0.8125 under
the unchanged 1.0 gate. Ordinary compilation uses time-only egglog reports;
verbose query-plan diagnostics remain opt-in.

The final run still spends 354.19 s in the `glumoe` ruleset. Candidate selection
also varies between the two builds under the same seed: B8 decode improves,
but prefill selects a generic BF16 output projection taking 171.15 ms in the
event profile despite an equivalent cuBLASLt candidate. The result establishes
attribution and bucket correctness, not a whole-engine performance improvement.
Reproducible candidate coverage and measured region selection are next.

[Startup preparation](../results/startup-preparation-20260913/README.md) now moves
bounded artifact-representative graph preparation before readiness. Separate
runtime residency, decoder representative-input and engine startup modules own
the lifecycle; their tests live under the corresponding `tests/` trees. The
H20 comparison holds capacity fixed within each off/on pair, covering capacities
two and one. All 24 C1 processes complete 352 requests and 13,824 tokens with
identical paired output and state drain. At capacity two, the short-output first
stream interval is 138.9→29.9 ms and P99 TPOT 37.7→24.5 ms; steady ITL remains
about 23.6 ms. These are narrow paired measurements, not a production-tail claim.
Four model processes compare 128 full-vocabulary output rows against an
independent reference with maximum absolute error 0.5, including first use,
repeated phase changes, eviction and 16 released request lifecycles.

The default capacity remains one. Preparation preserves existing materializations
before filling unused slots in artifact order. Its regression test verifies that
startup does not replace the loaded graph with a speculative bucket. The default
HTTP trace still shows P99 TPOT 42.4→43.9 ms and throughput
18.34→18.12 token/s; this does not establish a default-capacity performance benefit.
The earlier eviction regression retains its own source identity; final measurements use the
corrected build. Both prepared buckets are reused without a first-request full
graph rebuild when capacity is two. This is an explicit deployment policy;
automatic workload selection, long-prefill and multi-request coverage remain
separate gates. The earlier rejected 128-input-token attempt is preserved.

The child CUDA Graph result demonstrates a narrow dispatch optimization. The
released-hybrid residence experiment is the first narrow L5 compiler result:
ten paired release-mode epochs show a 27.8% live-payload reduction, a fixed-budget
boundary increase from 528 to 560, and a positive paired total-time interval.
It is not a comparative serving benefit. TTFT/TPOT/tails and internal concurrency
scaling are measured through C8, while fairness, soak, and the capacity limit
remain open. The current tuned comparison remains negative: at C2 OrbitKV
reaches 0.598x stock SGLang throughput with 1.59x TPOT and 4.79x TTFT. Its configured
persistent K/V tensor payload is 40.4% smaller on the same Gemma3 capacity
because SGLang disables hybrid SWA memory. Executor performance, not KV lifetime
correctness, is the next blocker.


Profile-directed local search is available through artifact-bound
`hotspot_candidates` (default zero). Extraction provenance survives loop copies
and CUDA fusion. A separate CUDA-event pass orders existing egraph choices;
complete uninstrumented candidates and deployment CUDA Graphs remain the ranking
objectives. The H20 regression requires a measured generated-GEMM/cuBLASLt local
transition beside exact persistent-state updates. It is bounded coordinate
exploration, with independent dependency-closure repair, joint layout search and
whole-model persistent kernels still open. See [search coverage](search-coverage.md).

The [27B hotspot qualification](validation/hotspot-search-20260914/README.md)
passes 296 reference comparisons with 0.8125 maximum absolute logit error and
all drains. It records 49 measured local neighbors with no state/resource
rejection and two measured prefill provider transitions. B8 diagnostic decode
is 37.29 ms; its seed already uses cuBLASLt, so the historical difference is not
a causal search or serving result. `glumoe` remains the dominant compile cost.
