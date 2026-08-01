# Roadmap

## Permanent Product Boundary

Loom does not implement matrix multiplication. Dense, quantized, sparse, and
grouped GEMM belong to cuBLASLt, CUTLASS, FlashInfer, or another
engine-selected vendor backend. Loom may prepare or consume their buffers and
fuse memory-bound work immediately around them, but it will not hide a second
matrix core behind a Loom API.

A new direction enters this roadmap only when all three statements are true:

1. the cost is memory traffic, launch overhead, layout conversion, or scheduling
   metadata rather than matrix arithmetic;
2. a named inference-engine path has a real gap that Loom can enter without
   copying tensors into a private format;
3. a real model or serving workload can close an engine, memory, or end-to-end
   exit gate.

Microbenchmark opportunity alone is not admission.

## Execution Order After K0.7

K0.7's first native-wheel matrix row is complete. Publication remains an
explicit release action, not an engineering prerequisite for starting the next
feature. New feature work follows this order:

| Order | Track | First deliverable | Required system proof |
| --- | --- | --- | --- |
| 1 | Quantization plumbing | scale, pack/unpack, dequant/requant, and layout transitions around vendor GEMM | one named quantized model removes an HBM pass or temporary tensor |
| 2 | MoE routing and movement | top-k routing, histogram/prefix sum, permutation, and inverse permutation | lower model-level MoE latency while grouped GEMM remains vendor-owned |
| 3 | Profile-gated KV movement | a named offload, beam, or compaction profile that exposes physical movement | fewer launches or less movement time without replacing an efficient driver/vendor path |
| 4 | Profile-gated speculative extensions | tree/stochastic/KV boundaries only after profiling exposes material non-GEMM cost | a named draft/target model pair improves decode latency or throughput |
| 5 | Minimal Rust decode proof | zero-copy Rust orchestration over vendor-produced tensors and Loom operators | one deterministic decode step uses borrowed memory and stream ownership without becoming an inference engine |

The original default-prefix/preemption movement candidate failed admission.
The [H20 vLLM V1 probe](results/h20-vllm-kv-movement-admission-rejected-20260727.json)
observed a 1,024-token prefix hit and three real preemptions, but none of the
instrumented swap or batch-copy entrypoints ran. Prefix caching reuses logical
blocks and default preemption frees blocks before recomputation. No public Loom
movement operator will be added for that path.

The seeded-sampling candidate passed admission, direct implementation, request
lifecycle, and engine gates. The
[source-pinned admission](results/h20-vllm-seeded-sampling-admission-20260727.json)
showed one exponential kernel per seeded row and a full F32 noise tensor. The
[ABI8 direct result](results/h20-categorical-sample-20260727.json) replaces
that boundary with one kernel and `512 B` of measured output allocation. It is
slower at 1–2 rows, then `1.15–5.40x` faster at 4–32 rows with exact Loom
replay and a passing statistical contract.

Persistent state now follows vLLM 0.24/0.25 cached requests through active
slot removal, condensation, resumption, and swaps. In process-isolated
[baseline-first](results/h20-vllm-engine-categorical-sample-20260727.json)
and
[Loom-first](results/h20-vllm-engine-categorical-sample-loom-first-20260727.json)
Qwen2.5-0.5B runs, every case exactly replays, Loom launches once per decode
step with no rejection, and batch 32 improves latency/throughput by
`5.7–8.1%` in both orders. Batch 1–4 pays `1.5–2.4%`; the adapter reports that
cost instead of switching an in-flight request to another RNG stream.
The current
[ABI11 matrix wheel](results/h20-native-wheel-clean-install-abi11-20260801.json)
passes 342 tests with each supported vLLM minor and 231 applicable tests on
PyTorch 2.10 from repository-free fresh environments, including the same
categorical subsystem.

Static FP8 KV-cache compression remains a K3 evidence track, not the next
kernel implementation. Its first pinned Qwen2.5 candidate is rejected below;
qualification resumes only for a distinct named model, backend, or cache
representation that can pass the same held-out quality precondition.

## K0: Backend Foundation

Status: complete.

- backend-independent Rust contracts and CPU oracle;
- safe CUDA resource ownership and C ABI;
- reproducible correctness and latency report format.

## K0.5: Publishable Rust Distribution

Status: complete for the Rust source crates in `1.0.0-alpha.1`.

- independent `loom-kernels`, `loom-cuda-sys`, and `loom-cuda` package
  metadata with versioned registry dependencies;
- handwritten CUDA sources packaged inside `loom-cuda-sys`, so an extracted
  crate does not depend on repository-relative files;
- package-specific READMEs, changelog, Cargo archive checks in CI, and a pure
  Rust H2D → CUDA → D2H oracle smoke example;
- clean archive rebuild of `loom-kernels` plus CUDA-enabled archive rebuild of
  `loom-cuda-sys` on NVIDIA H20;
- source-adapter Python metadata at `1.0.0a1`, which established the package
  name and entry point before K0.7 added native distribution.

Exit: a downstream Rust consumer can build the published source crates and run
an oracle-checked CUDA path without cloning the repository.

## K0.6: Engine-Owned Runtime Interop

Status: complete for the framework operator surface after
`v1.0.0-alpha.1`.

- ~~generic safe backend over owned or borrowed CUDA streams~~;
- ~~sealed read/write device-memory traits shared by owned buffers and borrowed
  tensor views~~;
- ~~zero-copy H20 oracle smoke on a borrowed stream and borrowed allocations,
  including non-destruction of framework-owned resources~~;
- ~~route one real framework adapter through the safe Rust boundary instead of
  calling the raw C ABI directly~~ — PyTorch/vLLM Add+RMSNorm and PyTorch
  RMSNorm+dynamic-FP8 now enter `loom-cuda-bridge` with actual buffer lengths
  and borrowed current-stream ownership;
- ~~validate external current-stream ordering, CUDA Graph capture, and engine
  fallback behavior through that Rust-owned path~~ — the H20 gate covers all
  three dtypes, odd widths, `torch.compile`, CUDA Graph replay, Add+RMSNorm
  vLLM IR invocation, and Rust-side invalid-buffer rejection for both paths.
- ~~move one proven decode-tail engine path through checked Rust~~ — contiguous
  greedy+sampled-logprob now uses typed borrowed Rust views, exact buffer
  lengths, disjoint-output validation, and the framework current stream;
- ~~route every remaining PyTorch operator through the same boundary~~ —
  standalone RMSNorm, activation/FP8, padded logits, selected-token logprobs,
  Min-P, RoPE+paged-KV, and base/split-K paged decode now use explicit Rust
  layout contracts; the ctypes, direct-CUDA, and unchecked dispatcher paths
  were removed as a breaking change.

Exit: an inference-engine call reaches checked Rust dispatch using its existing
tensor memory and CUDA stream, with no hidden copy, allocation, or ownership
transfer.

## K0.7: Framework Compatibility And Binary Distribution

Status: complete for the first Linux x86_64, CUDA 13.1, SM90 matrix row.

- ~~qualify the next vLLM minor without weakening adapter gates~~ — official
  vLLM 0.24.0 and 0.25.1 packages each pass the complete 293-test H20 GPU suite;
- ~~centralize runtime version admission and package metadata~~ — supported
  range is `vllm>=0.24,<0.26`, with registration-time series checks;
- ~~document the current binary boundary and Stable ABI decision~~ — the
  dispatcher target, runtime range, and revalidation rules are explicit;
- ~~replace the entire production dispatcher with PyTorch's Stable ABI~~ —
  every schema and kernel uses the boxed Stable ABI with a declared PyTorch
  2.10 target; the temporary probe and previous ATen dispatcher were deleted;
- ~~validate one binary across two PyTorch minor releases~~ — the exact H20
  `.so` built with PyTorch 2.11 passes on 2.10 and 2.11, including complete
  vLLM 0.24/0.25 suites on the qualified 2.11 stack;
- ~~automate the CUDA/PyTorch/Python matrix artifact~~ —
  `python/build_wheel.py` builds from a clean revision, packages exactly the
  two native libraries, emits their manifest, audits ELF/RPATH/symbols, and
  refuses an accidental source-only wheel;
- ~~prove repository-free H20 clean installs~~ — the current exact ABI11
  `py3-none-linux_x86_64` artifact passes fresh Python 3.11 venv gates on
  PyTorch 2.10/2.11 and vLLM 0.24/0.25, including `pip check`, package-local
  library loading, GPU smoke, and the applicable 342/231-test suites. ABI10
  and earlier artifacts remain historical evidence.

Exit: a qualified binary artifact installs without a repository checkout, uses
a declared PyTorch ABI boundary, and passes the same framework and H20 gates as
the source build. The first row reached this exit; it has not been published to
a package index.

## K1: Useful Normalization Family

Status: in progress.

1. ~~vectorized FP16 and BF16 RMSNorm~~ — H20 correctness gate complete;
2. ~~fused residual Add+RMSNorm~~ — double in-place H20 gate complete;
3. ~~RMSNorm plus dynamic per-token FP8 output quantization~~ — H20 and named
   vLLM bitwise/performance gates complete;
4. optional-residual RMSNorm plus symmetric dynamic per-token INT8 — ABI10,
   direct H20 correctness, a real Qwen2.5 W8A8 compiler route, unchanged
   Cutlass GEMM, and the matrix-wheel gate are complete. Exact model quality,
   an order-stable engine benefit, and default admission remain open;
5. ~~named vLLM baseline and engine integration~~ — IR provider, compilation,
   CUDA Graph, and synthetic-Qwen2 generate-loop gates complete;
6. ~~native CUDA/LibTorch wheel and clean-install matrix gate~~; a production
   model/workload gate remains.

Exit: one fused path improves a real decode workload, not only a microbenchmark.

## K2: MLP Activation And Quantization

Status: in progress.

1. ~~split-half SiLU-and-Mul for F32/FP16/BF16~~ — Rust, CUDA, PyTorch,
   vLLM layer override, and H20 compatibility gates complete;
2. ~~SiLU-and-Mul plus dynamic per-block FP8 output quantization~~ — groups
   64/128, exact vLLM compatibility, compiler-fusion registration, and H20
   named-baseline gates complete; pinned Qwen2.5 online-FP8 compilation,
   path-hit, CUDA Graph, exact-token, and order-reversed engine gates are also
   complete, while the measured 0.5B end-to-end result remains at parity;
3. ~~dynamic per-token INT8 output quantization~~ — ABI11 spans the CPU
   oracle, safe Rust, checked bridge, handwritten CUDA, Stable ABI PyTorch,
   direct benchmarks, and an explicit vLLM compiler pattern for the observed
   W8A8 `SiLU-and-Mul -> dynamic_scaled_int8_quant -> Cutlass` path. H20 source,
   exact compiled semantics, real-engine path, unchanged Cutlass GEMM, and
   32-prompt exact-quality gates pass. CUDA Graph ratios remain below parity
   and engine latency is not order-stable, so it is explicit-only; the ABI11
   matrix-wheel gate passes but does not overrule that performance rejection;
4. GELU/GELU-tanh and gated variants admitted by model coverage;
5. explicit handoff to engine-selected vendor GEMM, with Loom limited to
   memory-bound bias, activation, and quantization boundaries around it.

Exit: a fused activation+quantization path removes an HBM round trip and
improves a real model workload. Standalone SiLU parity alone does not close it.

## K2.5: Quantization Plumbing Around Vendor GEMM

Status: in progress; optional-residual RMSNorm-to-FP8 is source-, integration-,
H20-, and clean-wheel-qualified, and the current ABI11 matrix also distributes
INT8.

- ~~generalize RMSNorm-to-dynamic-per-token-FP8 to the exact optional-residual
  vLLM fusion schema~~ — both plain and Add+RMSNorm fusion keys now use one
  ABI9 Rust/CUDA path; all FP8 bytes, scales, and residual bytes match vLLM;
- ~~prove the first real vendor-GEMM handoff~~ — generated-source audits show
  native/Loom normalization calls swapping while both providers retain eight
  Cutlass scaled-mm call sites. Order-reversed Qwen2.5-0.5B prefill improves
  batch latency by `1.0066-1.0506x`; decode-heavy latency crosses parity and
  carries no acceleration claim;
- ~~implement and distribute the matching optional-residual
  RMSNorm-to-dynamic-per-token-INT8 path~~ — one ABI10 contract spans the CPU
  oracle, safe CUDA, checked bridge, Stable ABI PyTorch, and vLLM 0.24/0.25
  compiler adapter. A real
  Qwen2.5 W8A8 graph records `1440/0` Loom launches while preserving eight
  Cutlass scaled-mm sites on both providers; the same two-library wheel passes
  342 tests with each vLLM minor and 231 applicable PyTorch 2.10 tests;
- close the INT8 quality/default/performance admission gates — the real-layer
  shadow differs by one INT8 LSB across 688,128 elements with exact
  scales/residuals, but the 32-prompt
  one-step gate matches only `29/32` top-1 tokens and dual-order engine latency
  crosses parity. The route stays explicit opt-in despite its qualified ABI11
  distribution;
- per-token, per-channel, and per-block scale reduction for FP8 and INT8;
- pack/unpack and layout conversion for engine-selected quantized kernels;
- dequantize, requantize, scale conversion, and scale-layout transpose;
- fuse adjacent activation, normalization, or cache movement only when it
  removes a measured launch or HBM round trip;
- keep matrix multiplication and its tuning entirely in the vendor backend.

Exit: a named quantized model path passes bitwise or declared-tolerance gates,
records the vendor GEMM unchanged on both sides, and improves an engine-level
latency, memory, or temporary-allocation metric.

The first slice meets the full source, engine, and distribution exit on the
exact Qwen prefill boundary. K2.5 remains open only for additional
scale/pack/layout work admitted by another named vendor-kernel consumer.
The ABI10 INT8 candidate reaches implementation, real-engine invocation, and
binary distribution, but not the quality, stable-benefit, or default-admission
exit. Wheel qualification is not used to overrule those separate failures.

## K3: KV-Cache Update Family

Status: implementation and integration qualified; the first compression
system candidate and the default vLLM relocation candidate are rejected, while
the family-level system-value exit remains open.

- ~~RoPE plus paged-KV write~~ — Rust/CUDA/PyTorch, packed-QKV and NHD/HND
  layouts, vLLM compiler fusion, H20 named baseline, and exact-token Qwen2.5
  engine gates complete; operator benefit is measurable, model-level benefit
  remains open;
- ~~FP8 E4M3 quantize-on-write with explicit static per-tensor or per-head
  scales~~ — Rust contract/oracle, safe CUDA backend, checked bridge, Stable ABI
  PyTorch operator, vLLM adapter, exact-byte H20 comparison, named operator
  benchmark, current-stream/compile/graph checks, current ABI11 clean wheel, and
  order-reversed real-engine invocation are complete; the pinned Qwen2.5-7B
  candidate passes the operational and native-vLLM/Loom provider-equivalence
  gates but is rejected because both FP8 providers exceed the BF16 held-out
  perplexity limit;
- ~~process-isolated native/FP8/Loom system measurement harness~~ — cache
  capacity, CUDA memory, perplexity, TTFT, TPOT, throughput, token divergence,
  package provenance, calibrated checkpoint scale scheme, and path telemetry
  are captured under a pinned model and corpus contract; the first result
  records `1.99879x` cache-token capacity and a `1.00064` symmetric
  native-vLLM/Loom FP8 perplexity ratio, but an accepted order-reversed
  large-model artifact remains open;
- ~~reproducible calibration and quality-corpus preparation~~ —
  llm-compressor attention/KV calibration records source and output checkpoint
  digests, model config, tokenizer, stateful observer, scale layouts, packages,
  and corpus provenance; deterministic tokenizer-qualified JSONL selection
  verifies the same data/tokenizer and records its own source and output
  digests without adding calibration packages to the runtime wheel;
- [rejected Qwen2.5-7B system result](results/h20-fp8-kv-system-rejected-20260727.json)
  — on an 8-sequence, 1,016-scored-token early-stop slice, minmax per-head FP8
  gives native-vLLM/Loom FP8-to-BF16 perplexity ratios of `3.07370x` and
  `3.07173x`; quality fails before the dual-order TTFT/TPOT gate, so this
  result is not performance evidence;
- [rejected default vLLM movement result](results/h20-vllm-kv-movement-admission-rejected-20260727.json)
  — a real 0.24 V1 run records a 1,024-token logical prefix hit and three
  preemptions under a `1.2366x` over-capacity workload, yet zero physical
  swap/copy calls or bytes. Loom therefore exposes no default
  prefix/preemption relocation API;
- FlashAttention/FlashInfer consume the compressed cache directly, so Loom
  deliberately does not add a full-cache dequantize-on-read pass;
- dynamic per-token-head scale caches and INT8 follow only when a named
  engine/model path requires those distinct contracts;
- append/copy with layout conversion remains profile-gated for a named
  engine-native cache path;
- block copy, swap, gather, scatter, compact, and remap reopen only for a named
  offload, beam, or defragmentation workload that physically moves data and
  does not already use an efficient CUDA-driver or engine implementation;
- expose no private cache ownership: engine allocations, page tables, streams,
  and lifetime remain borrowed.

Exit: a real engine shows lower cache bytes and a larger admitted context or
batch size for compression, or lower scheduler movement time for relocation,
while preserving token/quality gates and reporting TPOT impact explicitly.

## K4: Decode Tail

Status: in progress.

- ~~greedy argmax plus sampled-token raw logprob~~ — Rust oracle, safe
  CUDA/C ABI, PyTorch, checked-Rust contiguous dispatch, and narrow vLLM
  0.24/0.25 integration complete; the vLLM 0.24 H20 named baseline and both
  real-engine provider orders show exact token/rank parity and material
  latency/TPOT benefit;
- ~~general selected-token raw logprob and rank~~ — vLLM continues to own
  penalties, top-k/top-p, RNG, and token selection; Rust/CUDA/PyTorch plus
  order-reversed Qwen2.5 H20 gates show exact token/rank parity and material
  operator and end-to-end benefit;
- ~~in-place min-p filtering~~ — Rust/CUDA/PyTorch and a vLLM 0.24/0.25 opt-in
  are complete; H20 evidence selects Loom only for at least 32 rows and a
  65,536+ vocabulary, while smaller shapes fall back because the
  one-block-per-row kernel is slower there;
- ~~sparse repetition/presence/frequency penalties~~ — exact Rust oracle,
  one-kernel CUDA hash, caller workspace, checked bridge, PyTorch
  compile/graph coverage, and explicit vLLM 0.24/0.25 registration are
  complete; the H20 operator gate is `5.82–34.30x` faster for rows 1–128 and
  replaces up to `427.85 MB` of vLLM temporaries with a `2 MiB` caller
  workspace; both vLLM 0.24 Qwen2.5-0.5B provider orders preserve every token,
  record `1440/0` Loom submissions, and measure `1.056–1.123x` batch-latency
  plus `1.068–1.126x` TPOT ratios;
- ~~sampled-token plus top-k raw logprobs~~ — Rust F32/FP16/BF16 oracles,
  deterministic two-stage CUDA with caller-owned workspace, checked PyTorch
  compile/graph coverage, and an exact vLLM 0.24 adapter are complete. The
  direct operator is `3.25x`, `2.60x`, and `1.19x` faster at 1/8/32
  Qwen-vocabulary rows and near parity at 128 rows. Both engine orders preserve
  tokens, returned top-k IDs/ranks, and values within `1.91e-6`, with
  `1440/0` Loom submissions; latency crosses parity under order reversal, so no
  model-level speedup is claimed;
- ~~exact in-place top-k filtering~~ — Rust F32/FP16/BF16 contracts and
  oracles, one device-only partition-sort/binary-count CUDA path for every
  valid `top_k`, caller-owned workspace below the PyTorch boundary, checked
  ABI5 dispatch, compile/graph coverage, and an opt-in vLLM 0.24/0.25 hook are
  complete. On H20 with 151,936 F32 logits and `top_k=50`, every admitted
  1–7-row case beats vLLM's full-sort path by `1.42–2.15x` while using
  `0.62–4.36 MB` rather than `4.90–47.01 MB` of peak temporary storage.
  Eight or more rows retain vLLM's Qrita Triton path because its duplicate-tie
  semantics select exactly `k` positions rather than preserving the threshold;
- ~~fused top-p filtering and renormalization~~ — Rust F32/FP16/BF16
  contracts and oracles, deterministic descending-token-ID ties, one
  partition-radix/device-threshold CUDA path, caller-owned workspace below the
  framework boundary, checked ABI6 dispatch, and PyTorch compile/graph coverage
  are complete. The explicit vLLM 0.24/0.25 route admits only F32 top-p-only
  requests with rows 2–7 and vocabulary at least 32,768; vLLM retains RNG,
  generators, token selection, joint top-k/top-p, and every unqualified shape.
  At the 32,768-vocabulary crossover the H20 operator is `1.15–1.34x` faster;
  at 151,936 it is `1.72–1.77x` faster and uses roughly one third of vLLM's
  peak temporary storage. An F32 cutoff may move by one boundary token because
  the implementations use different parallel accumulation orders; the
  qualified per-row probability L1 difference is below `1e-4`;
- ~~fused logits preprocessing~~ — one in-place F32 pass applies dense
  blocked-token masking, unique sparse additive bias, sparse suppression, and
  mixed-row temperature in vLLM order. Rust oracle, handwritten CUDA, safe
  dispatch, checked ABI7, Stable ABI PyTorch compile/graph coverage, and
  explicit mixed-sampling vLLM 0.24/0.25 registration are complete. The H20
  operator is exact and `3.26–7.30x` faster at 1–32 Qwen-vocabulary rows.
  Both real-engine provider orders preserve every token and record `720/0`
  Loom submissions; TPOT ratios are `1.010–1.084x`, while batch latency
  crosses parity at batch 32, so no stable model-level batch-latency claim is
  made;
- ~~deterministic counter-based RNG sampling without a host round trip~~ —
  ABI8-A `categorical_sample` consumes normalized contiguous F32 probabilities
  plus caller-owned `(seed, counter)` int64 state, emits one int64 token per
  row, and advances counters in place on the current stream. Rust and CUDA use
  the same fixed logical F32 CDF tree; checked bridge, Stable ABI PyTorch,
  compile/FakeTensor/current-stream/CUDA Graph coverage, and the 65,536-draw
  distribution gate pass. H20 direct sampling uses one kernel and is
  `1.15–5.40x` faster for 4–32-row cases without a
  probability-shaped temporary. It has no implicit global generator or
  seedless mode. The explicit vLLM 0.24/0.25 registration stores state on
  cached requests and contiguous batch slots, rejects unseeded random or
  speculative engines, and passes remove/condense/resume/swap lifecycle
  tests. Order-reversed Qwen2.5-0.5B evidence records exact replay, 640 Loom
  calls per run, no rejection, and a stable `1.057–1.081x` batch-32 ratio.
  See the [counter-based sampling design](design/counter-based-sampling.md),
  [direct result](results/h20-categorical-sample-20260727.json),
  [baseline-first engine result](results/h20-vllm-engine-categorical-sample-20260727.json),
  [Loom-first engine result](results/h20-vllm-engine-categorical-sample-loom-first-20260727.json),
  [ABI8 wheel result](results/h20-native-wheel-clean-install-abi8-20260727.json),
  and
  [admission result](results/h20-vllm-seeded-sampling-admission-20260727.json).

Exit: fewer launches and temporary tensors with exact token results where the
contract requires parity, or deterministic replay plus an explicitly declared
statistical sampling contract. The
selected-logprob exit gates are closed for pure greedy and engine-owned general
sampling requests with `logprobs=0`; the sparse-penalty gate is also closed for
the pinned deterministic Qwen workload. Exact top-k filtering closes its
operator gate for the admitted small-row vLLM path; top-k raw-logprob
correctness plus temporary reduction is closed without a stable engine-speedup
claim. Fused top-p/renormalization and mixed-sampling logits preprocessing
close their shape-gated operator and real-engine invocation exits. Loom-owned
deterministic RNG closes its direct, persistent-request-state, and
order-reversed engine exits for explicitly seeded non-speculative requests.
Its repository-free ABI8 clean-wheel distribution exit is also closed.

## K4.5: Speculative Decoding Support

Status: real-engine path complete; performance exit open and further extensions
are profile-gated.

- ~~verify flattened ragged greedy drafts and compact accepted/bonus tokens~~ —
  Rust contract and CPU oracle, one-warp handwritten CUDA, safe borrowed-Rust
  dispatch, PyTorch current-stream/compile/graph coverage, and explicit vLLM
  0.24/0.25 registration are complete; all 15 H20 benchmark shapes are
  bit-exact and reduce verifier-level latency by `9.2-11.3%`;
- ~~run a named draft/target model through isolated native and Loom
  providers~~ — Qwen2.5-1.5B target plus Qwen2.5-0.5B draft on vLLM 0.24
  preserves exact native/Loom speculative tokens and statistics, records
  `714/714` measured Loom calls per order, and isolates target/native/Loom in
  separate processes;
- profile result: the verifier is only `0.048-0.200%` of batch latency,
  native/Loom end-to-end ratios cross parity under order reversal, and this
  speculative configuration is `3.18-4.97x` slower than target-only. Do not
  spend the next milestone on verifier micro-optimization;

The remaining speculative boundaries require a new named workload that shows
material metadata, sampling, or KV-management cost:

- construct batched draft-verification metadata and tree/branch masks consumed
  by an engine-selected attention backend;
- implement stochastic residual-distribution acceptance/rejection using an
  explicit counter-based RNG state contract;
- update caller-owned sequence/KV metadata without host round trips;
- add cache commit/rollback or slot-remap primitives only where the selected
  engine exposes that boundary;
- keep draft/target model GEMM and verification attention in vendor libraries.

Exit: one named draft/target model pair reaches Loom from a real engine,
preserves the engine's declared sampling distribution and seeded behavior,
records path hits, and improves end-to-end decode latency or throughput in both
provider orders. A standalone acceptance-kernel benchmark does not close this
milestone. The current Qwen2.5 gate closes invocation and equivalence, but
explicitly does not close the performance clause.

## K5: MoE Routing And Movement

Status: in progress; the direct movement boundary and an explicit vLLM/Cutlass
engine route are qualified, while a production-workload exit remains open.

- top-k routing and renormalization remain a separate profile-driven slice;
- ~~stable expert-major permutation, local offsets, inverse permutation, and
  expert mapping~~ — Rust oracles, F32/FP16/BF16 plus byte-exact FP8 E4M3FN
  permutation, expert-parallel remote ordering, and exact vLLM
  production-scratch metadata are complete;
- ~~caller-owned metadata/workspace handoff into the engine-selected grouped
  GEMM, with no Loom matrix implementation~~ — safe Rust and ABI12 expose this
  boundary; the PyTorch convenience API owns only its explicit outputs and
  byte workspace;
- ~~weighted expert-output reduction~~ — F32 route accumulation with one final
  dtype conversion is complete; shared/routed output fusion remains planned
  only if a real engine profile shows removable traffic.
- ~~explicit vLLM movement admission without replacing grouped GEMM~~ — the
  opt-in adapter reuses vLLM caller-owned scratch/output tensors, routes only
  supported Cutlass/Humming contracts, records hits/rejections, and fails
  closed after an admitted Loom launch.

The [all-local](results/h20-moe-movement-20260801.json) and
[expert-parallel](results/h20-moe-movement-ep-20260801.json) H20 gates compare
the complete permute-plus-combine pipeline with vLLM 0.25.1 while keeping
grouped GEMM absent from both sides. CUDA Graph ratios span `0.962-1.163x` for
the all-local 1-2,048-token sweep and `1.013-1.191x` for the 64-global/32-local
32-2,048-token sweep. Source matrices pass vLLM 0.24/0.25 and PyTorch 2.10/2.11.
This is operator-boundary evidence, not a model claim.

The [vLLM engine gate](results/h20-vllm-engine-moe-movement-20260801.json)
runs isolated baseline and Loom `LLM.generate` processes over a synthetic
two-layer Qwen2-MoE checkpoint with vLLM 0.25.1 `fp8_per_channel` and the
Cutlass backend. Generated token IDs are exact; the Loom process records 48
FP8 permutation and 48 BF16 combine hits through caller-owned tensors with no
rejection, while grouped GEMM remains vLLM-owned. The median ratio is
`1.0205x`. This closes explicit engine admission only: a random tiny checkpoint
does not establish production-model, serving, or routing value.

Exit: a pinned production-representative MoE workload shows that movement, and
routing only if profiling admits it, reduce model-level latency on a named
engine. The vendor grouped GEMM is identical on both sides of the comparison.

## K6: Attention

Status: in progress.

- ~~paged MQA/GQA base contract and CPU oracle~~ — one query per request,
  native paged KV, MQA/GQA mapping, and block-table validation are fixed;
- ~~first handwritten short-context CUDA candidate~~ — F32/FP16/BF16 C ABI,
  safe Rust, current-stream PyTorch, randomized oracle, compile/graph gates,
  and an H20 FA3 comparison are complete;
- ~~GQA-packed 32/64-token optimization~~ — two/four query heads reuse each
  paged K/V load; compile-time partial tails support odd GQA ratios without
  adding hot-loop guards to full groups;
- ~~native vLLM cache layout and broad short-context qualification~~ — the C
  ABI accepts interleaved K/V block strides; a 156-case shape sweep and focused
  132-case batch sweep identify the exact winning envelope;
- ~~measured-shape vLLM 0.24/0.25 adapter with explicit FA3 fallback~~ — the
  opt-in route is limited to FP16/BF16 Hq/Hkv 32/8, D128, block 16/32,
  batch <=128, context <=32; 0.25 compatibility and the 0.24 direct-backend
  and stable-output synthetic-engine gates pass;
- pretrained-model gate and broader head geometry — the first Qwen2.5 `14/2`,
  D64 attempt hit the engine but failed exact-token and latency gates, so it
  remains intentionally unrouted;
- ~~tiled split-K/LSE optimization for 128-1024 tokens~~ — explicit
  caller-owned Rust/C workspace, stable partial-state merge, CUDA Graph-safe
  PyTorch dispatch, and H20 legacy/FA3 gates are complete for D128 batches
  1-8; it materially improves Loom but does not widen the vLLM route because
  FA3 remains faster;
- vendor attention integration where it wins;
- distributed split-KV/LSE merge, sliding-window variants, and MLA when a
  consumer exists.

Exit: hardware-qualified engine evidence determines admission; prior Loom
Attention prototype code is not carried forward automatically.

## K7: Communication-Aware Fusion

Status: planned after reproducible single-GPU and multi-GPU engine baselines.

- tensor-parallel reduction plus residual/norm epilogues;
- sharded-vocabulary sampling and selected-logprob merge;
- expert-parallel dispatch/combine integration.

Exit: end-to-end TP or EP goodput improves under an equivalent NCCL/transport
baseline; local adapters do not count as distributed evidence.

## K8: Engine-Neutral Rust Decode Proof

Status: planned after one new post-K0.7 feature reaches an engine.

- accept vendor- or engine-produced CUDA tensors through borrowed Rust device
  memory and a non-owning stream;
- chain a minimal decode slice such as cache update, logits processing,
  sampling, and token output through the existing safe Loom APIs;
- use a callback or external boundary for every GEMM and model-owned attention
  operation;
- allocate no private copy of framework tensors and own no scheduler, model
  weights, tokenizer, or KV-cache lifetime.

Exit: a reproducible Rust example performs one deterministic zero-copy decode
step, matches a reference token and state update, survives external-stream and
CUDA Graph gates where applicable, and demonstrates that Loom is engine-neutral
without growing into an inference engine.

The complete intended surface, including profile-gated layout primitives, is
tracked in the [operator catalog](operator-catalog.md).
