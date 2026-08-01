# Changelog

Loom Kernels follows Semantic Versioning. The Rust crates use Cargo's SemVer
spelling; Python package metadata uses the equivalent PEP 440 spelling.

## Unreleased

### Breaking

- replaced the mixed Python/ctypes, direct C++ CUDA, and partial Rust-bridge
  framework stack with one required
  `PyTorch -> C++ dispatcher -> Rust bridge -> safe Rust -> CUDA` path;
- removed `_native.py`, every `*_unchecked` dispatcher operator,
  `LOOM_KERNELS_CUDA_LIBRARY`, `libloom_kernels_cuda.so`, `adapter_backend()`,
  and the per-operator telemetry functions;
- replaced telemetry with `Operator`, `launch_count(operator)`, and
  `reset_launch_count(operator)`;
- changed Rust CUDA entrypoints for row-strided logits, paged decode, RoPE/KV,
  and activation FP8 to require explicit physical-layout objects;
- changed `PagedDecodeAttentionSpec::new` to accept
  `max_sequence_length` independently from block-table capacity;
- replaced the production ATen/c10 dispatcher with one boxed LibTorch Stable
  ABI implementation targeting PyTorch 2.10; no old dispatcher or experimental
  probe remains;
- removed `LOOM_KERNELS_TORCH_LIBRARY`: installed wheels load only their
  package-local native pair, while editable source checkouts use repository
  `build/`.
- renamed private Rust CUDA modules to `<domain>_dispatch.rs` and checked C
  modules to `<domain>_bridge.rs`; no legacy source-path aliases remain.
- changed `RopePagedKvWriteSpec::new` and the PyTorch
  `rope_paged_kv_write_` schema to carry an explicit cache encoding and K/V
  scale tensors; the checked bridge ABI is now version 2 and native wheels use
  a distinct `2cu...` build tag.
- replaced separate mutable K/V cache-view arguments in the PyTorch
  `rope_paged_kv_write_` schema with the engine's single packed
  `[blocks, 2, block, heads, head_size]` allocation; no legacy overload is
  retained.
- advanced the checked framework bridge to ABI 4 for sampled-token plus top-k
  logprobs; its wheel used a distinct `4cu...` build tag and retained an
  immutable clean-install record separate from the older ABI2 artifact.
- advanced the checked framework bridge to ABI 5 for exact per-row in-place
  top-k filtering and a caller-owned uint32 workspace; no ABI4 compatibility
  entrypoint or legacy workspace-free launch is retained.
- advanced the checked framework bridge to ABI 6 for fused top-p filtering and
  F32 renormalization; no ABI5 compatibility entrypoint or filter/softmax twin
  is retained.
- advanced the checked framework bridge to ABI 7 for fused mixed-sampling
  logits preprocessing; no ABI6 compatibility entrypoint or decomposed
  mask/bias/suppression/temperature twin is retained.
- advanced the checked framework bridge to ABI 8 for explicit-state
  categorical sampling; source dispatchers require ABI8 and no ABI7
  compatibility entrypoint is retained.
- advanced the checked framework bridge to ABI 9 by replacing the former
  RMSNorm-to-FP8 schema with vLLM's exact optional-residual mutation schema;
  no ABI8 normalization overload, boxed-op alias, or bridge compatibility shim
  is retained.
- advanced source to bridge ABI10 for optional-residual
  RMSNorm-to-dynamic-INT8; the dispatcher requires the matching ABI10 bridge
  and retains no ABI9 compatibility entrypoint. The exact ABI10 artifact now
  passes the repository-free clean-install matrix.
- advanced source to bridge ABI11 for fused SiLU-and-Mul-to-dynamic-INT8;
  the dispatcher requires the matching ABI11 bridge and retains no ABI10
  compatibility entrypoint. H20 source, real-engine, and repository-free
  matrix-wheel gates pass; ABI10 remains immutable historical evidence.
- advanced source to bridge ABI12 for stable MoE permutation and weighted
  combine; the dispatcher requires the matching ABI12 bridge and retains no
  ABI11 compatibility entrypoint or MoE-free fallback. The ABI11 wheel remains
  the latest qualified binary artifact until ABI12 clean-install gates close.

### Added

- stable F32/FP16/BF16 plus byte-exact FP8 E4M3FN MoE expert-major
  permutation and F32/FP16/BF16 weighted combine
  around unchanged vendor grouped GEMM, including Rust contracts/oracles,
  vLLM-compatible expert-parallel remote ordering, caller-owned safe CUDA and
  checked bridge paths, 16-byte vectorized handwritten CUDA with scalar
  fallback, allocating and caller-owned Stable ABI PyTorch APIs, compile/graph
  tests, and all-local plus expert-parallel H20 named-baseline evidence. An
  explicit vLLM 0.25.1 Cutlass adapter reaches the same operators in an
  isolated synthetic Qwen2-MoE `LLM.generate` gate with exact tokens and
  unchanged vendor grouped GEMM. Direct movement and engine admission are
  qualified; production-workload routing value and ABI12 wheel qualification
  remain open;
- complete bridge coverage for RMSNorm, Add+RMSNorm, RMSNorm+FP8/INT8,
  SiLU-and-Mul, SiLU-and-Mul+FP8/INT8, RoPE+paged-KV, greedy/selected logprobs,
  Min-P, and base/split-K paged decode;
- optional-residual Add+RMSNorm-to-dynamic-per-token-FP8 across the CPU oracle,
  safe Rust, checked bridge, handwritten CUDA, Stable ABI dispatcher, and both
  vLLM 0.24/0.25 fusion keys. H20 exact-byte, direct named-baseline,
  generated-source, unchanged-Cutlass-GEMM, and order-reversed Qwen prefill
  gates pass; decode-heavy latency crosses parity and carries no speedup claim;
- optional-residual Add+RMSNorm-to-symmetric-dynamic-per-token-INT8 across the
  CPU oracle, safe Rust, checked ABI10 bridge, shared handwritten CUDA,
  Stable ABI dispatcher, direct Python APIs, telemetry, and explicit vLLM
  0.24/0.25 compiler patterns. H20 source tests and real Qwen2.5 W8A8
  invocation pass with unchanged Cutlass GEMM. The real-layer shadow has one
  one-LSB difference across 688,128 INT8 elements with exact scales/residuals,
  while held-out one-step quality and dual-order latency keep the route
  explicit opt-in with no default, exact-output, or speedup claim. Its ABI10
  wheel distribution gate is qualified separately;
- split-half SwiGLU followed by symmetric dynamic per-token INT8 across the
  CPU oracle, safe Rust, checked ABI11 bridge, handwritten CUDA, Stable ABI
  PyTorch API, telemetry, benchmark, and an explicit vLLM 0.24/0.25 compiler
  pattern. H20 source suites pass, and a real Qwen2.5 W8A8 graph preserves all
  eight Cutlass scaled-mm sites while both provider orders exactly match 32/32
  top-1 tokens, top-20 sets, and common logprobs. Compiled CUDA Graph ratios
  remain below parity and engine latency is not order-stable, so the route is
  explicit-only with no speedup claim. Its ABI11 wheel distribution gate is
  qualified separately;
- explicit Rust physical-layout contracts for padded logits, packed QKV,
  NHD/HND caches, interleaved cache storage, scale layout, and FP8 scale upper
  bounds;
- standalone PyTorch `rms_norm` and `rms_norm_out` APIs;
- vLLM 0.25 support, an explicit compatibility matrix, H20 0.24/0.25 GPU-suite
  evidence, contribution guidance, and structured issue forms;
- a two-minor H20 binary gate proving the exact same dispatcher `.so` on
  PyTorch 2.10 and 2.11, plus a CI guard against unstable ATen/c10 C++ APIs;
- a clean-revision native wheel builder that packages exactly the Rust CUDA
  bridge and Stable ABI dispatcher, emits and validates their matrix manifest
  and hashes, and rejects accidental source-only wheels;
- immutable ABI4/ABI5/ABI6/ABI7/ABI8/ABI9/ABI10 history and qualified ABI11 H20
  wheel-install evidence for PyTorch 2.10/2.11 and vLLM 0.24/0.25.
- deterministic greedy speculative verification and accepted/bonus-token
  compaction over vLLM-compatible flattened ragged metadata, with Rust/CUDA/
  PyTorch coverage, explicit vLLM 0.24/0.25 registration, and H20 evidence.
- a process-isolated vLLM draft/target benchmark with exact native/Loom token
  and acceptance gates, measured launch coverage, post-timing CUDA boundary
  profiling, provider-order reversal, and pinned Qwen2.5 H20 evidence.
- static per-tensor/per-head FP8 E4M3 quantize-on-write in the existing fused
  RoPE+paged-KV path, including CPU oracles, byte-typed safe CUDA dispatch,
  checked Rust bridge, Stable ABI PyTorch schema, vLLM 0.24/0.25 admission, and
  native-versus-FP8 benchmark metadata. Exact-byte, framework, operator,
  clean-wheel, and real-engine invocation H20 gates pass; the system-level
  native-versus-FP8 quality/capacity/serving gate remains open.
- reproducible FP8 KV calibration and held-out corpus tools that pin the tool,
  checkpoint, model config, tokenizer, dataset, package, selected-row, scale,
  observer, and output digests without adding calibration dependencies to the
  runtime wheel; the attention/KV-only recipe requires a stateful multi-sample
  observer and leaves model weights unchanged.
- deterministic sampled-token plus top-k raw logprobs for F32/FP16/BF16,
  including CPU oracles, a two-stage handwritten CUDA reduction with
  caller-owned workspace, safe Rust and checked bridge dispatch, PyTorch
  compile/graph coverage, direct H20 evidence, and an exact vLLM 0.24 adapter
  that preserves engine `torch.topk` tie order.
- exact F32/FP16/BF16 in-place top-k filtering with threshold ties preserved,
  a single partition-radix-sort plus parallel binary-count CUDA algorithm for
  all valid `top_k` values, safe Rust and checked ABI5 dispatch, current-stream
  PyTorch compile/graph coverage, an opt-in vLLM 0.24/0.25 small-row adapter,
  and an H20 gate showing `1.42–2.15x` over the corresponding full-sort path.
- fused F32/FP16/BF16 top-p filtering plus contiguous F32 retained-prefix
  renormalization, with deterministic descending-token-ID ties, one
  partition-radix/device-selection CUDA algorithm, safe Rust and checked ABI6
  dispatch, current-stream PyTorch compile/graph coverage, and a measured
  vLLM 0.24/0.25 route for F32 top-p-only rows 2–7 at vocabularies of at least
  32,768. vLLM keeps RNG, generators, token selection, and unsupported policy;
  the H20 gate measures `1.72–1.77x` at 151,936 tokens and `1.15–1.34x` at
  the 32,768-token boundary.
- one exact ABI6 `py3-none-linux_x86_64` matrix wheel containing both native
  libraries; repository-free H20 installs pass 277 tests on each supported
  vLLM minor and 186 applicable tests on PyTorch 2.10. The artifact is
  qualified but not published.
- fused in-place F32 logits preprocessing for mixed greedy/random batches,
  combining dense blocked-token masking, unique sparse additive bias, sparse
  suppression, and per-row temperature in one handwritten CUDA pass. The
  checked ABI7 path includes safe Rust/oracle coverage, Stable ABI PyTorch
  compile/graph gates, conservative vLLM 0.24/0.25 registration, and exact
  H20 operator plus order-reversed Qwen2.5 evidence.
- one exact ABI7 `py3-none-linux_x86_64` matrix wheel containing both native
  libraries; repository-free H20 installs pass 286 tests on each supported
  vLLM minor and 193 applicable tests on PyTorch 2.10. The artifact is
  qualified but not published.
- a refreshed ABI7 wheel from revision `f98a931` that packages the final FP8
  KV adapter and passes the complete 286-test vLLM 0.24 H20 suite from a fresh
  repository-free environment; the remaining refresh matrix rows stay open.
- a source-pinned vLLM V1 KV-movement admission probe. A real prefix hit and
  three preemptions produced zero physical copies, so default
  prefix/preemption movement is rejected and the next candidate is the
  explicit-state counter-based sampling boundary.
- a source-pinned vLLM 0.24 seeded-sampling admission profiler and H20 result.
  The all-seeded sampling-only path reaches 34 kernels and `19.45 MB` of peak
  incremental storage at 32 rows, so the explicit-state ABI8-A categorical
  sampler was admitted for implementation without claiming an unbuilt
  speedup.
- the ABI8-A categorical sampler itself: canonical Philox4x32-10 with
  caller-owned `(seed, counter)` state, a fixed Rust/CUDA F32 CDF tree, one
  handwritten kernel, safe Rust and checked bridge dispatch, Stable ABI
  PyTorch mutation schema, direct Python API, compile/FakeTensor/current-stream/
  CUDA Graph coverage, and a 65,536-draw distribution gate. On H20 the direct
  boundary is `1.15–5.40x` faster than vLLM's all-seeded fallback at
  4–32-row cases with one kernel and no probability-shaped temporary.
- an explicit vLLM 0.24/0.25 categorical adapter with state owned by
  `CachedRequestState` and contiguous active `InputBatch` slots. State survives
  removal, condensation, resumption, and swaps; unseeded random requests and
  speculative engines fail before sampling. Order-reversed Qwen2.5-0.5B H20
  runs exactly replay each provider stream, launch Loom once per decode step,
  and measure an order-stable `1.057–1.081x` batch-32 engine ratio. Batch 1–4
  retains a measured `1.5–2.4%` cost.
- the exact ABI8 `e2c2982` two-library wheel and repository-free H20 matrix:
  293 tests pass with each supported vLLM minor, 199 applicable tests pass on
  PyTorch 2.10, every installed native hash matches the manifest, and no
  repository or library-path override is present. The artifact is qualified
  but not published.
- the exact ABI9 `7df4133` two-library wheel and repository-free H20 matrix:
  305 tests pass with each supported vLLM minor, 201 applicable tests pass on
  PyTorch 2.10, every installed native hash matches the manifest, and no
  repository or library-path override is present. The artifact is qualified
  but not published.
- the exact ABI10 `de28ceb` two-library wheel and repository-free H20 matrix:
  326 tests pass with each supported vLLM minor, 218 applicable tests pass on
  PyTorch 2.10, both native libraries load from each fresh venv, and no
  repository or library-path override is present. The artifact is qualified
  but not published.
- the exact ABI11 `afc54c4` two-library wheel and repository-free H20 matrix:
  342 tests pass with each supported vLLM minor, 231 applicable tests pass on
  PyTorch 2.10, all twelve vLLM-free SiLU-and-Mul-to-INT8 tests execute, both
  native libraries load from each fresh venv, and no repository or
  library-path override is present. The artifact is qualified but not
  published.

### Fixed

- the RMSNorm-to-INT8 Python oracle no longer imports vLLM unconditionally;
  vLLM-free PyTorch 2.10 now exercises direct, `torch.compile`, and CUDA Graph
  paths against an engine-independent reference while vLLM environments retain
  an additional IR-equivalence check;
- source-checkout library discovery now follows the packaged
  `crates/loom-cuda-sys/cuda` layout after removal of the legacy root `cuda`
  directory;
- constrained the wheel build backend to setuptools 80–81 to match PyTorch
  2.11's build dependency range.
- sized vectorized RMSNorm-quantization blocks from four-element pack count,
  avoiding hundreds of idle reduction threads at hidden size 896.
- made fused RoPE+KV auto-functionalization preserve the complete packed cache
  on PyTorch 2.10 by exposing one real mutable cache allocation instead of two
  storage-aliasing views.
- kept vLLM's static `quant_fp8` query quantization opaque alongside
  `rotary_embedding`, allowing the official RoPE+KV fusion pattern to match and
  reach Loom for FP8-cache models.
- additively registered vLLM's per-KV-head static-query RoPE+KV compiler form
  while preserving its original scalar pattern, so dataset-calibrated
  `attn_head` checkpoints reach the same Loom fusion boundary.

## 1.0.0-alpha.1 — 2026-07-22

First public alpha of Loom Kernels as a Rust-first CUDA operator backend for
LLM inference.

GitHub tag and Release name: `v1.0.0-alpha.1`. Cargo packages use the matching
Semantic Versioning spelling `1.0.0-alpha.1`.

### Included

- backend-independent Rust contracts, capability queries, and deterministic
  CPU oracles;
- safe Rust CUDA streams, buffers, events, checked dispatch, and a raw C ABI;
- non-owning `CudaStreamRef`, `DeviceSlice`, and `DeviceSliceMut` adapters for
  zero-copy execution over framework-controlled streams and device memory;
- sealed read/write memory traits shared by every owned and borrowed safe Rust
  operator entrypoint;
- handwritten CUDA for normalization/quantization, SwiGLU, RoPE plus paged-KV
  writes, decode-tail sampling and logprobs, Min-P, and paged decode attention;
- opt-in PyTorch and vLLM 0.24 adapters with explicit shape and policy gates;
- H20 correctness, framework, engine, and named-baseline evidence kept as
  machine-readable artifacts;
- self-contained Cargo source archives and a pure Rust CUDA smoke example that
  covers both owned and borrowed runtime resources on NVIDIA H20.

### Alpha boundaries

- APIs and admitted shape envelopes may change before 1.0 stable;
- CUDA is opt-in and requires a local NVIDIA toolkit at build time;
- Python packaging is source-adapter metadata, not a portable CUDA/LibTorch
  binary wheel;
- unsupported engine contracts intentionally fall back to the native backend.
