# Evidence

This directory stores machine-readable results from permanent Loom Infer
providers.

## Results

Two current 2026-08-06 records qualify unrolled 16-byte `cp.async` staging for
tiled eight-partition ragged GQA4 at correctness, sanitizer, and matched eager
gates. Earlier tiled, direct, and token-parallel records remain immutable
below. None contains a Graph, engine, or serving claim.

- [FlashInfer v0.6.16.post1 ragged cp.async matched eager performance](h20-flashinfer-v0.6.16.post1-ragged-prefill-cp-async-eager-performance-20260806.json):
  asynchronous K/V staging lowers Loom long-GQA latency to `48.232`
  microseconds, 1.148x below the previous tiled split-K result and 7.729x below
  direct. FlashInfer remains 2.206x lower-latency on stable long GQA.
  Short-MHA and mixed-MQA rankings are excluded because FlashInfer's order
  deltas are 10.643% and 14.097%. Both provider orders and all 600 raw samples
  are retained.

- [BF16 ragged cp.async H20 correctness](h20-bf16-ragged-prefill-cp-async-correctness-20260806.json):
  direct, token-parallel, and tiled two-kernel MHA/MQA/GQA cases pass the CPU
  oracle. Four Compute Sanitizer tools report no errors. The tiled partial uses
  226 registers, 16,384 bytes shared memory, unrolled 16-byte `cp.async.cg`
  copies, and no stack or spills.

The previous tiled split-K result remains immutable history:

- [FlashInfer v0.6.16.post1 ragged tiled split-K matched eager performance](h20-flashinfer-v0.6.16.post1-ragged-prefill-tiled-split-k-eager-performance-20260806.json):
  fused tensor-core tiling and eight-way split-K lower Loom long-GQA latency
  to `55.355` microseconds, 3.986x below the previous specialized result and
  6.734x below direct. FlashInfer remains 2.538x and 1.349x lower-latency on
  stable long GQA and mixed MQA. Short-MHA ranking is excluded because
  FlashInfer's order delta is 54.709%. Both provider orders and all 600 raw
  samples are retained.

- [BF16 ragged tiled split-K H20 correctness](h20-bf16-ragged-prefill-tiled-split-k-correctness-20260806.json):
  direct, eight/sixteen-warp, and tiled two-kernel MHA/MQA/GQA cases pass the
  CPU oracle. Four Compute Sanitizer tools report no errors. The tiled partial
  and merge kernels use 168/34 registers; the partial uses 16,384 bytes shared
  memory, and both assemble without stack or spills.

The earlier dual token-parallel and uniform eight-warp results remain
immutable history:

- [FlashInfer v0.6.16.post1 ragged dual token-parallel matched eager performance](h20-flashinfer-v0.6.16.post1-ragged-prefill-dual-token-parallel-eager-performance-20260806.json):
  sixteen-warp MQA lowers Loom latency by 1.254x versus the earlier eight-warp
  path and 7.245x versus direct. The previous long-GQA path was 1.689x below
  direct but remained 10.028x slower than FlashInfer. All 600 raw samples are
  retained.

- [BF16 ragged dual token-parallel H20 correctness](h20-bf16-ragged-prefill-dual-token-parallel-correctness-20260806.json):
  direct, eight-warp, and sixteen-warp MHA/MQA/GQA cases pass the CPU oracle.
  Four Compute Sanitizer tools report no errors. The eight-warp and
  sixteen-warp kernels use 37/38 registers and 4,160/8,320 bytes shared memory
  with no stack or spills.

- [FlashInfer v0.6.16.post1 ragged token-parallel matched eager performance](h20-flashinfer-v0.6.16.post1-ragged-prefill-token-parallel-eager-performance-20260806.json):
  eight-warp token parallelism lowers Loom mixed-MQA and long-GQA eager
  latency by 5.779x and 1.689x versus the immutable direct record. FlashInfer
  remains 10.114x lower-latency on stable long GQA. Short-MHA and mixed-MQA
  provider rankings are excluded because FlashInfer's order deltas are 64.03%
  and 15.21%. All 600 raw samples are retained.

- [BF16 ragged token-parallel H20 correctness](h20-bf16-ragged-prefill-token-parallel-correctness-20260806.json):
  direct and eight-warp MHA/MQA/GQA cases pass the CPU oracle with maximum
  BF16 output error `1.220703125e-4` and maximum log2-LSE error
  `2.861022949e-6`. Four Compute Sanitizer tools report no errors. The
  token-parallel kernel uses 37 registers and 4,160 bytes shared memory with
  no stack or spills.

- [FlashInfer v0.6.16.post1 ragged direct matched eager baseline](h20-flashinfer-v0.6.16.post1-ragged-prefill-direct-eager-performance-20260806.json):
  the immutable direct Loom source and pinned FlashInfer FA2 path use identical
  BF16 tensor and I32 indptr fixtures. The record retains both provider orders
  and all 600 raw samples.

- [BF16 ragged prefill H20 correctness](h20-bf16-ragged-prefill-correctness-20260806.json):
  the earlier correctness-only direct provider history remains immutable.
  BF16 NHD D128 MHA, MQA, and GQA batches pass against the CPU oracle with
  bit-exact BF16 output and maximum log2-LSE error `4.768371582e-7`.
  Separate query and KV `indptr` arrays, mixed lengths, bottom-right causal
  alignment, short metadata, and an invalid-metadata device guard are covered.
  Four Compute Sanitizer tools report no errors.

Four additional 2026-08-06 records preserve the direct paged baseline and
qualify the current token-parallel provider at device and matched eager gates.
They contain no Graph, engine, or serving claim.

- [FlashInfer v0.6.16.post1 paged token-parallel matched eager performance](h20-flashinfer-v0.6.16.post1-paged-token-parallel-eager-performance-20260806.json):
  eight-warp block-local merge lowers Loom MQA/GQA eager latency by 3.78x and
  3.32x versus the immutable direct record. Loom is 4.41x lower-latency for
  MHA and 2.35x lower-latency for MQA than FlashInfer. GQA remains excluded
  from stable ranking because FlashInfer's order delta is 60.62%. All 600 raw
  samples are retained.

- [BF16 paged token-parallel H20 correctness](h20-bf16-paged-batch-decode-token-parallel-correctness-20260806.json):
  direct MHA and eight-warp MQA/GQA produce BF16 output bit-exact with the CPU
  oracle. Four Compute Sanitizer tools report no errors. The token-parallel
  kernel uses 39 registers and 4,192 bytes shared memory with no stack or
  spills.

- [FlashInfer v0.6.16.post1 paged batch-decode matched eager performance](h20-flashinfer-v0.6.16.post1-paged-batch-decode-eager-performance-20260806.json):
  identical BF16 page-pool bits, `i32` page tables, preallocated buffers, CUDA
  events, 200 warm-up calls, 100 calls per sample, 50 samples per provider
  order, and both provider orders. Loom is 4.21x lower-latency for batch-1
  MHA; FlashInfer is 1.62x lower-latency for mixed-length batch-3 MQA. The
  batch-4 GQA ranking is excluded because FlashInfer's order delta is 52.49%.
  All 600 official raw samples and CUPTI diagnostic hashes are retained.

- [BF16 paged batch-decode H20 correctness](h20-bf16-paged-batch-decode-correctness-20260806.json):
  BF16 NHD D128 page-size-16 MHA, MQA, and GQA batches pass against the CPU
  oracle with bit-exact BF16 output and maximum log2-LSE error
  `4.768371582e-7`. Mixed lengths, partial and full tail pages, physical-page
  reordering and reuse, short metadata, and an invalid-page device guard are
  covered. Four Compute Sanitizer tools report no errors.

Three current 2026-08-05 records qualify the parallel merge, isolated kernel
decomposition, and matched eager performance. They do not replace the immutable
serial-merge and pre-split-K history below.

- [BF16 single-decode parallel-merge H20 correctness](h20-bf16-single-decode-parallel-merge-correctness-20260805.json):
  an eight-warp block-local F32 merge passes the tuned split 12, 16, and 64
  cases. Four Compute Sanitizer tools report no errors. The merge uses 4,160
  bytes of shared memory and assembles without stack or spills.

- [BF16 single-decode parallel-merge isolated profiling](h20-bf16-single-decode-parallel-merge-profiling-20260805.json):
  Nsight Systems CUPTI activity timing records the KV-length-4096 partial
  kernel at `31.104` microseconds and the parallel merge at `5.056`
  microseconds. The merge is 3.99x lower-duration than the recorded serial
  merge. Nsight Compute hardware counters are excluded because the host sets
  `RmProfilingAdminOnly=1`.

- [FlashInfer v0.6.16.post1 parallel-merge matched eager performance](h20-flashinfer-v0.6.16.post1-parallel-merge-eager-performance-20260805.json):
  parallel merge raises the complete Loom speedup over the direct baseline to
  5.39x at GQA KV length 127 and 38.19x at KV length 4096. FlashInfer remains
  1.17x and 2.09x lower-latency. MQA KV length 33 is approximately tied, but
  its FlashInfer provider-order delta exceeds five percent.

- [BF16 single-decode split-K H20 correctness](h20-bf16-single-decode-split-k-correctness-20260805.json):
  balanced partial states and stable F32 merge pass MQA and GQA cases, including
  KV length 4096 with 64 partitions. A one-command queue and short workspace
  fail before submission. Four Compute Sanitizer tools report no errors, and
  both SM90 kernels assemble without stack or spills.

- [FlashInfer v0.6.16.post1 split-K matched eager performance](h20-flashinfer-v0.6.16.post1-split-k-eager-performance-20260805.json):
  identical BF16 operand bits, preallocated buffers, CUDA events, 200 warm-up
  calls, 100 provider calls per sample, 50 samples per provider order, and both
  provider orders. Split-K lowers Loom median latency by 3.79x at GQA KV length
  127 and 26.79x at KV length 4096 relative to the recorded pre-split-K source.
  FlashInfer remains 1.69x and 3.00x lower-latency at those shapes.

- [Pre-split-K FlashInfer v0.6.16.post1 matched eager performance](h20-flashinfer-v0.6.16.post1-eager-performance-20260805.json):
  identical BF16 operand bit patterns, preallocated buffers, CUDA events, 200
  warm-up launches, 100 launches per sample, 50 samples per provider order,
  and both provider orders. FlashInfer is 6.29x lower-latency at GQA KV length
  127 and 80.08x lower-latency at GQA KV length 4096. Loom is lower-latency in
  the fixed M=1 cuBLASLt GEMM case. RMSNorm comparison is excluded because the
  unmodified FlashInfer release did not compile on the declared CUDA 13.1 host.
  The record preserves and explains one corrected raw Loom commit-metadata typo;
  its source tree, lockfile, raw files, and timing samples remain hash-bound.

Two 2026-08-03 records qualify the current source projection. One records the
new attention contract. The other reruns every provider affected by the shared
command resolver change. Earlier records remain immutable historical results.

- [Shared command-resolution H20 regression](h20-shared-command-regression-20260803.json):
  RMSNorm, BF16 GEMM, the fixed-address Graph, and BF16 single decode pass on
  one source projection. The declared Compute Sanitizer matrix reports no
  errors or device leaks.

- [BF16 single-decode attention H20 correctness](h20-bf16-single-decode-correctness-20260803.json):
  BF16 NHD D128 MHA, MQA, and GQA cases pass against the CPU reference.
  Exact-span and duplicate-binding checks reject invalid launches. Compute
  Sanitizer reports no errors or device leaks. The record contains no
  performance measurement.

- [Owned bindings and CUDA Graph H20 result](h20-owned-bindings-cuda-graph-correctness-20260803.json):
  RMSNorm and BF16 GEMM correctness pass. The final output after two fixed-address
  Graph replays matches the CPU fixture. Compute Sanitizer reports no errors or
  device leaks.

- [F32 RMSNorm H20 correctness](h20-rms-norm-f32-correctness-20260802.json):
  four shapes and exact buffer rejection pass on a non-default stream.
- [F32 RMSNorm H20 command scope](h20-rms-norm-f32-command-scope-20260802.json):
  checked bindings, queue reuse, two-command chaining, and a partial-scope
  rejection pass. The record contains no performance, graph, engine, or
  serving claim.
- [FP16 and BF16 RMSNorm H20 correctness](h20-rms-norm-low-precision-20260802.json):
  scalar and packed paths, typed heterogeneous bindings, short-buffer checks,
  signed zero, and reusable two-command scopes pass.
- [BF16 cuBLASLt GEMM H20 correctness](h20-bf16-cublaslt-correctness-20260802.json):
  fixed-plan standalone and transpose-sensitive cases, exact spans, command
  capacity, reuse, and RMSNorm-to-GEMM chaining pass. No performance claim is
  included.

## Claim levels

| Level | Required evidence |
| --- | --- |
| Correctness | Declared contract, oracle, error limit, and edge cases |
| Lifecycle | Stream order, resource retention, capacity, reuse, and completion behavior |
| Kernel | Matched buffers, streams, provider order, and raw device timings |
| Graph | Capture and replay behavior with fixed plans and declared binding policy |
| Engine | Real invocation, provider hit count, and model output |
| Serving | TTFT, TPOT, throughput, memory, and workload definition |

## Record format

Each JSON record includes:

- source commit, lockfile hash, and clean-worktree state.
- hardware, driver, compiler, CUDA, and `cuda-oxide` versions.
- operator contract, shapes, dtypes, layouts, and tolerances.
- commands and artifact hashes.
- accepted and excluded claims.

Performance records also include raw timing samples, summary statistics, and
provider order.

Use `h20-<operator>-<gate>-YYYYMMDD.json` for H20 results. A result file is
immutable after review. A changed source, contract, toolchain, or provider
requires a new record.
