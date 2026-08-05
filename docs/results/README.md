# Evidence

This directory stores machine-readable results from permanent Loom Infer
providers.

## Results

The 2026-08-05 matched record is the first performance result. It does not
replace the correctness and sanitizer records below.

- [FlashInfer v0.6.16.post1 matched eager performance](h20-flashinfer-v0.6.16.post1-eager-performance-20260805.json):
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
