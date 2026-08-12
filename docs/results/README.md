# Evidence index

This directory stores immutable machine-readable records. Each record applies
only to its source tree, contract, toolchain, device, and command matrix.

The project rename changed crate and native-provider identities. Records that
use the former name remain historical and unchanged. They do not qualify the
current Oxide source. See [rename provenance](../design/rename-provenance.md).

## Current qualification notice

The fused RoPE plus paged KV append contract now requires an authoritative
reference count for every physical page. Every write target must have reference
count one. The engine or KV pager must make shared tails private before enqueue.

All fused-append H20 records dated 2026-08-06 predate this rule. They remain
useful for implementation history and old-contract performance, but they do
not qualify the current source.

The 2026-08-12 phase 2 record below completes R1 for the exact declared H20
boundary. It binds all nine permanent runners, 89 machine-readable passing
case lines, 12 named Graph case lines, and 36 of 36 Compute Sanitizer cells to
clean source `477b47a` and nine recorded `sm_90a` binary hashes. Performance,
real-model engine execution, serving, and other GPUs remain outside R1.

Performance is qualified separately by the current-source R2 matched record
below. It does not broaden the R1 correctness or device boundary.

The current DeviceRegion work changed submission for RMSNorm, GEMM, decode,
prefill, and RoPE:

- Device and Graph records dated through 2026-08-07 predate that launch path.
- The phase 2 record qualifies only the current paths and exact cases emitted
  by its nine permanent runners; it does not revive broader historical claims.
- The 2026-08-11 experimental Loom GEMV record is a historical pre-rename row.
  It covers the post-change path only for its declared H20 `sm_90a` contract.

This notice includes the two paged-prefill records dated 2026-08-07. They apply
to clean source commit `8478ee9`, before the DeviceRegion submission path.

No other record gains a broader claim because it appears in this index. Check
the record's `accepted_claims` and `excluded_claims` fields.

Oxide Infer H20 gates use Rust CPU references.
FlashInfer comparison scripts use separate PyTorch F32 references.
No published result proves cross-provider reference-digest parity.
The summarizer rejects mixed contracts, fixtures, measurements, and execution identities.

Current Graph qualification requires every valid recipe to capture a
target-poison operator stage before the real operator stage. Both stages run
inside each replay and write the same output addresses.

Earlier records do not satisfy that rule. Benchmark `graph_nodes` fields are
handwritten metadata, not CUDA driver enumeration.

A binary FlashInfer wheel records its verified distribution version but no
source commit unless the artifact proves one.

Matched attention and attention-Graph records through 2026-08-07 predate these
checks and remain historical. Use the 2026-08-12 FlashInfer 0.6.17 record below
for current-source eager-provider rankings. No current Graph comparison has
been published.

## Current-source R1 phase 2

- [H20 phase 2 correctness, fixed-address Graph, and sanitizer qualification](h20-r1-phase2-qualification-477b47a-20260812.json)
  binds clean source `477b47a`, strict CUDA Clippy, seven release `sm_90a` lab
  tests, all nine permanent H20 runners, and all four sanitizer tools over each
  exact runner binary. It records 89 machine-readable passing case lines, nine
  valid-output Graph cases, three rejection Graph cases, and 36 passing
  sanitizer cells without filters or suppressions.

This record completes R1 at its declared boundary. It excludes performance,
native GEMV promotion, real mistral.rs model execution, serving metrics, a
valid-output paged-decode Graph, and all GPUs other than the recorded H20.

## Current-source R2 matched performance

- [Current long paged-GQA4 eager performance against FlashInfer 0.6.17](h20-flashinfer-v0.6.17-paged-prefill-current-gqa4-eager-performance-02faf27-20260812.json)
  binds source `02faf27`, source `49290b5` as a separate progression cohort,
  exact runner and artifact hashes, 400 raw samples, correctness, valid-output
  and rejection Graph replay, and all four Compute Sanitizer tools. In the
  provider cohort, Oxide measures 46.60 microseconds and FlashInfer 23.21
  microseconds, a 2.01x gap. In the source cohort, current Oxide is 18.45%
  lower latency than `49290b5`.

- [Optimized long ragged-GQA4 eager performance against FlashInfer 0.6.17](h20-flashinfer-v0.6.17-ragged-prefill-dual-tile-gqa4-eager-performance-f9b95b0-20260812.json)
  binds clean source `f9b95b0`, its source-bound parent, exact matched and gate
  runner hashes, both provider orders, 400 raw latency samples, correctness,
  fixed-address Graph replay, and all four Compute Sanitizer tools. The
  parent/candidate cohort moves from 39.42 to 37.04 microseconds. In the
  separate provider-comparison cohort, Oxide measures 36.94 microseconds and
  FlashInfer 21.93 microseconds, a 1.68x gap.

- [Native SM90a M=1 GEMV performance stop against Mistral.rs custom GEMV and cuBLASLt](h20-sm90a-m1-gemv-stop-ac2bd5a-20260812.json)
  binds the five-shape fixture, exact Oxide and Mistral.rs source commits,
  runner hashes, bit-exact output digests, and both process orders. Only one
  shape met the required 10% margin against both baselines, so the frozen
  native algorithm remains experimental and cuBLASLt remains selected. The
  record retains per-order summaries but not raw samples; it supports the
  conservative stop decision, not a complete performance qualification.

- [First optimized long paged-GQA4 eager performance against FlashInfer 0.6.17](h20-flashinfer-v0.6.17-paged-prefill-tiled-gqa4-eager-performance-49290b5-20260812.json)
  binds clean source `49290b5`, exact paged and ragged tiled-GQA4 runners, both
  provider orders, correctness and Graph limits, and all four Compute
  Sanitizer tools. Oxide moves from 265.66 to 57.60 microseconds on the recorded
  shape, a 4.61x speedup. FlashInfer remains lower at 23.35 microseconds, a
  2.47x gap. The current paged record above supersedes this exact row.

- [Matched BF16 attention eager-provider performance against FlashInfer 0.6.17](h20-flashinfer-v0.6.17-attention-eager-performance-7f3d08e-20260812.json)
  binds the unchanged product source at merged commit `7f3d08e`, the exact
  Rust runner binary, FlashInfer 0.6.17, both provider orders, and 2,800 raw
  latency samples. All 14 measured paged-decode, ragged-prefill, and
  paged-prefill shapes satisfy the five-percent order-stability limit. Oxide
  has lower combined median latency in eight shapes and FlashInfer in six.

The record is an eager-provider microbenchmark on one recorded H20. It does
not establish an overall library winner, isolated-kernel or Graph performance,
native GEMV promotion, model speed, serving throughput, or behavior on another
GPU. Its largest measured gap was long-context GQA4 paged prefill at 11.51x.
The optimized paged and ragged records above supersede only their exact rows.
Their remaining gaps stay explicit optimization targets.

## Historical R1 phase 1 precursor

- [H20 phase 1 correctness and fixed-address Graph](h20-r1-graph-correctness-8927ed7-20260812.json)
  binds clean source `8927ed7`, strict CUDA Clippy, seven lab library tests,
  and four of nine permanent H20 runners. The runners cover BF16 cuBLASLt GEMM,
  ragged prefill, paged prefill, RoPE, and fused paged append. They report 48
  passing case lines and six named Graph cases.

This record remains the immutable partial precursor to phase 2. It excludes
Compute Sanitizer, performance, the other five permanent runners, other GPUs,
and engine or serving execution.

## Runtime, normalization, and GEMM

- [F32 RMSNorm correctness](h20-rms-norm-f32-correctness-20260802.json)
  covers four shapes and exact-buffer rejection on a non-default stream.
- [F32 RMSNorm command scope](h20-rms-norm-f32-command-scope-20260802.json)
  covers checked bindings, queue reuse, chained commands, and partial-scope
  rejection.
- [Low-precision RMSNorm correctness](h20-rms-norm-low-precision-20260802.json)
  covers FP16 and BF16 scalar and packed paths.
- [BF16 cuBLASLt correctness](h20-bf16-cublaslt-correctness-20260802.json)
  covers one fixed contiguous GEMM plan and command reuse.
- [Owned bindings and fixed-address Graph correctness](h20-owned-bindings-cuda-graph-correctness-20260803.json)
  covers the RMSNorm-to-GEMM chain.
- [Historical pre-rename SM90a M=1 GEMV correctness and fixed-address Graph](h20-bf16-pre-rename-sm90a-simt-gemv-correctness-20260811.json)
  covers five census shapes, typed pre-submit rejection, plan reuse, a
  three-command poison-and-write Graph recipe, two observable outputs, and two
  replays on one H20. It does not qualify performance, Compute Sanitizer, SASS,
  or engine integration.
- [Shared command-resolution regression](h20-shared-command-regression-20260803.json)
  covers the source projection recorded on 2026-08-03.

These records contain no general external-allocation, engine, or serving
qualification.

## Single decode

- [Direct BF16 single-decode correctness](h20-bf16-single-decode-correctness-20260803.json)
  covers NHD D128 MHA, MQA, and GQA.
- [Split-K correctness](h20-bf16-single-decode-split-k-correctness-20260805.json)
  covers MQA and GQA with explicit partitions and caller-owned workspace.
- [Parallel-merge correctness](h20-bf16-single-decode-parallel-merge-correctness-20260805.json)
  covers the eight-warp block-local merge.
- [Parallel-merge CUPTI activity](h20-bf16-single-decode-parallel-merge-profiling-20260805.json)
  records isolated partial and merge durations. It contains no hardware
  counters.
- Historical [pre-split-K matched eager baseline](h20-flashinfer-v0.6.16.post1-eager-performance-20260805.json),
  [split-K matched eager](h20-flashinfer-v0.6.16.post1-split-k-eager-performance-20260805.json),
  and [parallel-merge matched eager](h20-flashinfer-v0.6.16.post1-parallel-merge-eager-performance-20260805.json)
  preserve the optimization sequence and both provider orders.

The records contain no single-decode CUDA Graph, engine, or serving result.

The current source has a simulated-engine H20 gate for direct single decode.
It uses external regions and an event-bridged stream without adapter copies.
The bridge returns its authority token after it enqueues the post-event wait.
Checked bindings remain opaque.
No reviewed JSON record or real model-runner result exists.

## Paged batch decode

- [Direct correctness](h20-bf16-paged-batch-decode-correctness-20260806.json)
  covers NHD D128, page size 16, MHA, MQA, GQA, mixed lengths, page order,
  read-only page reuse, and invalid-page guards.
- [Token-parallel correctness](h20-bf16-paged-batch-decode-token-parallel-correctness-20260806.json)
  covers direct MHA and eight-warp MQA and GQA.
- Historical [direct matched eager baseline](h20-flashinfer-v0.6.16.post1-paged-batch-decode-eager-performance-20260806.json)
  and [token-parallel matched eager](h20-flashinfer-v0.6.16.post1-paged-token-parallel-eager-performance-20260806.json)
  retain raw samples and both provider orders.

The batch-4 GQA provider ranking remains excluded because the recorded
FlashInfer order delta exceeds the acceptance limit. The records contain no
Graph, engine, or serving result.

Current source adds typed device metadata errors for paged decode. The records
in this section predate its validator, status readback, and HND layout.

## Ragged causal prefill

- [Direct correctness](h20-bf16-ragged-prefill-correctness-20260806.json)
  covers the first direct MHA, MQA, and GQA fixtures.
- [Eight-warp correctness](h20-bf16-ragged-prefill-token-parallel-correctness-20260806.json)
  and [sixteen-warp correctness](h20-bf16-ragged-prefill-dual-token-parallel-correctness-20260806.json)
  preserve the token-parallel stages.
- [Tiled split-K correctness](h20-bf16-ragged-prefill-tiled-split-k-correctness-20260806.json)
  and [cp.async correctness](h20-bf16-ragged-prefill-cp-async-correctness-20260806.json)
  cover the admitted long GQA4 tiled path.
- Historical [direct matched eager](h20-flashinfer-v0.6.16.post1-ragged-prefill-direct-eager-performance-20260806.json),
  [eight-warp matched eager](h20-flashinfer-v0.6.16.post1-ragged-prefill-token-parallel-eager-performance-20260806.json),
  [sixteen-warp matched eager](h20-flashinfer-v0.6.16.post1-ragged-prefill-dual-token-parallel-eager-performance-20260806.json),
  [tiled matched eager](h20-flashinfer-v0.6.16.post1-ragged-prefill-tiled-split-k-eager-performance-20260806.json),
  and [cp.async matched eager](h20-flashinfer-v0.6.16.post1-ragged-prefill-cp-async-eager-performance-20260806.json)
  retain the optimization sequence.
- [Tiled fixed-address Graph correctness](h20-bf16-ragged-prefill-cuda-graph-correctness-20260806.json)
  and [matched Graph performance](h20-flashinfer-v0.6.16.post1-ragged-prefill-graph-performance-20260806.json)
  cover one long GQA4 two-kernel plan.

Graph evidence does not cover direct, eight-warp, or sixteen-warp ragged
plans. Short-MHA and mixed-MQA rankings in the latest eager record remain
excluded because their provider-order deltas exceed the acceptance limit.

## Paged causal prefill

- [Direct correctness](h20-bf16-paged-prefill-correctness-20260807.json)
  covers NHD D128, page size 16, MHA, MQA, GQA, mixed lengths, page order,
  read-only page reuse, and metadata guards.
- Historical [matched eager performance](h20-flashinfer-v0.6.16.post1-paged-prefill-eager-performance-20260807.json)
  retains 600 raw samples and both provider orders.
- [Token-parallel correctness](h20-bf16-paged-prefill-token-parallel-correctness-20260807.json)
  covers sixteen-warp long MQA and eight-warp long GQA4 at source `8478ee9`.
- Historical [long-context matched eager performance](h20-flashinfer-v0.6.16.post1-paged-prefill-long-eager-performance-20260807.json)
  retains both provider orders for those two token-parallel cases.
- [Fixed-address Graph correctness](h20-bf16-paged-prefill-cuda-graph-correctness-20260807.json)
  and [matched Graph performance](h20-flashinfer-v0.6.16.post1-paged-prefill-graph-performance-20260807.json)
  cover one direct GQA4 page-reorder fixture.

The Graph records do not cover MHA, MQA, mutable metadata, graph updates, token-parallel plans, engine execution, or serving.
The token-parallel records do not qualify the merged DeviceRegion and typed-status path.
Paged-prefill algorithm selection is now explicit.
The long-context record's FlashInfer commit is an unverified script assertion.
The installed wheel proves only its distribution version.

## Standard RoPE

- [BF16 standard RoPE correctness](h20-bf16-rope-pos-ids-correctness-20260806.json)
  covers NHD D128 NeoX split-half rotation with explicit I32 positions.
- [Matched eager performance](h20-flashinfer-v0.6.16.post1-bf16-rope-pos-ids-eager-performance-20260806.json)
  compares independent references under one BF16 error limit.

The records do not qualify other dimensions, layouts, Llama 3.1 scaling,
cached cosines and sines, or fused storage formats.

## Historical fused-append records

The following records use the earlier contract, which did not require the
current exclusive-page reference-count input:

- [One-token correctness](h20-bf16-rope-paged-kv-append-correctness-20260806.json)
- [One-token matched eager performance](h20-flashinfer-v0.6.16.post1-bf16-rope-paged-kv-append-eager-performance-20260806.json)
- [Explicit one-through-64-token correctness](h20-bf16-rope-paged-kv-append-tokens-correctness-20260806.json)
- [Explicit-token matched eager performance](h20-flashinfer-v0.6.16.post1-bf16-rope-paged-kv-append-tokens-eager-performance-20260806.json)
- [Explicit-token fixed-address Graph correctness](h20-bf16-rope-paged-kv-append-tokens-cuda-graph-correctness-20260806.json)
- [Explicit-token matched Graph performance](h20-flashinfer-v0.6.16.post1-bf16-rope-paged-kv-append-tokens-graph-performance-20260806.json)

Do not cite these files as current append correctness or performance evidence.
Their measurements describe only the recorded old-contract source projection.

## Claim levels

| Level | Required evidence |
| --- | --- |
| Host contract | Rust validation, CPU reference, edge cases, and error behavior |
| Device correctness | Declared GPU contract, independent oracle, numerical limit, and output sentinels |
| Lifecycle | Stream order, retained resources, completion, reuse, and failure settlement |
| Sanitizer | Declared Compute Sanitizer tools and commands |
| Performance | Matched inputs, streams, timed regions, provider order, and raw samples |
| Graph | Captured plan, fixed or mutable binding policy, replay behavior, and completion boundary |
| Engine | Real engine call site, provider hit count, no-copy proof, and model output |
| Serving | Workload, TTFT, TPOT, throughput, and memory |

Host validation does not prove device correctness. Device correctness does not
prove Graph correctness. Graph correctness does not prove performance.
Operator performance does not prove an engine or serving improvement.

## Record rules

Each JSON record includes source and lockfile identity, environment, operator
contract, commands, artifact hashes, accepted claims, and excluded claims.
Performance records also include raw samples and provider order.

Use `h20-<operator>-<gate>-YYYYMMDD.json` for H20 results. Do not edit a
reviewed record. A changed source, contract, toolchain, provider, or evidence
script requires a new file.
