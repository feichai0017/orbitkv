# Experimental SM90 M=1 GEMV contract

The first planned Loom dense-GEMM algorithm is
`LoomSm90SimtGemvM1N16K64`. This document freezes its experimental contract
before kernel work starts.

The current source contains no Loom GEMM provider or kernel. This freeze
controls the first implementation and its evidence. It does not admit device
support or a performance claim.

## Census basis

The design uses one complete, untimed H20 census from the paired Mistral.rs
Qwen2.5-1.5B path at tensor parallel size one.

| Item | Identity |
| --- | --- |
| Producer source | Mistral.rs `b0d0cbffb71d17e22e2a215a82020e2d3d4cd7b1` |
| Schema source | Loom `8b971b064d4246b2cd5cbc74f9902a51c720aefa` |
| Raw record SHA-256 | `bb063790374493eda03a1e04295a4e5ac831d8d17b329a0ac32f977b68f1b3f7` |
| Summary SHA-256 | `c0630b3388b830d2e993c41a86539a1b258951c4c67b24d6f4b1229080a245f3` |
| Hardware | NVIDIA H20, compute capability 9.0, `sm_90a` build target |
| Workload | 42 prompt tokens, eight completion tokens, one prefill forward, seven decode forwards |

Mistral.rs evidence commit `fd88a5b04fe1c7182c436ee351aa29d117b11a3e`
contains the [validation record](https://github.com/feichai0017/mistral.rs/blob/fd88a5b04fe1c7182c436ee351aa29d117b11a3e/mistralrs/examples/advanced/loom_gemm_census/h20-gemm-shape-census-b0d0cbff-20260811.json),
[raw record](https://github.com/feichai0017/mistral.rs/blob/fd88a5b04fe1c7182c436ee351aa29d117b11a3e/mistralrs/examples/advanced/loom_gemm_census/results/h20-gemm-shape-census-b0d0cbff-20260811.raw.jsonl),
and [deterministic summary](https://github.com/feichai0017/mistral.rs/blob/fd88a5b04fe1c7182c436ee351aa29d117b11a3e/mistralrs/examples/advanced/loom_gemm_census/results/h20-gemm-shape-census-b0d0cbff-20260811.summary.json).

The summary contains ten buckets and 1,352 dense-linear host calls. Prefill
accounts for 169 calls, and decode accounts for 1,183 calls.

Calls with `M=1` account for 1,184 calls, or 87.574% of all calls. They account
for 22,076,719,104 FLOPs, or 16.708% of recorded FLOPs.

All 1,184 `M=1` calls used Mistral.rs custom CUDA GEMV. The remaining 168
`M=42` calls used Candle CUDA flattened matmul.

| M | N | K | Host calls | Observed phase |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1,536 | 1,536 | 392 | Decode |
| 1 | 256 | 1,536 | 392 | Decode |
| 1 | 17,920 | 1,536 | 196 | Decode |
| 1 | 1,536 | 8,960 | 196 | Decode |
| 1 | 151,936 | 1,536 | 8 | Seven decode and one prefill |

The five logical shapes occupy six census buckets. The `lm_head` call has
separate prefill and decode activation views.

The census records successful host dispatch before backend selection. It
compiles and executes no Loom or cuda-oxide kernel. It proves no latency,
throughput, code generation, HBM behavior, or Graph behavior.

## Frozen operator contract

The algorithm reuses the existing `Bf16DenseGemmSpec` semantics:
`D[M,N] = A[M,K] * W[N,K]^T`. Storage is BF16, each dot product accumulates in
F32, and the completed output rounds once to BF16.

`GemmPlanner` may create this experimental plan only when every condition
holds:

| Field | Requirement |
| --- | --- |
| Algorithm | `LoomSm90SimtGemvM1N16K64` |
| Provider | `Loom` |
| Device | NVIDIA H20 with an `sm_90a` artifact |
| Shape | `M=1`, `N % 16 = 0`, and `K % 64 = 0` |
| Activation | BF16, row-major contiguous `[1,K]`, not transposed |
| Weight | BF16, row-major contiguous `[N,K]`, interpreted as transposed |
| Output | BF16, row-major contiguous `[1,N]` |
| Region offset | Allowed when the bound logical span remains exact and contiguous |
| Post-ops | None, including no bias or fused activation |
| Workspace | Zero required bytes |
| Aliasing | Existing `CommandScope` read and exclusive-write rules |

The planner returns a typed planning error for every unsupported condition. It
does not enqueue work, change algorithms, or switch to cuBLASLt.

The first H20 gate must cover all five census shapes. A later contract change
requires a new algorithm identity or an explicit revision to this freeze.

## Framework placement

cuda-oxide is the device compiler, not the provider identity. The provider is
`Loom`, which matches the existing framework vocabulary.

| Lifecycle role | Placement |
| --- | --- |
| `Spec` | Reuse `crates/loom-infer/src/gemm/mod.rs::Bf16DenseGemmSpec` |
| `Provider` and `Algorithm` selection | Extend `crates/loom-infer-cuda/src/gemm/planner.rs` |
| Immutable `Plan` facade | Extend `crates/loom-infer-cuda/src/gemm/plan.rs` |
| Loom provider owner | Add `crates/loom-infer-cuda/src/gemm/provider/loom/` |
| SM90 implementation | Add `crates/loom-infer-cuda/src/gemm/provider/loom/sm90/` |
| `Operands`, `CommandScope`, and `Completion` | Reuse the current dense-GEMM path |

The source tree keeps the current singular `provider` directory:

```text
gemm/
|-- mod.rs
|-- planner.rs
|-- plan.rs
`-- provider/
    |-- mod.rs
    |-- cublaslt.rs
    `-- loom/
        |-- mod.rs
        `-- sm90/
            `-- mod.rs
```

The implementation must not add a second public GEMM execution path. It must
also keep planning and fallback decisions outside enqueue.

## Algorithm choice

The first experiment uses SIMT work partitioning for the measured `M=1`
contract. A 64-row WGMMA tile would leave 63 rows unused or compute discarded
work on every recorded shape.

The pinned cuda-oxide revision
`868f8ec4ef900bae7e67e7f9508b2da66eee5472` also has a limited WGMMA lowering
surface. WGMMA stays outside this algorithm. A measured larger-M workload can
justify a separate WGMMA algorithm later.

This choice predicts no speedup. Operator and engine measurements decide
whether the SIMT candidate advances.

## Qualification gates

The candidate remains experimental until one clean source and artifact pass
all applicable H20 gates. Each cuda-oxide build and gate must set
`CUDA_ARCH=sm_90a` explicitly instead of using the repository default.

- Host tests accept all five shapes and reject every condition outside the
  frozen contract.
- Device correctness covers all five shapes against the independent reference.
  It also checks output sentinels and the declared numerical limit.
- Lifecycle tests cover plan reuse, command capacity, ordinary stream order,
  completion settlement, and retained resource leases.
- Compute Sanitizer runs memcheck with leak checking, racecheck, synccheck, and
  initcheck over the permanent runner.
- SASS and compiler reports show no register spill or local-memory traffic for
  the admitted artifact.
- Fixed-address Graph tests cover capture, poisoned output, replay, completion,
  owner drop, and lease retention for every admitted shape.
- Plan metadata reports zero required workspace, and the provider allocates no
  hidden storage.
- Negative gates reject unsupported shapes, layouts, dtypes, post-ops, and
  device targets before submission.

### Operator performance

Performance qualification uses both existing baselines:

- Mistral.rs custom CUDA GEMV on the same engine tensor contract.
- Loom `CublasLt` with `CublasLtHeuristic` on the same tensor bits.

Each shape needs raw H20 samples against each baseline in both provider orders.
The ranking is invalid when either provider's order medians differ by more
than 5%.

For each shape, the candidate's combined median must be at least 10% lower
than each baseline's combined median. Every timed interval must include all
provider-private device work required by the declared operator boundary.

### Engine performance

The paired Mistral engine gate holds model, requests, scheduler, and output
checks constant. Define the noise method before execution, then require lower
TPOT and higher throughput beyond that bound. Record TTFT, peak memory,
provider hits, and non-Loom dense-linear routes.

## Stop conditions

Any condition in this list stops promotion:

- a host, numerical, lifecycle, sanitizer, or negative gate fails.
- memcheck or a sentinel detects an HBM overread, overwrite, or use after free.
- the admitted artifact spills registers or uses unexpected local memory.
- fixed-address Graph capture, replay, owner drop, or lease retention fails.
- either baseline comparison lacks two stable provider orders.
- any declared shape misses the 10% median threshold against either baseline.
- TPOT or throughput does not improve beyond measured real-model noise.

After a hard safety failure, the artifact cannot remain selectable. A corrected
artifact requires the full gate matrix again. After a performance failure, the
algorithm remains experimental and `CublasLt` remains the selected plan.
