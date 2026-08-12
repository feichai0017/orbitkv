# Experimental SM90a M=1 GEMV contract

The first implemented Oxide dense-GEMM algorithm is
`OxideSm90SimtGemvM1N16K64`. It remains experimental. This document fixes its
contract and records the completed performance stop decision.

The source contains an explicit Oxide provider and cuda-oxide kernel. It does
not select Oxide by default or fall back to cuBLASLt. `CublasLtHeuristic`
remains the selected production plan.

## Census basis

The design uses one complete, untimed H20 census from the paired Mistral.rs
Qwen2.5-1.5B path at tensor parallel size one.

| Item | Identity |
| --- | --- |
| Producer source | Mistral.rs `b0d0cbffb71d17e22e2a215a82020e2d3d4cd7b1` |
| Historical schema source | Loom `8b971b064d4246b2cd5cbc74f9902a51c720aefa` |
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
compiles and executes no Oxide or cuda-oxide kernel. It proves no latency,
throughput, code generation, HBM behavior, or Graph behavior.

## Frozen operator contract

The algorithm reuses the existing `Bf16DenseGemmSpec` semantics:
`D[M,N] = A[M,K] * W[N,K]^T`. Storage is BF16, each dot product accumulates in
F32, and the completed output rounds once to BF16.

`GemmPlanner` may create this experimental plan only when every condition
holds:

| Field | Requirement |
| --- | --- |
| Algorithm | `OxideSm90SimtGemvM1N16K64` |
| Provider | `Oxide` |
| Device | NVIDIA H20 with an `sm_90a` artifact |
| Shape | `M=1`, `N % 16 = 0`, and `K % 64 = 0` |
| Activation | BF16, row-major contiguous `[1,K]`, not transposed |
| Weight | BF16, row-major contiguous `[N,K]`, interpreted as transposed |
| Output | BF16, row-major contiguous `[1,N]` |
| Region offset | Allowed when the bound logical span is exact, contiguous, and four-byte aligned |
| Post-ops | None, including no bias or fused activation |
| Workspace | Zero required bytes |
| Aliasing | Existing `CommandScope` read and exclusive-write rules |

The planner returns a typed planning error for every unsupported condition. It
does not enqueue work, change algorithms, or switch to cuBLASLt.

The first H20 gate must cover all five census shapes. A later contract change
requires a new algorithm identity or an explicit revision to this freeze.

## Framework placement

cuda-oxide is the device compiler, not the provider identity. The provider is
`Oxide`, which matches the current framework vocabulary.

| Lifecycle role | Placement |
| --- | --- |
| `Spec` | Reuse `crates/oxide-infer/src/gemm/mod.rs::Bf16DenseGemmSpec` |
| `Provider` and `Algorithm` selection | `crates/oxide-infer-cuda/src/gemm/planner.rs` |
| Immutable `Plan` facade | `crates/oxide-infer-cuda/src/gemm/plan.rs` |
| Oxide provider owner | `crates/oxide-infer-cuda/src/gemm/provider/oxide/` |
| SM90 implementation | `crates/oxide-infer-cuda/src/gemm/provider/oxide/sm90/` |
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
    `-- oxide/
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

The candidate remains experimental. Each cuda-oxide build and gate must use
`sm_90a` explicitly instead of the repository default.

The pre-rename implementation passed these gates on one NVIDIA H20:

- Host admission accepts all five census shapes and rejects unsupported M, N,
  K, buffer length, and alignment. The successful device run also confirms the
  positive NVIDIA H20, compute capability 9.0, and `sm_90a` artifact path.
- Device correctness covers all five shapes against the independent reference.
  A positive exact fixture checks indexing and sentinels. A separate
  varying-scale cancellation fixture checks the declared mixed tolerance.
- Lifecycle coverage includes plan reuse, command capacity, completion, and
  retained resource leases.
- Fixed-address Graph coverage uses a three-command poison-and-write recipe,
  two guarded outputs, and two replays for every census shape.
- Plan metadata reports zero required workspace. The provider allocates no
  hidden workspace.
- Standard RoPE passed its permanent H20 runner with both `sm_90` and
  `sm_90a` after the shared device-math compiler contract changed.

Those records remain historical. The current-source R1 phase 2 record reran
the five shapes under the Oxide identity and passed correctness,
fixed-address Graph replay, and all four Compute Sanitizer tools. The R2
matched comparison then triggered the performance stop gate. SASS review and
real-engine performance were not run because they cannot promote this frozen
candidate after the operator-level stop.

### Contract evidence boundaries

| Boundary | Evidence status |
| --- | --- |
| Alternate layouts, dtypes, and post-ops | The fixed `Bf16DenseGemmSpec` cannot express them. These are type-level boundaries, not runtime rejection results |
| Device, compute capability, and artifact target | The provider checks them before loading. The permanent gate covers only the successful H20 `sm_90a` path |

### Operator performance

Performance qualification uses both existing baselines:

- Mistral.rs custom CUDA GEMV on the same engine tensor contract.
- Oxide Infer `CublasLt` with `CublasLtHeuristic` on the same tensor bits.

Each shape needs raw H20 samples against each baseline in both provider orders.
The ranking is invalid when either provider's order medians differ by more
than 5%.

For each shape, the candidate's combined median must be at least 10% lower
than each baseline's combined median. Every timed interval must include all
provider-private device work required by the declared operator boundary.

The 2026-08-12 paired runs used 200 warmups, 100 launches per CUDA-event
sample, 50 samples per order, and both process orders. Positive percentages
below mean lower Oxide latency. All provider-order median drift was at most
2.66%, below the 5% limit.

| M | N | K | Versus Mistral.rs | Versus cuBLASLt | Both gates |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1,536 | 1,536 | +20.49% | +20.17% | Pass |
| 1 | 256 | 1,536 | -30.39% | +24.38% | Fail |
| 1 | 17,920 | 1,536 | +63.99% | -13.43% | Fail |
| 1 | 1,536 | 8,960 | -45.11% | -47.76% | Fail |
| 1 | 151,936 | 1,536 | +69.47% | -4.98% | Fail |

Only one of five declared shapes beats both baselines by the required margin.
The [machine-readable stop record](../results/h20-sm90a-m1-gemv-stop-ac2bd5a-20260812.json)
retains per-order summaries, fixture digests, source identities, and artifact
hashes. Its raw samples were reviewed but not archived, so it supports the
conservative stop decision rather than a full performance qualification.

### Engine performance

The paired Mistral engine gate was not run. The operator gate already stopped
this exact algorithm, so engine routing, TPOT, throughput, and peak-memory work
would not change its promotion result. A future algorithm identity must define
its own engine gate after it passes operator performance.

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

The 2026-08-12 comparison triggered the performance stop: four of five shapes
missed the required margin against at least one baseline. Do not add
shape-aware production routing for this candidate. A future M=1 design or a
larger-M experiment needs a new algorithm identity and evidence matrix.
