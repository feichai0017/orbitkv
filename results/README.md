# Validation Records

The [Capability Matrix](../docs/capability-matrix.md) is the normative support
boundary. This directory is append-only evidence: every record qualifies only
the exact source, ABI, engine, hardware, commands, and outputs bound by its
manifest.

The live tree is the single native-session product at wire 14: 48 exported
typed symbols and 78 frozen ctypes layouts. The raw manager ABI, neutral Python
runtime, reference adapter, structured route, and second-engine surface are no
longer active products. The current source contract admits five narrow
SGLang-backed profiles; their host, real-device, benefit, and production claims
remain independently gated. Manager and fixed-state pool handles are separate,
so only fail-stop containment—not cross-handle atomicity—is provided. All
earlier-wire, sealed, or dirty records below qualify only their own source
closures and do not transfer to the live wire.
OrbitKV is an attention-state compiler plus transactional ownership runtime;
none of these records makes it a full SGLang replacement or a mature L5
production system.

## Native bucketed decoder diagnostic

| Record | Scope |
| --- | --- |
| `bucketed-decoder-correctness-20260904` | Exact clean `903f8ce` parent source and pinned `beb24ac` Luminal fork; released dense decoder checkpoint and one H20; one real two-bucket search followed by prefill and repeated decode in one runtime/KV arena, plus persistent-input relocation correctness; no matched performance, memory, capacity, outer CUDA Graph, or production claim |

[Bucketed decoder correctness diagnostic](bucketed-decoder-correctness-20260904/README.md)

## Current-wire Full+Sliding diagnostic

| Record | Scope |
| --- | --- |
| `h20-sglang-wire14-full-sliding-diagnostic-20260902` | Current wire 14, pinned SGLang v0.5.17, released GPT-OSS 20B, one H20, three roomy stock/manager epochs and nine paired samples. Paired correctness and native current-stream/SWA lifecycle gates passed. Throughput gate failed: paired median regression `3.9195%`, bootstrap upper regression `5.2764%`. A subsequent exact-floor correction leaves the archive unsealed and capacity unqualified. |

Every manager epoch directly observed 72 Sliding retirement certificates, 72
reclaimed pages, 12 wraps, 88 page-generation reuse events, a current-forward-
stream completion frontier, zero fail-stops, and complete final drain. This is
the first real current-wire execution of the final native-session product path,
but it is deliberately indexed as a diagnostic: it establishes no speedup,
capacity advantage, memory saving, production readiness, or sealed current-HEAD
L4 result.

[Current-wire Full+Sliding diagnostic](h20-sglang-wire14-full-sliding-diagnostic-20260902/README.md)

## Current sealed and qualified engine records

| Record | Scope |
| --- | --- |
| `h20-sglang-v0517-abi8-full-hybrid-20260823` | Sealed exact `6f62a23` ABI8 on the official SGLang v0.5.17 base, with the manager-side canonical loader patch bound by the record, and one H20; Scoped L4 correctness for Qwen Full and GPT-OSS Full+SWA Prefix B1/B4; `performance_go=false` |
| `h20-sglang-v0517-abi8-token-relocation-20260825` | Sealed exact clean `7e029310` ABI8 on the official SGLang v0.5.17 base plus its manifest-bound canonical loader patch and one observed H20; scoped Qwen2.5-0.5B request-private Full token-relocation correctness and lifecycle at B1/B4; `hardware_attested=false`, `performance_go=false` |

The Prefix record's exact source commit is
`6f62a23b9abaa9bf12e9b060389259fa9185e70f`. The qualified profile is page16
BF16 NHD, eager FA3, and TP/PP/DP/DCP = 1. All 12 manager/stock pairs pass
across three epochs; manager and stock each contain 126 measured request traces
and 4,158 output tokens. Every manager case records Prefix publication, warm
hits and attach, eviction, and a clean final drain. Every Hybrid case has
positive SWA retirement-certificate and reclaimed-page counters.

Mean manager-over-stock time is +7.3678% for Full B1, +11.8684% for Full B4,
+3.9865% for Full+SWA B1, and +4.4476% for Full+SWA B4. These measurements are
diagnostic and `performance_go=false`; they are not a speedup claim. The record
makes no memory-saving claim and does not establish a complete SGLang
replacement or production readiness. Token relocation, MLA, fixed state,
overlap, CUDA Graphs, speculation, distributed execution, and performance
qualification are explicitly excluded.

The token-relocation record's exact source commit is
`7e02931036123c0f830bcca7130a43543c9e6eb1`. Its manifest records
`source_clean=true`, `preflight_bound=true`, `sealed=true`, and
`qualified=true`. The profile is Qwen2.5-0.5B request-private Full attention,
the official SGLang v0.5.17 base plus its manifest-bound canonical loader
patch, page16 BF16 NHD eager FlashInfer, and one observed H20. Four
order-balanced epochs produce 16 process records and eight
Naive/Relocate B1/B4 pairs. All eight pairs have exact output tokens, the
required relocation lifecycle, complete final drain, and zero
failure/quarantine/fail-stop counters. The bundled component suite passes 7/7
H20 cases.

| Case | Relocate throughput | Mean latency | Median latency | p95 latency |
| --- | ---: | ---: | ---: | ---: |
| B1 | -1.3467% | +1.3651% | +2.0033% | +0.5141% |
| B4 | +2.8096% | -2.7328% | +0.9810% | -19.8635% |

This second seal qualifies only scoped Full token-relocation correctness and
lifecycle. It records `hardware_attested=false` and `performance_go=false`,
and the mixed measurements do not establish a general speedup. It makes no
capacity, memory-saving, production-readiness, full-replacement, Prefix,
Hybrid/SWA, MLA, async-overlap, Graph, speculation, distributed, or multi-GPU
claim. The two sealed records do not inherit or combine scopes.

[Sealed Qwen2.5-0.5B token-relocation qualification](h20-sglang-v0517-abi8-token-relocation-20260825/README.md)

## Qwen3.5 fixed-state pair-verification record

| Record | Scope |
| --- | --- |
| `h20-sglang-v0517-abi8-qwen35-fixed-state-pair-verification-20260823` | Exact `7385ee5` ABI8 on official SGLang v0.5.17 and one recorded H20; Qwen3.5-0.8B request-private Full+GDN fixed-state B1/B4 pair verification only; not qualified, not independently hardware-attested, and `performance_go=false` |

Across three epochs, all six fresh-prompt stock/manager pairs match
token-for-token and every manager record reports fixed-state lifecycle events
and a complete final drain. The profile is page16 BF16 NHD, eager FA3 for Full
attention, Triton GDN execution, FP32 recurrent state, Radix disabled, and
TP/PP/DP/DCP = 1. The aggregate manager overhead is +0.5074% for B1 and
+3.7394% for B4; after excluding each B4 process's first iteration, the
diagnostic overhead is +7.0451%. These measurements are not a speedup claim.

This archive does not qualify performance, Prefix/Radix sharing, a fixed-state
copy or replacement trigger, CUDA Graphs, overlap scheduling, speculation,
distributed or multi-GPU execution, other Qwen3.5 profiles, or production
readiness. Its `hardware_attested=false` boundary remains authoritative even
though all raw runtime snapshots consistently record one NVIDIA H20.

## Qwen3.8-27B-FP8 diagnostic pair run

This run is deliberately indexed as a diagnostic, not as a qualified result.
It uses official `Qwen/Qwen3.8-27B-FP8` revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`: 66/66 indexed weight shards,
30,866,866,928 bytes, and 1,606 indexed tensors. It pins official SGLang
`v0.5.17` revision `29481685462732237d80d86076d6563e1f658102` and records
NVIDIA H20 UUID `GPU-3a35e57b-fc54-5620-56ee-deaf5a9c40d3`. The execution
contract is explicit `fp8_gemm_runner_backend=triton`, eager single GPU, and
page16 BF16 NHD KV storage. The shared runtime policy is a structural GDN plus
convolution capability check; this evidence pin is specific to Qwen3.8.

There are four fully order-balanced epochs: 1 and 3 run manager then stock,
while 2 and 4 run stock then manager. Each epoch contains one B1 and one B4
pair, yielding eight pairs total; every process runs five iterations. All eight
pairs pass the verifier and match tokens exactly; every
manager final census is zero and all failure/fail-stop counts are zero. Hot
statistics exclude each process's iteration 0, leaving 16 samples per mode and
batch size.

| Case | Stock mean / median / p95 | Manager mean / median / p95 | Latency / throughput delta | Per-epoch latency delta |
| --- | --- | --- | --- | --- |
| B1 | 2.6296723178 / 2.5374011379 / 3.0721712420 s | 2.8031814888 / 2.6451683380 / 3.3369778013 s | +6.5981% / -6.1897% | +2.7822%, +0.7129%, +26.3407%, -1.9835% |
| B4 | 2.9576119229 / 2.9602836296 / 3.0001941137 s | 3.0395950049 / 3.0345056280 / 3.0800264925 s | +2.7719% / -2.6972% | +1.8441%, +2.8850%, +5.0070%, +1.3885% |

The B1 epochs contain substantial jitter and a +26.3407% outlier; the pooled
result supports no positive performance claim. B4 is also slower. Stock and
manager have the same configured tensor-arena capacity and report the same
KV-cache reservation, so the observed reservation difference is **0%**. This
is not a qualified end-to-end memory-saving result. The source was dirty, no
qualification preflight was run, and the
result is `hardware_attested=false` and `qualified=false`. A separate default-
auto DeepGEMM attempt loaded all 66/66 shards, but E2E remained incomplete
because lengthy precompilation was externally terminated.

[Qwen3.8-27B-FP8 diagnostic archive](h20-sglang-v0517-abi8-qwen38-fp8-diagnostic-20260824/README.md)

## Historical superseded Qwen2.5-0.5B token-relocation diagnostic

| Record | Scope |
| --- | --- |
| `h20-sglang-v0517-token-relocation-diagnostic-20260825` | Historical/superseded dirty-source diagnostic retained for auditability; official SGLang v0.5.17, Qwen2.5-0.5B Full/FlashInfer, page16 BF16 NHD eager B1/B4 Naive-versus-byte-exact-Relocate; recorded H20 observation only; unsealed, independently unattested, unqualified, and `performance_go=false` |

All 8/8 pairs match output tokens exactly. Each process runs five iterations,
each iteration reaches two reclamation rounds, relocation/reclaimed-page
counters match the contract, the final manager census drains fully, and all
failure, quarantine, and fail-stop counters are zero. Epoch order alternates,
and excluding iteration 0 leaves 16 hot samples per mode and batch size.

| Case | Relocate throughput delta | Relocate mean latency delta | Relocate p95 latency delta |
| --- | ---: | ---: | ---: |
| B1 | +7.629% | -7.088% | -22.844% |
| B4 | -1.232% | +1.247% | +2.880% |

B1 has substantial epoch-to-epoch jitter, while B4 is slightly slower. These
numbers are descriptive and remain `performance_go=false`. The archive is
`diagnostic_only`, `sealed=false`, `source_dirty=true`,
`hardware_attested=false`, and `qualified=false`. Reclaimed pages prove that
relocation occurred for the workload, not that configured capacity or
end-to-end memory use fell. A relocate-only FA3 E2E smoke passed, but the
sparse Naive oracle is invalid under FA3 and now fails closed; the paired
comparison therefore uses FlashInfer for both modes.

[Qwen2.5-0.5B relocation diagnostic archive](h20-sglang-v0517-token-relocation-diagnostic-20260825/README.md)

## Historical frozen ABI5-v5 record

| Record | Historical scope |
| --- | --- |
| `h20-sglang-v0517-abi5-v5-grouped-release-20260821` | Frozen `9233c06d…` ABI5-v5 on official SGLang v0.5.17 and one H20; scoped L4 correctness for Qwen Full and GPT-OSS Full+SWA B1/B4; grouped release; same-cap memory reduction 0%; `performance_go=false` |

The ABI5-v5 record contains four PRIMARY manager records and their stock
references. All eight JSON records pass independent verification, all 84
request traces match stock token-for-token, and every Full/SWA arena drains.
Each B4 manager record releases 20 requests using five release/recycle
transactions.

The qualified profile is page16 BF16 NHD, eager ChunkCache, TP/PP/DP/DCP = 1,
Qwen2.5-7B Full/FlashInfer, and GPT-OSS-20B ordered Full+SWA128/FA3 with
SGLang's built-in Triton MoE. Radix/Prefix, overlap, Graph, speculation,
disaggregation, streaming, hierarchical cache, and remote cache were disabled.

The compared manager and stock processes reserve equal configured KV tensor
capacity, so the observed reservation difference is **0%**; this is not a
qualified end-to-end memory-saving result. The single epoch reports
B4 steady manager overhead of +4.1932% for Qwen and -5.2048% for GPT-OSS;
Qwen B1 is +5.0009%. With no repeated-epoch statistics,
`performance_go=false`; the GPT result is not a general speedup claim.

Nothing in this historical record qualifies ABI8 Prefix/COW, Python, SGLang
integration, relocation, fixed-state model execution, Graph, or distributed
execution. The separate sealed ABI8 record above is the sole source for its
narrower current claim.

## Earlier records

| Record | Historical boundary |
| --- | --- |
| `h20-sglang-v0517-abi5-full-hybrid-20260821` | Frozen ABI5-v4 official-release Full/Hybrid epoch |
| `h20-sglang-v0517-full-hybrid-20260821` | Frozen ABI4 official-release Full/Hybrid epoch |
| `h20-canonical-manager-20260820` | ABI3/development-pin Mistral pure-SWA lifecycle and memory accounting |
| `h20-rust-owned-pages-20260820` | Rust-selected physical SWA pages through SGLang allocation kernels |
| `h20-dense-sglang-20260819` | Pure-SWA 128-token Dense page binding through SGLang |
| `dense-runtime-20260819` | Fixed-capacity ownership reference and differential benchmark |
| `h20-cuda-event-overlap-20260819` | Historical request-scoped CUDA-event frontier experiment |
| `h20-radix-prefix-20260819` | Historical component-aware Full+SWA Prefix experiment |
| `h20-transactional-binding-20260819` | Historical prepare/load/commit hydration transaction |
| `h20-runtime-state-plan-20260819` | Historical shared runtime-artifact experiment |
| `h20-hybrid-capsule-20260818` | Historical GPT-OSS Full+SWA continuation experiment |
| `h20-live-tail-capsule-20260818` | Historical pure-SWA live-tail continuation experiment |
| `h20-capsule-export-20260818` | Historical checkpoint KV export/host restore experiment |
| `applicability-h20-20260817` | Historical Qwen Full, Mistral bounded, and GPT-OSS Hybrid geometry |
| `h20-gpt-oss-20b-real-20260817` | Historical real-checkpoint systems experiment |
| `lifetime-normalization-20260817` | Per-head window and retention-amplification analysis |
| `chunked-local-20260817` | Same-chunk to resettable-arena compiler proof |
| `sink-sliding-20260817` | Sink plus local lifetime partitioning |
| `retention-ir-20260817` | Declarative retention IR and legacy equivalence |
| `h20-generation-vmm-20260817` | Historical generation-aware CUDA VMM lifecycle |
| `owner-ffi-20260817` | Historical in-process Owner ABI |

Other directories are earlier calibration records retained for auditability.

## Interpretation rules

- Historical manifests and raw outputs are never rewritten to claim a later
  source tree. A provenance correction is a separate append-only amendment.
- A breaking ABI cannot inherit a prior ABI's L2 or L4 result.
- A host microbenchmark is not model throughput, and a single process epoch is
  not a performance GO.
- Smaller admitted capacity is not KV compression. Same-capacity comparisons
  must account for actual tensor-arena bytes, padding, and temporary headroom.
- Token compaction means exact K/V relocation. It is not quantization or a
  numerical compression claim.
- Compiler/reference proofs are not GPU or engine qualification.
- An isolated VMM record does not prove that VMM backs live SGLang KV tensors.

`h20-canonical-manager-20260820/provenance-amendment.json` records that its old
workspace snapshot was not originally sealed; its original manifest remains
unchanged. The later ABI5-v5 record carries the exact sealed source closure
used by its run.
