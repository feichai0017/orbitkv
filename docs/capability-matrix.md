# Capability Matrix

This is the normative boundary for the live source tree. A historical result
qualifies only the source closure named by its manifest; breaking ABI8 work
cannot inherit ABI5 hardware evidence.
OrbitKV's system boundary is an attention-state compiler plus a transactional
ownership runtime. It does not replace SGLang's model execution stack, and the
live project is not a mature L5 production system.
The sealed Prefix and token-relocation records below are independent scoped
qualifications; neither widens the other's model, feature, hardware-attestation,
performance, or production boundary.

## Levels

| Level | Meaning |
| --- | --- |
| L1 Compiler | Semantics parse and compile into checked programs. |
| L2 Host/ABI | A core, wire, or adapter surface passes host correctness and fault gates. |
| L3 GPU Primitive | An isolated primitive passes exact-source hardware tests. |
| L4 Engine E2E | A pinned engine and released checkpoint pass exact-source end-to-end gates. |
| L5 Production | Pressure, cancellation, feature combinations, and a version matrix are qualified. |

## Live ABI8 source

| Capability | Level | Exact boundary | Evidence |
| --- | --- | --- | --- |
| Retention and plan compiler | L1 | Checked Full, sliding, and retained IR/compiler relations | `src/retention.rs`, `src/plan/`, compiler tests |
| Heterogeneous attention-state compiler | L1 GO | Separates token KV, MLA latent+RoPE components, recurrent Mamba/GDN/KDA/linear state, and convolution state; token-only manager projection preserves explicit storage/component geometry and excludes fixed-width state | `src/attention_state.rs`, `compile-state-plan`, `compile-state-manager-plan`, mixed-state example/tests |
| Qwen `qwen3_5` dense-config-family HF frontend | L1 GO | Exact official Qwen3.5-0.8B and non-FP8 Qwen3.8-27B config fixtures compile into Full token KV plus request-private GDN/convolution state; malformed/defaulted or schedule-inconsistent configs fail closed | `src/hf_config.rs`, `fixtures/qwen3.5-0.8b/config.json`, `fixtures/qwen3.8-27b/config.json`, CLI tests |
| Strict token-only HF frontend | L1 | For supported profiles outside that dense config family, emits the sole token-addressable `KvPlanInput`; unknown semantics fail closed | `src/hf_config.rs`, `tests/canonical_cli.rs` |
| Identity and arena ownership | L2 GO | Generation-checked request, snapshot, page, step, submission, Prefix, reclamation, and relocation leases; independent class pools | `src/kv_manager/identity.rs`, `arena.rs`, host tests |
| Persistent snapshots | L2 GO | Immutable class roots, expected-head CAS, stale-head rejection, incremental path-copy, no hot full-root materialization | `src/kv_manager/persistent_snapshot.rs`, host/property tests |
| Append transactions | L2 GO | Failure-atomic acquire/fork/prepare/submit/complete, compact write/copy intents, abort and quarantine | `src/kv_manager/append_transaction.rs`, fault tests |
| Prefix, request fork, and joint COW core | L2 GO | Page-aligned dense Prefix lookup/publish/publish-release/attach/evict/recycle; generation-safe dense/packed request fork; dense and mixed-layout joint COW, including shared packed Full partial tails | `src/kv_manager/prefix.rs`, append transactions, Prefix/COW tests |
| Page-owned reclamation | L2 GO | Request/Prefix refs, reader pins and writer state jointly gate detach, certificates, ACK, and reuse | `src/kv_manager/reclamation.rs`, lifecycle/fault tests |
| Token virtualization and relocation core | L2 GO | Canonical token views; semantic-death/policy-eviction evidence; ABI8 batch prepare/submit/complete primitives; repeated append-mark-full-evacuation-ACK for one private Full class; exact copy receipts; packed publication/fork/shared partial-tail COW; quarantine and precise release | `src/kv_manager/token_virtualization.rs`, `relocation_transaction.rs`, repeated host/property/fault/lifecycle and packed-COW tests |
| Recurrent/convolution checkpoint pool | L2 host | Fixed-width generation-checked initial/replace/retire/ACK lifecycle with exact copy receipts, completion gating, abort, and quarantine; no token ids or TokenMove surface | `src/state_checkpoint.rs`, host lifecycle/fault tests |
| Typed C ABI8 wire | L2 GO | Exactly 40 typed symbols: 29 canonical-manager plus 11 independent state-pool symbols; per-handle transactions, C/C++ layout checks, reserved-field, capacity, short-buffer, stale-lease, and receipt validation; no cross-handle atomicity | `crates/orbitkv-ffi/include/orbitkv.h`, FFI tests, CI symbol diff |
| ABI8 Python FFI/runtime | L2 GO | Exact-40 ctypes loader; 73 frozen layouts; bounded manager workspaces; independent state-pool client; typed retry/fail-stop and force-destroy teardown. The ABI8-preserving multi-request relocation path calls native mark/prepare/submit/complete once each, then performs one aggregate page-registry commit followed by one aggregate request-head replacement; the scalar API is a singleton batch compatibility wrapper | Python FFI/runtime success-path tests against the release library at B1/B4, including packed B4 fork/COW/append; B1/B4/B32 stale-member preflight and full-copy CUDA conformance pass, with the exact component boundary listed below |
| Engine-neutral adapter SPI and reference arena | L2 host/package | `orbitkv-runtime` defines the typed data-plane contract; `orbitkv-reference` implements external CPU/CUDA tensor arenas, exact append/relocate effects, completion evidence, mirror cleanup, and ACK-gated generation reuse | Separate wheels build and clean-install in CI; the reference is a contract oracle/reusable arena adapter, not a scheduler, model runner, allocator, attention kernel, or complete engine; SGLang has not migrated to this SPI |
| Request-private pressure telemetry | L2 host only | Opt-in event samples separate consumed capacity, resident data, request-reachable bytes, semantic-live bytes, free-space minima, high-water marks, and retention amplification | Host runtime and async-schedule tests only; no real asynchronous GPU pressure run. Fixed-state bytes are excluded, and shared Prefix/request-fork retention amplification fails closed |
| Official SGLang source contract | L2 | Official `v0.5.17` base at peeled commit `29481685462732237d80d86076d6563e1f658102`; pristine stock checkout and manager checkout with the manifest-bound canonical loader patch are checked separately | pinned-checkout tests |
| SGLang `OrbitKVPrefixCache` | L2 GO | Official cache seam; nodes contain token/digest/LRU plus opaque Prefix leases only; warm attach, lock/ref accounting, Full+SWA COW, grouped release, eviction, and hostile fault paths pass host gates | pinned `v0.5.17` contract and plugin integration tests; the exact sealed H20 subset is listed below |
| ABI8 H20 Prefix path | Scoped L4 correctness | Qwen2.5-7B Full and GPT-OSS-20B Full+SWA, B1/B4, page16 BF16 NHD eager FA3 on one H20; Prefix lifecycle and final drain qualified; Qwen3.5 and fixed state are excluded | `results/h20-sglang-v0517-abi8-full-hybrid-20260823` at exact `6f62a23` |
| SGLang token relocation | L2 host plus scoped sealed correctness/lifecycle qualification | The explicit/default-off eager path preserves one multi-request scheduler batch: all moves are flattened into one backend move and one completion event, every ReqToToken/LUT/request mirror plan is validated before the first write, and all retirements receive one batch ACK. The qualified engine scope is request-private Full only; common-victim-set Full+SWA remains host-tested, not part of this seal | Host plugin/runtime tests plus `results/h20-sglang-v0517-abi8-token-relocation-20260825` at exact clean source `7e029310…`. Completion is eager and host-blocking, not asynchronously overlapped. Packed fork/shared-tail COW is newer host/raw-ABI8/Python-FFI functionality outside the seal; packed Prefix fails closed |
| Engine-neutral CUDA opaque-byte relocation harness | H20 component conformance: 7 passed / qualification pending | B1/B4/B32; two cycles on the same requests/cursors; independent live-token/payload oracle; 257-byte coordinate-bearing records; distinct real append/copy/consumer streams; exact 3-page-to-2-page evacuation and 24 moves per request/cycle; event-ordered byte readback; no reuse before ACK, same-page/higher-generation reuse after ACK, and final drain | `integrations/sglang/tests/relocation_conformance.py`, `test_cuda_relocation_conformance.py`; all seven H20 cases pass: payload uniqueness, stale-member atomicity at B1/B4/B32, and CUDA copy/consumer conformance at B1/B4/B32. This is component evidence, not sealed L3/L4, performance, capacity, or complete-engine qualification |
| Qwen2.5-0.5B SGLang token relocation | Scoped correctness + lifecycle qualified / performance pending | Exact clean source `7e02931036123c0f830bcca7130a43543c9e6eb1`; official SGLang v0.5.17 base plus the manifest-bound canonical loader patch; request-private Full; BF16 NHD page16 eager FlashInfer; four alternating-order B1/B4 epochs; five iterations/process; 16 hot samples/mode/group after excluding iteration 0 | [sealed qualification](../results/h20-sglang-v0517-abi8-token-relocation-20260825/README.md); 16 records / 8 pairs have exact tokens, expected relocation lifecycle, full drain, and zero failures. `source_clean=true`, `preflight_bound=true`, `sealed=true`, `qualified=true`, `hardware_attested=false`, `performance_go=false`; no capacity, memory-saving, general speedup, production, or replacement claim |
| SGLang MLA relocation seam | L2 host / L4 pending | Pure BF16 Full-retention MLA, page16, eager, single GPU, FlashInfer or FA3; compiler geometry is checked against `MLATokenToKVPool` and relocation copies every layer's combined latent+RoPE row | real pinned `MLATokenToKVPool` host copy/config tests; no H20/model evidence; DSA, FP4, DCP, Hybrid Linear, Prefix performance pending |
| SGLang fixed-state seam | Scoped L2 host plus recorded-device evidence / L4 pending | Runtime admission is structural for the currently implemented exact GDN+convolution contract: request-private state maps to `slot_id + 1`, allocation/initial clear, forward completion events, and release-time wait/retire/clear/exact-ACK; it admits neither Prefix sharing nor a fixed-state copy trigger | Qwen3.5 has scoped pair verification from recorded H20 runtime snapshots; Qwen3.8 has diagnostic-only recorded-device evidence. Both are `qualified=false` and `hardware_attested=false`, and neither promotes the seam to L4 |
| Qwen3.5 SGLang pair-verification contract | Scoped pair verification from recorded H20 runtime snapshots / independently unattested / L4 pending | Both modes disable Radix and use fresh unique prompts; manager reads back exact token/state plans and descriptors, reports zero cached tokens and copies, satisfies exact prepare/clear/event/retire/ACK counts, and drains cleanly | `results/h20-sglang-v0517-abi8-qwen35-fixed-state-pair-verification-20260823`; six B1/B4 pairs across three epochs pass, but the record is `qualified=false` and `hardware_attested=false` |
| Qwen3.8-27B-FP8 diagnostic | Weight-backed recorded-device diagnostic / L4 pending | The diagnostic declares FP8 repository revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` and binds the downloaded config, index, and all shards by hash and byte count; repository provenance is not independently online-attested. The plan has 16 Full token-KV layers and 48 request-private GDN/convolution layers; four epochs for B1 and B4 yield eight verifier- and token-exact pairs | [diagnostic archive](../results/h20-sglang-v0517-abi8-qwen38-fp8-diagnostic-20260824/README.md); `diagnostic_only`, `sealed=false`, dirty source, `preflight_bound=false`, `hardware_attested=false`, `qualified=false`, `performance_go=false`, no positive performance claim |
| Stable-address CUDA VMM primitive | L2 host | Isolated reserve/map/remap/unmap backend; not the manager data plane and not SGLang tensor storage | `crates/orbitkv-cuda/` host tests |
| General SGLang replacement | Not L5 | The separately scoped Prefix and Full-relocation seals do not qualify MLA, fixed state, async pressure/overlap, Graph, speculation, distributed execution, performance, or a release matrix | this matrix |

The relocation batch boundary is collective but not rollback-capable end to end.
After a disposition mark succeeds, a later prepare, copy, submit, complete,
publication, mirror, or ACK failure or uncertain return fail-stops the runtime;
it does not restore the pre-mark disposition snapshot or an older request head.
An explicitly unobserved copy can release the relocation reservations, but it
does not undo the already committed mark. Async copy/consumer overlap remains
unimplemented.

### Strict Qwen3.5-0.8B host/runtime profile

This profile is deliberately narrower than generic Qwen3.5 or generic Hybrid
Attention support. The checked official manifest has 24 text layers:

- Full attention at layers `3, 7, 11, 15, 19, 23`, with BF16 token-addressable
  K and V components of 1,024 bytes each per token per layer; and
- GDN at the other 18 layers, with 1,048,576 bytes of FP32 recurrent state and
  36,864 bytes of BF16 convolution history per layer. The convolution width is
  four and the persistent history holds the preceding three positions.

The compiler sends only Full KV to `CanonicalKvManager`. GDN recurrent and
convolution state go to the independent generation-checked
`StateCheckpointPool` and remain request-private. In the SGLang adapter the
server runs with Radix disabled; the OrbitKV cache seam remains installed only
to preserve canonical release handling and never publishes or attaches a
Prefix entry for this profile. The production gate requires Full FA3, Triton
linear-attention execution for the general, prefill, and decode selectors,
configured and actual temporal FP32 state,
`mamba_radix_cache_strategy=no_buffer`, and `disable_radix_cache=true`. Any
drift fails before mutation.

SGLang still owns tensor allocation, attention/linear kernels, scheduling, and
model execution. OrbitKV owns the accepted plan and identities plus page/state
transactions and completion-gated reclamation. A resource becomes reusable
only after the Semantic Frontier proves it unreachable and the Execution
Frontier proves that prior GPU use has completed. The two proofs are not
interchangeable.

This remains below Qwen3.5 L4 qualification, but it is no longer host-only. A
scoped archive now records six stock/manager pairs from H20 runtime snapshots with equal
output-token totals, exact lifecycle counters, CUDA stream/event completion, and final
drain. The archive is explicitly `qualified=false`, `hardware_attested=false`,
and `performance_go=false`; the recorded H20 runtime snapshots are
independently unattested. It therefore establishes no performance gain, configured-arena or
end-to-end memory reduction, production readiness, or complete SGLang
replacement.

The runtime policy is intentionally generic at this boundary: it admits the
structural GDN/convolution capability proven by the compiled plan and live
tensor geometry. Evidence is narrower and remains pinned to the exact model,
checkpoint revision, engine revision, backend choices, and workload recorded
for each run.

### Exact ABI8 C surface

The dynamic library must export these 40 symbols and no other `orbitkv_*`
symbol:

```text
orbitkv_abi_version
orbitkv_manager_abort_relocations_batch
orbitkv_manager_abort_steps_batch
orbitkv_manager_acknowledge_reclamations_batch
orbitkv_manager_arena_identities
orbitkv_manager_arena_stats
orbitkv_manager_complete_batch
orbitkv_manager_complete_relocation_batch
orbitkv_manager_create
orbitkv_manager_destroy
orbitkv_manager_mark_token_dispositions_batch
orbitkv_manager_prefix_attach_batch
orbitkv_manager_prefix_evict_batch
orbitkv_manager_prefix_lookup_batch
orbitkv_manager_prefix_publish_batch
orbitkv_manager_prefix_publish_release_batch
orbitkv_manager_prefix_recycle_batch
orbitkv_manager_prepare_batch
orbitkv_manager_prepare_relocation_batch
orbitkv_manager_quarantine_steps_batch
orbitkv_manager_quarantine_submissions_batch
orbitkv_manager_recycle_requests_batch
orbitkv_manager_release_batch
orbitkv_manager_request_acquire_batch
orbitkv_manager_request_fork_batch
orbitkv_manager_stats
orbitkv_manager_submit_batch
orbitkv_manager_submit_relocation_batch
orbitkv_manager_token_views_batch
orbitkv_state_pool_abort_batch
orbitkv_state_pool_acknowledge_batch
orbitkv_state_pool_complete_batch
orbitkv_state_pool_create
orbitkv_state_pool_current_batch
orbitkv_state_pool_destroy
orbitkv_state_pool_identity
orbitkv_state_pool_prepare_batch
orbitkv_state_pool_retire_owners_batch
orbitkv_state_pool_stats
orbitkv_state_pool_submit_batch
```

The ABI5 scalar-shaped names `abort_steps`, `quarantine_steps`,
`quarantine_submissions`, `acknowledge_reclamations`, and `recycle_requests`
are removed. CI fails if an active source surface reintroduces them. Frozen
headers inside `results/` remain unchanged.

## Current sealed ABI8 engine evidence

Two sealed records qualify disjoint ABI8 engine scopes. Qualification does not
transfer between them.

### Prefix scope

`results/h20-sglang-v0517-abi8-full-hybrid-20260823` binds exact source commit
`6f62a23b9abaa9bf12e9b060389259fa9185e70f` (short id `6f62a23`) to:

- official SGLang `v0.5.17` base at peeled commit
  `29481685462732237d80d86076d6563e1f658102`, with the manager-side
  canonical loader patch bound by the record;
- one NVIDIA H20, page16 BF16 NHD storage, eager FA3, and TP/PP/DP/DCP = 1;
- Qwen2.5-7B Full and GPT-OSS-20B Full+SWA at B1 and B4; and
- 12 passed manager/stock pairs across three epochs, comprising 126 measured
  request traces and 4,158 output tokens in each mode.

All measured manager outputs match their stock pair. Every manager case
records Prefix publication, warm hits and attach, eviction, and a clean final
manager/arena drain. Every Hybrid manager case records positive SWA retirement
certificates and reclaimed pages. This is Scoped L4 correctness for only that
boundary.

| Profile | Mean manager over stock |
| --- | ---: |
| Full B1 | +7.3678% |
| Full B4 | +11.8684% |
| Full+SWA B1 | +3.9865% |
| Full+SWA B4 | +4.4476% |

These timings are diagnostic: `performance_go=false`. The record does not
claim a speedup and it does not qualify a complete SGLang replacement or
production readiness. The compared manager and stock runs reserve equal
configured KV tensor arenas, so the observed configured arena reservation difference is
**0%**; this is not a qualified end-to-end memory-saving result. This
seal covers only Qwen2.5 Full and GPT-OSS Full+SWA; Qwen3.5, token relocation,
MLA, fixed state, overlap, CUDA Graphs, speculation, distributed execution,
and performance qualification are explicitly excluded.

### Token-relocation scope

`results/h20-sglang-v0517-abi8-token-relocation-20260825` binds exact clean
source `7e02931036123c0f830bcca7130a43543c9e6eb1` to the official SGLang
`v0.5.17` base plus its manifest-bound canonical loader patch, Qwen2.5-0.5B
request-private Full attention, page16 BF16 NHD eager
FlashInfer, and single-GPU B1/B4 Naive-versus-Relocate comparison. Its manifest
records `source_clean=true`, `preflight_bound=true`, `sealed=true`, and
`qualified=true`. Across four order-balanced epochs, all 16 process records and
8 pairs have exact output-token equality, expected relocation lifecycle,
complete final drain, and zero failure/quarantine/fail-stop counters. The
bundled H20 component result passes 7/7 cases.

| Case | Relocate throughput delta | Mean latency delta | Median latency delta | p95 latency delta |
| --- | ---: | ---: | ---: | ---: |
| B1 | -1.3467% | +1.3651% | +2.0033% | +0.5141% |
| B4 | +2.8096% | -2.7328% | +0.9810% | -19.8635% |

The seal qualifies only scoped Full token-relocation correctness and lifecycle.
It records `hardware_attested=false` and `performance_go=false`; the mixed
timing results establish no general speedup. It does not qualify capacity or
memory savings, production readiness, a complete SGLang replacement,
Prefix/Radix sharing, Hybrid/SWA, MLA, asynchronous pressure or overlap, CUDA
Graphs, speculation, distributed execution, or multi-GPU.

## Qwen3.5 pair-verification evidence from recorded H20 runtime snapshots

`results/h20-sglang-v0517-abi8-qwen35-fixed-state-pair-verification-20260823`
records scoped verification rather than an L4 qualification seal. Its exact
boundary is:

- official `Qwen/Qwen3.5-0.8B`;
- official SGLang `v0.5.17` at peeled commit
  `29481685462732237d80d86076d6563e1f658102`;
- one recorded NVIDIA H20 runtime snapshot with UUID prefix `GPU-3a35…`, but
  without independent hardware attestation;
- page16 BF16 NHD, eager Full FA3, Triton general/prefill/decode linear
  backends, Radix disabled, fresh prompts, and `cached_tokens=0`;
- B1: one request, fixed-state capacity two, one iteration; B4: four requests,
  fixed-state capacity four, five iterations; and
- three epochs for each case, or six stock/manager pairs total.

Every pair has equal stock/manager output-token totals. Token and fixed-state
lifecycle counters, CUDA stream/event completion, and final drain pass exactly.
Per epoch, fixed-state prepare, clear, retire, and ACK counts are one in B1 and
20 in B4; fixed-state copies are zero.

| Case | Manager aggregate | Stock aggregate | Manager over stock |
| --- | ---: | ---: | ---: |
| B1 | 1.5898847853 s | 1.5818591726 s | +0.507353% |
| B4 | 0.9498731474 s | 0.9156339097 s | +3.739403% |

Both manager aggregates are slower. The archive therefore records
`performance_go=false`; the observed configured arena reservation
difference is **0%**, not a qualified end-to-end memory-saving result. It also
records `qualified=false` and `hardware_attested=false`; H20 runtime snapshots
are present, but they are independently unattested. This closes the scoped
recorded-H20 pair-verification step only. L4 qualification,
a performance benefit, complete SGLang replacement, same-owner state copy, and
broader Hybrid Attention support remain unproven.

## Qwen3.8-27B-FP8 H20 diagnostic evidence

The current Qwen3.8 run is diagnostic evidence, not an L4 qualification
record. It declares `Qwen/Qwen3.8-27B-FP8` repository revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`; its raw records bind the
downloaded config, index, and all 66 weight shards by hash and byte count
(30,866,866,928 bytes, 1,606 tensors), while repository provenance is not
independently online-attested. It also records official SGLang `v0.5.17` revision
`29481685462732237d80d86076d6563e1f658102`, and recorded NVIDIA H20 UUID
`GPU-3a35e57b-fc54-5620-56ee-deaf5a9c40d3`. Both modes use explicit
`fp8_gemm_runner_backend=triton`, eager single-GPU execution, and page16 BF16
NHD KV storage.

Four epochs use a completely balanced process order: epochs 1 and 3 run
manager then stock, and epochs 2 and 4 run stock then manager. Each epoch has
one B1 and one B4 pair, yielding eight total; every process runs five
iterations. All eight pairs pass the verifier and match output tokens
exactly; manager final census and failure/fail-stop counts are zero. Hot
statistics exclude iteration 0 from every process, giving 16 samples per mode
and batch size.

| Case | Stock mean / median / p95 | Manager mean / median / p95 | Latency delta | Throughput delta | Epoch latency deltas |
| --- | --- | --- | --- | ---: | --- |
| B1 | 2.6296723178 / 2.5374011379 / 3.0721712420 s | 2.8031814888 / 2.6451683380 / 3.3369778013 s | +6.5981% | -6.1897% | +2.7822%, +0.7129%, +26.3407%, -1.9835% |
| B4 | 2.9576119229 / 2.9602836296 / 3.0001941137 s | 3.0395950049 / 3.0345056280 / 3.0800264925 s | +2.7719% | -2.6972% | +1.8441%, +2.8850%, +5.0070%, +1.3885% |

The B1 epochs are visibly noisy and include a +26.3407% outlier, so the pooled
result cannot support a positive claim; B4 is also slower. Stock and manager
hold the same configured tensor-arena capacity and report the same configured
KV-cache reservation, yielding an observed reservation difference of **0%**
for those configured arenas. This is
not a qualified end-to-end memory-saving result. The evidence comes from a
dirty source closure with `sealed=false`, `preflight_bound=false`,
`hardware_attested=false`, `qualified=false`, and `performance_go=false`. The
default-auto DeepGEMM path separately loaded all 66/66
shards, but lengthy precompilation was externally terminated before E2E
completion, so it contributes no correctness or performance pair.

[Qwen3.8-27B-FP8 diagnostic archive](../results/h20-sglang-v0517-abi8-qwen38-fp8-diagnostic-20260824/README.md)

## Historical superseded Qwen2.5-0.5B token-relocation diagnostic

`results/h20-sglang-v0517-token-relocation-diagnostic-20260825` records a
verified diagnostic rather than a qualification. It is retained for
auditability but is historical and superseded by the clean-source sealed
qualification above. It pins official
SGLang `v0.5.17` revision
`29481685462732237d80d86076d6563e1f658102`, Qwen2.5-0.5B, Full attention,
page16 BF16 NHD storage, eager single-GPU execution, and FlashInfer as the
token-indexed same-policy oracle. Four epochs alternate process order; every
epoch contains B1 and B4 Naive/Relocate pairs, and every process runs five
iterations. Excluding iteration 0 leaves 16 hot samples per mode and group.
All 8/8 pairs have exact output-token equality, two reclamation rounds per
iteration, expected relocation/copy/reclaimed-page counters, complete final
drain, and zero failure, quarantine, and fail-stop counters.

| Case | Relocate throughput delta | Relocate mean latency delta | Relocate p95 latency delta |
| --- | ---: | ---: | ---: |
| B1 | +7.629% | -7.088% | -22.844% |
| B4 | -1.232% | +1.247% | +2.880% |

B1 has material inter-epoch jitter, including both a slower epoch and a much
faster epoch, while B4 is slightly slower in the pooled diagnostic. The result
therefore remains `performance_go=false`. It is also `diagnostic_only`,
`sealed=false`, dirty-source, `hardware_attested=false`, and
`qualified=false`: the 64 snapshots are recorded NVIDIA H20 observations, not
independent hardware attestation. Reclaimed-page counters establish relocation
for this workload, not a configured-capacity or end-to-end memory reduction.
A relocate-only FA3 end-to-end smoke passed, but sparse Naive+FA3 is invalid
and now fails closed; the paired matrix uses FlashInfer in both modes.

[Historical Qwen2.5-0.5B relocation diagnostic archive](../results/h20-sglang-v0517-token-relocation-diagnostic-20260825/README.md)

## Historical frozen ABI5-v5 evidence

`results/h20-sglang-v0517-abi5-v5-grouped-release-20260821` is the preceding
frozen engine record. It binds exact source closure `9233c06d…` to:

- official SGLang `v0.5.17` at peeled commit
  `29481685462732237d80d86076d6563e1f658102`;
- one NVIDIA H20, page16 BF16 NHD storage, eager ChunkCache, and
  TP/PP/DP/DCP = 1;
- Qwen2.5-7B Full attention with FlashInfer;
- GPT-OSS-20B ordered Full+SWA128 with FA3 and SGLang's built-in Triton MoE;
- B1 and B4×5; prompt 513 plus decode 33; and
- radix/Prefix, overlap, Graph, speculation, disaggregation, streaming,
  hierarchical cache, and remote cache disabled.

Within that boundary, eight JSON records pass independent verification, all
84 request traces match stock token-for-token, and every arena drains. B4
grouped release uses five release/recycle transactions for 20 requests.

Manager and stock use equal configured tensor-arena capacity, so the observed
configured arena reservation difference is **0%**; this is not a qualified end-to-end
memory-saving result. One epoch reports B4 steady manager overhead
of +4.1932% for Qwen and -5.2048% for GPT-OSS, and Qwen B1 is +5.0009%. No
profile has repeated-epoch statistics. Therefore `performance_go=false`; the
negative GPT diagnostic is not a general speedup claim.

This is scoped historical L4 correctness for ABI5-v5. It does not qualify the
live ABI8 core, C wire, Python runtime, Prefix path, relocation, fixed-state
path, or performance.

## Earlier records

- `results/h20-sglang-v0517-abi5-full-hybrid-20260821` is the preceding frozen
  ABI5-v4 epoch.
- `results/h20-sglang-v0517-full-hybrid-20260821` is the preceding ABI4
  official-release epoch.
- `results/h20-canonical-manager-20260820` is older ABI3/development-pin pure
  SWA evidence; its reported 62.89% reduction compares different admission
  capacities and is not compression.

These records remain append-only calibration and provenance. None can qualify
a later ABI.

## Not qualified

- ABI8 SGLang/H20 Prefix profiles outside the sealed Qwen Full and GPT-OSS
  Full+SWA B1/B4 boundary, and all Prefix performance qualification;
- independent hardware attestation or performance qualification for the sealed
  request-private Full SGLang token-relocation scope; retained-slot
  attention/logit correctness beyond the exact-token oracle; packed Prefix;
  H20/model qualification of packed shared-tail COW; compaction capacity
  benefit; or compaction
  performance qualification;
- H20-qualified SGLang MLA latent+RoPE relocation; the fixed-state same-owner
  replacement production trigger; KDA/ShortConv/linear-attention families
  outside the strict Qwen3.5 profile; and Qwen3.5 L4 qualification, independent
  hardware attestation, or performance qualification beyond scoped pair
  verification from recorded H20 runtime snapshots;
- Qwen3.8-27B-FP8 L4 or performance qualification beyond the dirty-source,
  no-preflight diagnostic pair set;
- overlap scheduling, multiple completion domains, or CUDA Graph replay;
- real asynchronous GPU pressure coverage, fixed-state pressure accounting, or
  shared-Prefix/request-fork retention amplification;
- speculative branches, rollback, beam search, or cancellation pressure;
- cross-attention, dynamic sparse attention, production Mamba/SSM profiles
  outside the strict request-private Qwen3.5 path, vLLM,
  VMM-backed
  engine tensors, multi-GPU, disaggregation, remote memory, or production
  version/pressure matrices; and
- a same-capacity memory reduction, numerical compression, or general
  throughput/latency improvement.

The engine-neutral `orbitkv-runtime` and `orbitkv-reference` wheels establish a
host-tested/package-tested SPI and reference tensor-arena adapter only. The
reference is not a complete engine, and the current SGLang integration has not
migrated to the SPI.

Unsupported profiles must fail closed before mutation. The implementation and
qualification order is specified in the
[Token Virtualization and Attention Expansion Roadmap](token-virtualization-and-attention-roadmap.md).
