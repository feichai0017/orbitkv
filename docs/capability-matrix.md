# Capability Matrix

This is the normative boundary for the live source tree. A historical result
qualifies only the source closure named by its manifest; breaking ABI8 work
cannot inherit ABI5 hardware evidence.

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
| Strict HF manager-plan frontend | L1 | Emits the sole `KvPlanInput`; unknown semantics fail closed | `src/hf_config.rs`, `tests/canonical_cli.rs` |
| Identity and arena ownership | L2 GO | Generation-checked request, snapshot, page, step, submission, Prefix, reclamation, and relocation leases; independent class pools | `src/kv_manager/identity.rs`, `arena.rs`, host tests |
| Persistent snapshots | L2 GO | Immutable class roots, expected-head CAS, stale-head rejection, incremental path-copy, no hot full-root materialization | `src/kv_manager/persistent_snapshot.rs`, host/property tests |
| Append transactions | L2 GO | Failure-atomic acquire/fork/prepare/submit/complete, compact write/copy intents, abort and quarantine | `src/kv_manager/append_transaction.rs`, fault tests |
| Prefix and joint COW core | L2 GO | Page-aligned lookup/publish/publish-release/attach/evict/recycle; request fork; Full+SWA partial-tail joint COW | `src/kv_manager/prefix.rs`, Prefix/COW tests |
| Page-owned reclamation | L2 GO | Request/Prefix refs, reader pins and writer state jointly gate detach, certificates, ACK, and reuse | `src/kv_manager/reclamation.rs`, lifecycle/fault tests |
| Token virtualization and relocation core | L2 GO | Canonical token views; semantic-death/policy-eviction evidence; profitable private Full evacuation; exact copy receipts; packed publication; quarantine and precise release | `src/kv_manager/token_virtualization.rs`, `relocation_transaction.rs`, property/fault/lifecycle tests |
| Recurrent/convolution checkpoint pool | L2 host | Fixed-width generation-checked initial/replace/retire/ACK lifecycle with exact copy receipts, completion gating, abort, and quarantine; no token ids or TokenMove surface | `src/state_checkpoint.rs`, host lifecycle/fault tests |
| Typed C ABI8 wire | L2 GO | Exactly 40 typed symbols: 29 canonical-manager plus 11 independent state-pool symbols; per-handle transactions, C/C++ layout checks, reserved-field, capacity, short-buffer, stale-lease, and receipt validation; no cross-handle atomicity | `crates/orbitkv-ffi/include/orbitkv.h`, FFI tests, CI symbol diff |
| ABI8 Python FFI/runtime | L2 GO | Exact-40 ctypes loader; 73 frozen layouts; bounded manager workspaces; independent state-pool client; typed retry/fail-stop and force-destroy teardown | Python FFI/runtime tests against the release library |
| Official SGLang source contract | L2 | Official `v0.5.17`, peeled commit `29481685462732237d80d86076d6563e1f658102`, checked required hooks and fail-hard patch | pinned-checkout tests |
| SGLang `OrbitKVPrefixCache` | L2 GO | Official cache seam; nodes contain token/digest/LRU plus opaque Prefix leases only; warm attach, lock/ref accounting, Full+SWA COW, grouped release, eviction, and hostile fault paths pass host gates | pinned `v0.5.17` contract and plugin integration tests; the exact sealed H20 subset is listed below |
| ABI8 H20 Prefix path | Scoped L4 correctness | Qwen Full and GPT-OSS Full+SWA, B1/B4, page16 BF16 NHD eager FA3 on one H20; Prefix lifecycle and final drain qualified | `results/h20-sglang-v0517-abi8-full-hybrid-20260823` at exact `6f62a23` |
| SGLang token relocation | L2 host / L4 pending | Explicit/default-off eager path covers Full and ordered Full+SWA while they share one victim set; Full relocates physically, SWA keeps class-specific placement through a checked LUT; compact ReqToToken, split absolute/active lengths, decode continuation, and fail-closed release are host-tested | host plugin/runtime tests; no H20/E2E evidence; compact Hybrid fails closed once SWA visibility diverges |
| SGLang MLA relocation seam | L2 host / L4 pending | Pure BF16 Full-retention MLA, page16, eager, single GPU, FlashInfer or FA3; compiler geometry is checked against `MLATokenToKVPool` and relocation copies every layer's combined latent+RoPE row | real pinned `MLATokenToKVPool` host copy/config tests; no H20/model evidence; DSA, FP4, DCP, Hybrid Linear, Prefix performance pending |
| SGLang fixed-state seam | Scoped L2 host / production incomplete | The production request-owned Mamba path connects exact `slot_id + 1` mirrors, initial `MambaPool.clear_slots`, the forward completion event, and release-time retire/clear/ACK only | Same-owner `MambaPool.copy_from` replacement is coordinator/real-CPU-tensor host-tested, but its production trigger is pending; GDN/KDA/ShortConv/linear-attention family bindings and all real CUDA/model/H20/performance evidence are pending; separate token/state handles provide fail-stop containment, not cross-handle atomicity |
| Stable-address CUDA VMM primitive | L2 host | Isolated reserve/map/remap/unmap backend; not the manager data plane and not SGLang tensor storage | `crates/orbitkv-cuda/` host tests |
| General SGLang replacement | Not L5 | Scoped Prefix L4 does not qualify relocation, MLA, fixed state, overlap/Graph, speculation, distributed execution, pressure, performance, or a release matrix | this matrix |

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

`results/h20-sglang-v0517-abi8-full-hybrid-20260823` is the sealed and
latest engine record. It binds exact source commit
`6f62a23b9abaa9bf12e9b060389259fa9185e70f` (short id `6f62a23`) to:

- official SGLang `v0.5.17` at peeled commit
  `29481685462732237d80d86076d6563e1f658102`;
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
claim a speedup or memory saving, and it does not qualify a complete SGLang
replacement or production readiness. Token relocation, MLA, fixed state,
overlap, CUDA Graphs, speculation, distributed execution, and performance
qualification are explicitly excluded.

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

Same-capacity intrinsic KV-memory reduction is **0%** because manager and stock
use equal tensor-arena capacity. One epoch reports B4 steady manager overhead
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
- H20-qualified SGLang token relocation, retained-slot attention correctness, or compaction performance;
- H20-qualified SGLang MLA latent+RoPE relocation; the Mamba same-owner
  replacement production trigger; every GDN/KDA/ShortConv/linear-attention
  family binding; and real CUDA/model/H20/performance qualification for fixed
  state;
- overlap scheduling, multiple completion domains, or CUDA Graph replay;
- speculative branches, rollback, beam search, or cancellation pressure;
- cross-attention, dynamic sparse attention, production Mamba/SSM state, vLLM,
  VMM-backed
  engine tensors, multi-GPU, disaggregation, remote memory, or production
  version/pressure matrices; and
- a same-capacity memory reduction, numerical compression, or general
  throughput/latency improvement.

Unsupported profiles must fail closed before mutation. The implementation and
qualification order is specified in the
[Token Virtualization and Attention Expansion Roadmap](token-virtualization-and-attention-roadmap.md).
