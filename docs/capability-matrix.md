# Capability Matrix

This is the normative boundary for the live source tree. A historical result
qualifies only the source closure named by its manifest; breaking ABI7 work
cannot inherit ABI5 hardware evidence.

## Levels

| Level | Meaning |
| --- | --- |
| L1 Compiler | Semantics parse and compile into checked programs. |
| L2 Host/ABI | A core, wire, or adapter surface passes host correctness and fault gates. |
| L3 GPU Primitive | An isolated primitive passes exact-source hardware tests. |
| L4 Engine E2E | A pinned engine and released checkpoint pass exact-source end-to-end gates. |
| L5 Production | Pressure, cancellation, feature combinations, and a version matrix are qualified. |

## Live ABI7 source

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
| Typed C ABI7 wire | L2 GO | Exactly 29 batch-only symbols; C/C++ layout checks; token/relocation spans; reserved-field, capacity, short-buffer, stale-lease, and receipt validation | `crates/orbitkv-ffi/include/orbitkv.h`, FFI tests, CI symbol diff |
| ABI7 Python FFI/runtime | L2 GO | Exact-29 ctypes loader; 58 frozen layouts; bounded hot/cold workspaces; optional relocation capability; typed retry/fail-stop and force-destroy teardown | Python FFI/runtime tests against the release library |
| Official SGLang source contract | L2 | Official `v0.5.17`, peeled commit `29481685462732237d80d86076d6563e1f658102`, checked required hooks and fail-hard patch | pinned-checkout tests |
| SGLang `OrbitKVPrefixCache` | L2 GO | Official cache seam; nodes contain token/digest/LRU plus opaque Prefix leases only; warm attach, lock/ref accounting, Full+SWA COW, grouped release, eviction, and hostile fault paths pass host gates | pinned `v0.5.17` contract and plugin integration tests; no H20 evidence |
| SGLang token relocation | L2 host / L4 pending | Explicit/default-off eager path covers Full and ordered Full+SWA while they share one victim set; Full relocates physically, SWA keeps class-specific placement through a checked LUT; compact ReqToToken, split absolute/active lengths, decode continuation, and fail-closed release are host-tested | host plugin/runtime tests; no H20/E2E evidence; compact Hybrid fails closed once SWA visibility diverges |
| SGLang MLA relocation seam | L2 host / L4 pending | Pure BF16 Full-retention MLA, page16, eager, single GPU, FlashInfer or FA3; compiler geometry is checked against `MLATokenToKVPool` and relocation copies every layer's combined latent+RoPE row | real pinned `MLATokenToKVPool` host copy/config tests; no H20/model evidence; DSA, FP4, DCP, Hybrid Linear, Prefix performance pending |
| Stable-address CUDA VMM primitive | L2 host | Isolated reserve/map/remap/unmap backend; not the manager data plane and not SGLang tensor storage | `crates/orbitkv-cuda/` host tests |
| General SGLang replacement | Not L5 | ABI7 H20 Prefix/relocation E2E, overlap/Graph, speculation, distributed execution, pressure, performance, and a release matrix are pending | this matrix |

### Exact ABI7 C surface

The dynamic library must export these 29 symbols and no other `orbitkv_*`
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
```

The ABI5 scalar-shaped names `abort_steps`, `quarantine_steps`,
`quarantine_submissions`, `acknowledge_reclamations`, and `recycle_requests`
are removed. CI fails if an active source surface reintroduces them. Frozen
headers inside `results/` remain unchanged.

## Historical frozen ABI5-v5 evidence

`results/h20-sglang-v0517-abi5-v5-grouped-release-20260821` is the latest
engine record. It binds exact source closure `9233c06d…` to:

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
live ABI7 core, C wire, Python runtime, Prefix path, relocation, or performance.

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

- ABI7 SGLang/H20 Prefix correctness or Prefix warm-hit performance;
- H20-qualified SGLang token relocation, retained-slot attention correctness, or compaction performance;
- H20-qualified SGLang MLA latent+RoPE relocation and any recurrent/convolution state checkpoint integration;
- overlap scheduling, multiple completion domains, or CUDA Graph replay;
- speculative branches, rollback, beam search, or cancellation pressure;
- cross-attention, dynamic sparse attention, Mamba/SSM state, vLLM, VMM-backed
  engine tensors, multi-GPU, disaggregation, remote memory, or production
  version/pressure matrices; and
- a same-capacity memory reduction, numerical compression, or general
  throughput/latency improvement.

Unsupported profiles must fail closed before mutation. The implementation and
qualification order is specified in the
[Token Virtualization and Attention Expansion Roadmap](token-virtualization-and-attention-roadmap.md).
