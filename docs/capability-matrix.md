# Capability Matrix

This is the normative boundary for the live source tree. A historical result
qualifies only the source closure named by its manifest; breaking ABI6 work
cannot inherit ABI5 hardware evidence.

## Levels

| Level | Meaning |
| --- | --- |
| L1 Compiler | Semantics parse and compile into checked programs. |
| L2 Host/ABI | A core, wire, or adapter surface passes host correctness and fault gates. |
| L3 GPU Primitive | An isolated primitive passes exact-source hardware tests. |
| L4 Engine E2E | A pinned engine and released checkpoint pass exact-source end-to-end gates. |
| L5 Production | Pressure, cancellation, feature combinations, and a version matrix are qualified. |

## Live ABI6 source

| Capability | Level | Exact boundary | Evidence |
| --- | --- | --- | --- |
| Retention and plan compiler | L1 | Checked Full, sliding, and retained IR/compiler relations | `src/retention.rs`, `src/plan/`, compiler tests |
| Strict HF manager-plan frontend | L1 | Emits the sole `KvPlanInput`; unknown semantics fail closed | `src/hf_config.rs`, `tests/canonical_cli.rs` |
| Identity and arena ownership | L2 GO | Generation-checked request, snapshot, page, step, submission, Prefix, and reclamation leases; independent class pools | `src/kv_manager/identity.rs`, `arena.rs`, host tests |
| Persistent snapshots | L2 GO | Immutable class roots, expected-head CAS, stale-head rejection, incremental path-copy, no hot full-root materialization | `src/kv_manager/persistent_snapshot.rs`, host/property tests |
| Append transactions | L2 GO | Failure-atomic acquire/fork/prepare/submit/complete, compact write/copy intents, abort and quarantine | `src/kv_manager/append_transaction.rs`, fault tests |
| Prefix and joint COW core | L2 GO | Page-aligned lookup/publish/publish-release/attach/evict/recycle; request fork; Full+SWA partial-tail joint COW | `src/kv_manager/prefix.rs`, Prefix/COW tests |
| Page-owned reclamation | L2 GO | Request/Prefix refs, reader pins and writer state jointly gate detach, certificates, ACK, and reuse | `src/kv_manager/reclamation.rs`, lifecycle/fault tests |
| Typed C ABI6 wire | L2 GO | Exactly 23 batch-only symbols; C/C++ layout checks; reserved-field, span, capacity, short-buffer, stale-lease, and receipt validation | `crates/orbitkv-ffi/include/orbitkv.h`, FFI tests, CI symbol diff |
| ABI6 Python FFI/runtime | L2 GO | Exact-23 ctypes loader, bounded hot/cold workspaces, incremental snapshot/page/identity journals, collective mirror cleanup, typed retry/fail-stop, and force-destroy teardown | Python FFI/runtime tests against the release library |
| Official SGLang source contract | L2 | Official `v0.5.17`, peeled commit `29481685462732237d80d86076d6563e1f658102`, checked required hooks and fail-hard patch | pinned-checkout tests |
| SGLang `OrbitKVPrefixCache` | L2 GO | Official cache seam; nodes contain token/digest/LRU plus opaque Prefix leases only; warm attach, lock/ref accounting, Full+SWA COW, grouped release, eviction, and hostile fault paths pass host gates | pinned `v0.5.17` contract and plugin integration tests; no H20 evidence |
| Stable-address CUDA VMM primitive | L2 host | Isolated reserve/map/remap/unmap backend; not the manager data plane and not SGLang tensor storage | `crates/orbitkv-cuda/` host tests |
| General SGLang replacement | Not L5 | H20 Prefix E2E, overlap/Graph, speculation, distributed execution, pressure, performance, and a release matrix are pending | this matrix |

### Exact ABI6 C surface

The dynamic library must export these 23 symbols and no other `orbitkv_*`
symbol:

```text
orbitkv_abi_version
orbitkv_manager_abort_steps_batch
orbitkv_manager_acknowledge_reclamations_batch
orbitkv_manager_arena_identities
orbitkv_manager_arena_stats
orbitkv_manager_complete_batch
orbitkv_manager_create
orbitkv_manager_destroy
orbitkv_manager_prefix_attach_batch
orbitkv_manager_prefix_evict_batch
orbitkv_manager_prefix_lookup_batch
orbitkv_manager_prefix_publish_batch
orbitkv_manager_prefix_publish_release_batch
orbitkv_manager_prefix_recycle_batch
orbitkv_manager_prepare_batch
orbitkv_manager_quarantine_steps_batch
orbitkv_manager_quarantine_submissions_batch
orbitkv_manager_recycle_requests_batch
orbitkv_manager_release_batch
orbitkv_manager_request_acquire_batch
orbitkv_manager_request_fork_batch
orbitkv_manager_stats
orbitkv_manager_submit_batch
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
live ABI6 core, C wire, Python runtime, Prefix path, or performance.

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

- ABI6 SGLang/H20 Prefix correctness or Prefix warm-hit performance;
- token-exact relocation/compaction;
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
