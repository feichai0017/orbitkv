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
Capability-oriented qualification and verifier entry points are routing
facades over manifest-bound implementations. Their generic names add no new
profile or claim. Legacy hardware- or model-named paths remain compatibility
and evidence entry points; immutable copies under `results/` preserve existing
source closures and archive reproducibility.

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
| Dense Hybrid GDN HF frontend | L1 GO | The current frontend admits one exact external architecture/model-type discriminator tuple, then compiles its explicit geometry into Full token KV plus request-private GDN/convolution state. Structurally similar but unrecognized families are not admitted; malformed, defaulted, or schedule-inconsistent configs fail closed | `src/hf_config.rs`, fixture-backed CLI tests |
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
| Request-private pressure telemetry | L2 host plus unarchived device diagnostic | Opt-in event samples separate consumed capacity, resident data, request-reachable bytes, semantic-live bytes, free-space minima, high-water marks, and retention amplification | Host runtime/async-schedule gates and a real single-device diagnostic have run; no append-only, sealed, or qualified pressure record is published. The device observation is not an allocator peak or a performance/capacity result. Fixed-state bytes are excluded, and shared Prefix/request-fork retention amplification fails closed |
| Official SGLang source contract | L2 | The pinned stock checkout and manager checkout with its manifest-bound loader patch are checked separately | pinned-checkout tests |
| SGLang `OrbitKVPrefixCache` | L2 GO | Official cache seam; nodes contain token/digest/LRU plus opaque Prefix leases only; warm attach, lock/ref accounting, Full+SWA COW, grouped release, eviction, and hostile fault paths pass host gates | pinned engine-contract and plugin integration tests; the exact sealed subset is linked below |
| ABI8 Prefix engine path | Scoped L4 correctness | Prefix correctness and lifecycle are qualified only for the model, backend, and workload boundary frozen by the result; fixed state and broader engine support are excluded | [sealed Prefix record](../results/h20-sglang-v0517-abi8-full-hybrid-20260823/README.md) |
| SGLang token relocation | L2 host plus scoped sealed correctness/lifecycle qualification | The explicit/default-off eager path preserves one multi-request scheduler batch: all moves are flattened into one backend move and one completion event, every ReqToToken/LUT/request mirror plan is validated before the first write, and all retirements receive one batch ACK. The qualified engine scope is request-private Full only; common-victim-set Full+SWA remains host-tested, not part of this seal | Host plugin/runtime tests plus the [sealed relocation record](../results/h20-sglang-v0517-abi8-token-relocation-20260825/README.md). Completion is eager and host-blocking, not asynchronously overlapped. Packed fork/shared-tail COW is newer host/raw-ABI8/Python-FFI functionality outside the seal; packed Prefix fails closed |
| Engine-neutral CUDA opaque-byte relocation harness | Component conformance / qualification pending | Independent payload oracle, real stream ordering, exact evacuation, ACK-gated generation reuse, and final drain | `integrations/sglang/tests/relocation_conformance.py`, `test_cuda_relocation_conformance.py`; component evidence only, not sealed L3/L4, performance, capacity, or complete-engine qualification |
| SGLang token-relocation sealed record | Scoped correctness + lifecycle qualified / performance pending | Request-private Full relocation against the pinned engine closure; the manifest carries the exact model, backend, hardware, batch, and sampling boundary | [sealed qualification](../results/h20-sglang-v0517-abi8-token-relocation-20260825/README.md); `source_clean=true`, `preflight_bound=true`, `sealed=true`, `qualified=true`, `hardware_attested=false`, `performance_go=false`; no capacity, memory-saving, general speedup, production, or replacement claim |
| SGLang MLA relocation seam | L2 host / L4 pending | The compiler geometry is checked against the engine's latent-KV pool and relocation copies each combined latent+RoPE row | Real pinned-pool host copy/config tests; broader precision, distributed, Hybrid Linear, Prefix, and engine qualification pending |
| SGLang fixed-state seam | Scoped L2 host plus recorded-device evidence / L4 pending | Runtime admission is structural for the implemented request-private GDN+convolution contract: allocation/clear, completion, retirement, exact ACK, and fail-closed exclusion of Prefix sharing and an unimplemented copy trigger | Host gates plus the linked fixed-state records below; recorded-device evidence does not promote the seam to L4 |
| Fixed-state pair-verification record | Scoped pair verification / independently unattested / L4 pending | Fresh-prompt execution validates token/state plans, lifecycle, completion, and final drain for the manifest-bound frontend profile | [pair-verification record](../results/h20-sglang-v0517-abi8-qwen35-fixed-state-pair-verification-20260823/README.md); `qualified=false`, `hardware_attested=false`, `performance_go=false` |
| Fixed-state weight-backed diagnostic | Recorded-device diagnostic / L4 pending | Model artifacts and execution closure are recorded in the archive; the run is diagnostic rather than a qualification gate | [diagnostic archive](../results/h20-sglang-v0517-abi8-qwen38-fp8-diagnostic-20260824/README.md); `diagnostic_only`, `sealed=false`, `preflight_bound=false`, `hardware_attested=false`, `qualified=false`, `performance_go=false` |
| Stable-address CUDA VMM primitive | L2 host | Isolated reserve/map/remap/unmap backend; not the manager data plane and not SGLang tensor storage | `crates/orbitkv-cuda/` host tests |
| General SGLang replacement | Not L5 | The separately scoped Prefix and Full-relocation seals do not qualify MLA, fixed state, async pressure/overlap, Graph, speculation, distributed execution, performance, or a release matrix | this matrix |

The relocation batch boundary is collective but not rollback-capable end to end.
After a disposition mark succeeds, a later prepare, copy, submit, complete,
publication, mirror, or ACK failure or uncertain return fail-stops the runtime;
it does not restore the pre-mark disposition snapshot or an older request head.
An explicitly unobserved copy can release the relocation reservations, but it
does not undo the already committed mark. Async copy/consumer overlap remains
unimplemented.

### Fixed-state host/runtime profile

The frontend recognizes one exact external discriminator tuple and compiles its
explicit layer schedule and byte geometry; arbitrary structurally similar
checkpoints are not currently admitted. After compilation, runtime admission is
not keyed by a marketing model name: it checks the structural contract for
token-addressable Full KV plus request-private GDN recurrent and convolution
state, and fails closed on missing, defaulted, or inconsistent semantics.

The compiler sends only Full KV to `CanonicalKvManager`. GDN recurrent and
convolution state go to the independent generation-checked
`StateCheckpointPool` and remain request-private. In the SGLang adapter the
server runs with Radix disabled; the OrbitKV cache seam remains installed only
to preserve canonical release handling and never publishes or attaches a
Prefix entry for this profile. The adapter also checks backend, dtype, cache,
and state-layout requirements from the accepted plan before mutation.

SGLang still owns tensor allocation, attention/linear kernels, scheduling, and
model execution. OrbitKV owns the accepted plan and identities plus page/state
transactions and completion-gated reclamation. A resource becomes reusable
only after the Semantic Frontier proves it unreachable and the Execution
Frontier proves that prior GPU use has completed. The two proofs are not
interchangeable.

This remains below L4 qualification. A scoped archive records matching outputs,
exact lifecycle counters, completion, and final drain, but is explicitly
`qualified=false`, `hardware_attested=false`, and `performance_go=false`. See
the [fixed-state pair-verification record](../results/h20-sglang-v0517-abi8-qwen35-fixed-state-pair-verification-20260823/README.md).

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

The sealed Prefix record binds the exact source closure, engine,
frontend profiles, backends, hardware, and workloads. Matching outputs, Prefix
lifecycle, SWA retirement where applicable, and final drain establish Scoped
L4 correctness only for that boundary. Timings remain diagnostic with
`performance_go=false`, and the record is not a qualified end-to-end
memory-saving result. It does not qualify fixed state, relocation, broader
engine support, or production readiness. See the
[sealed Prefix record](../results/h20-sglang-v0517-abi8-full-hybrid-20260823/README.md).

### Token-relocation scope

The token-relocation record binds a clean, preflighted source closure to a
pinned request-private Full engine path. Exact outputs, relocation lifecycle,
final drain, and component conformance pass. The seal is `qualified=true` only
for scoped correctness and lifecycle; `hardware_attested=false` and
`performance_go=false`. It establishes no capacity, memory-saving, general
speedup, production, Prefix, Hybrid/SWA, MLA, overlap, Graph, distributed, or
multi-GPU claim. See the
[sealed relocation record](../results/h20-sglang-v0517-abi8-token-relocation-20260825/README.md).

## Fixed-state engine evidence

The first record completes scoped pair verification for the manifest-bound
fixed-state frontend profile: outputs, token/state lifecycle, stream/event
completion, and final drain match. It remains `qualified=false`,
`hardware_attested=false`, and `performance_go=false`, so L4, performance,
memory-saving, same-owner copy, and broader-family claims remain pending. See
the [pair-verification record](../results/h20-sglang-v0517-abi8-qwen35-fixed-state-pair-verification-20260823/README.md).

A later weight-backed run is diagnostic only. Its artifact hashes, source and
engine closure, hardware observation, workloads, and timings live in the
[diagnostic archive](../results/h20-sglang-v0517-abi8-qwen38-fp8-diagnostic-20260824/README.md).
It remains dirty-source, `sealed=false`, `preflight_bound=false`,
`hardware_attested=false`, `qualified=false`, and `performance_go=false`; no
positive correctness, performance, memory, or production claim is promoted.

## Historical records

Superseded diagnostics and earlier ABI records remain append-only for
auditability. Their exact source, engine, model, device, workload, measurements,
and exclusions live only in the [Results Index](../results/README.md). A
historical record qualifies neither the live ABI8 tree nor any capability
outside its original manifest.

## Not qualified

- Prefix profiles outside the sealed manifest boundary, and all Prefix
  performance qualification;
- independent hardware attestation or performance qualification for the sealed
  request-private Full SGLang token-relocation scope; retained-slot
  attention/logit correctness beyond the exact-token oracle; packed Prefix;
  engine qualification of packed shared-tail COW; compaction capacity
  benefit; or compaction
  performance qualification;
- engine-qualified MLA latent+RoPE relocation; the fixed-state same-owner
  replacement production trigger; KDA/ShortConv/linear-attention families
  outside the currently bound structural profile; and L4 qualification,
  independent hardware attestation, or performance qualification beyond the
  scoped fixed-state records;
- overlap scheduling, multiple completion domains, or CUDA Graph replay;
- sealed or qualified asynchronous GPU pressure coverage, fixed-state pressure accounting, or
  shared-Prefix/request-fork retention amplification;
- speculative branches, rollback, beam search, or cancellation pressure;
- cross-attention, dynamic sparse attention, broader production recurrent-state
  profiles, additional engine adapters, VMM-backed
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
