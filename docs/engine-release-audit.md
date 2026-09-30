# Released engine interface audit

This S5.1 audit fixes the upstream comparison at the official, non-prerelease
releases checked on 2026-09-29. It is a source and interface audit, not an
OrbitKV engine upgrade or a serving qualification.

The subsequent 2026-09-30 adapter upgrade selects 0.30.0 and consumes its native
transfer-result API. See [S5.1's current delivery and qualification status](completion-plan.md#s51--release-and-interface-audit);
the audit table and conclusions below retain the earlier reviewed baseline.

| Project | Release | Commit | OrbitKV status |
| --- | --- | --- | --- |
| vLLM | [`v0.30.0`](https://github.com/vllm-project/vllm/releases/tag/v0.30.0) | [`ced6857a`](https://github.com/vllm-project/vllm/tree/ced6857afa0ea7b2e3f0846a62e1394e90f15607) | Upgrade target; the package pin, submodule and serving support remain `0.29.0` |
| SGLang | [`v0.5.20`](https://github.com/sgl-project/sglang/releases/tag/v0.5.20) | [`94602c9c`](https://github.com/sgl-project/sglang/tree/94602c9c2b7cbdb8efd5c52802dac6a1c180089e) | Current source and serving baseline |
| LMCache | [`v0.5.5`](https://github.com/LMCache/LMCache/releases/tag/v0.5.5) | [`05a013b2`](https://github.com/LMCache/LMCache/tree/05a013b29da78cf2321b9b46ec5039dde2fb0bb0) | Interface and comparison reference only |

The immutable links below identify the reviewed release source; the delivery
handoff records the external local snapshot and its hashes. vLLM main
`af5b4857e1353c01fd6bf41bc3cb9f84dc82dd89` and SGLang main
`f731e82f09be137cfc5001c732f044dd44740c1d` were inspected only for later fixes.
Nothing available only at those revisions is treated as a released contract.

## Released integration boundaries

vLLM 0.30.0 exposes the experimental `KVConnectorBase_V1` contract and a
connector factory that accepts an explicit module path. Scheduler callbacks own
lookup, allocation handoff, request completion and engine block retention. Worker
callbacks own registration, preemption before overwrite, load/save, layer waits,
terminal transfer results and shutdown. `SupportsHMA` adds all-group completion.
The release also adds `finish_forward`, `reset_capture_state`, failed-receive
reporting and `bind_kv_cache_manager`. These are the right public boundary for an
OrbitKV adapter even though the API is still explicitly marked experimental.

| vLLM callback family | Released calls consumed or evaluated by OrbitKV |
| --- | --- |
| Scheduler lookup/allocation | `get_num_new_matched_tokens`, `update_state_after_alloc`, `on_new_request`, `build_connector_meta` |
| Scheduler completion/ownership | `requires_kv_delivery`, `supports_divergent_local_hybrid_hits`, `update_connector_output`, `request_finished`, `request_finished_all_groups`, `register_finished_partial_tail`, `has_pending_block_frees`, `has_pending_push_work` |
| Worker registration/order | `register_kv_caches`, `set_host_xfer_buffer_ops`, `handle_preemptions`, `start_load_kv`, `wait_for_layer_load`, `finish_forward`, `reset_capture_state` |
| Worker save/completion | `save_kv_layer`, `wait_for_save`, `get_transfer_results`, `get_block_ids_with_load_errors`, `build_connector_worker_meta`, `shutdown` |
| Topology and integration | `bind_kv_cache_manager`, `bind_gpu_block_pool`, `get_required_kvcache_layout`, `requires_piecewise_for_cudagraph`, handshake setters, stats/metrics/events and `reset_cache` |

These two capability properties need explicit values during the upgrade. Ordinary
OrbitKV read/write and `save_only` cache publication is best effort: a dropped
save becomes a future miss, so `OrbitKVConnector.requires_kv_delivery` must be
false. A reliable P/D producer must return true until its handoff is delivered or
fails terminally; a consumer does not own producer delivery. `MultiConnector`
returns true when **any** child requires delivery, ensuring a best-effort cache
cannot weaken the P/D producer. The base divergent-hybrid property defaults to
false, NIXL reports true, and `MultiConnector` reports true only when **all**
children do. OrbitKV must remain false until its released 0.30.0 adapter proves
that divergent local Full/SWA/Mamba hits are completed atomically across every
required group; the existing block-pool suppression cannot be removed first.

vLLM's 0.30.0 LMCache entry is not one fixed built-in implementation. At import
time it prefers `LMCacheMPConnector` from the installed LMCache package and falls
back to a retained vLLM copy on any `ImportError`; an environment switch can
force the built-in copy. LMCache 0.5.5 implements the public connector with
separate scheduler and worker adapters, request trackers, optional dispatcher,
MP service connections and standard completion callbacks. OrbitKV should follow
the scheduler/worker responsibility split, but should not copy the import-time
version fallback, dispatcher, cache policy or MP transport.

vLLM 0.30.0 also registers `FlexKVConnectorV1`, `NixlPullConnector`,
`NixlPushConnector` and `MultiConnector` in the same factory. The connector
module-path mechanism is sufficient for a separately installed OrbitKV package;
an upstream registration entry is useful for discoverability, not required for
correct loading. The 0.30.0 `MultiConnector` contains the real-block allocation
fix from vLLM PR #46865: the selected loader receives the external-token count,
while every other connector receives the request's actual blocks with zero
external tokens. LMCache's 0.5.5 P/D recipe requires that fix, and the fix is
present in the selected vLLM release.

SGLang 0.5.20 exposes a package entry point, `register_radix_cache_backend`, the
`UnifiedRadixCache` construction path and `UnifiedCacheLinker`. The linker owns
lookup, queued load, layer-start, cancellation, completion, offload, reset and
close; the engine retains the radix tree, allocation and request locks. The
released Hook registry is a supported plugin mechanism, but a string target such
as `_commit_transfer_to_req` remains a dependency on an internal method. LMCache's
released `LMCRadixCache` subclasses the older native `RadixCache`, maintains its
own in-flight node/load markers and supports MP/IP modes. OrbitKV keeps the
released Unified tree/linker ownership instead of importing that second tree or
LMCache's CUDA-IPC service protocol.

At the recorded SGLang main revision, an unreleased
[`LMCacheUnifiedRadixCache`](https://github.com/sgl-project/sglang/blob/f731e82f09be137cfc5001c732f044dd44740c1d/python/sglang/srt/mem_cache/storage/lmcache/lmcache_unified_radix_cache.py)
adds Unified-tree load, abort and Mamba flows. It is useful evidence for upstream
interface design, but is absent from 0.5.20 and cannot replace OrbitKV's released
linker or qualify a deployment. No main-only vLLM or SGLang API is selected by
this audit.

## vLLM 0.30.0 P/D assessment

The released NIXL connectors are useful replacements for much of the current
OrbitKV-owned P/D control plane, but they are not yet a drop-in correctness
replacement for TENT plus OrbitKV state contracts.

| Requirement | Released vLLM 0.30.0 behavior | Decision before removing OrbitKV code |
| --- | --- | --- |
| Layout and ranks | Dense transfer requests `LBHNC`; MLA uses the default layout. NIXL exchanges PP/TP/DCP metadata and implements heterogeneous block-size, region/group and head-placement paths. | Implement a narrow TENT transport backend or equivalent released construction point, then compare its descriptors with OrbitKV HND/BHNC, MLA and heterogeneous-TP mappings. Reject unsupported resharding. |
| Hybrid state | `SupportsHMA`, SWA clipping, Mamba speculative-slot clipping and `mamba_cache_mode` handling are present. Full/SWA/Mamba groups are selected together at the connector boundary. NIXL opts into divergent local hybrid hits; `MultiConnector` requires every child to opt in. | Keep OrbitKV's divergent-hit property false until exact Full + SWA + recurrent boundaries and state bytes pass against compiled recovery demand. A capability flag or group count does not prove atomic all-state readiness. |
| Cancellation and preemption | Scheduler paths clean aborted/preempted requests; workers defer failure until submitted handles finish and retain a handle when release fails. `requires_kv_delivery` makes preempted reliable handoffs recompute, and `MultiConnector` requires delivery when any child does. | Reuse these engine lifecycle semantics. Keep the earlier OrbitKV preemption fence, generation checks and current destination/source retention until real abort, partial-submit and restart gates prove the TENT adaptation. |
| Completion and failure | `KVConnectorTransferResults` distinguishes finished sends, finished receives and failed receives. NIXL polls transfer state and releases completed handles. | Map TENT terminal status into this result exactly once. A timeout, heartbeat loss or lease expiry must not stand in for native drain. The native NIXL TTL behavior is not evidence for OrbitKV source reclamation. |
| Cache composition | `MultiConnector` now gives non-loading caches real blocks and tracks extra asynchronous saves. LMCache documents NIXL handoff plus LMCache offload using this path. | Test cold, partial and full handoff with OrbitKV `save_only` and ordinary read/write cache modes. Exactly one connector may load/write each destination; all required save completions must delay block free. |
| Request routing | The vLLM router and NIXL own the released P/D request flow. | Native P/D now uses the upstream router with the exact patched engine revision in S5.4; OrbitKV's custom proxy is removed. Routing remains outside Cache Manager policy. |

Two released ordering gaps remain decisive. In vLLM 0.30.0, the model runner
updates requests and performs page zeroing/COW before
`ActiveKVConnector.pre_forward` calls `handle_preemptions`; moving OrbitKV's
preemption fence there could let an asynchronous save read an overwritten page.
The runner also executes Mamba `preprocess_state` before `pre_forward`, so a
recurrent restore submitted there is too late. OrbitKV 0.29.0's `runtime.py`
drains preempted saves before `update_requests`, then starts restore after page
initialization/COW and before recurrent preprocessing. A later upgrade may use
`pre_forward` for dense load submission only after proving its page ownership,
but it cannot replace either current ordering fence. The upgrade needs released
pre-update and post-update/preprocess lifecycle boundaries, or must retain one
narrow release-specific owner until equivalent hooks are released and qualified.

## Current adapter inventory and removal gates

Every runtime file below has one named caller and disposition. “Keep” means the
file owns engine translation or a physical resource. “Remove” is gated by a
consumed released replacement and tests; this audit deletes no protection.

| File | Actual caller and owned state/resource | Replacement or removal gate |
| --- | --- | --- |
| `vllm/__init__.py` | Package import surface for connector classes | Keep a small public export surface |
| `vllm/plugin.py` | `vllm.general_plugins` entry point; registers OrbitKV connector names | Prefer one official factory entry; remove duplicate registration only when the selected release resolves the class without it |
| `vllm/connector.py` | vLLM factory; public callback adapter and role construction; owns selected client/context | Keep public class; reduce it to configuration plus scheduler/worker delegation. Explicitly return false for best-effort delivery and divergent hits until the latter is qualified |
| `vllm/config.py` | Connector construction and scheduler/worker helpers; owns immutable identity/topology values | Keep engine-specific identity and rank mapping; the unused service-state field is removed |
| `vllm/layout.py` | Worker registration and scheduler boundary code; maps released cache groups to physical layouts | Keep while layouts are consumed; qualify against 0.30.0 HMA/MLA/Mamba specs |
| `vllm/metadata.py` | Scheduler-to-worker and worker-to-scheduler connector callbacks | Keep; migrate to 0.30.0 transfer results without compatibility aliases |
| `vllm/scheduler.py` | Released scheduler callbacks; owns pending queries, leases, save intents and prepared state | Keep. Replace the `BlockPool.get_cached_block` monkey patch only after all-group native-hit correctness is consumed |
| `vllm/worker.py` | Released worker callbacks; owns GPU registrations, restore handles, CUDA dependencies and save thread | Keep. Port completion/failure callbacks to 0.30.0 and retain drain ownership |
| `vllm/runtime.py` | Plugin-installed `GPUModelRunner.update_requests` monkey patch; drains preempted saves before page overwrite and starts restore after page setup but before recurrent preprocessing | `pre_forward` is too late for both fences. Remove only after released pre-update and post-update/preprocess contracts pass page-reuse, dense and recurrent GPU gates |
| `vllm/metrics.py` | vLLM metric callbacks and scheduler/worker aggregation | Keep metrics with production consumers; remove fields only with their producer and dashboard |
| `vllm/tp_shards.py` | Scheduler multi-Manager query fan-out for same-host TP shards | Keep bounded local responsibility; cross-host fan-out belongs to S5.5 Manager work |
| `client/__init__.py` | Both adapters import the shared client and CUDA registration surface from here | Keep the small public export surface aligned with the native type stubs |
| `client/connection.py` | Both engine adapters; owns node-local UDS clients, shard sockets and close | Keep shared transport configuration owner |
| `client/gpu.py` | Engine registration paths; serializes framework CUDA IPC wrappers | Replace in S3 with explicit validated registration data, then remove pickle-compatible representation |
| `vllm/transport.py` | Native MooncakeConnector factory | Thin TENT payload backend; custom P/D request states, handshake and proxy removed in S5.4 |
| `sglang/__init__.py` | Package import surface | Keep a small public export surface |
| `sglang/config.py` | `linker.py`; owns model/adapter/representation identity | Keep engine-specific identity; add live-weight invalidation before supporting it |
| `sglang/layout.py` | `plugin.py`/`linker.py`; maps Unified pools to GPU regions | Keep; reject unknown DSA/draft/auxiliary layouts |
| `sglang/linker.py` | `UnifiedRadixCache`; owns OrbitKV query/load/offload queues, registrations and terminal close | Keep as the released external-linker implementation |
| `sglang/events.py` | Graph initialization Hook and `OrbitKVLinker`; owns stable CUDA events and forward activation counters | Keep event lifetime and pre-capture registration; import GPU dependencies only when the backend is selected |
| `sglang/recovery.py` | `plugin.py`; wraps linker/tree and overrides Mamba component checkpoint behavior | Move generic checkpoint lifecycle into released engine components before shrinking; keep safety checks until consumed |
| `sglang/admission.py` | Pending-query admission plus opt-in enqueue preparation; cancellation uses native cache finish/linker release | Replace the remaining two targets with public pending-lookup/enqueue callbacks |
| `sglang/completion.py` | Public decode-owned `PDTransferEvent` observer | Six private P/D Hooks removed; reports only terminal observations and never authorizes page release |
| `sglang/pd.py` | Plugin; registers the TENT factory explicitly on the patched engine | Factory implemented and tested; official 0.5.20 lacks the API. Keep native bootstrap and request states; qualify S3 drain before claiming safe fault recovery |
| `sglang/plugin.py` | Backend/factory registration; two ordinary cache Hooks and one optional enqueue Hook | Native P/D uses public observations; ordinary cache internal Hooks remain S5.3 work |

The subsequent S5.2 cleanup removes `vllm/state_manager.py`, its context field,
health thread and mocks because no production query consumed its availability.
Restore failures still propagate and retain page ownership. Scheduler and worker
implementations are named `SchedulerAdapter` and `WorkerAdapter`; their request
states remain unchanged. This cleanup does not establish a released replacement
for the remaining lifecycle dependencies inventoried above. A follow-up removes
`pd/base_connector.py`, `pd/async_runner.py` and `pd/prefill_tasks.py`: public
callbacks live on the connector, role workers directly own their request states,
and task records/accounting live with the executors that consume them. No
compatibility aliases preserve the removed internal classes.

The current SGLang Hook targets are deliberately explicit. Graph capture uses
`TpModelWorker.init_cuda_graphs`. Request admission uses
`PrefillAdder.add_one_req`; opt-in queue preparation uses
`Scheduler._add_request_to_queue`. Cancellation already has a released
replacement: `BasePrefixCache.finish(ABORT)` calls
`UnifiedRadixCache.release_aborted_request`, which calls the installed linker's
`release_request`. The redundant `Scheduler._release_aborted_request` Hook is
removed. The subsequent S5.4 cutover removes the six private P/D observation
Hooks and consumes public `PDTransferEvent` callbacks plus the explicit TENT
factory. The module-global transfer-engine class is not replaced. These are
fork APIs at the exact revisions in [P/D setup](pd.md), not released support.
The two ordinary cache Hook targets and optional enqueue Hook still need
consumed public replacements.

The two vLLM monkey patches are also separate. `runtime.py` replaces
`GPUModelRunner.update_requests` for preemption and recurrent restore ordering;
`scheduler.py` replaces one `BlockPool.get_cached_block` method on the bound pool
to suppress unsafe partial HMA hits. Neither is replaced merely by registering
the connector in the public factory.

## Configuration inventory

Current adapters parse and validate these without old/new aliases.

| Owner | Current entries |
| --- | --- |
| vLLM cache endpoint/session | `orbitkv.host`, `orbitkv.port`, `orbitkv.bootstrap_socket`, `orbitkv.tp_shard_endpoints`, `orbitkv.tp_shard_bootstrap_sockets`, `orbitkv.timeout_ms`, `orbitkv.spin_iterations`; environment overrides `ORBITKV_HOST`, `ORBITKV_PORT`, `ORBITKV_INSTANCE_ID` |
| vLLM cache behavior and identity | `orbitkv.mode`, `orbitkv.transfer_backend`, `orbitkv.wait_for_full_prefix`; `ORBITKV_CROSS_LAYER_BLOCKS`, `ORBITKV_LOAD_TIMEOUT_SECONDS`, `ORBITKV_PREPARE_REQUESTS`, `ORBITKV_QUEUE_WARMUP`, `PYTHONHASHSEED`, `CUDA_VISIBLE_DEVICES` |
| SGLang | `ORBITKV_SGLANG_ENDPOINT`, `ORBITKV_TRANSFER_BACKEND`, `ORBITKV_PREPARE_REQUESTS`, `ORBITKV_QUEUE_WARMUP`, `SGLANG_MOONCAKE_TRANSFER_ENGINE`, `ORBITKV_SGLANG_TENT_TIMEOUT_S`, `SGLANG_ENABLE_FAILED_SESSION_PROBE`, `MC_FORCE_TCP`; standard SGLang flags select the plugin/backend and native P/D mode |

The custom P/D configuration, request fields and proxy CLI are retired with
that implementation. Native `kv_role`, `mooncake_protocol`, `device_name` and
`transfer_engine_factory` configure vLLM P/D; let each start choose a new native
engine identity. GPU visibility, model/cache identity, Manager connection,
physical backend and cache timeout settings remain OrbitKV responsibilities.

## Execution and remaining qualification

[The completion plan](completion-plan.md) is the only delivery queue. This audit
began against the 0.29.0 baseline; current ordinary cache pins are vLLM 0.30.0
and SGLang 0.5.20. Native P/D requires the separate patched revisions in
[P/D setup](pd.md). Its source/drain/output gates and legacy removal belong to
S5.4; ordinary hybrid ordering and public cache lifecycle remain S5.3. Fork
implementation, local hardware qualification, independent acceptance, upstream
merge and released support remain distinct statuses.
