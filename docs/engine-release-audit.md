# Released engine interface audit

This S5.1 audit fixes the upstream comparison at the official, non-prerelease
releases checked on 2026-09-29. It is a source and interface audit, not an
OrbitKV engine upgrade or a serving qualification.

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
| Request routing | The vLLM router and NIXL own the released P/D request flow. | Remove OrbitKV's proxy only after an executable released router scenario carries the required request IDs, rank metadata, errors and cancellation. Routing remains outside Cache Manager policy. |

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
| `vllm/pd/__init__.py` | vLLM factory loads split P/D public connector classes | Retire after a released native connector plus TENT backend passes all gates |
| `vllm/pd/scheduler.py` | Split connector scheduler callbacks; owns active waits/pushes and release reasons | Replace with native P/D request lifecycle after equivalence tests |
| `vllm/pd/worker.py` | Shared GPU layer registration, rank identity and TENT construction; no role handlers | Retain only consumed TENT/layout adapter responsibilities at native cutover |
| `vllm/pd/decode_worker.py` | Decode worker; owns handshake, page grants, generation-scoped waits, async transfer completion and prefill dispatch | Native cancellation/completion must preserve destination generations and terminal drain before removal |
| `vllm/pd/prefill_worker.py` | Prefill worker; owns target authorization, per-layer push plans, async writers/finalizers and source releases | Native producer lease plus TENT completion must pass partial-submit/preemption/restart before removal |
| `vllm/pd/mooncake.py` | `worker.py`; thin Rust TENT engine adapter and request-generation bookkeeping | Keep only the explicit TENT payload adapter required by the released native connector; rename at cutover rather than preserve Mooncake compatibility |
| `vllm/pd/layout.py` | P/D worker registration | Merge only with a consumed engine layout mapper; keep explicit layout validation |
| `vllm/pd/layout_mapping.py` | Prefill target planning and decode rank fan-in | Keep for heterogeneous TP until native released mapping proves equivalent; reject unsupported mappings |
| `vllm/pd/metadata.py` | Split scheduler/worker and proxy handshake | Replace with released handshake/result types when all required fields have owners |
| `vllm/pd/kv_params.py` | Split schedulers parse request transfer parameters | Remove after released router/native connector supplies the same validated request contract |
| `vllm/pd/chunk_tracker.py` | Prefill handler tracks per-layer submitted/completed chunks | Remove only when native completion owns identical per-range progress |
| `vllm/pd/prefill_async.py` | Prefill worker task pools and completion statistics | Remove with custom pipeline; retain required statistics in the native adapter only if consumed |
| `vllm/pd/prefill.py` | Decode-side HTTP sender to the custom prefill endpoint | Replace with released router/native request flow, then delete |
| `vllm/pd/proxy.py` | Standalone HTTP proxy/router and its metrics/client pools | Replace with pinned upstream router scenario, then delete rather than keep a legacy mode |
| `vllm/pd/metrics.py` | Split P/D connector metrics callbacks | Migrate only metrics consumed by the released connector; delete with old state owners |
| `sglang/__init__.py` | Package import surface | Keep a small public export surface |
| `sglang/config.py` | `linker.py`; owns model/adapter/representation identity | Keep engine-specific identity; add live-weight invalidation before supporting it |
| `sglang/layout.py` | `plugin.py`/`linker.py`; maps Unified pools to GPU regions | Keep; reject unknown DSA/draft/auxiliary layouts |
| `sglang/linker.py` | `UnifiedRadixCache`; owns OrbitKV query/load/offload queues, registrations and terminal close | Keep as the released external-linker implementation |
| `sglang/events.py` | Graph initialization Hook and `OrbitKVLinker`; owns stable CUDA events and forward activation counters | Keep event lifetime and pre-capture registration; import GPU dependencies only when the backend is selected |
| `sglang/recovery.py` | `plugin.py`; wraps linker/tree and overrides Mamba component checkpoint behavior | Move generic checkpoint lifecycle into released engine components before shrinking; keep safety checks until consumed |
| `sglang/admission.py` | Pending-query admission plus opt-in enqueue preparation; cancellation uses native cache finish/linker release | Replace the remaining two targets with public pending-lookup/enqueue callbacks |
| `sglang/completion.py` | Six Hook-registry targets in native P/D receiver/queue; observes page handoff, DecodeReady, abort/failure/release | Replace with explicit lifecycle callbacks; telemetry must not become release authority |
| `sglang/pd.py` | Plugin; substitutes the native Mooncake transfer class with a TENT adapter | Replace with a released transport factory/backend boundary; keep native bootstrap and request states |
| `sglang/plugin.py` | Backend registration; two ordinary cache Hooks, one optional enqueue Hook, six opt-in P/D Hooks | Keep public registration; remove remaining internal Hooks with consumed replacements |

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
removed. P/D evidence, registered only when TENT is enabled, uses
`MooncakeKVReceiver.send_metadata`, `MooncakeKVReceiver.abort`,
`MooncakeKVReceiver.failure_exception`, `DecodeTransferQueue.add`,
`DecodeTransferQueue._commit_transfer_to_req` and
`DecodeTransferQueue._do_release`. The plugin also replaces the module-level
`MooncakeTransferEngine` class. The Hook registry itself is released; these
targets and the class assignment are version-coupled dependencies. Their
replacement is a backend factory plus enqueue, abort, allocation, DecodeReady,
failure and deferred-release callbacks carrying the same ownership evidence.

The two vLLM monkey patches are also separate. `runtime.py` replaces
`GPUModelRunner.update_requests` for preemption and recurrent restore ordering;
`scheduler.py` replaces one `BlockPool.get_cached_block` method on the bound pool
to suppress unsafe partial HMA hits. Neither is replaced merely by registering
the connector in the public factory.

## Configuration inventory

The 0.30.0 upgrade must parse and validate these once, without old/new aliases.

| Owner | Current entries |
| --- | --- |
| vLLM cache endpoint/session | `orbitkv.host`, `orbitkv.port`, `orbitkv.bootstrap_socket`, `orbitkv.tp_shard_endpoints`, `orbitkv.tp_shard_bootstrap_sockets`, `orbitkv.timeout_ms`, `orbitkv.spin_iterations`; environment overrides `ORBITKV_HOST`, `ORBITKV_PORT`, `ORBITKV_INSTANCE_ID` |
| vLLM cache behavior and identity | `orbitkv.mode`, `orbitkv.transfer_backend`, `orbitkv.wait_for_full_prefix`, `orbitkv.pd_tail_save`, `orbitkv.pd_tail_load`; `ORBITKV_CROSS_LAYER_BLOCKS`, `ORBITKV_LOAD_TIMEOUT_SECONDS`, `ORBITKV_PREPARE_REQUESTS`, `ORBITKV_QUEUE_WARMUP`, `PYTHONHASHSEED`, `CUDA_VISIBLE_DEVICES` |
| vLLM custom P/D connector | `orbitkv.pd.mooncake.bind_host`, `orbitkv.pd.mooncake.rank_map`, `orbitkv.pd.prefill_tp_size`, `orbitkv.pd.prefill_sender_worker_count`, `orbitkv.pd.push_worker_count`, `orbitkv.pd.push_finalizer_worker_count`, `orbitkv.pd.validate_runtime_layout`, `orbitkv.pd.completion_observation_socket`, `orbitkv.pd.completion_observation_instance_id` |
| vLLM custom P/D request fields | Consumer: `do_remote_prefill`, `prefill_url`, `remote_request_id`, `done_request_id`, `prefill_max_tokens`, `proxy_start_ts_ns`. Producer: `do_remote_prefill_sender`, `target_engine_id`, `target_request_id`, `pd_handshakes`, `pd_consumer_abort_returns_ack` |
| vLLM custom P/D proxy CLI | `--listen-host`, `--listen-port`, `--prefill-url`, `--decode-url`, `--prefill-urls`, `--decode-urls`, `--routing-policy`, `--timeout-s`, `--prefill-max-tokens`, `--decode-warmup-connections`, `--log-file` |
| SGLang | `ORBITKV_SGLANG_ENDPOINT`, `ORBITKV_TRANSFER_BACKEND`, `ORBITKV_PREPARE_REQUESTS`, `ORBITKV_QUEUE_WARMUP`, `ORBITKV_SGLANG_TENT`, `ORBITKV_SGLANG_TENT_TIMEOUT_S`, `SGLANG_ENABLE_FAILED_SESSION_PROBE`, `MC_FORCE_TCP`; standard SGLang flags select the plugin/backend and native P/D mode |

`PYTHONHASHSEED` is part of cache identity and is required by partial-tail reuse;
`CUDA_VISIBLE_DEVICES` affects GPU ordinal/UUID resolution. Preserve both inputs
until their consumers have an explicit replacement. The proxy CLI and request
fields retire with the custom router/control plane. P/D worker counts, rank maps
and completion-observation settings retire with the custom connector unless a
released TENT backend still consumes them. Cache Manager socket, timeout,
physical backend and model/adapter identity configuration remain OrbitKV
responsibilities.

## Upgrade and contribution sequence

1. Upgrade vLLM as a standalone commit: set the optional dependency to `0.30.0`,
   update the lock and submodule together, and adapt only to the released API.
   Do not change the public support matrix until source-only, installed-wheel,
   cold/partial/full, restart, eager/graph and overhead gates pass.
2. Move completion reporting to `KVConnectorTransferResults`. Use `pre_forward`
   only for load submission proven safe at that point. Retain preemption drain
   before `update_requests` and restore after page initialization/COW but before
   `preprocess_state`; propose those two narrow ordering boundaries upstream.
3. Replace the blanket HMA block-pool override with released divergent-hit and
   all-group readiness only after dense+recurrent state is proven atomic. Keep
   the property false until then. Mark ordinary cache delivery best effort and
   P/D producer delivery reliable; test their any/all aggregation in
   `MultiConnector`.
4. Upstream one vLLM registration/configuration/tests/docs change, then any
   generic recurrent-ordering or TENT transport work separately. Record submitted,
   merged and released states separately.
5. For SGLang, upstream external-linker construction/registration separately
   from request lifecycle and TENT transport. Replace each internal Hook with a
   named lifecycle callback and the class substitution with a transport factory.
6. Only after native vLLM P/D plus TENT passes layout, hybrid, cancellation,
   completion and cache-composition gates may the custom `vllm/pd/` controller,
   HTTP sender and proxy be deleted in the same change.

No adapter or P/D implementation is removed by this audit. vLLM 0.29.0 remains
the only qualified vLLM baseline, and SGLang 0.5.20 remains the current SGLang
baseline. CUDA compute failed in the current container, so no new engine cell is
qualified here.
