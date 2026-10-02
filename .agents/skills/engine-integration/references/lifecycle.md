# Trace engine state recovery

All paths below are relative to the repository root. Start with the actual
adapter and pinned engine; search function names instead of assuming old logs.

| Boundary | Entry point and evidence |
| --- | --- |
| Scheduler query | `python/orbitkv/vllm/scheduler.py`: `get_num_new_matched_tokens`, query deferral, common prefix and recovery boundary |
| TP common coverage | `python/orbitkv/vllm/tp_shards.py`: `query_prefetch` returns `QueryLoading` or `QueryReady`; verify each retained lease and common prefix |
| Allocation handoff | Scheduler `update_state_after_alloc` and `build_connector_meta`; inspect `LoadIntent` in `metadata.py` |
| GPU submission | Worker `start_load_kv` calls native `start_restore` and `wait_restore_enqueued`; retain `RestoreTask` and destination ownership |
| Compute readiness | Worker `wait_for_layer_load` and forward hooks; follow the group/layer mapping and eager or graph stream dependency |
| Terminal ownership | Worker final Restore wait and `crates/orbitkv-channel/src/cache_client.rs`; enqueue, per-layer readiness, final completion and source retirement are distinct |
| Save acknowledgement | Worker `get_finished` reports completed sends; in the current cache connector it returns no receive-completion set |

Read `docs/engine-local-restore.md` for coarse dependencies in recurrent,
packed cross-layer and multipart layouts. Pending query coordination lives in
`crates/orbitkv-server/src/endpoint/pending.rs`, not an old Python engine facade.

For a preemption/restart failure, follow the pinned vLLM scheduler under
`third-party/vllm/vllm/v1/core/sched/` and correlate page ownership with the native
operation. Distinguish a cache miss from a failed submitted copy. A stale lease,
registration change or canceled request must not release an allocation before
its GPU/native work is drained. Check unrelated requests still make progress.

## SGLang

Trace `plugin.py:create_cache` into native tree match and
`RecoveryLinkerWrapper.match/load_back`, then `OrbitKVLinker.lookup/load` and
`start_layer_wise_loading`. Track tree locks, full/window/checkpoint boundaries,
actual pool indices and rank-common agreement before diagnosing a byte copy.
Follow the layer counter and native Restore final wait separately. On abort or
reset, inspect queued-load cancellation and submitted-work drain before releasing
request/tree slots. Generic checkpoint operations belong to engine components;
OrbitKV's compiled recovery requirements do not allocate the engine's pages.

## P/D

Official vLLM NIXL/MultiConnector and SGLang native disaggregation own live P/D.
OrbitKV's cache is independent: P may restore, D saves completed state while
native P/D owns incoming writes. See `docs/pd.md` for the candidate support gates.
Trace producer CUDA readiness, authorized decoder ranges, native transfer
completion, rank agreement and engine readiness. Engine metrics and output
alone cannot prove remote writes have drained. No fork transport adapter or
private P/D observation Hook remains in the wheel.

These paths describe the present implementation. Re-audit them against the
selected release before replacing internal hooks or adopting newer result APIs.
