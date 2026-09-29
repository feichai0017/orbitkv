# Trace a vLLM cache load

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
