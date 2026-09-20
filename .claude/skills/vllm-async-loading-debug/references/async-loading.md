# vLLM Async KV Loading Reference

## Primary Files

- `python/orbitkv/vllm/scheduler.py`
- `python/orbitkv/vllm/worker.py`
- `.project-plans/scheduler.py`

## End-to-End Async Load Flow (KV Connector)

1) Scheduler checks cache hit and prefetch state
- `SchedulerConnector.get_num_new_matched_tokens()` calls `_count_available_block_prefix()`.
- If `prefetch_state == "loading"`, returns `None` to defer scheduling.
- Otherwise returns `hit_blocks` and logs hits.

2) Scheduler allocates blocks and builds load intents
- `update_state_after_alloc()` consumes `LoadIntent` from `RequestTracker` and stores it in `_pending_load_intents`.
- `build_connector_meta()` attaches `load_intents` and `save_intents` to `SchedulerOutput`.

3) Scheduler schedules async load
- In `.project-plans/scheduler.py`:
  - For WAITING requests, `connector.get_num_new_matched_tokens()` may set `load_kv_async`.
  - If `load_kv_async` is true, request transitions to `WAITING_FOR_REMOTE_KVS` and is not scheduled for compute.

4) Worker initiates async load
- `WorkerConnector.start_load_kv()` batches all `block_ids` and `block_hashes` across requests.
- Calls `engine_client.load(...)` with a `PyLoadState` shared-memory token.
- Tracks `self._pending_loads` and `self._pending_load_reqs` by shm name.

5) Worker reports completion
- `WorkerConnector.get_finished()` polls `PyLoadState.is_ready()`.
- When ready, it returns `finished_recving` request IDs; scheduler consumes these in `_update_from_kv_xfer_finished()`.

6) Scheduler resumes WAITING_FOR_REMOTE_KVS requests
- `_update_waiting_for_remote_kv()` moves requests back to `WAITING`.
- If the request was previously preempted, it resumes as `PREEMPTED`.

## Prefetch and Query Behavior

- `SchedulerConnector._count_available_block_prefix()` uses `engine_client.query()`.
- Dict responses include `prefetch_state` and `hit_blocks`.
- `prefetch_state == "loading"` returns `None` (scheduler retries later).
- Service errors mark the connector unavailable and return 0 cache hits.

## Preemption Interactions

- `_preempt_request()` frees KV blocks, resets `num_computed_tokens`, and sets status to `PREEMPTED`.
- Preempted requests are reinserted into the waiting queue.
- When async loads complete, requests are resumed via `WAITING_FOR_REMOTE_KVS -> WAITING` and then scheduled as `PREEMPTED`.
- `reset_prefix_cache(reset_running_requests=True)` preempts all running requests and discards latest async tokens.

## Failure Paths and Retries

- `kv_connector_output.invalid_block_ids` triggers `_handle_invalid_blocks()`.
- Async loads are retried by marking `failed_recving_kv_req_ids`.
- If `kv_load_failure_policy` is `fail`, affected requests finish with errors.

## Log Markers (grep targets)

- `get_num_new_matched_tokens`: "hit_blocks" and "need_to_compute_tokens"
- Prefetch: "Prefetch completed" and `prefetch_state == "loading"`
- Worker load: "started async load"
- Completion: "finished loading KV" and "Finished recving KV transfer"
- Preemption: "Preempted" and "preempted and moved to the waiting queue"
