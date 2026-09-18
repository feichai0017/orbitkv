---
name: vllm-async-loading-debug
description: Debug vLLM async KV loading and OrbitKV connector behavior. Use when investigating async KV loads, WAITING_FOR_REMOTE_KVS states, load/save intents, connector metadata flow, prefetch behavior, or preemption interactions in vLLM scheduler/worker code.
---

# Vllm Async Loading Debug

## Overview

Trace the async KV loading path between vLLM scheduler and worker connectors, including prefetch and preemption edges.

## Workflow

### 1) Confirm async-load symptom

- Record request IDs and statuses: `WAITING`, `RUNNING`, `WAITING_FOR_REMOTE_KVS`, `PREEMPTED`.
- Check async-load logs and whether prefetch is in progress.

### 2) Follow scheduler-side decisions

- `python/orbitkv/connector/scheduler.py`: prefetch query + `LoadIntent` creation.
- `.project-plans/scheduler.py` (private): `load_kv_async` gating and `WAITING_FOR_REMOTE_KVS` transitions.

### 3) Follow worker-side lifecycle

- `python/orbitkv/connector/worker.py`: `start_load_kv()` calls `engine_client.load()` and tracks `PyLoadState`.
- `get_finished()` polls `is_ready()` and emits `finished_recving`.

### 4) Check preemption and retries

- `.project-plans/scheduler.py` (private): `_preempt_request()` and `reset_prefix_cache()` behavior.
- `invalid_block_ids` handling can trigger recompute or failure based on `kv_load_failure_policy`.

## Reference Files

- Read `references/async-loading.md` for full call flow and log markers.
