---
name: vllm-async-loading-debug
description: Diagnose OrbitKV vLLM query deferral, prefix leases, Restore readiness, save completion, preemption and page reuse. Use for async cache-loading or scheduler/worker lifecycle failures.
---

# vLLM loading diagnosis

Read `AGENTS.md` and `docs/vllm-request-state-machine.md` from the Git root.
Use the checked-out `third-party/vllm` version, not a private scheduler snapshot.
Read [the tracing guide](references/async-loading.md) when following a request.

Identify which boundary is stuck: query admission, remote/SSD materialization,
lease handoff, Restore enqueue, layer readiness, final drain, or source retirement.
Track request/operation identities, registration generations and owned bytes.
Keep logs outside the checkout and reproduce with frozen native artifacts.

Treat query deferral, readiness for computation and terminal completion as
separate facts. Do not assume an upstream `WAITING_FOR_REMOTE_KVS` mechanism is
used by this connector, or change `get_finished()` to report a load merely because
an enqueue acknowledgement arrived. Establish the consumed callback contract.

Regress the relevant cancellation, preemption, graph/eager and page-reuse case,
then rerun the pinned vLLM correctness gate. A timeout must not free destinations
that a submitted native transfer can still write.
