# Experimental vLLM P/D transfer with Mooncake

OrbitKV's vLLM-only P/D connector streams each completed prefill layer directly into the
decode worker's KV pages. The layout and request state machine remain owned by
OrbitKV; all remote byte movement is performed by the pinned upstream Mooncake
TENT engine.

## Data and control flow

```text
Decode worker                         Prefill worker
-------------                         --------------
allocate KV pages
register pages with Mooncake
publish endpoint + allowed layout ---> validate request and layout
                                      compute layer i
                                      Mooncake batch WRITE --------> KV pages
Mooncake notification <-------------- done / failed / aborted
start decode after all producers
```

The P/D HTTP request remains the out-of-band framework control channel. Its
handshake contains: request identity, TP rank/size, destination Mooncake
endpoint, authorized block IDs, and per-layer address/stride geometry. It does
not contain rkeys, QP descriptors, or an RDMA-IMM identifier.

The native boundary exposed to Python is `MooncakeTransferEngine`:

- `endpoint` advertises the local Mooncake segment name;
- `register_memory` registers long-lived vLLM KV tensors once;
- `write` submits and waits for a batch of source/destination ranges;
- `send_notification` emits request completion or failure;
- `open_notification_scope`, `wait_for_status` and
  `close_notification_scope` keep notification polling, counting and
  close/reopen generation fencing in Rust while the GIL is released;
- `nic_load_stats` exposes TENT rail inflight bytes and EWMA bandwidth for
  diagnostics, without claiming which transport a specific batch used.

Layer writes execute on the connector's existing background worker pool, so
prefill computation can continue while earlier layer tasks move data. The
first implementation waits inside each worker task for its Mooncake batch; a
future optimization may retain batch handles and poll completions without
changing the wire contract.

## vLLM 0.29 layout and callback contract

Registration consumes the pinned runner's four-dimensional
`[blocks, heads, states, content]` view. Attention content packs K and V
in one region; MLA and indexer caches use the same raw registration shape.
The adapter validates the shape against `KVCacheSpec`, dense HNC inner
strides, non-overlapping pages, and the actual backing-storage bounds.
It copies content bytes only, preserving page padding and the physical
block stride. Kernel-block splitting is rejected explicitly.
The pinned MLA attention callback squeezes its singleton head axis;
that callback view must retain the registered strides and base address.

Prefill prepares each step's sends in `bind_connector_metadata`, before
forward. The V2 runner can call `start_load_kv` after forward for steps with
no synchronous loads; using that callback to prepare a layerwise send misses
the current step. Layer callbacks record CUDA events, and sender tasks wait
for the event before reading the source pages. Prefill requires piecewise
CUDA graphs so those callbacks are not skipped during full-graph replay.

The decode waiter is the sole publisher of receive completion. The TENT
port validates the native terminal notification but does not independently
mark pages ready for the scheduler. This prevents a second completion queue
from racing the decode owner's completion processing. Writes validate each
destination block grant and source registration bounds before admission;
source and destination page strides may differ.

## Configuration

Each TP rank configures a routable bind host and may select an RDMA NIC. When
`rank_map` is omitted, Mooncake chooses the available transport and may use TCP.
Keys are **TP ranks**, never CUDA ordinals or host GPU indices. A TP=1 replica
always uses key `"0"`, including when `CUDA_VISIBLE_DEVICES` selects GPU 4.
An explicit map must contain that rank and a nonempty NIC. OrbitKV reads the
pinned vLLM `kv_connector_extra_config` directly; there is no legacy
`extra_config` field or NIXL host-environment alias. TENT is the required backend,
so no backend-enabled switch is needed.

Example:

```json
{
  "kv_connector": "PdDecodeConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "orbitkv.vllm.pd",
  "engine_id": "d0",
  "kv_connector_extra_config": {
    "orbitkv.pd.mooncake.bind_host": "10.0.0.2",
    "orbitkv.pd.completion_observation_socket": "/run/orbitkv/orbitkv.sock",
    "orbitkv.pd.completion_observation_instance_id": "decode-instance",
    "orbitkv.pd.mooncake.rank_map": {
      "0": {"nic": "mlx5_0"}
    }
  }
}
```

Use `PdPrefillConnector` in the prefill process. The former role-selecting
`PdConnector` facade is not retained; `engine_id` identifies the instance and
no longer chooses connector behavior.

The two completion-observation settings are optional and decode-only. When
present, the named instance/device must already be registered with that
node-local Manager. The authenticated process channel consumes a bounded report
after the decode-side TENT waiter reaches a terminal state. Reporting failure
disables later reports in that worker but never changes P/D completion or page
ownership. The transfer generation is validated as nonzero freshness evidence
and is deliberately excluded from estimator keys; vLLM supplies its TENT
notification-scope generation.

Mooncake uses `P2PHANDSHAKE` for peer metadata exchange. No external Mooncake
Store or metadata service is required for this path. OrbitKV does not use
Mooncake Store as its state authority.

## Safety contract

- The destination advertises only ranges allocated for the request.
- A transfer notification is accepted only for the matching request ID.
- Closing or replacing a request generation wakes its old native waiter without
  allowing it to complete the replacement.
- The decode side waits for the expected number of producer notifications.
- An optional Cache Manager observation is emitted only by that decode-side
  completion owner; source-side write return is not relabelled as decode-ready.
- vLLM captures its generation-fenced waiter depth and live TENT rail snapshot.
  SGLang captures its admitted decode queue in `DecodeTransferQueue.add` and
  reports success only after `_commit_transfer_to_req` passes metadata and
  optional HiCache restore gates.
- Failure/abort notifications never publish the destination as complete.
- Queued writes carry the producer request generation and captured destination
  authorization. Reusing a request ID cannot redirect an old task into new
  pages. Overlapping chunks share a generation only while their authorization
  and target mapping remain unchanged.
- Producer release waits for admitted writes to drain even after another task
  reports an error; an error alone does not permit retiring authorization.
- vLLM retains ownership of source/destination HBM pages; OrbitKV's proposed
  semantic and execution frontiers are not enforced by the current connector.

## Qualification gates

- host-safe Mooncake loopback byte correctness;
- GPU memory registration and GPUDirect WRITE correctness on the H20 host;
- same-TP and heterogeneous-TP P/D correctness;
- cancellation, timeout, and peer-restart behavior;
- throughput and TTFT comparison with vLLM's supported NIXL connector.

The September 28, 2026 Qwen3-8B qualification used vLLM 0.29.0,
bfloat16, TP=1, 64-token pages, FlashAttention 2, eager execution, a 576 MiB
KV budget per replica, and greedy 16-token outputs with EOS ignored. Three
natural-language prompts of 129, 513 and 1025 tokens required information
from the first cache page. The model revision was
`b968826d9c46dd6066d109eabc6255188de91218`.

| Deployment | Actual KV transfer | Complete output versus A100 monolithic | Completion |
| --- | --- | --- | --- |
| Two replicas on one A100 | 261 MiB TENT TCP WRITE | 3/3 identical | Sends, finalizers and waits drained |
| H20 prefill to A100 decode | 261 MiB TENT TCP WRITE; 1044 GPU ranges have identical source/destination SHA-256 | 1/3 identical; all three retrieve the requested key | Sends, finalizers and waits drained |

The byte diagnostic reads source GPU pages after their CUDA events and target
GPU pages after TENT completion, before the Decode owner publishes readiness.
An earlier diagnostic incorrectly raced decode writes because the port had a
second receive-completion queue; that queue has been removed. The final byte
comparison includes the transferred partial tail pages. Hashing synchronizes
GPU reads, so this run is correctness evidence, not an overlap or latency
benchmark. Raw logs and launch commands are under
`/root/orbitkv-artifacts/s1-evidence-20260929/legacy-results/runs/two-host-natural-20260928/vllm-pd-same-a100/` and
`vllm-pd-byte-probe/`.

The heterogeneous strict-output gate **fails** and is not waived by byte
correctness. Identical transported bytes and the same-A100 control narrow the
remaining discrepancy to computation/configuration across the two GPU types;
they do not establish a general model-quality tolerance. Multi-GPU TP/PP,
hybrid-model handoff, full cancellation/restart qualification, CUDA-graph
replay, and RDMA/GPUDirect still require their own execution gates. Neither
available container exposes `/dev/infiniband`.
