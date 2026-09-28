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

The Rust/Python compile gates and host loopback currently pass. GPU and
cross-machine P/D qualification remain required before this path is called
production-ready.
