# OrbitKV P/D Transfer with Mooncake

OrbitKV's P/D connector streams each completed prefill layer directly into the
decode worker's KV pages. The layout and request state machine remain owned by
OrbitKV; all remote byte movement is performed by the pinned upstream Mooncake
Transfer Engine.

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
- `take_notifications` drives the decode-side waiter.

Layer writes execute on the connector's existing background worker pool, so
prefill computation can continue while earlier layer tasks move data. The
first implementation waits inside each worker task for its Mooncake batch; a
future optimization may retain batch handles and poll completions without
changing the wire contract.

## Configuration

Each TP rank configures a routable bind host and may select an RDMA NIC. When
`rank_map` is omitted, Mooncake chooses the available transport and may use TCP:

```json
{
  "kv_connector": "PdConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "orbitkv.vllm.pd",
  "engine_id": "d0",
  "kv_connector_extra_config": {
    "orbitkv.pd.mooncake.bind_host": "10.0.0.2",
    "orbitkv.pd.mooncake.rank_map": {
      "0": {"nic": "mlx5_0"}
    }
  }
}
```

Mooncake uses `P2PHANDSHAKE` for peer metadata exchange. No external Mooncake
Store or metadata service is required for this path. OrbitKV does not use
Mooncake Store as its state authority.

## Safety contract

- The destination advertises only ranges allocated for the request.
- A transfer notification is accepted only for the matching request ID.
- The decode side waits for the expected number of producer notifications.
- Failure/abort notifications never publish the destination as complete.
- Page reuse still requires OrbitKV's semantic and execution frontiers.

## Qualification gates

- host-safe Mooncake loopback byte correctness;
- GPU memory registration and GPUDirect WRITE correctness on the H20 host;
- same-TP and heterogeneous-TP P/D correctness;
- cancellation, timeout, and peer-restart behavior;
- throughput and TTFT comparison with vLLM's supported NIXL connector.

The Rust/Python compile gates and host loopback currently pass. GPU and
cross-machine P/D qualification remain required before this path is called
production-ready.
