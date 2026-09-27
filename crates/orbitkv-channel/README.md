# orbitkv-channel

Versioned request/response channel between inference processes and a Cache Manager
on the same host. The control message is fixed at 64 bytes; KV payloads remain
in registered CUDA IPC pages or Cache Manager storage. The channel does not
select RAM, SSD, or a remote replica: the Cache Manager resolves each cache
request and uses Mooncake when a remote fetch is available.

The production server owns a thread-safe iceoryx2 service and currently serves
`Ping`, `QueryBundle`, `Publish`, `Restore`, `Release`, and `Shutdown`, including
session-epoch fencing. A mode-0600 Unix socket verifies peer credentials and
passes sealed descriptor and restore-result memfds
plus a notification eventfd. Every client owns one arena slot guarded by a client
token and a monotonic request/response generation. Python exposes diagnostics
through `ChannelProbeClient` and cache operations through the PyO3
`CacheManagerClient`. Rust `CacheClient` owns query revisions, warming interests,
an independent publish connection and restore handles bound to their issuer.
Restore uses a generation-tagged shared result slot plus the bootstrapped eventfd
with bounded fallback polling. The native client consumes and acknowledges the
result without a terminal Poll RPC. An outcome waiter publishes directly after
GPU drain instead of waiting for a dispatcher scan. Waiting releases the GIL and never blocks the Cache Manager dispatcher;
a deadline does not release GPU destinations. The low-level Rust `ChannelClient`
owns descriptor framing and session failure, without a second Python facade.

Bootstrap version 3 and channel ABI 6 require rebuilding both client and Manager.
Each result mapping has 1024 records and at most 4096 error bytes per record;
mapping admission includes disconnected sessions retained by outstanding work.

Run the two-process latency harness in separate terminals:

```bash
cargo run -r -p orbitkv-channel --bin orbitkv-channel-echo -- orbitkv/bench
cargo run -r -p orbitkv-channel --bin orbitkv-channel-bench -- orbitkv/bench 100000
```

The benchmark client sends `Shutdown` after the run so the echo process exits.
