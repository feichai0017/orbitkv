# orbitkv-channel

Versioned request/response channel between inference processes and a Cache Manager
on the same host. The control message is fixed at 64 bytes; KV payloads remain
in registered CUDA IPC pages or Cache Manager storage. The channel does not
select RAM, SSD, or a remote replica: the Cache Manager resolves each cache
request and uses Mooncake when a remote fetch is available.

The production server owns a thread-safe iceoryx2 service and currently serves
`Ping`, `QueryBundle`, `Publish`, `Restore`, `Release`, and `Shutdown`, including
session-epoch fencing. A mode-0600 Unix socket verifies peer credentials and
passes a sealed memfd arena
plus a liveness eventfd. Every client owns one arena slot guarded by a client
token and a monotonic request/response generation. Python exposes diagnostics
through `ChannelProbeClient` and cache operations through `ChannelClient`. Restore uses an
operation ID plus the bootstrapped eventfd so GPU completion never blocks the
Cache Manager's channel thread.

Run the two-process latency harness in separate terminals:

```bash
cargo run -r -p orbitkv-channel --bin orbitkv-channel-echo -- orbitkv/bench
cargo run -r -p orbitkv-channel --bin orbitkv-channel-bench -- orbitkv/bench 100000
```

The benchmark client sends `Shutdown` after the run so the echo process exits.
