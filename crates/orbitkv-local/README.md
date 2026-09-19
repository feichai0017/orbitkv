# orbitkv-local

Versioned local request/response transport between inference processes and an
OrbitKV sidecar. The control message is fixed at 64 bytes; KV payloads remain
in CUDA IPC or registered shared-memory pages.

The production server owns a thread-safe iceoryx2 service and currently serves
`Ping`, `QueryBundle`, `Publish`, `Release`, and `Shutdown`, including
session-epoch fencing. A mode-0600 Unix socket verifies peer credentials and
passes a sealed memfd arena
plus a liveness eventfd. Every client owns one arena slot guarded by a client
token and a monotonic request/response generation. Python exposes lifecycle and
query paths through `LocalControlClient` and `LocalQueryClient`. Restore uses an
operation ID plus the bootstrapped eventfd so GPU completion never blocks the
sidecar's local-control thread.

Run the two-process latency harness in separate terminals:

```bash
cargo run -r -p orbitkv-local --bin orbitkv-local-echo -- orbitkv/bench
cargo run -r -p orbitkv-local --bin orbitkv-local-bench -- orbitkv/bench 100000
```

The benchmark client sends `Shutdown` after the run so the echo process exits.
