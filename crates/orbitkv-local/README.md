# orbitkv-local

Versioned local request/response transport between inference processes and an
OrbitKV sidecar. The control message is fixed at 64 bytes; KV payloads remain
in CUDA IPC or registered shared-memory pages.

The current crate is a transport foundation. The production sidecar handlers
and Python adapter bindings are tracked in the repository TODO.

Run the two-process latency harness in separate terminals:

```bash
cargo run -r -p orbitkv-local --bin orbitkv-local-echo -- orbitkv/bench
cargo run -r -p orbitkv-local --bin orbitkv-local-bench -- orbitkv/bench 100000
```

The benchmark client sends `Shutdown` after the run so the echo process exits.
