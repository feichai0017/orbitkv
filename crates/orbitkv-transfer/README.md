# OrbitKV Transfer

`orbitkv-transfer` is OrbitKV's single remote byte-movement implementation. It
wraps a pinned upstream Mooncake Transfer Engine and contains no private verbs
stack.

OrbitKV owns state identity, recovery completeness, replica choice, leases, and
generation validation. Mooncake owns segment discovery, memory registration,
RDMA/TCP selection, multi-rail routing, transfer completion, and notification.

```text
OrbitKV authorized transfer plan
        │
        ▼
Mooncake segment + remote address + batch operation
        │
        ├── RDMA / GPUDirect when available
        └── TCP fallback or forced TCP for validation
```

The public Rust API is deliberately small:

- `TransferEngine::new` creates one upstream Mooncake engine.
- `register_memory` and `unregister_memory` expose stable host or device memory.
- `submit_and_wait` executes a READ or WRITE batch.
- `submit_and_notify` couples a batch to a peer notification.
- `take_notifications` and `send_notification` implement P/D completion and
  failure signals.

The upstream source is pinned by the `third-party/mooncake` submodule.
`orbitkv-mooncake-provider` builds and stages `libtransfer_engine.so`,
`libmooncake_common.so`, and `libasio.so`; the libraries use an `$ORIGIN`
runpath so the three files can be bundled together in a wheel or container.

## Build and smoke test

Initialize the nested source dependencies once:

```bash
git submodule update --init --recursive third-party/mooncake
```

Then run:

```bash
cargo test -p orbitkv-transfer --no-default-features --features cuda-13
```

The host-safe smoke test forces Mooncake TCP and verifies registration, segment
opening, WRITE completion, and byte equality. RDMA/GPUDirect qualification is a
separate hardware gate.
