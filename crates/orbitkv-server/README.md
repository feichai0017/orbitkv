# OrbitKV Cache Manager

This crate runs the cache engine and exposes it to inference processes. The
default single-node path uses `endpoint/` for iceoryx2 commands and UDS lifecycle
frames. `endpoint/pending.rs` owns in-flight query
state; `cache/lifecycle.rs` serializes
registration and cleanup. `wire.rs` converts shared protobuf registration
messages into cache-layer inputs. The distributed listener serves only peer
transfer control RPCs. Mooncake transfers KV bytes between nodes.

## Building

The binary embeds CPython via PyO3 so it can reconstruct registered CUDA tensors with Torch. Before running cargo commands, point PyO3 to the exact interpreter you want (usually the repo's `.venv`) so linking works and the runtime can import `orbitkv.client.gpu`:

```bash
# Explicitly set your Python interpreter path if needed:
export PYO3_PYTHON="$(pwd)/.venv/bin/python"

export PYTHONPATH="$(pwd)/python:$PYTHONPATH"

cargo run -r --bin orbitkv-cache-manager -- --pool-size 30gb
```

Adjust the Python path if your venv uses a different minor version.

For peer control, configure `--etcd-endpoints`, `--node-id`, matching `--catalog-nodes`, and a routable `--addr`. The embedded catalog shares the peer port.
`--devices` selects CUDA device IDs; omitting it detects available devices
automatically. Keep the Python extension and Cache Manager from
the same build because the bootstrap protocol is versioned.
