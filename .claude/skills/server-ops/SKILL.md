---
name: server-ops
description: >
  Use when configuring, deploying, or troubleshooting orbitkv-server/orbitkv-catalog.
  Covers CLI flags, pool sizing, SSD cache, NUMA, metrics, and multi-node setup.
---

# Server Operations

## Starting the Server

```bash
# Auto-detect all GPUs (default)
cargo run -r --bin orbitkv-server -- --addr 0.0.0.0:50055 --pool-size 30gb

# Specify devices
cargo run -r --bin orbitkv-server -- --addr 0.0.0.0:50055 --devices 0,2,4 --pool-size 30gb
```

## Running Examples

```bash
uv run python examples/basic_vllm.py
uv run python examples/bench_kv_cache.py --model /path/to/model --num-prompts 10
```

## CLI Flags — orbitkv-server

| Flag | Default | Description |
|---|---|---|
| `--addr` | `127.0.0.1:50055` | Bind address |
| `--devices` | auto-detect all | CUDA device IDs, comma-separated |
| `--pool-size` | `30gb` | Pinned memory pool size |
| `--hint-value-size` | — | Hint for typical value size to tune cache/allocator |
| `--use-hugepages` | `false` | Use huge pages (requires `/proc/sys/vm/nr_hugepages`) |
| `--enable-lfu-admission` | `false` | Enable TinyLFU admission (default: plain LRU) |
| `--disable-numa-affinity` | `false` | Disable NUMA-aware allocation |
| `--http-addr` | `0.0.0.0:9091` | HTTP server for health check and Prometheus |
| `--enable-prometheus` | `true` | Enable `/metrics` endpoint |
| `--metrics-otel-endpoint` | — | OTLP metrics export endpoint |
| `--metrics-period-secs` | `5` | Metrics export period (OTLP only) |
| `--log-level` | `info` | `trace`/`debug`/`info`/`warn`/`error` |
| `--ssd-cache-path` | — | Enable SSD cache with file path |
| `--ssd-cache-capacity` | `512gb` | SSD cache capacity |
| `--ssd-write-queue-depth` | `8` | Max pending write batches |
| `--ssd-prefetch-queue-depth` | `2` | Max pending prefetch batches |
| `--ssd-write-inflight` | `2` | Max concurrent block writes |
| `--ssd-prefetch-inflight` | `16` | Max concurrent block reads |
| `--max-prefetch-blocks` | `800` | Backpressure for SSD prefetch |
| `--trace-sample-rate` | `1.0` | Sampling rate 0.0–1.0 (requires `--features tracing`) |
| `--etcd-endpoints`, `--node-id`, `--catalog-nodes` | — | Distributed membership and immutable catalog placement; requires a concrete peer `--addr` |

## Key Files

- `crates/orbitkv-server/src/lib.rs`: Manager startup and embedded peer services
- `crates/orbitkv-server/src/bin/orbitkv-router.rs`: P/D request router
- `crates/orbitkv-server/src/cluster/`: etcd registration and Watch
- `crates/orbitkv-catalog/src/service.rs`: Catalog gRPC service
- `crates/orbitkv-catalog/src/store.rs`: Multi-owner block hash store with TTL sweep (backed by DashMap)

## Environment Variables

- `ORBITKV_ENGINE_ENDPOINT`: gRPC endpoint (default: `127.0.0.1:50055`)
- `RUST_LOG`: Rust logging (e.g., `info,orbitkv_core=debug,orbitkv_server=debug`)
