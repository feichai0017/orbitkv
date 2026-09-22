# Cache Manager Configuration

## OrbitKV Cache Manager

```bash
orbitkv-cache-manager
```

### Options

- `--addr`: Peer control bind address in distributed mode and local socket port seed (default: `127.0.0.1:50055`)
- `--channel-service`: optional iceoryx2 name prefix. Every Manager startup appends a unique incarnation and advertises it through the stable UDS bootstrap socket. Old clients cannot block a restart by retaining the previous service.
- `--devices`: CUDA device IDs to initialize, comma-separated (default: auto-detect all available GPUs, e.g., `--devices 0,1,2,3`)
- `--pool-size`: Pinned memory pool size (default: `30gb`, supports: `kb`, `mb`, `gb`, `tb`)
- `--hint-value-size`: Hint for typical value size to tune cache and allocator (optional, supports: `kb`, `mb`, `gb`, `tb`)
- `--use-hugepages`: Use huge pages for pinned memory (default: `false`, requires pre-configured `/proc/sys/vm/nr_hugepages`)
- `--enable-lfu-admission`: Enable TinyLFU cache admission policy (default: plain LRU)
- `--disable-numa-affinity`: Disable NUMA-aware memory allocation (default: enabled)
- `--blockwise-alloc`: Allocate each block separately instead of contiguous batch allocation. Reduces memory fragmentation when blocks are freed in different order (default: `false`)
- `--log-level`: Log level: `trace`, `debug`, `info`, `warn`, `error` (default: `info`)

### HTTP & Metrics

- `--http-addr`: HTTP server address for health check and Prometheus metrics (default: `0.0.0.0:9091`, always enabled)
- `--enable-prometheus`: Enable Prometheus `/metrics` endpoint (default: `true`)
- `--metrics-otel-endpoint`: OTLP metrics export endpoint (optional, leave unset to disable)
- `--metrics-period-secs`: Metrics export period in seconds (default: `10`, only used with OTLP)

### HTTP Endpoints

- `GET /health`: Health check.
- `GET /metrics`: Prometheus metrics, when `--enable-prometheus` is enabled.
- `GET /instances`: List registered instance IDs.
- `POST /instances/cleanup[?id=<instance_id>]`: Remove one instance, or all instances when `id` is omitted.
- `POST /cache/memory/cleanup`: Evict resident in-memory cache blocks while preserving backing-store data. `evicted_bytes` is the cache footprint removed from residency; `reclaimed_bytes` is the pinned-pool memory actually released immediately.

### SSD Cache

- `--ssd-cache-path`: Enable SSD cache by providing cache file path (optional)
- `--ssd-cache-capacity`: SSD cache capacity (default: `512gb`, supports: `kb`, `mb`, `gb`, `tb`)
- `--ssd-write-queue-depth`: SSD write queue depth, max pending write batches (default: `8`)
- `--ssd-prefetch-queue-depth`: SSD prefetch queue depth, max pending prefetch batches (default: `2`)
- `--ssd-write-inflight`: SSD write inflight, max concurrent block writes (default: `2`)
- `--ssd-prefetch-inflight`: SSD prefetch inflight, max concurrent block reads (default: `16`)
- Full read queues wait for capacity while retaining the query's byte reservation.
  Concurrent queries for an identical prefix and storage identity can share a read.

### Query ownership budgets

- `--query-budget`: Maximum query-owned bytes across preparation, ready leases,
  and submitted GPU loads. Defaults to 75% of `--pool-size`.
- `--query-instance-budget`: Limit for one registered instance, shared across its
  client sessions. Defaults to the global query budget.

Both accept the same memory units as `--pool-size`, with
`0 < instance budget <= global budget <= pool size`. Reservations use the
registered storage group's padded block bytes, including its physical layout
and shards. Partial results shrink their reservation. Each owner is charged
independently even when pages or reads are shared; the pinned allocator tracks
physical use separately. The resident cache can still use the full pool.

Temporary budget pressure leaves a query pending. A single query larger than
the configured limit returns an empty restore result and increments
`orbitkv_query_budget_bypasses_total`; it does not wait indefinitely or record
an authoritative backing-store miss. Cancellation drains submitted I/O before
returning its bytes. A delivered lease remains charged until release, session
disconnect, expiry, or completion of every GPU consumer.

`orbitkv_query_reserved_bytes{phase="preparing|ready|restoring"}` tracks ownership
through these phases. Preparation includes queued reads and reconstruction;
it is a conservative payload reservation, not a measurement of a particular
device's buffers. `orbitkv_query_budget_waits_total` counts admission attempts
delayed by bytes, and `orbitkv_query_coalesced_reads_total` counts joined reads.
The old block-count prefetch limit has been removed.

### Cross-Node (Multi-Node Setup)

- `--nics`: Optional Mooncake RDMA rail allow-list (e.g., `--nics mlx5_0,mlx5_1` or `--nics mlx5_0 mlx5_1`). Omit it to let Mooncake select the available transport, including TCP fallback.
- `--etcd-endpoints`: comma-separated HTTP etcd endpoints; enables distributed cache. Requires `--node-id` and `--catalog-nodes`. Peers must reach the concrete `--addr` endpoint.
- `--catalog-nodes`: identical set of 1–16 stable catalog host Node IDs on every Manager. Placement is immutable; incompatible joins fail. Missing members do not remap shards.
- `--catalog-budget`: accounted index and retained retry bytes across this Manager's assigned shards; defaults to 256 MiB. This is not a process RSS cap.
- `--cluster-name`: etcd namespace, default `orbitkv`. `--membership-ttl-secs` defaults to 30 and accepts 12–3600; remote admission uses half the acknowledged TTL. See [deployment](p2p.md#leased-manager-membership).
- `--transfer-lock-timeout-secs`: Transfer lock timeout in seconds (default: `120`). Blocks held for a Mooncake transfer are locked for at most this duration before being force-released.
- `--inventory-journal-bytes`: Retained residency-change bytes (default: `16777216`, 16 MiB). Lag beyond this history triggers a paginated inventory resnapshot.

## Embedded catalog

The same Cache Manager binary hosts its assigned directory shards on the peer
gRPC port. There is no directory executable or separate HTTP service. Start all
Managers with the same etcd namespace and catalog host set:

```bash
orbitkv-cache-manager --addr <routable-ip>:50055 \
  --etcd-endpoints http://<etcd-host>:2379 \
  --node-id cache-a --catalog-nodes cache-a,cache-b
```

Use the other host's address and Node ID there. Catalog placement is immutable in
this stage; each shard has one metadata copy. See [deployment and failure
behavior](p2p.md) and the [recovery protocol](../crates/orbitkv-catalog/README.md).
