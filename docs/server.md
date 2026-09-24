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
- `--cache-protected-percent`: Maximum percentage of pinned pool bytes in the demand-protected replacement segment, `0`–`100` (default: `0`, disabled). See [retention and admission](cache-policies.md).
- `--disable-numa-affinity`: Disable NUMA-aware memory allocation (default: enabled)
- `--blockwise-alloc`: Allocate each layer/page segment independently in DRAM-only mode (default: `false`). SSD-backed Managers always use this allocation policy so reads and writes share the same reclaimable units; surviving prefix pages do not pin other pages from a batch.
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
- `POST /cache/sync`: Wait for already submitted saves and acknowledged catalog residency. Returns 503 on synchronization failure or 504 after 30 seconds; it does not make SSD payloads restart-durable.

### SSD Cache

- `--ssd-cache-path`: Enable SSD cache by providing cache file path (optional)
- `--ssd-cache-capacity`: SSD cache capacity (default: `512gb`, supports: `kb`, `mb`, `gb`, `tb`). Selecting cuFile reserves this physical space before serving; allocation errors fail startup. io_uring uses logical sizing without upfront reservation.
- `--ssd-write-queue-depth`: SSD write queue depth, max pending write batches (default: `8`)
- `--ssd-write-policy`: `all` writes newly saved pages; `reuse` admits foreground-returned pages or repeated publications within a bounded history (default: `all`). Selective admission can require recomputation on the first reuse after DRAM eviction.
- `--ssd-backend`: `auto` (default) tries native cuFile on ext4/XFS and falls back
  to io_uring if capability initialization fails; a cuFile operation failure switches new
  operations to io_uring until Manager restart. `uring` uses pinned DRAM;
  `cufile` writes complete GPU state groups and restores leased SSD sources
  through 8 MiB GPU staging per instance/device and follows NVIDIA's compatibility
  configuration without automatic backend fallback. Native GDS must be [qualified separately](gds.md).
  Fragmented/multi-writer saves and speculative preparation retain io_uring.
  GPU-direct writes hold Publish pages until storage completion.
- `--ssd-prefetch-queue-depth`: SSD prefetch queue depth, max pending prefetch batches (default: `2`)
- `--ssd-write-inflight`: SSD write inflight, max concurrent block writes (default: `2`)
- `--ssd-prefetch-inflight`: SSD prefetch inflight, max concurrent block reads (default: `16`)
- Full io_uring read queues wait for capacity while retaining the query's byte reservation.
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

`orbitkv_query_reserved_bytes` tracks total ownership under the budget lock;
`orbitkv_query_reserved_bytes_by_phase{phase="preparing|ready|restoring"}` diagnoses
the individual stages. Summed phase samples are not an atomic budget snapshot.
Ownership remains charged through these phases. Preparation includes queued
reads and reconstruction;
it is a conservative payload reservation, not a measurement of a particular
device's buffers. `orbitkv_query_budget_waits_total` counts admission attempts
delayed by bytes, and `orbitkv_query_coalesced_reads_total` counts joined reads.
The old block-count prefetch limit has been removed.

Opt-in consumer preparation adds `preloading` and `prepared` phases. Both stay
within the speculative quarter-budget until a foreground claim; expiry never
releases an unfinished I/O's buffers. `--query-read-batch`,
`--query-read-timeout-ms`, and `--query-read-max-batches` control ordinary read
submission and recompute fallback. Defaults preserve ordinary demand behavior.
See [request preparation](request-preparation.md) for limits and control runs.

### Cross-Node (Multi-Node Setup)

- `--nics`: Optional Mooncake RDMA rail allow-list (e.g., `--nics mlx5_0,mlx5_1` or `--nics mlx5_0 mlx5_1`). Omit it to let Mooncake select the available transport, including TCP fallback.
- `--etcd-endpoints`: comma-separated HTTP etcd endpoints; enables distributed cache. Requires `--node-id` and `--catalog-nodes`. Peers must reach the concrete `--addr` endpoint.
- `--catalog-nodes`: identical set of 1–16 stable catalog host Node IDs on every Manager. Placement is immutable; incompatible joins fail. Missing members do not remap shards.
- `--catalog-budget`: accounted index and retained retry bytes across this Manager's assigned shards; defaults to 256 MiB. This is not a process RSS cap.
- `--cluster-name`: etcd namespace, default `orbitkv`. `--membership-ttl-secs` defaults to 30 and accepts 12–3600; remote admission uses half the acknowledged TTL. See [deployment](p2p.md#leased-manager-membership).
- `--transfer-lock-timeout-secs`: Mark source transfers overdue after this many seconds (default: `120`). Timeout never releases memory still exposed to a remote READ.
- `--transfer-budget`: Source allocation reservations, defaulting to half the pinned pool. Entire allocations are charged once per session, including overdue sessions. At most 1024 sessions can be retained. New authorizations fail when either limit is exhausted; permanent requester loss still requires safe transport revocation or coordinated teardown.
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
