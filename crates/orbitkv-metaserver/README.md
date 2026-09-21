# OrbitKV MetaServer

The current multi-node directory is a separate, in-memory gRPC service.
Cache Managers continuously synchronize their authoritative DRAM inventories;
a directory restart is repaired from surviving Managers without restarting the
engines. It remains a single service with no HA replication. The
[distributed cache design](../../docs/distributed-cache.md) plans an embedded
catalog with etcd membership in the next milestone.

## Run

```bash
cargo run --release -p orbitkv-metaserver -- \
  --addr 0.0.0.0:50056 --http-addr 0.0.0.0:9092

orbitkv-cache-manager --addr <manager-ip>:50055 \
  --metaserver-addr http://<directory-ip>:50056
```

| Directory option | Default | Meaning |
| --- | --- | --- |
| `--addr` | `127.0.0.1:50056` | gRPC listener |
| `--http-addr` | `0.0.0.0:9092` | Health, metrics and maintenance listener |
| `--log-level` | `info` | Log verbosity |
| `--node-stale-secs` | `30` | Hide owners after this much inactivity |
| `--ttl-minutes` | `120` | Remove inactive owners and their inventories |
| `--sweep-interval-secs` | `600` | Background lifecycle cleanup interval |
| `--inventory-bytes-per-node` | `268435456` | Accounted metadata limit per owner |

The Manager's `--inventory-journal-bytes` defaults to `16777216` (16 MiB).
It bounds retained residency changes. Lag beyond this history triggers a
snapshot, without blocking cache insertion or pinning payloads. No inventory
index or journal is allocated when distributed discovery is disabled.

## Recovery protocol

The [protobuf schema](../orbitkv-proto/proto/engine.proto) defines four RPCs:

| RPC | Behavior |
| --- | --- |
| `HeartbeatNode` | Register/refresh an owner session; return the directory epoch and acknowledged inventory progress |
| `SyncInventory` | Begin a replacement view, send bounded snapshot pages and ordered deltas, then commit the complete view |
| `QueryPrefixBlocks` | Plan a contiguous prefix from live, committed remote owners, excluding the requester |
| `UnregisterNode` | Remove only the matching owner session and its entries |

A Manager process creates a random `node_id`. Heartbeats run approximately
three times per stale interval, including when the cache is idle. A different
process may take over an endpoint after the current session becomes stale.
Every directory process has a fresh `catalog_epoch`. Inventory generations
increase for successive snapshot attempts within a Manager process; owner
sequences order actual insertions and removals across all its namespaces.
These are the D0 stream identities, before logical sharding and etcd fencing.

Recovery proceeds as follows:

1. Record the owner's starting sequence and begin a hidden replacement view.
2. Scan the current resident index with ordered key cursors. Each entry includes
   its insertion sequence; pages contain at most 1,024 records and 512 KiB of
   accounted record bytes. Oversized records fail synchronization explicitly.
3. Capture an end sequence and replay every intervening change in order. A
   newer entry sampled during the scan survives an earlier replayed removal.
4. Commit only at the acknowledged cut, then stream subsequent changes.

A resnapshot discards this owner's previous directory entries and hides its
replacement until commit. This avoids holding two whole inventory views.
Other owners remain visible. Gaps or ambiguous snapshot replies restart with a
higher generation. A lost live-delta reply can resume from heartbeat progress.
The receiver accepts an exact retry of the immediately preceding operation;
older/conflicting operations and sequence gaps are rejected. Repair uses
bounded batches and exponential retry backoff with jitter.

`OrbitKVEngine::flush_saves_and_inventory()` first drains saves and then waits
for a fresh heartbeat and acknowledgement through the captured residency
sequence. Synchronization timeout returns an error after 30 seconds. This is
an observation barrier: eviction can still remove a block afterwards, and a
source must validate and pin blocks before Mooncake reads them.

## Ownership and limits

- Actual cache insert/evict operations update residency and sequence under the
  same lock. Rejected admissions and duplicate inserts create no new event.
- Publish, SSD-to-DRAM restore and peer-to-DRAM restore use this common path.
  SSD-only replicas are not advertised in D0.
- The directory maintains both key-to-owner and owner-to-key indexes. Cleanup
  walks only the affected owners' inventories. A healthy heartbeat preserves
  old resident entries; registration age is not an expiration rule.
- Different owners update independently. Queries check liveness and inventory
  readiness. Advisory reclaim hints require two other visible copies and are
  applied only to the matching local residency sequence.
- Per-owner accounting charges `192 + 2 * (namespace bytes + hash bytes)` per
  key, plus one bounded retry operation outside that budget. This is an index
  admission budget, not a hard process-RSS limit or a cluster-wide capacity cap.
  The owner's resident index scales with its real cached blocks.
- Directory recovery does not recover lost payloads. Manager restart and durable
  SSD recovery, etcd membership, catalog replication, local candidate caching
  and qualified cross-host transfer lifetimes remain separate work.

## Operations

The HTTP listener exposes `GET /health`, `GET /metrics`, and
`POST /admin/sweep-expired-nodes`. The latter runs the same node-inactivity
cleanup as the background sweep and returns removed owner/key counts. It does
not delete still-resident evidence simply because it is old.

Monitor Manager counters `orbitkv_inventory_snapshots_started`,
`orbitkv_inventory_snapshots_completed`, `orbitkv_inventory_history_gaps`,
`orbitkv_inventory_sync_failures`, `orbitkv_inventory_records_sent`, and
`orbitkv_metaserver_heartbeat_failures`. Repeated starts without commits indicate
repair is not converging; inspect connectivity, owner byte limits and retained
history. Directory redundancy gauges count stored entries, including incomplete
views; they are not a guarantee of durable payload copies.

This protocol is a breaking pre-1.0 change. Upgrade Managers and the directory
together; the former independent registration/removal RPCs and queue-depth
option have been removed.

## Validate

```bash
cargo test --release -p orbitkv-metaserver
cargo test --release -p orbitkv-core --no-default-features --features cuda-13,mooncake \
  internode::metaserver_client::tests
cargo bench -p orbitkv-metaserver --bench unregister_node
```

The synchronization tests use the real gRPC service with restart, outage,
response-loss and snapshot-overflow injection. Store tests cover concurrent
owners, stale sessions/epochs, ordering, byte limits and visibility at the cut.
The cleanup benchmark source is [benches/catalog.rs](../../benches/catalog.rs).
