# OrbitKV catalog

This library embeds replica discovery in Cache Managers. It owns fixed shard
placement, cached membership, the inventory index, and the peer gRPC catalog
service. There is no catalog executable, Python launcher, or separate HTTP port.
See [deployment](../../docs/p2p.md) and the [distributed roadmap](../../docs/distributed-cache.md).

## Placement and admission

The v1 protocol has 16 logical shards. SHA-256 over a domain-separated,
length-framed StateKey selects a shard; equal-weight rendezvous hashing of the
shard and configured Node IDs chooses one catalog host. All Managers supply the
same `--catalog-nodes` set (1–16 stable Node IDs). Order is normalized. etcd
atomically creates `/orbitkv/v1/<cluster>/placement`; incompatible joins fail.

Placement is immutable in this stage. Watch observes member and configuration
changes. A removed or changed placement fences the runtime; this is not an
online migration API. Use a new cluster name and restart the Managers for a
planned configuration replacement. Losing a member leaves its shards unavailable
until that Node ID returns, without reassigning them to other live members.

Every catalog RPC carries the shard, placement fingerprint and destination
runtime UUID. `LocateBlocks` carries one route for every included shard, allowing
one bounded batch to cover multiple shards hosted by the same Manager. The receiver checks its own admission and assignment. Inventory
publishers must also match the cached member view, and every record/query key
must belong to the requested shard. These are consistency checks in a trusted
cluster, not network authentication. TLS/auth integration remains open.

## Inventory recovery

Each Manager maintains an independently ordered DRAM inventory and bounded
journal per shard. Actual insertions and removals update residency and sequence
under the cache lock. Duplicate insertions and rejected admissions produce no
event; journals retain no payload references. SSD-only replicas are not advertised.

The [schema](../orbitkv-proto/proto/engine.proto) exposes `HeartbeatNode`,
`SyncInventory`, `LocateBlocks`, and `UnregisterNode` on the Manager's peer port.
Each catalog shard has a runtime epoch. A changed member incarnation, catalog
epoch, sequence gap or ambiguous snapshot reply triggers recovery:

1. Capture the owner/shard starting sequence and hide its replacement view.
2. Enumerate resident keys using bounded pages and insertion sequences.
3. Capture an end sequence and replay the complete intervening journal interval.
4. Commit at the acknowledged cut, then stream subsequent changes.

Newer sampled records survive older replay events. Exact retries of the preceding
operation are accepted; gaps and conflicting order are rejected. Overflow restarts
the snapshot with backoff. Other owners remain queryable during repair. Idle owners
heartbeat approximately every ten seconds, so lost catalog evidence also recovers
without new cache writes. Endpoint takeover by a different owner incarnation may
wait for the old inventory's 30-second heartbeat staleness interval.

`flush_saves_and_inventory()` drains saves and waits for fresh acknowledgement
through the captured sequence of every shard, against its current catalog owner.
The barrier fails after 30 seconds if a shard cannot catch up. Local Publish does
not wait for this barrier; remote visibility is asynchronous.

## Bounds and discovery

| Resource | Current limit |
| --- | --- |
| Owner journal | 16 MiB per Manager, divided across 16 shards |
| Candidate index | 16 MiB; positive entries only; five-second TTL |
| Catalog index admission | `--catalog-budget`, default 256 MiB, divided across assigned shards |
| Inventory page/delta | 1,024 records, 512 KiB of accounted record bytes |
| Cold discovery batch | 128 keys, 64 KiB of namespace/hash bytes, one catalog host |
| Candidate row | At most four endpoint/incarnation/insertion-sequence hints |
| Cold query | Three-second deadline including coalescing; at most four hosts queried concurrently |
| Catalog gRPC message | 4 MiB |
| Concurrent catalog operations | 16 per Manager |

The index budget accounts for both key indexes, owner records and the retained
retry operation across all owners of a shard. It is logical admission accounting,
not a process-RSS cap; allocator overhead and bounded in-flight RPC buffers are
additional. Metadata exhaustion rejects the operation atomically. Expired owner
inventories are swept every 30 seconds, with a two-hour retention TTL.

Cold queries group missing keys by catalog host and preserve position alignment.
A host batch validates every shard route before looking up any keys. Channels are
shared per current host incarnation and stale entries are removed on cold lookups. Positive
candidate hits issue no catalog lookup. An unavailable shard produces missing
hints, not authoritative absence; already-known prefix evidence remains usable.
Source authorization still checks exact runtime and insertion sequences while
pinning data before Mooncake reads. Catalog hints never authorize memory access.

## Operations and tests

Metrics share the Manager's `/metrics` endpoint. `orbitkv_catalog_store_entries`,
`orbitkv_catalog_block_owners` and `orbitkv_catalog_metadata_bytes` have a `shard`
label. Inventory snapshot, history-gap, sync-failure and catalog RPC counters expose
repair progress. There is no separate catalog health or maintenance listener.

```bash
cargo test --release -p orbitkv-catalog
cargo test --release -p orbitkv-core --no-default-features --features cuda-13,mooncake \
  peer::catalog::tests
cargo bench -p orbitkv-catalog --bench unregister_node
```

Tests cover two catalog endpoints, shard isolation, wrong destinations, incarnation
replacement, idle restart repair, lost replies, overflow, source generations and
bounded accounting. Private units stay under `tests/unit/`; the cleanup workload is
[benches/catalog.rs](../../benches/catalog.rs).

Each shard currently has one metadata copy. Replication, weighted/versioned
handoff, subscriptions and remote SSD discovery remain planned. Recovery rebuilds
metadata from surviving owners; it does not recover lost payloads or revoke
orphaned source transfers. Single-host tests do not qualify cross-host HA.
