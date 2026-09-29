# OrbitKV local global index

This library owns the complete in-memory block-location index and cached member
admission for each distributed Manager. It performs no networking. Server cluster
code publishes owner residencies to etcd and applies fixed-revision snapshots and
Watch updates here. The old sharded service, placement and TTL candidate cache
are removed.

`GlobalIndex::lookup` returns bounded remote candidates per medium. Entries are
qualified by runtime UUID, exact residency sequence, medium, representation and
known stored bytes. Publisher readiness and current membership gate visibility.
Source grants still validate and pin actual storage before exposing addresses.

Index overflow or malformed/conflicting updates clear readiness and the entire
view; they never leave an advertised complete index with silently missing rows.
The configured budget is conservative logical accounting, not a process RSS cap.
DRAM and SSD are independent records. A batch source rejection does not mutate
the shared index; only background metadata synchronization does.

See [distributed design](../../docs/distributed-cache.md) for key layout,
publication/retry fencing, snapshot/Watch recovery and remaining production
qualification. See [deployment](../../docs/p2p.md) for commands and tests.

```bash
cargo test -p orbitkv-catalog --lib
```
