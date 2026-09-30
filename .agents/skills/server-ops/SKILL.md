---
name: server-ops
description: Configure or diagnose OrbitKV Cache Managers, local process transport, DRAM/SSD budgets, etcd global indexes and peer transfers. Use for deployment and runtime incidents.
---

# OrbitKV Manager operations

Resolve paths from the Git root and read `AGENTS.md`. Use the built
`orbitkv-cache-manager --help` and `docs/server.md` for current flags; do not copy
an old flag list. The source binary is `orbitkv-cache-manager`; the wheel exposes
the same console command and bundles `orbitkv-cache-manager-py`.

Start with `docs/deployment.md` for topology and `docs/p2p.md` for distributed
startup. Every node has an independent Manager. Distributed mode needs a concrete
peer address, etcd endpoints, a unique stable node ID and the same cluster name.
`orbitkv-catalog` is an embedded local index, not a separate service.

Trace an incident through its owner:

1. Check process health and the engine's UDS/iceoryx2 registration before diagnosing
   a cache miss as a transport failure. See `crates/orbitkv-server/src/endpoint/`.
2. Check `/cache/metadata`: registration validity, index availability/revision and
   publisher progress. `POST /cache/sync` is a source publication barrier; a
   requester must separately observe that revision. See `src/cluster/` in Server.
3. Check source grants and generation checks in `crates/orbitkv-core/src/peer/`,
   then TENT READ/completion. Index freshness does not authorize an address.
4. Inspect query, source export, SSD staging and GPU admission budgets using
   `docs/metrics.md`. Lease expiry or a deadline cannot establish native drain.

Use native TENT for payloads and OrbitKV source-control RPCs. Do not revive
Catalog placement/lookup RPCs or change etcd's transport to fix a payload problem.
Use test-owned processes for faults; preserve other workloads and collect logs
outside the repo. Keep same-host TCP, physical two-host TCP and RDMA evidence
separate. Build first, then test with native binaries/libraries frozen.

For repeated-key metadata traffic, `--inventory-publish-coalesce-ms` selects a
0–5 ms quiet window; the default is 0. `/cache/sync` interrupts intentional wait
but still requires complete input-interval publication. Compare inventory input/
output counters and `orbitkv_metadata_watch_key_value_bytes_bytes_total` with measured
observer visibility and effective hits. Watch payload accounting excludes framing
and snapshot RPCs; etcd network counters cover broader traffic. A lower mutation
count alone does not establish serving improvement.

For inference interference, measure the physical NIC/direction, PCIe/NUMA and
engine collective traffic as well as TENT transfers. Manager query-byte budgets
are not a node-wide network scheduler; engine-local P/D WRITE is a separate
submitter. Receiver credit/pacing is planned in S6, not an existing runtime flag.
Source authority, network credits and destination lifetime need separate drain
evidence. Report that distinction when diagnosing congestion or retained memory.
