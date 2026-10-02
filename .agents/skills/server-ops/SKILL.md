---
name: server-ops
description: Configure or diagnose OrbitKV Cache Managers, local process transport, DRAM/SSD budgets, etcd membership, inventory-fed global indexes and peer transfers. Use for deployment and runtime incidents.
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

For remote metadata scope, omission means `AllNamespaces`. Repeat
`--metadata-namespace` only with complete storage namespaces logged by actual
registration, or use `--metadata-empty-scope` for an explicit empty set. Scope is
startup configuration: change it by restarting and fully bootstrapping the
Manager. Never infer it from model display names, prefixes or query misses. The
allowlist reduces record bytes, not peer-session fanout, and is not tenant auth.

Trace an incident through its owner:

1. Check process health and the engine's UDS/iceoryx2 registration before diagnosing
   a cache miss as a transport failure. See `crates/orbitkv-server/src/endpoint/`.
2. Check `/cache/metadata`: membership validity/revision, explicit coverage,
   scope digest/kind/count, active/staging index bytes, installed views and stream
   queues. Distinguish a complete empty owner view from a missing view. A
   scope-outside remote lookup is unavailable even when local cache service works.
   `/cache/sync` returns a source inventory fence; a controlled requester must
   install it with the same scope via `/cache/metadata/await`. A received frame or
   source head is not that watermark.
3. Check source grants and generation checks in `crates/orbitkv-core/src/peer/`,
   then TENT READ/completion. Index freshness does not authorize an address.
4. Inspect query, source export, SSD staging and GPU admission budgets using
   `docs/metrics.md`. Lease expiry or a deadline cannot establish native drain.

Use native TENT for payloads and OrbitKV source-control RPCs. Do not revive
Catalog placement/lookup RPCs or change etcd's transport to fix a payload problem.
Use test-owned processes for faults; preserve other workloads and collect logs
outside the repo. Keep same-host TCP, physical two-host TCP and RDMA evidence
separate. Build first, then test with native binaries/libraries frozen.

For a normal Manager stop, require SIGTERM to fence membership before gRPC waits
for inventory streams, then drain lifecycle ownership and revoke the member lease.
Do not accept a test helper's SIGKILL fallback as graceful shutdown evidence.
Verify the member key is absent before reusing a node ID, and require a larger
epoch plus a different incarnation after restart. Current/peak inventory session
counters are abort-safe diagnostics: a replaced follower must return current
sessions to the real live count even when its task is aborted.

For repeated-key metadata traffic, `--inventory-stream-coalesce-ms` selects a
0–5 ms quiet window; the default is 0. A requested fence interrupts intentional
wait but still requires complete input-interval installation. Compare inventory
input/output counters and stream encoded bytes with measured observer visibility
and effective hits. Etcd network counters now cover membership/configuration,
not block propagation. A lower mutation count alone does not establish serving
improvement.

For scoped-stream qualification, pair all-domain and scoped runs with the same
exact namespace/key/payload mutations and coalescing. Report input/output filter
records and CPU, encoded bytes/frames, scope-bound coverage, active/staging/index
bytes, RSS, queue peaks, visibility, bootstrap/repair, and lookup/update lock
wait/hold time. Empty filtered intervals must advance only source-proven coverage.

For inference interference, measure the physical NIC/direction, PCIe/NUMA and
engine collective traffic as well as TENT transfers. Manager query-byte budgets
are not a node-wide network scheduler; engine-local P/D WRITE is a separate
submitter. Receiver credit/pacing is planned in S6, not an existing runtime flag.
Source authority, network credits and destination lifetime need separate drain
evidence. Report that distinction when diagnosing congestion or retained memory.
