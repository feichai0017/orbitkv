# Distributed KV cache

Every Manager keeps a complete local global index of advertised block locations.
etcd replicates locations and membership; background publication and snapshot/Watch
maintain the local views. Mooncake TENT carries KV bytes. Source grants and release
remain OrbitKV gRPC. See the [protocol](distributed-cache.md).

## Leased Manager membership

All Managers use the same `--cluster-name` and etcd cluster; each has a distinct
stable `--node-id`. Supply multiple etcd endpoints for failover. Production metadata
HA requires a quorum deployed across failure domains. Index replication does not
replicate a payload held only by a failed Manager.

On host A:

```bash
orbitkv-cache-manager \
  --addr 10.0.0.1:50055 --pool-size 30gb \
  --etcd-endpoints http://10.0.0.101:2379 \
  --cluster-name inference --node-id cache-a
```

On host B:

```bash
orbitkv-cache-manager \
  --addr 10.0.0.2:50055 --pool-size 30gb \
  --etcd-endpoints http://10.0.0.101:2379 \
  --cluster-name inference --node-id cache-b
```

Use concrete peer addresses reachable from the other hosts. Peer gRPC and
Mooncake's P2P handshake/data endpoints use separate ports on that host. Optional
`--nics mlx5_0,mlx5_1` filters RDMA rails; omitted, Mooncake selects an available
transport including TCP. For a same-host TCP test set `MC_FORCE_TCP=1`; OrbitKV
then disables TENT RDMA/NVLink/MNNVL for that engine creation and temporarily
suppresses `MC_TENT_CONF` so a general config cannot invalidate the forced-TCP
control.

There is no separate MetaServer process or `--metaserver-addr` flag. Upgrade all
Managers together for this breaking protocol change. Engine processes retain the
same UDS/iceoryx2 and CUDA IPC integration described in the
[vLLM and SGLang deployment guide](deployment.md).

## Lookup and transfer

```mermaid
sequenceDiagram
    participant A as Source Manager
    participant E as etcd quorum
    participant B as Requester Manager
    A->>A: Seal DRAM or commit SSD; update inventory
    A-->>E: Background fenced publication
    E-->>B: Fixed-revision snapshot / Watch
    B->>B: Local global-index lookup and route planning
    B->>A: Authorize exact incarnation and residency sequences
    A->>A: Pin sources and reserve bytes
    A-->>B: Source grant and registered-memory descriptors
    B->>A: Mooncake TENT READ
    B->>A: Release after native completion
    B->>B: Restore to engine HBM
    B-->>E: Publish new local residency
```

State identity must match model artifacts, computation, rank topology and storage
geometry. Source authorization, rather than metadata freshness, protects memory
reads. Validate vLLM-to-vLLM and SGLang-to-SGLang separately; these tests do not
establish cross-engine byte compatibility or hybrid-state completeness.

The requesting Manager authorizes the next planned segment while the current
segment's READ runs. There is one execution strategy, with bounded lookahead.
Remote source NUMA identifiers describe the source host only. The receiver
allocates each fetched slot beside its own registered GPU, using the sealed
local layer/group topology. Receiver placement is excluded from the storage
namespace, included in pending-read coalescing, and retained when fetched
blocks are published again. A source/receiver slot-count mismatch fails before
allocating or submitting a READ.

Only one READ and one following authorization can be active per fetch plan;
the following segment allocates destination memory only when consumed. A failed
or partial current READ discards the unused grant. A speculative authorization
failure is retried on demand after the current READ drains, subject to the
existing admission limits. On resource exhaustion, authorization may wait up to
three seconds for that peer's releases already in progress before the attempt,
then retry once. It does not wait for unrelated active READs; a rejected ticket's
own cleanup cannot satisfy that wait.
The three-second bound covers only this release wait; authorization RPCs retain
their own existing deadlines.
Cancellation uses the existing known-ticket cleanup
owner, including when the authorization response was lost.

Source grants awaiting release acknowledgement can outlive those two active
stages; the existing per-source 64 and global 1024 completion limits bound them.
Lookahead can retain source memory earlier, so benchmark with identical source
budgets and delayed release ACKs. `prepared_wait` records residence between
authorization and consumption; speculative grants do not train the sequential
composite route estimate. No distributed throughput improvement is claimed yet.
See the [communication implementation sequence](completion-plan.md#s4--finish-communication-execution-and-demonstrate-gains) for
batched metadata and engine-side execution work that remains planned.

## Failure and configuration behavior

- Registration requires an absent live Node ID, advances its persistent epoch,
  and binds the endpoint/runtime UUID to an etcd lease. Lease renewal, publication
  and Watch are independent tasks.
- Every Manager builds a complete snapshot at revision R and Watches from R + 1.
  Incomplete or over-budget indexes disable remote discovery rather than silently
  dropping metadata. Compaction or malformed metadata triggers a rebuild.
- Publisher cursors fence retries and delete/recreate operations. Journal overflow
  withdraws that owner's readiness, reconciles its entire inventory and replays
  concurrent changes before making it discoverable again.
- Source UUID and generation checks prevent stale rows from authorizing reused
  addresses. Both DRAM and SSD can be advertised for one StateKey.
- Losing a Manager does not remove another Manager's index. A sole lost payload
  still requires another replica or recomputation. Removing its leased membership
  hides candidates even before all block-delete events arrive.
- Losing an etcd member permits recovery through surviving endpoints if quorum
  remains. Losing quorum stops updates; new remote admission stops when local
  lease validity expires. Local DRAM/SSD operations continue, and submitted
  transfers retain their owners through completion.
- Registration is valid for half the acknowledged TTL measured from request send
  time. Once expired/fenced it cannot revive; restart for a new incarnation.
  Lease expiry or requester disappearance is not proof of native transport drain.
- The connector currently uses HTTP etcd endpoints, without exposed TLS/auth
  configuration. UUIDs and transfer tickets are consistency fences, not peer
  authentication. Use a trusted cluster network.

## Catalog availability and etcd

There are two local structures: the **owner inventory** records real storage,
and the **global index** records advertised locations from all owners. etcd
replicates the latter's source metadata. There are no assigned directory hosts,
shards, migration protocol, TTL hint cache or request-time directory fallback.

The versioned prefix is `/orbitkv/v2/<cluster>/`. Upgrade all Managers together;
there is no v1 synchronization path. Old metadata under v1 is not read or changed
by new Managers. See [identity and publication](distributed-cache.md#identity-and-publication).

## Limits and observability

`--inventory-journal-bytes` defaults to 16 MiB per owner; overflow triggers
reconciliation. `--index-budget` defaults to 256 MiB of logical accounting for
this Manager's complete global index. No measured cluster-scale capacity SLO is
established; full replication pays update CPU and index memory on every Manager.

`POST /cache/sync` returns `published_revision`. Wait for an available requester
index with `GET /cache/metadata` at or above that revision when a test requires
an explicit remote-visibility barrier. Ordinary query paths do not wait on etcd.
Metadata status includes publisher readiness and inventory/publication progress;
`/metrics` retains payload, source admission and completion measurements.

## Validation

For independent engine replicas, see [shared-cache qualification](shared-cache-qualification.md).
The source reservation budget is `--transfer-budget` (default: half the pinned
pool), with a maximum of 1024 active or overdue sessions. It charges entire pinned
allocations, deduplicated within each session, so a small slice cannot retain an
unaccounted large slab. Concurrent sessions each reserve their full allocation
footprint. Exhaustion returns a bounded miss to the requesting cache path.

Run native builds and runtime gates sequentially in a checkout; builds restage
shared Mooncake libraries. The etcd gate starts isolated temporary processes:

```bash
ETCD_BIN=/path/to/etcd cargo test --release \
  --no-default-features --features cuda-13,mooncake \
  --lib cluster::tests::etcd -- --ignored

ETCD_BIN=/path/to/etcd MC_FORCE_TCP=1 cargo test --release -p orbitkv-server \
  --no-default-features --features cuda-13,mooncake \
  --lib cluster::tests::p2p_mooncake -- --ignored
```

The etcd gate covers duplicate identities, incarnation restart, compaction,
leader loss and quorum-loss fencing. Run `--lib cluster::publish::tests -- --ignored`
for lost replies, delayed retry fencing and independent medium propagation.
The GPU gate exercises real publication/Watch, source authorization and Mooncake
READ across 260 blocks, plus encoded payloads. Set `ORBITKV_NVCOMP_LIBRARY` for the
ANS case. These gates run on one host; physical host loss and RDMA need their own
qualification.

The Python process gate starts real etcd plus two Manager binaries, uses the
shared engine-facing client to restore GPU bytes remotely, then verifies local
restoration after stopping etcd. Set the binary path to prevent runtime tests
from starting a concurrent Cargo build:

```bash
cd python
ETCD_BIN=/path/to/etcd \
ORBITKV_CACHE_MANAGER_BINARY=/path/to/orbitkv-cache-manager \
PYTHONPATH=. ../.venv/vllm-release/bin/python -m pytest -m integration \
  tests/integration/test_distributed_cache.py
```
