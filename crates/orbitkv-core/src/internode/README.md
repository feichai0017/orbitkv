# Cross-node coordination

`metaserver_client.rs` owns the current directory synchronization state machine,
heartbeat and bounded `LocateBlocks` queries through the positive candidate
index in `discovery.rs`. `p2p_service.rs` authorizes and pins
source blocks for Mooncake and releases completed transfer holds.

The engine channel remains UDS/iceoryx2. These network services run only when
`--metaserver-addr` enables distributed discovery and transfer.

## Inventory flow

```text
Publish / SSD restore / peer restore / eviction / cleanup
                         |
              ReadCache mutation lock
                         |
        resident index + bounded sequence journal
                         |
            coalesced wakeup, no event queue
                         |
              inventory sync state machine
                         |
       HeartbeatNode + SyncInventory over gRPC
                         |
               current standalone directory
```

The journal stores metadata, without retaining payload Arcs. The write and
prefetch paths have no directory client dependency. Rejected admission and
repeated insertion do not advertise a new residency episode.

The background worker sends bounded snapshot pages and contiguous deltas.
Directory epoch changes, missing history and ambiguous snapshot replies trigger
reconstruction. Heartbeats verify progress even when no cache writes occur.
Requests have deadlines, and retries use backoff with jitter. An inventory
flush waits for an actual acknowledgement or returns an error; there is no
success path that silently drops pending changes.

The requesting Manager plans transfers in `backing/fetch_plan.rs`. A 16 MiB
logical-byte LRU retains positive evidence for five seconds without extending
the TTL on hits. Cold lookups are coalesced and split at 128 keys / 64 KiB.
There is no negative cache. The source validates its runtime UUID and exact
insertion sequences atomically with payload pinning. Rejected versions are
invalidated locally; up to two alternate-source retries use existing candidates.
Mooncake transfers payload bytes. The blocking operation retains destination
buffers and the source-release guard across caller cancellation. Source
timeout/revocation and cross-host failure qualification remain open. The directory never grants direct memory access.

## Configuration

```bash
orbitkv-cache-manager --addr 10.0.0.1:50055 --pool-size 30gb \
  --metaserver-addr http://10.0.0.100:50056 \
  --inventory-journal-bytes 16777216
```

Without distributed discovery, no resident inventory index or journal is
allocated. See [directory protocol and limits](../../../orbitkv-metaserver/README.md)
and the [distributed cache plan](../../../../docs/distributed-cache.md).
