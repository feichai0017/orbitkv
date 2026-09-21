# Cross-node coordination

`metaserver_client.rs` owns the current directory synchronization state machine,
heartbeat and remote prefix-plan queries. `p2p_service.rs` authorizes and pins
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

The requesting Manager still calls `query_plan` on a local miss. Moving plan
construction and a bounded candidate index into the Manager belongs to D1.
Mooncake is responsible for payload bytes, while the source service owns block
validation and holds. The directory never grants direct memory access.

## Configuration

```bash
orbitkv-cache-manager --addr 10.0.0.1:50055 --pool-size 30gb \
  --metaserver-addr http://10.0.0.100:50056 \
  --inventory-journal-bytes 16777216
```

Without distributed discovery, no resident inventory index or journal is
allocated. See [directory protocol and limits](../../../orbitkv-metaserver/README.md)
and the [distributed cache plan](../../../../docs/distributed-cache.md).
