# External KV tiers

OrbitKV treats an external cache as a byte-storage and transfer provider, not as
a second KV block manager. OrbitKV remains authoritative for semantic object
identity, local page generations, request and Prefix ownership, retirement, and
reuse.

## Current export transaction

The core now exposes a backend-neutral immutable-export transaction:

```text
ready immutable request snapshot
        |
        v
prepare_external_export
        | pin exact PageLease generations
        | emit logical-page copy records
        v
adapter executes backend-specific transfers
        |
        v
exact per-page checksum + durable receipt
        | monotonic completion frontier
        v
publish ExternalReplica and release local read pins
```

OrbitKV privately binds each transfer-local copy ordinal to an exact source
page generation. The public copy record contains the ordinal, class, logical
position, manager-authored backend page index, aggregate payload bytes,
visible-token geometry, and an opaque external destination; it exposes no
`PageLease`. The backend index is not a byte address. An executor adapter
expands it through model-independent layer/component bindings into the iovecs
required by a particular transport. The resulting tensor offsets use the same
absolute backend-page indexing as normal writes and token relocation.
An object key also carries the compiled plan fingerprint, so equal token bytes
under incompatible model/KV layouts cannot produce a restore hit.

Malformed receipts do not release pins or publish a replica. An export may be
aborted only when the adapter proves that no backend operation observed it.
Ambiguous export or restore outcomes have explicit fail-stop quarantine paths.
Published catalog entries are removed only after exact deletion acknowledgement.
Deletion is rejected while a restore still references the object.
External transfers share the manager's monotonic completion-domain frontier
with CUDA work, so a transport cannot advance lifecycle state using a stale
completion value.

The initial transaction keeps the source request busy while exporting. This is
intentional: it avoids racing source retirement before OrbitKV has independent
snapshot-object ownership. A later async export can relax this after snapshots
have their own public lease and pin lifecycle.

## Current restore transaction

Restore is supported for an empty request and a cataloged object whose plan
fingerprint and exact manager-authored page geometry match the current manager.
The tested closure currently covers request-private Full token KV:

```text
cataloged ExternalReplica + empty request
        |
        v
native prepare_append_batch allocates fresh PageLease generations
        |
        v
ExternalRestorePlan names opaque source ranges and local backend indices
        |
        v
adapter fills per-layer K/V iovecs and returns exact checksums/order evidence
        |
        v
native bind validation -> submit -> completion -> publication -> ACK
```

Bad restore receipts leave the native append prepared and safely abortable. An
ambiguous write outcome quarantines the native destinations. External object
deletion is blocked while a restore still references it. Partial tail pages are
stored compactly in the external object and restored into the start of the new
full-stride local page; valid/visible token metadata prevents unwritten bytes
from entering attention.

## Mooncake adapter mapping

Mooncake's Transfer Engine can implement the copy executor:

| OrbitKV contract | Mooncake responsibility |
| --- | --- |
| `ExternalExportCopy` | expand one logical page into registered-memory transfer iovecs |
| `ExternalReplicaTarget` | identify a Store object/segment and byte offset |
| `ExternalExportReceipt` | report exact copied range, checksum, and durable Store commit |
| `ExternalRestorePlan` | read compact object ranges into manager-authored local destinations |
| `ExternalRestoreReceipt` | report exact checksum and ordering before native publication |
| deletion evidence | confirm the exact Store object/replica was removed |

Mooncake Master metadata, replica placement, leases, and eviction are useful
storage-plane mechanisms. They are observations or policy inputs to OrbitKV;
they do not create or retire `PageLease` values. Prefill/decode disaggregation
can build on the restore transaction, but still requires a concrete Mooncake
adapter and end-to-end scheduling/transport qualification.

## Dynamo and NIXL adapter mapping

Dynamo KVBM usefully separates runtime connectors, logical block management,
and NIXL-backed storage/transfer. OrbitKV should reuse that separation while
keeping its stronger semantic compiler boundary:

| OrbitKV contract | Dynamo/NIXL responsibility |
| --- | --- |
| manager-authored logical page record | connector translation only |
| opaque external storage domain/index | registered NIXL volume and descriptor |
| export copy | NIXL read/write or backend volume operation |
| restore copy | NIXL volume read into manager-authored local K/V iovecs |
| completion evidence | exact transport completion, never inferred from enqueue |
| external deletion | volume/backend deletion acknowledgement |

Useful Dynamo ideas are the storage-interface boundary, heterogeneous volume
registration, asynchronous transfer descriptors, event-driven routing, and
separating runtime-specific connectors from the storage backend. OrbitKV adds
the compiler-derived retention program and the Semantic/Execution Frontier rule;
an LRU or remote eviction decision alone cannot prove a local generation safe
to reuse.

## Not implemented yet

- restoring directly as a shared Prefix snapshot (request-private restore works);
- independent restore qualification for Sliding, Full+Sliding, and Chunked views;
- remote replica lease renewal and concurrent external eviction races;
- cross-node request handoff and failure recovery;
- Mooncake, NIXL, RDMA, TCP, NVMe, or object-store adapter crates;
- matched offload/restore latency, throughput, capacity, or cost benefit.

Those features must preserve the implemented transaction pattern: OrbitKV
prepares generation-checked destinations, the adapter moves bytes,
device/transport completion is proven, and only then does OrbitKV publish the
new view.
