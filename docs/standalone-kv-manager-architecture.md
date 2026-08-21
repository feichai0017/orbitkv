# Standalone KV Manager Architecture

The normative qualification boundary is the
[Capability Matrix](capability-matrix.md). This document describes the live
ABI6 design; the H20 records under `results/` are historical evidence for
older frozen ABIs.

## Objective and authority

OrbitKV is an engine-independent, semantics-compiled KV state manager. It is
the sole authority for:

- request, snapshot, Prefix, page, step, submission, and reclamation identity;
- logical-to-physical KV bindings and physical page generation;
- immutable published roots and mutable transaction candidates;
- Prefix residency, sharing, attachment, and eviction;
- backend writers, GPU reader pins, and completion evidence; and
- detach, retirement, acknowledgement, and page reuse.

An inference engine may allocate registered tensor arenas and execute its own
attention kernels. It must not independently assign, free, or reuse a page in
those arenas. Engine page tables and LUTs are checked mirrors, never a second
ownership authority.

## Layering

```text
attention-retention semantics
    -> checked KvPlanInput / compiled classes
    -> CanonicalKvManager
         identity + arena
         persistent snapshot
         append transaction
         Prefix
         reclamation
    -> ABI6 typed batch wire
    -> engine adapter and checked device mirrors
    -> backend tensor arenas and attention kernels
```

The Rust core, C wire, ABI6 Python adapter, and SGLang `OrbitKVPrefixCache` are
host-qualified L2. The H20 Prefix path is not yet qualified.

## Module boundaries

The canonical manager is split by invariant rather than by call count:

| Module | Responsibility |
| --- | --- |
| `identity.rs` | Opaque generation-checked leases and semantic Prefix keys |
| `arena.rs` | Slot arenas, class/pool identity, physical page state and counts |
| `persistent_snapshot.rs` | Immutable class roots, path-copy and materialized cold views |
| `append_transaction.rs` | Prepare, submit, complete, abort, quarantine, tail policy and COW |
| `prefix.rs` | Request fork and page-aligned Prefix lookup/publish/attach/evict |
| `reclamation.rs` | Request release, detach, certificates, ACK and recycle |
| `transaction_validation.rs` | Batch-wide preflight and ref-count deltas |
| `protocol.rs` | Backend-independent request/result types |
| `facade.rs` | Construction, public queries and stable core facade |
| `manager_state.rs` | Private state records shared by the transaction modules |
| `test_model.rs`, `tests/` | Test-only executable model, full-scan oracles and fault traces |

Production Rust and Python modules are limited to 1,500 lines; test and
benchmark modules are limited to 2,000. CI applies this only to active source,
never to frozen source closures in `results/`.

## Identities and snapshots

Every authority crossing a boundary is opaque and generation checked:

```text
RequestLease        = (engine_epoch, slot, generation)
SnapshotLease       = (engine_epoch, slot, generation)
PrefixLease         = (engine_epoch, slot, generation)
StepLease           = (engine_epoch, slot, generation)
SubmissionLease     = (engine_epoch, slot, generation)
ReclamationLease    = (engine_epoch, slot, generation)
PageLease           = (engine_epoch, pool_epoch, pool_id,
                       page_id, page_generation)
```

A request contains only its current `SnapshotLease` head. A snapshot contains
immutable per-class persistent roots. An update path-copies changed search
paths; an old snapshot never changes and a released lease becomes stale.
Every mutation supplies the expected head, so a concurrent or replayed caller
receives a retryable conflict before state mutation.

Materializing all snapshot pages is a cold operation used by fork/attach
lowering and validation. The hot append path emits only class tail actions,
copy intents, fresh write intents, detached bindings, and reclamation
certificates.

## Physical page state

```text
Free
  -> Reserved(step)
  -> Live { writer?, request_refs, prefix_refs, reader_pins }
  -> Retiring(reclamation)
  -> Free(next generation, after exact backend ACK)

ambiguous backend/GPU outcome -> Quarantined
generation exhaustion         -> Exhausted
```

`request_refs`, `prefix_refs`, `reader_pins`, and the active writer live on the
physical page state. Reclamation is global and page-owned: detaching one
request does not produce a certificate while another request, Prefix, reader,
or writer still holds the page. A shared page is certified once, when its final
reference disappears.

## Append and COW transaction

Every lifecycle mutation is batch-only and preflights the complete item set,
flat spans, reserved fields, and output capacities before core mutation.

### Prepare

`prepare_batch` validates each request and expected head, constructs private
candidate snapshots, reserves exact destination pages, and emits:

- one `TailAction` per class: none, in-place, fresh, or copy-on-write;
- exact `CopyIntent` records for shared or pinned partial tails; and
- ordered `WriteIntent` records for manager-selected fresh pages.

No candidate becomes visible. A short buffer reports the required counts and
leaves manager state unchanged.

### Submit

`submit_batch` accepts exact backend bind and copy receipts. A COW receipt must
prove the expected source/destination leases, offsets, token count, backend
indices, that the copy was observed and completed, and that it is ordered
before new writes. A semantic mismatch enters fail-stop quarantine; uncertainty
is never treated as success or an ordinary abort.

Successful submit pins every page that the backend can read or write and
returns lease-only submissions. The private target snapshot is still not the
request head.

### Complete and publish

`complete_batch` validates a shared completion point and all ordered
submissions, removes writer/reader pins, applies retention detaches, and swaps
all request heads atomically. It returns publication scalars, detached mirror
actions, and any page-owned reclamation certificates.

`DetachedBinding` distinguishes `CLEAR` from `REPLACE`. This is necessary when
COW changes an engine mirror but the shared source page cannot yet be retired;
mirror maintenance is not inferred from the presence of a reclamation
certificate.

### Abort and quarantine

An unsubmitted transaction can abort only with proof that the backend did not
observe its destinations. Ambiguous binding, copy, launch, or event outcomes
quarantine the affected generations and fail-stop the lifecycle. They are
never converted to completion or reuse.

## Fork and joint COW

`request_fork_batch` shares an immutable source snapshot with acquired empty
target requests and aggregates page refs across the whole batch. The returned
cold `MaterializedRequestView` includes class, logical ordinal, exact
`PageLease`, backend identity, temporal cell/cycle, and valid/visible token
ranges so an adapter can rebuild mirrors without inventing ownership state.

When any class has a shared or pinned partial tail, every partial-tail class in
the same Hybrid request enters the COW transaction. This prevents Full and SWA
views from observing different publication boundaries. A copy failure leaves
the source snapshot live and quarantines the uncertain destination/operation.

## Prefix ownership

The core exposes page-aligned Prefix operations:

```text
lookup(key) -> generation-checked hint
attach(empty request, expected head, hint) -> materialized request view
publish(request, expected head, key) -> PrefixLease
publish_release(...) -> atomic request-to-Prefix ref transfer
evict(PrefixLease) -> detach + possible certificates
recycle(PrefixLease) -> generation-safe slot reuse
```

The key binds namespace, token digest, and page-aligned boundary. Lookup hints
are not ownership proofs; attach revalidates the candidate under the manager
lock. Structural Radix splits cannot fabricate a Prefix at an unaligned Hybrid
boundary.

The intended SGLang seam is a registered `OrbitKVPrefixCache` whose Radix nodes
store token/digest metadata, an opaque `PrefixLease`, and LRU policy only. They
must not store authoritative tensor indices, page generations, or free-list
state. This adapter passes host lifecycle and hostile-fault gates; exact-source
H20 engine qualification remains pending.

## Reclamation order

The engine-facing order is:

```text
manager detach/release/evict
  -> preflight every mirror CLEAR/REPLACE
  -> commit mirror updates
  -> establish completion/synchronization dependency
  -> send exact-generation reclamation ACKs
  -> recycle request/Prefix identities
  -> allow physical page generation reuse
```

An adapter exception with no typed pre-commit outcome is unknown. The runtime
must fail-stop; it must not retry the operation or repair state with a private
side map.

## Complexity contract

- hot append/complete: `O(C + Δ log R + Δ log Δ)`, roughly
  `O(C + Δ log R)`;
- Prefix lookup: `O(B log N + B·C)`; and
- cold fork/attach/publish/release/evict ref aggregation: `O(P log U)`.

`C` is class count, `Δ` the changed-page count, `R` resident pages, `B` lookup
boundaries, `N` Prefix entries, `P` materialized pages, and `U` unique physical
pages. A partial-tail update at an 8,192-page resident root must not traverse
or materialize all 8,192 entries.

## Compatibility and acceptance

ABI6 exports exactly 23 `orbitkv_*` symbols listed in the
[Capability Matrix](capability-matrix.md). There are no ABI5 lifecycle aliases,
older loaders, or silent native-allocation fallback paths.

An engine profile becomes a replacement claim only after its native allocator
and Prefix owner cease to be authoritative; all fault and pressure gates pass;
and an append-only manifest binds the exact manager, wire, adapter, engine
release, hardware, commands, and outputs. The frozen ABI5-v5 H20 record does
not satisfy those gates for ABI6.
