# Standalone KV Manager Architecture

The normative qualification boundary is the
[Capability Matrix](capability-matrix.md). This document describes the live
ABI8 design. Sealed Prefix and relocation records qualify only their disjoint,
manifest-bound engine scopes. Fixed-state pair verification and weight-backed
diagnostics remain below L4, and records for older ABIs remain historical.

## Objective and authority

OrbitKV is an engine-independent attention-state compiler and transactional
ownership runtime. It is the sole authority for:

- request, snapshot, Prefix, page, step, submission, reclamation, and relocation identity;
- logical-to-physical KV bindings and physical page generation;
- immutable published roots and mutable transaction candidates;
- Prefix residency, sharing, attachment, and eviction;
- backend writers, GPU reader pins, and completion evidence; and
- detach, retirement, acknowledgement, and page reuse.

An inference engine may allocate registered tensor arenas and execute its own
attention kernels. It must not independently assign, free, or reuse a page in
those arenas. Engine page tables and LUTs are checked mirrors, never a second
ownership authority.
OrbitKV is not a full replacement for SGLang's scheduler, kernels, tensor
allocation, or model execution, and the current evidence does not establish a
mature L5 production system.

## Layering

```text
attention-retention semantics
    -> checked KvPlanInput / compiled classes
    -> CanonicalKvManager
         identity + arena
         persistent snapshot
         append transaction
         token relocation
         Prefix
         reclamation
    -> ABI8 typed manager + fixed-state wire
    -> engine adapter and checked device mirrors
    -> backend tensor arenas and attention kernels
```

The Rust token/fixed-state core, C wire, ABI8 Python adapter, independent
fixed-state client, and SGLang `OrbitKVPrefixCache` retain their scoped host L2
qualification. The currently bound frontend profile has a scoped
request-private GDN+convolution host implementation: request
allocation, initial clear, forward completion-event registration, and
release-time wait/retire/clear/exact-ACK are connected and host-tested.
Same-owner replacement has coordinator/real-CPU-tensor host tests only and no
production trigger. Other GDN profiles, KDA, ShortConv, and other
linear-attention family bindings remain pending. Scoped pair verification does
not provide independent hardware attestation, L4 qualification, or performance
qualification.

The separate `orbitkv-runtime` and `orbitkv-reference` packages expose an
engine-neutral data-plane SPI and a reusable external tensor-arena reference
adapter. Both wheels build and clean-install in CI. The reference is a contract
oracle, not a scheduler, model runner, allocator, attention kernel, or complete
engine; the SGLang adapter has not migrated to this SPI.

Opt-in request-private pressure telemetry is host-tested. It distinguishes
consumed, resident, request-reachable, and semantic-live bytes and reports
retention amplification. No append-only, sealed, or qualified asynchronous GPU
pressure record is published; fixed-state bytes are excluded, and shared
Prefix/request-fork retention amplification fails closed.

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
| `token_virtualization.rs` | Canonical token views, dispositions, planning, and packed placement |
| `relocation_transaction.rs` | Batch relocation subtransaction, provably-unobserved abort, and fail-stop quarantine |
| `transaction_validation.rs` | Batch-wide preflight and ref-count deltas |
| `protocol.rs` | Backend-independent request/result types |
| `facade.rs` | Construction, public queries and stable core facade |
| `manager_state.rs` | Private state records shared by the transaction modules |
| `test_model.rs`, `tests/` | Test-only executable model, full-scan oracles and fault traces |
| `state_checkpoint.rs` | Independent request-owned fixed-width state replace, retirement, ACK, abort, and quarantine |

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

Physical generation reuse requires semantic unreachability (the Semantic
Frontier), execution completion (the Execution Frontier), and exact backend
ACK; none of the three alone authorizes reuse.

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

## Token relocation and packed boundary

The live runtime preserves ABI8 while executing relocation as one
multi-request scheduler batch. It freezes all candidate views and mirrors, then
invokes native disposition mark, prepare, submit, and complete exactly once
each. After complete output and readback validation, it commits one aggregate
page-registry plan and one aggregate request-head replacement. The scalar
`relocate_tokens` API is retained only as a singleton compatibility wrapper
over this collective path.

The SGLang plugin receives the ordered prepared batch, flattens every request's
moves into one backend move, and records one relocation completion event. It
constructs and validates every ReqToToken, Full-to-SWA LUT, and request-mirror
plan before the first mirror write; only after all plans pass does it commit the
mirrors and issue one ACK for the flattened batch retirement set. A
producer-to-copy event orders relocation. The current completion model then
synchronizes eagerly on the host before publication. An asynchronous
consumer-stream wait and copy/compute overlap are not implemented.

The host-qualified repeated profile remains one request-private Full class under
a full-evacuation policy:

```text
append -> mark dispositions -> relocate -> publish -> exact ACK -> repeat
```

Host tests cover a first evacuation, append into the packed publication, a
second evacuation, exact retirement spans, ACK-gated generation reuse, and
final drain. The SGLang adapter also has host coverage for its periodic trigger:
after each reclamation it derives the next absolute boundary from the current
active length, and a missed boundary fails closed.

This does not compose all dense-root capabilities with packed roots. Relocation
admits private, unpinned, non-Prefix sources. Generation-safe packed request
fork and packed shared partial-tail COW append are host-tested through Rust,
raw ABI8, and Python FFI, including repeated COW generations. Packed Prefix
operations remain unsupported. These newer COW paths are outside the sealed
engine qualification. The SGLang trigger
likewise rejects a nonempty Prefix mirror before manager mutation.

This collective transaction is fail-stop, not end-to-end rollback. Before the
mark, admission failures leave the batch unchanged. Once the batch mark has
succeeded, any later prepare/copy/submit/complete/publication/mirror/ACK failure
or uncertain return fail-stops the runtime and does not restore the old
dispositions or heads. A callback that proves no copy was observed can abort the
relocation reservations, but that abort cannot undo the prior mark.

The engine-neutral CUDA harness exercises an independent payload oracle,
stale-member atomicity, real stream ordering, exact evacuation, completion
evidence, ACK-gated generation reuse, and final drain. This is narrow component
conformance, not sealed L3/L4, capacity, performance, or production evidence.

A separate clean, preflight-bound record qualifies only scoped request-private
Full relocation correctness and lifecycle. Independent hardware attestation,
performance, capacity, memory-saving, production, and broader engine features
remain unqualified. See the
[sealed relocation qualification](../results/h20-sglang-v0517-abi8-token-relocation-20260825/README.md).

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
state. This adapter passes host lifecycle and hostile-fault gates; engine
correctness is sealed only for the manifest-bound Prefix profile. Fixed-state
pair verification is not part of that seal, and performance remains
unqualified.

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

## Fixed-state checkpoint protocol

At the compiler and core-protocol level, recurrent Mamba/GDN/KDA/linear-
attention state and finite convolution state do not enter token snapshots or
TokenMove. This taxonomy does not imply that every family has an SGLang
binding. ABI8 exposes a separate
`OrbitKvStatePoolHandle` with request-owned fixed-width slots:

```text
prepare(owner, expected slot)
  -> source? + reserved destination + transition
backend clear/copy on the consuming CUDA stream
  -> exact observed/written receipt
confirmed completion event
  -> atomically publish destination + retirement certificate for source?
physical source clear
  -> exact ACK -> next generation may reuse the slot
```

This is the core/coordinator protocol. Initial publication has no source and
requires a backend clear before submit. Core replacement copies the complete
model-specific state from the current slot. Receipt mismatch or unknown
observation quarantines the affected owner and destination.

The restricted SGLang seam maps the pool's zero-based identity to
physical Mamba slot `slot_id + 1`, preserving slot zero as the dummy slot. It
connects request allocation, initial `MambaPool.clear_slots`, the forward
completion event, and release-time wait/retire/clear/exact-ACK. The currently
bound frontend profile separates token-addressable Full KV from
request-private GDN recurrent and convolution state. Prefix-state sharing and
the native Mamba free-list authority stay disabled. Startup fails closed unless
the compiled schedule, backend, dtype, state layout, and cache mode match the
accepted structural contract.
The production policy is generic over this structural GDN/convolution
capability; qualification evidence remains pinned to a specific model and
checkpoint rather than being inferred from the shared structure.

Scoped pair evidence verifies outputs, token/fixed-state lifecycle, completion,
and final drain, but remains `qualified=false`, `hardware_attested=false`, and
`performance_go=false`. This is pair verification, not L4 qualification. See
the [fixed-state pair-verification record](../results/h20-sglang-v0517-abi8-qwen35-fixed-state-pair-verification-20260823/README.md).

Same-owner
replacement through `MambaPool.copy_from` is covered only by coordinator and
real-CPU-tensor host tests; no production trigger exists yet. Other GDN
profiles, KDA, ShortConv, and other linear-attention family bindings remain
pending, as do L4 and performance qualification.

A later weight-backed run exercises the same ownership seam but remains
diagnostic-only because its source and qualification preflight do not meet the
release gate. Exact artifacts, environment, workload, and observations remain
in the [diagnostic archive](../results/h20-sglang-v0517-abi8-qwen38-fp8-diagnostic-20260824/README.md).

The token manager and state pool are independent handles. The adapter can
contain a partial failure by fail-stopping the process, but it provides no
cross-handle atomic commit or rollback.

## Complexity contract

- hot append/complete: `O(C + Δ log R + Δ log Δ)`, roughly
  `O(C + Δ log R)`;
- Prefix lookup: `O(B log N + B·C)`; and
- cold fork/attach/publish/release/evict ref aggregation: `O(P log U)`.

`C` is class count, `Δ` the changed-page count, `R` resident pages, `B` lookup
boundaries, `N` Prefix entries, `P` materialized pages, and `U` unique physical
pages. A partial-tail update must not traverse or materialize the full resident
root.

## Compatibility and acceptance

ABI8 exports exactly 40 `orbitkv_*` symbols listed in the
[Capability Matrix](capability-matrix.md): 29 canonical-manager symbols and 11
independent state-pool symbols. There are no ABI5 lifecycle aliases, older
loaders, or silent native-allocation fallback paths.

An engine profile becomes a replacement claim only after its native allocator
and Prefix owner cease to be authoritative; all fault and pressure gates pass;
and an append-only manifest binds the exact manager, wire, adapter, engine
release, hardware, commands, and outputs. Neither historical records, scoped
ABI8 seals, nor unqualified diagnostics satisfy those gates for a complete
SGLang replacement.
