# RuntimeSession

`RuntimeSession` is the transactional boundary between compiled attention-state
semantics and device execution. It owns a `CanonicalKvManager` and, for hybrid
models, generation-checked recurrent/convolution pools. Callers receive plans
and opaque operation identities, never allocator authority.

## Owned state

- request, Prefix, batch, publication, release, control, and external-transfer IDs;
- immutable request heads and manager-authored page views;
- page identity and generation;
- prepared and submitted append transactions;
- Prefix lookup, attach, publish, eviction, fork, and COW state;
- execution completion high-water marks;
- retirement certificates, acknowledgement, quarantine, and reuse.
- fixed-state source/destination slots, private transition capabilities, and
  the last real completion frontier that published each owner.

The session is synchronous state-machine code. The async device executor owns
streams and events and calls session transitions in their required order. This
keeps CUDA dependencies out of the manager without weakening ownership.

## Append transaction

```text
acquire request
  -> prepare append batch
  -> lower manager-selected pages and copies
  -> execute copies, writes, attention, and sampling
  -> submit execution evidence
  -> observe device completion
  -> publish request heads and detach dead state
  -> acknowledge exact retirements
  -> recycle eligible generations
```

Validation covers the complete batch before physical effects begin. A prepared
transaction may be aborted only when the executor proves it was unobserved. An
ambiguous post-mutation failure is quarantined or fail-stopped; OrbitKV does not
invent rollback evidence.

For hybrid fixed-state sessions, each append step contains token-KV actions and
one state plan per recurrent/convolution class. The session preflights candidate
pools before mutating the KV manager, validates all state receipts before either
side is submitted, and publishes both sides from the same completion domain and
value. A contradiction quarantines the complete request batch. Request release
retires fixed slots against their last confirmed completion and makes them
reusable only after semantic release. Prefix sharing is rejected for these
sessions until recurrent-state sharing semantics are defined.

The public plan exposes state id, source/destination slot, and byte count but
never the private transition or retirement capability. The ordinary token-KV
executor cannot manufacture a fixed-state success receipt. The CUDA state
executor resolves those leases into stable arena ranges, initializes each
destination from its published source (or zero), uploads manager-selected slot
ids, and executes the Luminal graph. Only an opaque receipt tied to the exact
runtime binding and recorded after both graph execution and any selected
same-stream copy-back can release observed write evidence. Recurrent state
currently requires direct in-place mutation; convolution history may use a
measured copy-back schedule until a fused candidate is qualified. Rebinding the
same pointer creates a new identity, so an older receipt cannot certify it.

## Prefix and copy-on-write

Shared Prefix is an explicit cache policy. A Prefix stores an opaque snapshot
lease and indexes immutable state; it does not contain a page allocator.
Extending a shared partial page prepares a copy-on-write destination and exact
copy intent. The executor proves copy ordering before RuntimeSession publishes
the new request head. Request-private sessions reject Prefix operations.

## Frontiers

The Semantic Frontier proves that admitted future queries cannot read a state
generation. The Execution Frontier proves that already-submitted device work no
longer references it. A generation becomes reusable only after both frontiers
advance and the executor acknowledges the exact retirement certificate.

## External replicas

External export is a RuntimeSession transaction, not a raw storage callback. It
pins the exact immutable snapshot pages, blocks mutation of that request during
the initial export, validates exact durable copy receipts and a monotonic
completion point, and only then publishes an external replica catalog entry. A
failed receipt retains pins; abort requires proof that the backend was
unobserved. Concrete stores and transports never receive page-lifecycle authority.
Restore into an empty request is the symmetric transaction: the normal append
allocator selects fresh local pages, an adapter fills only those destinations,
and exact restore receipts are submitted through the normal binding and
completion path before the request head becomes visible.

Host tests cover ordering, stale identities, hostile evidence, abort,
quarantine, Prefix/COW, class-specific retirement, external transfer, joint
token-KV/fixed-state completion, and repeated generation reuse. Append
completion evidence is still supplied by the embedding runtime. Released-model
tests establish narrow correctness and lifecycle benefits; they do not establish
broad model or serving superiority.
