# RuntimeSession

`RuntimeSession` is the transactional boundary between compiled attention-state
semantics and device execution. It owns a `CanonicalKvManager`; callers receive
plans and opaque operation identities, never allocator authority.

## Owned state

- request, Prefix, batch, publication, release, control, and relocation IDs;
- immutable request heads and token placement views;
- page identity and generation;
- prepared and submitted append transactions;
- Prefix lookup, attach, publish, eviction, fork, and COW state;
- semantic token disposition;
- execution completion high-water marks;
- retirement certificates, acknowledgement, quarantine, and reuse.

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

## Prefix and copy-on-write

Shared Prefix is an explicit cache policy. A Prefix stores an opaque snapshot
lease and indexes immutable state; it does not contain a page allocator.
Extending a shared partial page prepares a copy-on-write destination and exact
copy intent. The executor proves copy ordering before RuntimeSession publishes
the new request head. Request-private sessions reject Prefix operations.

## Relocation

Relocation starts from canonical token views and explicit token dispositions. A
collective relocation transaction selects destinations, emits component-aware
copy work, waits for device completion, atomically publishes changed request
views, retires source generations, and requires exact acknowledgement before
reuse.

Relocation is not automatically profitable. Full state should move only when a
fragmentation or pressure policy predicts a net benefit. Sliding and Chunked
state normally obtain reuse from their compiled lifetime boundaries without
copying live tokens.

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
quarantine, Prefix/COW, class-specific retirement, relocation, and repeated
generation reuse. The executor now gates relocation success evidence on a real
CUDA event, and real-device tests cover relocation followed by packed decode.
Ordinary append completion evidence is still supplied by the embedding runtime,
and no current test establishes matched model performance.
