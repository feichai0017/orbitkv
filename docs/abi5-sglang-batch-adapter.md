# Historical ABI5 SGLang Adapter and ABI6 Migration

Status: **ABI5 and the subsequent ABI6 migration are historical. The live tree
is ABI8.**

This page is retained because it documents the historical ABI5 protocol and
its evidence boundary. It does not define the current C ABI and must not be used as a
compatibility guide for the live ABI8 tree. Current capabilities are normative
only in the
[Capability Matrix](capability-matrix.md).

## Frozen ABI5-v5 boundary

The append-only record binds the exact source, engine, frontend profiles,
backends, hardware, and workload. Within that frozen scope, outputs, arena
drain, Hybrid retirement, and grouped release pass. It remains a historical
scoped correctness result with `performance_go=false`; no general speedup,
same-capacity memory benefit, or live-ABI qualification follows. See the
[ABI5-v5 result](../results/h20-sglang-v0517-abi5-v5-grouped-release-20260821/README.md).

## What ABI5 did

ABI5 kept the canonical root inside Rust and exposed a compact batch lifecycle:

```text
acquire request
  -> prepare: previous tail + fresh write intents
  -> backend bind
  -> submit: request/submission leases only
  -> GPU forward and shared event
  -> complete: publication scalars + retirement certificates
  -> mirror clear + synchronization
  -> acknowledgement
  -> grouped release/recycle
```

Python retained incremental `RequestCursor` and `PageShadow` validation state.
ReqToToken and the Full-to-SWA LUT were checked mirrors of manager-selected
pages. They were not free lists or allocation authorities.

The frozen adapter preflighted ordered spans and output capacities before
manager mutation. Unknown backend or CUDA outcomes entered quarantine. A
certificate could authorize reuse only after mirror cleanup and an exact
generation-bearing acknowledgement.

## Historical ABI6 migration

ABI6 was the intermediate migration that kept the ownership principles while
replacing the identity and transaction model required for sharing:

| ABI5 | ABI6 |
| --- | --- |
| One mutable request publication lineage | Request head points to an immutable `SnapshotLease` |
| Private tail or fresh-page lowering | Explicit none/in-place/fresh/COW `TailAction` |
| Bind receipts | Bind plus exact copy receipts ordered before writes |
| Request-owned resident pages | Page-owned request refs, Prefix refs, reader pins and writer |
| Release implied mirror clearing from certificates | Separate `DetachedBinding` CLEAR/REPLACE actions |
| No public fork or Prefix ownership | Batch request fork and page-aligned Prefix lifecycle |
| Scalar-shaped abort/quarantine/ACK/recycle names | Consistent `_batch` names |

At that historical stage, the ABI6 C library was host-qualified L2 with exactly
23 exported symbols and did not retain ABI5 aliases. Those counts do not
describe the live ABI8 wire; see the [Capability Matrix](capability-matrix.md)
for the current surface.

## ABI6 Python/runtime split

The ABI6 migration separated the Python source into three layers:

```text
ffi/
  layouts + symbol loader + workspaces + typed manager calls

runtime/
  identities + snapshot shadow + journal + completion + reclamation

plugin/
  SGLang validation + lowering + mirror cleanup + hooks + facade
```

This organization removes the former monoliths and prevents
ctypes layout, lifecycle journals, and SGLang hook policy from becoming one
review surface. Its historical L2 gate required ABI6 host tests, fault paths,
exact-symbol checks, and stale-lease cases against the frozen source under
qualification.

## Historical ABI6 SGLang Prefix seam

The ABI6 adapter required an `OrbitKVPrefixCache` at the released SGLang cache
factory seam. A Radix node could store:

- token/digest metadata used for lookup;
- an opaque `PrefixLease`; and
- engine-local LRU metadata.

It could not store authoritative page IDs or generations, free pages, fabricate
Hybrid SWA roots at structural split points, or publish an unaligned endpoint.
Only a manager-returned lookup hint can be attached, and attach revalidates it
against an acquired empty request and expected snapshot head.

On a warm hit, the adapter rebuilds ReqToToken/LUT mirrors from the manager's
cold materialized view. On divergence, exact COW copies the shared partial tail
before new writes. On eviction, mirror detaches, completion ordering,
reclamation ACKs, and Prefix recycle follow the manager result rather than a
Radix free operation.

## Historical ABI6 adapter qualification gate

Before that surface could be marked L2, it had to pass:

- ctypes size/alignment/offset parity with `orbitkv.h` and ABI version 6;
- exact 23-symbol loading with no compatibility alias;
- Full and Full+SWA acquire/fork/prepare/submit/complete/release lifecycles;
- Prefix lookup, stale hint, publish, publish-release, attach, evict, ACK, and
  recycle;
- COW bind/copy receipt ordering and source/destination mismatch quarantine;
- batch-global ref aggregation when requests and Prefixes share pages;
- short buffers and malformed spans with zero mutation;
- collective mirror CLEAR/REPLACE preflight before any mutation;
- exceptions and ambiguous GPU outcomes entering fail-stop; and
- pinned SGLang hook checks with all unsupported modes rejected.

The next gate at that stage was a fresh manifest binding the ABI6 core, C
library, Python runtime, SGLang patch, model profiles, commands, and outputs.
ABI5-v5 results could not be copied forward, just as neither ABI5 nor ABI6
evidence qualifies the live ABI8 tree.
