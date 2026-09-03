# RuntimeSession Architecture

This is the native ownership architecture inside the single OrbitKV product.
The filename is retained as a stable link; `RuntimeSession` is not a standalone
engine or a generic multi-engine service. It is the Rust KV lifecycle authority
embedded in the complete pinned SGLang source product.

## Objective

For an admitted profile, exactly one component must decide which physical page
backs logical attention state and when that page can be reused. OrbitKV assigns
that responsibility to one Rust `RuntimeSession`. SGLang continues to schedule
requests, allocate tensors, run kernels, and execute CUDA work.

The design must provide:

- immutable, generation-checked request and Prefix snapshots;
- transactionally prepared physical effects;
- explicit semantic-death and execution-completion evidence;
- exact mirror-cleanup and retirement acknowledgement;
- fail-closed admission and stale-identity rejection; and
- no Python shadow allocator or fallback lifecycle authority.

The former neutral Python runtime, reference arena adapter, structured data
plane, and generic adapter SPI are removed history. They are not alternate
routes through this architecture.

## Layering

```text
declarative attention-state / retention semantics
  -> OrbitKV compiler
  -> RuntimeManifest
  -> RuntimeTarget + RuntimeBinding admission
  -> CanonicalKvManager
  -> Rust RuntimeSession
  -> typed session wire
  -> SGLang SessionRuntime coordinator
  -> reviewed overlay seams
  -> SGLang tensors, tables, kernels, and CUDA execution
```

The compiler produces checked physical programs. `CanonicalKvManager` provides
the generation-safe ownership primitives. `RuntimeSession` closes those
primitives behind engine-scoped identities and enforces cross-operation phase
ordering. The admitted SGLang route uses the session wire's engine-facing
effects rather than native manager capabilities. The raw manager C ABI has been
removed and is not a second SGLang lifecycle route. `SessionRuntime` coordinates session effects with SGLang. Exact current
wire and layout counts are maintained in the
[Capability Matrix](capability-matrix.md).

## Ownership model

One `RuntimeSession` owns by value:

- one `CanonicalKvManager`;
- session-private request and operation ID namespaces;
- request phase records;
- prepared and submitted append batches;
- pending publications, releases, and Prefix controls;
- completion-domain high-water requirements; and
- poison/quarantine state.

The engine sees stable request IDs and session-scoped operation IDs. It does
not receive `RequestLease`, `SnapshotLease`, `PageLease`, or other manager
capabilities that could be replayed through a second owner. Foreign, stale,
future, duplicated, reordered, or cross-session identities are rejected.

Python may retain only coordination state: SGLang request keys, positive
request-pool rows, submitted tickets, events, completion high-water marks, and
the minimal journal needed to reconcile an uncertain call. This state cannot
mint a native identity or make a page reusable.

## Admitted profile shapes

The current source contract admits five narrow native-session shapes:

| Shape | Physical/lifecycle characteristics | Product evidence |
| --- | --- | --- |
| Full token KV | Append-only page classes, shared page-aligned Prefix, COW | Host-tested; latest earlier-wire accelerator record has exact-source correctness and no benefit; live-wire device qualification pending |
| Ordered Full+Sliding token KV | Full class followed by Sliding class, two pools, shared Prefix, joint COW, checked Full-to-Sliding LUT | Host-tested; current real-device qualification pending |
| Pure Sliding token KV | Periodic slots, request-private cache, Sliding-leaf COW, semantic retirement | Host-tested; current real-device retirement/reuse qualification pending |
| Exact Chunked token KV | Canonical Retention IR, request-private resettable arena, epoch-end retirement | Host-tested; kernel/scheduler execution and device qualification pending |
| Full latent KV | Component-aware latent/RoPE geometry, request-private lifecycle | Host-tested; device and engine-E2E qualification pending |

All use the same ownership route. Cache-sharing policy and physical plan vary,
but neither creates another runtime. Unsupported topologies fail before arena
creation or mutation.

## Identities and snapshots

Logical identities are separate from physical locations. A request refers to
an immutable snapshot head. A snapshot contains class roots, and a root maps
logical blocks to generation-checked pages. Mutation path-copies only the
affected structure and publishes a new root after its physical effects are
confirmed. Readers may continue to pin an older root.

A page is identified by arena, physical index, and generation. Recycling the
same physical slot increments its generation, so a stale snapshot cannot name
new contents accidentally. Request, Prefix, reader, and writer references are
tracked independently.

The engine's `ReqToToken` row and, where applicable, Full-to-Sliding LUT are
derived mirrors. Logical token IDs and native snapshots remain stable even when
the mirrored physical binding changes.

## Physical page states

A page progresses through ownership states conceptually equivalent to:

```text
Free -> Reserved -> Writing -> Published -> Retiring -> Free(next generation)
                         \-> Quarantined
```

The exact internal representation is implementation-specific, but these rules
are invariant:

- reservation does not make data readable;
- publication requires confirmed writes and copies;
- semantic detach does not make storage reusable;
- pending readers or engine execution block reuse;
- retirement receipts must exactly match the pages offered for retirement;
- ACK precedes generation reuse; and
- uncertain observed mutation is contained, never guessed successful.

## Append transaction

### Prepare

The engine submits the admitted request identities and logical append ranges.
Rust checks request phase, expected snapshot heads, class geometry, capacity,
and copy-on-write requirements. It reserves pages and returns a complete ordered
effect plan containing physical writes, required copies, and mirror updates. No
new root is visible yet.

### Submit

After validating the entire effect plan, the bridge applies it to SGLang-owned
tensors and tables and submits execution evidence. Rust transitions the exact
prepared operation to submitted. Evidence count, order, request, class, page,
generation, and logical range must match.

### Complete and publish

SGLang creates the completion event on its execution stream. After observing
that event, the bridge asserts completion for the submitted ticket. Rust checks
the session, batch, completion domain, and monotonic value, then publishes the
new immutable request roots and returns any semantic retirements.

### Abort and quarantine

An operation can be aborted only while the backend effect is proven
unobserved. Once execution may have observed a destination, uncertainty cannot
be rolled back safely. The session quarantines the affected operation or
fail-stops according to the protocol so that ambiguous pages are never reused.

## Copy-on-write and Prefix sharing

Full and ordered Full+Sliding profiles support page-aligned Prefix lookup,
publish, attach, eviction, and atomic publish-and-release. Prefix entries own
opaque native leases, not raw page arrays.

Forking a request or extending a shared partial tail can require copy-on-write.
Rust prepares all affected class copies together. In the ordered two-class
profile, Full and Sliding effects retain their canonical class order and the
bridge validates both pools and the Full-to-Sliding LUT before writing either.
The new root is published only after all required copies and writes complete.

Pure Sliding, exact Chunked, and Full latent KV are request-private. They reject
every Prefix control rather than accepting an operation that their physical
plan cannot preserve.

## Semantic and execution frontiers

Let `Fs` denote the Semantic Frontier and `Fe` the Execution Frontier. A page
may return to the free pool only when both conditions hold:

```text
SemanticDead(page, Fs) && ExecutionComplete(page, Fe)
```

Semantic evidence is derived from the admitted plan: release, Sliding-window
retirement, epoch end, Prefix eviction, or an explicit token disposition.
Execution evidence comes from SGLang's real event/stream ordering. Neither can
substitute for the other.

Rust validates the shape and monotonicity of completion assertions. The bridge
is responsible for associating each assertion with the actual SGLang CUDA event
for the submitted ticket. This division preserves the real execution owner
without weakening the native lifecycle invariant.

## Release protocol

Release first proves that all submitted work has completed. Rust then prepares
the exact detach set and required device-mirror effects. The bridge validates
and clears only those rows and lookup entries, establishes completion of that
cleanup, and confirms exact ordered receipts. Rust ACKs the retirements and
recycles the request. Only after that result may SGLang free the request-pool
row.

Waiting-request cancellation follows the same authority rule: a request that
has begun a native Prefix attach or other control operation must be canceled or
finalized through the corresponding session operation. Dropping only the
SGLang object would leak or prematurely release native ownership.

## Sliding and resettable lifetimes

For Sliding state, the compiler derives periodic physical slots from the window
and page size. Completion can advance the semantic frontier and retire pages
whose logical range has left the window. Reuse still waits for exact execution
completion and ACK. Pure Sliding has no shared Prefix state or Full-to-Sliding
LUT.

For exact Chunked state, canonical Retention IR lowers to a request-private
resettable arena. An epoch boundary clears exactly the old epoch's logical
columns. Host tests cover retirement, confirmed completion, exact ACK,
generation-incrementing reuse, release, and drain. The host admission checks do
not observe the actual attention kernel or scheduler, so this is not a device
qualification.

## Relocation migration

Relocation preserves the semantic retained-token set while changing physical
placement. Its correct transaction requires a canonical token view, explicit
dispositions, a complete move plan, exact copy receipts, engine mirror
publication, source retirement, and ACK-gated reuse. A batch is collective:
after a disposition mark commits, a later failure is contained or fail-stopped,
not rolled back to an invented earlier snapshot.

Rust relocation logic, opaque `RuntimeSession` wire operations, and an SGLang
CUDA copy implementation exist. Product integration is still migrating to that
single session route. Its current qualification is host-only; an independent
CUDA conformance harness is component evidence rather than native-session E2E.
No capacity, memory-saving, latency, throughput, or production claim follows
from code existence.

## Fixed state

Recurrent and convolution state is not token-relocatable. The existing
generation-checked checkpoint pool keeps independent identities and supports
prepare, submit, completion, replacement retirement, release, exact ACK, abort,
and quarantine at host level. It is not yet unified atomically with token state
inside the admitted product profiles. Separate handles imply fail-stop
containment, not cross-handle commit or rollback.

## Failure model

The session rejects or contains:

- malformed, duplicate, foreign, stale, or reordered identities;
- short buffers and outputs exceeding fixed workspace bounds;
- unexpected page, generation, class, or logical-range effects;
- non-monotonic or wrong-domain completion evidence;
- partial mirror mutation and uncertain native returns;
- retirement receipt mismatch;
- reuse before cleanup and ACK; and
- unsupported cache policy or topology.

Retry is allowed only where the protocol defines an ID-only replay with no
duplicate physical effect. The first fatal uncertainty poisons the session; a
later operation cannot clear that poison by succeeding.

## Complexity contract

Hot operations are bounded by the affected batch, classes, tokens, pages, or
retirements. Native and Python workspaces are sized from admitted capacities. A
short-buffer result on a configured session is treated as a contract failure,
not permission to allocate an unbounded replacement buffer in the hot path.
Persistent roots avoid full-history copies for ordinary append.

## Acceptance boundary

Host tests establish transaction, fault, identity, and lifecycle behavior. The
latest earlier-wire Full engine evidence additionally establishes output
correctness, real forward-stream event observation, and final drain for its
exact closure, while showing no performance benefit. It does not qualify the
live wire. Full+Sliding and pure Sliding remain pending real-device
qualification. Relocation remains host-verified migration state.

See the [Capability Matrix](capability-matrix.md) for normative status and the
[SGLang Integration Contract](sglang-compatibility.md) for engine-side duties.
