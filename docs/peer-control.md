# Batched peer control over TENT

**Status: target design; the binary peer-control protocol is not implemented or
qualified.** This document defines one replacement for the current peer metadata
RPC path. It does not describe a deployed transport or establish a performance
improvement. See [the communication plan](communication-plan.md) for the wider
implementation sequence and [peer sharing](p2p.md) for current behavior.

## Current boundary

OrbitKV uses Mooncake TENT for remote payload READs. Catalog discovery, export
window creation, exact-residency authorization and export completion currently
use gRPC. Inventory snapshots and deltas also use gRPC. etcd supplies membership,
incarnations and placement through background registration and Watch; it is not
a per-request block directory.

The peer execution design uses one active READ and at most one authorization
lookahead. Payload buffers and the export guard move into the blocking READ
owner. Cancellation of the caller does not drop those owners before native
drain. A separate completion owner retries release until the source acknowledges
it. Local admission is bounded by 64 tickets per source and 1024 tickets globally.
The target retains those ownership limits and the single bounded-lookahead
strategy; there is no sequential-mode runtime branch in this design.

Catalog rows are candidate evidence. Only the source's exact-residency check and
export pin authorize a READ. A faster directory reply, a successfully posted
notification, a native SEND completion, a membership timeout and an application
completion ACK are different events. None of the first four proves that a
separately submitted payload READ has drained.

SGLang's TENT adapter uses synchronous WRITE through the same native loader,
without TENT notifications. SGLang retains its own P/D bootstrap and room
protocol; the Manager peer-control session does not replace them. The adapter's
`session_id` is a transfer endpoint string, not the fenced session nonce defined
below. vLLM P/D is a notification consumer and must migrate with the binary ABI.

## One target transport

Use binary, length-aware TENT notifications for peer control. Keep one session
protocol over both TENT RDMA notifications and TENT TCP notifications. Select the
physical notification transport explicitly for each peer at bootstrap. There is
no gRPC fallback for a failed or unsupported hot request and no parallel legacy
peer-control runtime. An unsupported protocol/build fails bootstrap. Ordinary
cache misses, unavailable candidates and recomputation remain planner outcomes.

| Operation | Target path | Responsibility |
| --- | --- | --- |
| Session bootstrap, health and low-frequency catalog heartbeat/unregister | gRPC | Discover the TENT endpoint, establish session identity and limits, report inventory progress, perform management |
| Batched catalog Locate / LocateReply | Binary session, foreground lane | Return bounded, ordered candidate rows under the current placement |
| Batched Grant / GrantReply | Binary session, foreground lane | Authorize exact source residencies and retain source allocations |
| Batched Complete / CompleteAck | Binary session, reserved cleanup lane | Retire known tickets after requester READ drain or cancellation before submission |
| Receipts, credit snapshots and session retirement | Binary session, reserved control capacity | Bound reliable delivery state and finish existing cleanup |
| Inventory Begin / Page / Delta / Commit and progress replies | Binary session, background lane | Maintain and repair directory evidence without competing for foreground admission |
| Membership registration, renewal and Watch | etcd, background only | Fence node incarnations and publish placement |
| KV payload READ | TENT data plane | Transfer authorized bytes and report native terminal status |

Inventory repair means replaying a retained journal or rebuilding a snapshot
after a gap, catalog restart or placement change. It is not payload replication
or a new replication policy. Its messages use the same binary session with a
separate budget and scheduling weight. Keeping inventory traffic on gRPC in the
final implementation would leave an unnecessary second metadata data path.
The retained low-frequency heartbeat is a management/progress operation; it
must not become the per-batch inventory ACK or the per-lookup dependency.

## Native prerequisites

The inspected Mooncake source is
`719735896c86b56fabec6cf3e825fb2ea640597a`, the revision pinned by
`orbitkv-mooncake-sys` (`v0.3.13.post1`). Its C++ notification support is useful,
but the current C ABI and receive path do not satisfy this protocol.

| Observed behavior | Required change before cutover |
| --- | --- |
| `tent_send_notifs` builds strings from NUL-terminated pointers. `tent_recv_notifs` copies into `name[256]` and `msg[4096]`, truncates to 255/4095 bytes and does not explicitly terminate a full-length copy. Its returned handle is always zero. | Provide one length-aware binary ABI for send, submit-with-notify and receive. Return explicit lengths and validated transport peer identity. Remove the old string ABI from OrbitKV's loaded API. Raw protobuf must never pass through `CString` or lossy UTF-8 conversion. |
| Send returns through the first installed notification-capable transport. Receive also reads only the first such transport. Neither operation implements peer-specific selection or receive aggregation. | Select the negotiated transport for each destination and drain all installed notification ingress paths fairly. Do not infer notification routing from the transport selected for a payload batch. |
| RDMA and TCP append received notifications to unbounded native vectors. The C receive wrapper drains and allocates for the entire vector. | Bound ingress at native enqueue by bytes and records, including per-peer quotas and reserved cleanup capacity. Expose a bounded receive operation; reject or drop overflow with observable counters and application retries. A bounded Rust channel alone is insufficient. |
| RDMA send waits on a condition variable for one of 256 SEND slots without an application deadline. | Expose nonblocking `try_send` admission or a native deadline-aware send. Distinguish accepted, busy, oversized and unavailable. Never make an async dispatcher wait indefinitely inside this call. |
| The notification QP uses 64 KiB buffers and up to 256 pending sends. Its send and receive rings can consume roughly 32 MiB per initialized endpoint before other resources. | Account for native endpoint memory and limit warm peer connections. Make ring capacity configurable for the pinned control workload; a small wire frame does not by itself reduce these allocations. |

These changes belong in the pinned native runtime and a small C ABI extension
built with it. A shim around the existing string functions cannot fix truncation,
transport selection or an already unbounded native queue. Do not add a second
TransferEngine to work around the selector: it would duplicate endpoint state,
registration ownership and progress resources.

The binary ABI must expose a build/protocol capability check, explicit buffer
ownership, bounded poll count/bytes and per-peer admission results. On successful
send admission, native code must own a copy of the frame until SEND completion;
Rust may then reuse the supplied slice. If an implementation instead retains the
slice, the Rust API must return an owner that keeps it alive to native completion.
Receive buffers must be returned exactly once, including decode failures.

RDMA enqueue success is not an application ACK. The TCP implementation calls
TENT's `ControlClient::notify`, so this design removes OrbitKV's hot gRPC method
dispatch but does not turn the TCP backend into an RPC-free transport. Both
backends must implement the same limits, delivery identity and application
protocol. Mixed RDMA/TCP hosts are a qualification case, not an assumed property
of the current selector.

Relevant pinned implementation:

- [C notification ABI](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/mooncake-transfer-engine/tent/src/transfer_engine_c.cpp#L262-L298).
- [Notification transport selection](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp#L2264-L2300).
- [RDMA SEND, receive enqueue and completion](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/mooncake-transfer-engine/tent/src/transport/rdma/endpoint.cpp#L1112-L1235).
- [RDMA transport receive queue](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/mooncake-transfer-engine/tent/src/transport/rdma/rdma_transport.cpp#L780-L794).
- [TCP notification path](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/mooncake-transfer-engine/tent/src/transport/tcp/tcp_transport.cpp#L299-L324).

## Bootstrap, identity and fencing

Replace export-window-only bootstrap with a peer-session bootstrap. The request
and response bind both node incarnations, the expected destination incarnation,
both TENT endpoints, a fresh session identifier, the protocol/build identity,
the selected notification transport and negotiated limits. The source returns
the export window identifier before any Grant can be sent. This removes the
current circular dependency in which the requester learns the transfer endpoint
from a successful grant response.

The session belongs to an exact pair of incarnations and a fresh random nonce.
The receiver binds the native peer connection to that session; sender identity
in an untrusted payload alone is insufficient. This is an extension of the
existing trusted cluster boundary, not a new authentication mechanism. Session
bootstrap must use the deployment's existing authenticated/trusted peer policy.
Do not claim cryptographic authentication from an incarnation or nonce.

Each frame includes sender incarnation, receiver incarnation, session identifier,
lane, message sequence, logical-message identifier, fragment index/count and
declared lengths. Each grant entry additionally carries its window, slot and
generation; each lookup/inventory entry carries its catalog placement/epoch and
route. Session identity does not replace any of those semantic fences.

Message sequences increase monotonically per direction and lane. Counters and
ticket generations never wrap; exhaustion stops new admission and establishes
a fresh session after existing ownership has been accounted for. Reconnecting
to the same address does not make an old frame valid for a new process.

A replacement session can admit new work only within the same node-wide budgets.
An old session with outstanding exports or cleanup remains a bounded retiring
owner. Its exact Complete/CompleteAck traffic is still accepted. It cannot issue
new grants, revive old lookup evidence or transfer a ticket into the new session.
Membership fencing rejects new Locate, Grant and inventory mutations while
allowing already-authorized cleanup to finish.

If the requester dies without proving READ drain, membership expiry alone must
not free its exported memory. The initial implementation retains those bounded
exports and exposes the retained bytes/tickets. Any automatic crash reclamation
requires a separately qualified native connection revocation/drain barrier that
proves the old peer can no longer access them. A new incarnation is not that
barrier. A source restart invalidates old addresses; requesters stop submitting,
drain local native batches and treat the route as failed.

## Wire format and fragmentation

Use one versioned binary schema in `orbitkv-proto`, carried by a fixed-size
envelope and length-delimited bodies. Decode the envelope before allocating or
decoding a body. Use explicit integer byte order, checked arithmetic and exact
16-byte UUID/incarnation representations. Reject unknown versions, invalid
lengths, wrong receiver/session, invalid kinds and counts before dispatch.

The first implementation uses **bounded notification fragmentation**, not a
registered metadata arena. An arena would introduce remote-writable slot
ownership, publication ordering and another generation/reuse protocol before
the grant protocol is even replaced. Payloads remain on the existing TENT READ
path; no new one-sided metadata WRITE ring is needed for this increment.

Initial engineering limits are 16 KiB per complete notification frame, 1 MiB per
logical message, and at most 128 block keys per logical Locate or Grant entry.
The fragment count is derived with checked arithmetic from the payload capacity
after the envelope; it is not `1 MiB / 16 KiB` with the header forgotten. These
are proposed limits to qualify against supported layouts, not measurements or
claims about existing maximum response sizes. The existing 64 KiB discovery
query limit and endpoint/namespace limits remain independently enforced.

Batch by peer, operation and existing logical plan entries. Preserve a result
for every request entry; do not silently collapse partial grant failures into a
single batch status. Admission and layout validation must estimate or bound the
encoded response before pinning a source. Split oversized groups by block and
byte budget. If one supported block descriptor cannot fit, resolve that layout
limit before enabling the protocol for it; do not pin first and discover an
unsendable reply afterward. Bound codec metadata and repeated slot counts as
well as hashes and request bytes.

Only a complete, validated logical message can enter a semantic handler. A
fragment cannot authorize a partial export or advance an inventory cursor.
Reassembly reserves the full declared logical length against peer and global
limits before accepting the first fragment. Reject overlapping or conflicting
fragments, invalid offsets, mismatched headers and a different payload for an
already observed message identifier. Duplicate identical fragments are harmless.
Retain bounded replay state so expiry of a reassembly buffer cannot turn an old
message into a fresh request.

Use one retry owner for each logical message. After a bounded retransmission
interval it resends the same identity and bytes; it never invents a new ticket
because the reply was lost. Partial reassembly may expire without touching
export ownership. Exceeding a foreground deadline returns a transport outcome
to its caller and transfers any known ticket to cleanup. It does not free the
native send buffer or any submitted READ owner.

## Admission, credit and progress

There are three scheduling classes: cleanup/control, foreground and inventory.
Reserve cleanup capacity at every stage: native receive admission, native send
admission, Rust inbound/outbound queues and semantic dispatch. Bulk backpressure
must not consume the last slot needed to close an existing export. Cleanup gets
priority, while bounded scheduling weights ensure foreground and inventory make
progress under sustained completion traffic.

Credits count records **and bytes**, not just notifications. Session bootstrap
grants explicit per-lane receive/reassembly capacity, subject to global limits.
A request reserves its request bytes, response/replay capacity and work slot
before execution. The sender bounds pending requests, outgoing bytes and retained
retry buffers too. The source's export bytes, staging bytes and ticket slots are
separate budgets; receiving a tiny Grant does not authorize unlimited source
memory. Retain the existing catalog lookup coalescing, global four-owner limit
and per-owner admission instead of replacing them with an unbounded dispatcher.

Delivery receipts and credit snapshots are monotonic, idempotent state updates
within a session. They carry cumulative progress plus a bounded selective bitmap
for out-of-order messages. Never add a returned credit twice on duplicate receipt.
They are coalesced into bounded per-peer control state and do not recursively
require acknowledgements of acknowledgements. A retry or management refresh can
recover a lost snapshot. Credit state cannot be reset by reconnecting while old
messages still own receive/replay capacity.

Native overflow must be visible and bounded even when peers duplicate traffic,
the Rust pump stalls or a sender violates credits. The initial native policy is
to reject/drop excess frames before enqueue, with per-peer/lane counters; the
application retries under the same message identity. Avoid blocking a shared
CQ/progress thread behind one full peer queue. Native ingress classification
must validate the small envelope and mapped session before charging a reserved
cleanup quota. Merely labelling a bulk message as cleanup must not bypass limits.

Maintain one receive pump per TransferEngine. It drains bounded native batches
and routes frames to the session owner. Independent callers must not race on
`take_notifications` and steal each other's frames. Use bounded native IO work,
not one unbounded `spawn_blocking` task per frame. Heavy catalog work, SSD staging
and payload READs run outside the pump. Warm endpoint setup and polling CPU are
explicit resources; measure idle and busy polling modes before selecting policy.

## Grant and completion state machine

The requester reserves its global permit, peer slot and next generation before
it sends authorization. The resulting known ticket has a cleanup owner even
when the Grant request or GrantReply is lost. A batch contains independently
identified tickets; retry and retirement apply per entry.

| State/event | Required action |
| --- | --- |
| New Grant | Validate current owner/session, exact residency sequence and query limits; reserve source admission once; pin DRAM or reserve and stage SSD once |
| Duplicate Grant while pending | Attach to the existing result; do not run admission or SSD staging again |
| Duplicate Grant after reply | Replay the same result while that ticket remains authorized; never create a second export |
| Grant accepted by requester | Validate every returned key, layout, byte range and identity before READ; retain source guard and destination owners with the native operation |
| Complete before Grant, or before full Grant reassembly | Install a generation tombstone; a later Grant for that ticket cannot resurrect it |
| Complete during SSD staging | Mark the ticket retiring; suppress Grant delivery and retain staging buffers until the staging owner drains |
| Complete after requester READ drain | Retire the matching export and free its source owners; then make CompleteAck available |
| Duplicate Complete | Return the same terminal ACK, or join the pending retirement; do not affect another generation |
| Late GrantReply after local cancellation | Never submit READ; the known ticket remains in cleanup until CompleteAck |
| CompleteAck consumed | Free the matching local completion slot and global permit atomically with respect to waiters; publish the release wake afterward |

The source's current authorization operation is single-use by ticket. It cannot
simply be called again on a network retransmission: a second call can return a
stale-ticket error after the first call succeeded. Add bounded in-flight/result
deduplication around that authority, keyed by session, request entry and ticket,
with payload identity validation. Source generation watermarks remain the final
protection against old requests after a replay entry is retired.

A received-message receipt only releases transport/replay resources. It is not
CompleteAck. The target CompleteAck means the exact ticket is irreversibly closed
and its source resources have finished retirement. For cancellation during SSD
staging this can be later than the current release method's return; use a
retirement completion owned by the staging operation. Do not block the receive
pump waiting for it. Closed-ticket tombstones still permit bounded idempotent
ACK replay without retaining the original large GrantReply.

Retain reply state until its consumption receipt or semantic retirement. Keep
retired request sequence watermarks so a delayed duplicate cannot be reexecuted
after reply eviction. Complete supersedes a stored GrantReply: a source must
never replay addresses after that ticket is closed. A late reply already in
flight is rejected by the requester's ticket state. Sequence replay windows,
reply storage and retiring sessions all need global byte/count admission.

Requester completion starts only after an unsubmitted grant is abandoned or the
actual native READ owner reports drain, including partial submission, status
failure, timeout and cancellation. Preserve `run_with_buffers` ownership. Do not
replace that fence with TENT submit-with-notify or native SEND completion.

Completion is asynchronous to payload drain. The native READ thread enqueues
retirement and returns; it does not wait for remote ACK. A caller timing out does
not cancel the cleanup owner. Cleanup retries use backoff and bounded storage
until application ACK or a qualified native revocation barrier; elapsed time is
never a release proof.

Keep the existing resource-pressure rule: snapshot only already-releasing
`(slot, generation)` entries before attempting authorization. On source budget
rejection, wait at most three seconds for that snapshot and retry admission once.
Do not wait for active READs or extend the snapshot with later tickets. Register
the wake before checking the predicate, regard a changed generation as completion
of the old ticket, and release local permit/slot under the same state lock before
notifying. Foreground request deadlines and this pressure wait are separate bounds.

## Catalog discovery and repair

Locate keeps the current cache, duplicate-lookup coalescing, bounded fanout and
ordered one-row-per-key response. Each request carries the shard routes and
placement identity. The server checks assignment before work; the requester
rechecks it before accepting a reply. A complete reply with zero candidates is
directory evidence, while timeout, malformed replies and stale placement remain
distinct transport/route failures. None proves global absence.

Extract the catalog's existing validation and store operations into the behavior
owner and call it from the binary dispatcher. Do not maintain copies in a tonic
adapter and a new handler. The same applies to source authority and transfer
descriptor serialization.

Inventory preserves the existing source incarnation, catalog epoch, inventory
generation, page cursor, journal sequence and explicit Commit-ready transition.
The session envelope is an additional fence. An InventoryReply acknowledges
applied progress, including reclaimable records; notification receipt cannot
advance the source cursor or satisfy `flush`. Retrying Begin/Page/Delta/Commit
must resolve from the operation's original result or current validated progress,
never apply a delta twice or skip a page because delivery succeeded.

After a catalog epoch or placement change, restart or resume using the existing
progress rules. A journal gap starts a new snapshot generation. Background
transfer fragmentation is below the inventory operation boundary: a half-delivered
page is not partially applied. Store jobs and serialization/replay bytes have
background quotas; catalog repair must not delay a CompleteAck behind a large
snapshot. Management heartbeat verifies liveness and reconciles progress after
ambiguity, while individual inventory messages carry their own application ACKs.

## Modules and startup ownership

| Module | Concrete change |
| --- | --- |
| Pinned TENT source and `orbitkv-mooncake-sys` | Implement/load only the new binary notification ABI; add bounded native queues, per-peer transport selection and bounded send/poll semantics; verify the required native build capability |
| `orbitkv-transfer/src/engine.rs` and `types.rs` | Own length-aware frame buffers and explicit send outcomes; expose bounded receive; retain native engine/buffer lifetime through completion |
| `orbitkv-proto` | Define one peer-control envelope/body schema, bootstrap request/response and transport-independent error codes; remove migrated hot RPC methods |
| `orbitkv-core/src/peer/control/` (new) | Own sessions, framing/reassembly, batching, credits, deduplication, deadlines and cleanup scheduling; one owner, not a forwarding facade over the old RPC client |
| `peer/completion.rs` | Replace `EngineClient`/tonic windows with session tickets and batched CompleteAck; preserve permit, generation, release snapshot and READ guard ownership |
| `peer/export.rs` | Retain exact-residency authority; add duplicate-result ownership and retirement completion without changing source pin/drain rules |
| `peer/catalog/lookup.rs` and `sync.rs` | Send foreground Locate and background inventory messages through the session; preserve cache/fanout/flush/epoch semantics |
| `orbitkv-catalog/src/service.rs` | Expose the existing semantic handlers to binary dispatch; retain only low-frequency management RPC adapters |
| `orbitkv-server/src/peer.rs` and `lib.rs` | Bootstrap sessions, wire source/catalog handlers, start and stop the control owner with Manager lifecycle; remove migrated tonic service methods |
| `python/src/mooncake.rs` and vLLM P/D notification users | Move to the same binary native ABI; perform any required text conversion explicitly at the Python API boundary, without retaining the old C string binding |
| `python/orbitkv/sglang/pd.py` | Keep SGLang's synchronous WRITE contract and bootstrap ownership; qualify the shared native loader/API change with adapter units and the two-GPU P/D gate |

Construction currently creates catalog workers before TENT and creates server
catalog handlers after the core engine. Make control startup explicit: construct
storage and TENT, construct the semantic source/catalog owners, attach the
dispatcher once, then start inventory workers and mark bootstrap ready. Avoid a
strong-reference cycle from engine to pump to server engine. Embedded tests must
exercise the same explicit startup and shutdown contract.

Shutdown stops new bootstrap/Grant/Locate admission first, drains or cancels
foreground tasks through their existing owners, and keeps cleanup/native progress
alive until safe retirement. A shutdown deadline reports retained ownership; it
does not authorize early unregister/free of memory still reachable by native
work. Destroy native progress and registrations only after their owners drain.

## Cutover and deletion

Implement and qualify native prerequisites first, then the session and semantic
handlers. Cut over Locate, Grant, Complete and inventory together for an enabled
binary protocol build. Old and new nodes are not an interoperability target;
rebuild/deploy the communicating cohort together. Use an immutable previous
commit for the performance baseline, not a runtime legacy-transport switch.

Delete the following when their replacement is connected and qualified:

- Hot `EngineClient` authorization/release calls and `QueryBlocksForTransfer` /
  `ReleaseTransferLock` RPC service methods. Replace window-only bootstrap with
  the single session bootstrap.
- Catalog `LocateBlocks` and `SyncInventory` RPC methods, hot tonic client caches
  and helpers used exclusively by them. Retain management heartbeat/unregister
  clients only where they own those operations.
- Duplicate gRPC/binary request conversions, obsolete RPC-specific test servers
  and message definitions that have no remaining consumer. Share semantic test
  fixtures and authority handlers instead.
- Loaded string notification C ABI symbols, fixed-string receive decoding and
  any fallback loader for the prior notification ABI. Update submit-with-notify
  and P/D consumers in the same change.

Already removed in the current communication worktree:

- The `ORBITKV_PEER_PIPELINE` runtime selection and sequential execution branch.
  Bounded lookahead is the sole execution strategy; overlap, ownership,
  cancellation and contiguous-prefix tests remain.
- Historical `restore_delivered` timeline parsing, `completion_delivery_ms`,
  legacy-delivery coverage fields and their fixtures. The current completion
  path has no such event. Client submit-to-ready, worker completion,
  publication/notification and observation coverage metrics remain; missing
  required observations are missing, never zero.

Rename metadata metrics such as `candidate_lookup_rpcs` and `discovery_rpc` to
describe logical requests/stages. Count physical notifications, retransmissions,
fragments and native overflow separately. A batched packet is not a catalog
query, and a retry is not a new logical transfer-plan attempt.

## Qualification and acceptance

All rows below are required evidence for their scope. None is a claim that the
new protocol has already passed. Reuse the ownership/fault workloads in
[shared-cache qualification](shared-cache-qualification.md) and
[fault qualification](fault-qualification.md).

| Gate | Required evidence |
| --- | --- |
| Binary ABI | Embedded NUL/non-UTF-8, exact frame limit and overflow, zero/invalid lengths, ownership on every error path, old ABI rejected at startup, P/D notification consumers rebuilt |
| P/D consumers | vLLM notification protocol gates plus SGLang adapter units and two-GPU P/D correctness after the shared loader/API change; the SGLang TCP gate does not qualify RDMA |
| Native boundedness | Pause Rust receive while flooding valid/duplicate/invalid traffic; cap native and Rust bytes/counts; cleanup progresses under bulk exhaustion; busy send has a bounded return path |
| Framing | Reordering, duplicate/lost/conflicting fragments, checked length overflow, wrong session/incarnation, expiry and replay-window pressure; no semantic execution before complete decode |
| Grant idempotence | Lost GrantReply, duplicate in-flight SSD grant, conflicting duplicate identity, Complete before first/last fragment, closed-ticket reply replay, source-byte accounting held exactly once |
| Completion | Lost/duplicate Complete and ACK, active READ cancellation, partial submit/status failure/timeout, SSD staging cancellation, stale generations, multiple waiters and release wake races |
| Admission pressure | Three source segments with a slab-sized budget; delayed old release, a rejected new ticket's ACK arriving first, bounded wait expiry and slot reuse; no active-READ wait or permit race |
| Session lifecycle | Bootstrap duplication, endpoint reconnect, both peers restarting independently, old/new sessions overlapping, membership fencing, maximum retiring sessions; no address replay or timeout-based unpin |
| Catalog | Ordered Locate rows, coalescing/fanout budgets, stale placement, inventory Begin/Page/Delta/Commit reply loss, epoch takeover, journal gap repair and `flush` application progress |
| Transport matrix | Same-host TCP, cross-host TCP, cross-host RDMA, RDMA/TCP mixed peers and multiple active native transports; verify actual notification transport and all ingress queues |
| Payload correctness | Existing GPU/CPU byte gates, DRAM and owner-staged SSD, multi-segment prefix preservation and source/destination lifetime; metadata success alone is insufficient |
| Installed artifact | Required native symbols/capabilities in the wheel/image and installed Manager; no dependency on an unrelated source checkout or legacy library |
| Performance | Matched previous-commit workload: end-to-end restore/TTFT/ITL and goodput, metadata p50/p95/p99, batch occupancy, queue wait, retransmits, retained source bytes, RSS/pinned endpoint memory and polling CPU |

Report native SEND admission, application GrantReply, payload drain, source
retirement and CompleteAck timings separately. Retain a caller-observed complete
restore measurement. Start with the existing real multi-segment transfer and
serving gates; measure isolated notification latency only as supporting evidence.
A same-host TCP result does not qualify RDMA or mixed-cluster behavior. No
transport speedup should be claimed until the same cache state, request mix,
memory budgets and CPU resources pass matched end-to-end comparison.
