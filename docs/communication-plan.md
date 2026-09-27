# Communication implementation sequence

The goal is to reduce the time between a valid recovery decision and the
engine consuming the required state. Transport ping latency alone is not the
acceptance criterion. A complete engine-owned HBM hit must not require a new
synchronous Manager call; external recovery must be compared with native
offloading under the same CPU, HBM, host-memory and quality budgets.

## Ownership boundaries

| Resource | Owner | Completion needed for reuse |
| --- | --- | --- |
| Engine GPU pages | Inference engine | All submitted readers and writers have completed |
| Local descriptor slot | Process channel session | The matching descriptor response has been consumed |
| Local restore result slot | Process channel session | The matching terminal result has been acknowledged |
| Host allocation and query lease | Cache Manager | Submitted copies have drained, including partial failures |
| Remote export | Source Manager | The requester proves its submitted READ batch has drained |

Descriptor/result generations protect communication slots. They do not replace
allocation ownership or GPU page generations. A timeout, lost notification or
membership expiry is not evidence of DMA completion.

## Current increment

There is one supported implementation of each completed path. Protocol changes
require clients and Managers from the same revision; retired wire decoders and
runtime implementation selectors are removed. Performance controls run the
baseline revision in a separate checkout with matched workloads and budgets.

### Restore identity and ambiguous submission

The client reserves a session-local operation ID in shared completion memory
before sending its descriptor. The Manager authenticates and consumes the
descriptor, then atomically claims that ID before decoding or consuming leases.
Cancellation competes with that claim: a cancelled reservation cannot execute;
once claimed, even a lost or malformed submission ACK returns the original
handle and preserves the operation until its actual terminal result. Requests
are not retransmitted. An ambiguous descriptor channel is closed, while already
claimed results remain readable through their retained mapping.

Each of the 1024 records moves through Reserved, Executing, Succeeded/Failed,
and Acknowledged. Only Reserved can be cancelled without a completion fence;
only consuming a terminal result reclaims a claimed slot. Preparation failures
also return a handle with a Failed result. A closed UDS does not complete a
restore: pending polls use the Manager pidfd, and recheck for a final publication
after observing process exit. Notification waits use eventfd without repeatedly
waking on a disconnected UDS; the existing bounded rescan covers lost wakes.

The fixed submission ACK only acknowledges descriptor consumption. The old
Restore response encoder/decoder and Manager-wide operation counter are gone.
Identity and trace keys include Manager epoch, client session token and operation
ID, since different sessions may reserve the same numeric ID. This implements
submission identity for the current Manager-owned CUDA path. Handles also retain
their native client issuer, fencing a replacement mapping even if an operator
reuses a configured Manager epoch. This does not yet
grant an engine permission to submit CUDA or prove engine-local DMA drain.

### Restore preparation and destination binding

Raw resident Restore now compiles GPU/host copy descriptors directly from the
prepared sources and validated layouts. Its worker task retains each selected
leased source once, instead of expanding source `Arc`s into every layer/block
pair and rebuilding the same descriptors in the worker. Encoded, SSD and mixed
plans retain their layer metadata for the physical work that still needs it;
they are selected from source representation, not after a failed raw submission.

Raw admission sorts descriptors by device address and rejects invalid or
overlapping destinations before enqueueing GPU work. This groups split K/V
regions so adjacent ranges can coalesce without changing a source/destination
pairing. Merging still requires the same
host and device allocation identities. The shared descriptor builder checks
every host subrange, including page-first offsets and split segments, before
pointer arithmetic. Descriptor construction/admission follows lease consumption;
failures there consume the prepared batch, as previous worker admission did.
Source owners and byte reservations remain held until GPU
drain, including partial enqueue failures. CUDA submission still runs in the
Manager; this is not the engine-local grant protocol.

The existing Restore command now resolves group/TP/page-first geometry once,
checks registered GPU destinations before consuming leases, and prepares source
ownership separately from local GPU address binding. `PreparedRestore` owns the
source references, reservations, selected route and group targets; consuming it
builds the current worker task without another topology lookup. This is the
preparation boundary for an engine-local executor, not that executor itself.

A batch consumes leases under one lock after all tokens, source counts, storage
slots and route choices validate. Repeated tokens in one batch are rejected.
Validation failure preserves every valid lease's remaining consumer count;
subsequent worker admission failure still consumes the successfully prepared
batch. The hot path checks only requested leases for expiry, leaving table-wide
reclamation to the existing sweep/create/release paths. Session cleanup cannot
revoke sources or reservations already transferred to preparation or GPU work.

`KVCacheGeometry` validates final strides, split ranges and host padding without
GPU addresses. `KVCacheLayout::bind` checks a process-local allocation against
that geometry. Registration, Publish and Restore use this single implementation;
the old layout constructor, mutating stride/padding builders and forwarding
geometry getters are removed. No alternate wire protocol or executor selector
is introduced. Arena/allocation identifiers will be added with their actual
cross-process grant consumer rather than as unused fields.

The native Python result also no longer wraps a lease byte vector in a second
type or clones it before constructing Python `bytes`. Query intent moves into
the request state machine, and the unused cloning bootstrap getter is removed.

### Request dispatch and encoding

A required iceoryx2 request event wakes the Manager after a command is enqueued.
The Manager clears stale events, briefly spins on the request queue, then waits
until an event or its next maintenance deadline. The fixed 50 us idle sleep is removed;
notification failure after enqueue never permits early Publish source reuse.
Ordinary replies remain on the request/response queue. A dedicated per-session
reply eventfd wakes Publish after its response is queued; the client waits on
that FD and the Manager pidfd instead of sleeping for 100 us. Restore completion
keeps its separate eventfd so the two waiters cannot consume each other's wakes.
Bootstrap version 5 and channel ABI 8
reject previous clients rather than retaining a runtime polling mode.

Request encoding uses exact payload sizes. Oversized Publish requests are
partitioned by their encoded lengths and encoded from borrowed block ranges,
without repeatedly cloning and encoding binary-search candidates. Each channel
still owns one descriptor slot, so its request/response lock remains necessary.

The [measured local comparison](communication-performance.md) records matched
Query, Publish, Restore and raw IPC latency with CPU accounting.

### Shared restore completions

Restore submission still uses the authenticated iceoryx2 command channel.
Terminal results are published into a session-owned sealed memfd and read by
the native client without a terminal Poll RPC. The client acknowledges a slot
only after consuming the matching result. The outcome waiter publishes after
the GPU worker supplies its terminal outcome, then signals eventfd directly;
the control dispatcher does not scan outstanding restores to discover GPU
completion.

The result mapping contains 1024 cache-line records and separate, lazily
touched error storage with a 4096-byte limit per slot. Successful operations do
not touch error pages. The Manager admits at most 64 mappings, including
disconnected sessions still retained by outcome waiters. Clients and Managers must be
rebuilt together after the bootstrap and command ABI change.

The completion timeline distinguishes worker completion from result
publication/notification and reports missing observations explicitly. The
terminal-Poll event decoder and its report fields have been removed. Reproduce
historical reports with the revision that produced their logs.

### Shared pinned payload backing

Every pinned-pool shard now uses a size-sealed memfd and `MAP_SHARED`. Manager
NUMA first-touch runs before its CUDA registration. Regular and reserved huge
pages are page policies of the same backing; private anonymous mappings,
`cudaHostAlloc` pool allocation and the `cpu_readable` selector are removed.
Existing allocation owners and TENT memory-registration owners still determine
when the Manager may reuse or unregister its mapping.

The memfd can outlive the allocating process through another process's own
mapping and CUDA registration. This establishes backing lifetime, not permission
to read an allocation that the Manager might reuse. Production bootstrap does
not yet export payload FDs or execute restore plans in the engine process.
Arena identity, allocation generations and source grants belong to the next
connected implementation, not unused public APIs in this increment.

### One-segment peer authorization lookahead

Authorization of the next planned segment overlaps the current segment's READ.
Bounded lookahead is the single execution strategy. There is at most one
speculative grant per fetch plan;
destination allocation waits until that segment is consumed. Existing global
and per-source completion limits continue to bound grants and cleanup.

Only a fully returned contiguous segment advances the prefix. Failure or
cancellation drops unconsumed grants through the existing known-ticket cleanup
owner. Submitted READs retain their source and destination owners until the
transport drains. A speculative admission rejection may be retried on demand
after the preceding READ completes; it does not by itself invalidate a source.
If resource pressure coincides with source releases already in progress before
an authorization attempt, authorization waits up to three seconds for those
release ACKs and retries once. It neither waits for active READs nor moves the
release RPC onto the blocking payload transfer's completion path. The same
bounded pressure recovery also covers ordinary demand authorization.
The three-second bound is for release waiting, not the entire fetch; each
authorization retains its own RPC deadline. `release_wait` records this wait,
and fetch-plan attempt counts describe logical segment attempts rather than
individual RPC retries.

This increment uses the existing source-control RPC and Mooncake TENT READ backend.
It does not introduce speculative payload reads, pre-authorized persistent
hotspot replicas, or a new metadata transport. Prepared-grant residence time
must not train the complete-route service-cost estimator.

## Next: engine-side GPU executor

The smallest useful slice is exact/raw DRAM-to-HBM restoration on one GPU.
The Manager remains the allocator and source-lease owner; a native executor in
the engine process owns CUDA submission using the engine's current GPU pages.

The shared backing is implemented. The next boundary exports payload FDs and
establishes per-process mappings, CUDA registration, arena identity and
allocation-generation validation before executing plans. The descriptor memfd
and restore-result memfd remain separate control-plane resources.

A prepared plan holds its sources and describes bounded arena ranges. Engine
destination pages are bound and validated when allocated. The executor first
preserves the whole-operation completion fence, then adds layer-group
dependencies so early groups can be consumed while later groups are restored.
Both eager execution and the supported CUDA graph replay modes must pass the
same page-reuse and cancellation gates. Moving CUDA submission into the engine
does not remove the physical HBM/host-memory transfer.

The [engine-local restore design](engine-local-restore.md) specifies source and
destination ownership, prepared/claimed/drained transitions, process-death
quarantine, the exact/raw first slice, layer-group dependencies and graph gates.

## Next: batched remote metadata messages

Keep etcd membership, epochs and placement outside the per-request lookup
path. Retain ordinary RPC for bootstrap and low-frequency management. Define
batched grant, completion, acknowledgement and credit messages around the
existing source authority on one bounded TENT control session. Replace the
hot source-control RPC methods when that session passes its qualification gates;
do not retain a second runtime protocol or an automatic gRPC fallback.

The pinned TENT source contains an RDMA SEND/RECV notification backend, but its
current C ABI truncates C strings, its native receive queues are unbounded,
and notification transport selection is not peer-specific. Its TCP backend
uses a control RPC. Native length-aware framing, bounded queues, peer transport
selection and bounded submission must be implemented before moving OrbitKV's
metadata authority onto it. A bounded Rust channel alone does not bound native
memory. A successful notification send is not proof that the receiver consumed
it or that a separate payload READ drained.

Only after revocation/drain is qualified should hotspot grants be issued ahead
of demand. Each grant must hold the precise source allocation and consume a
bounded budget. A local directory hint is not an authorization. Unused and
in-flight grants need distinct retirement paths; lease expiry cannot release
memory still accessible to submitted READs.

The [peer-control design](peer-control.md) defines the target messages, credits,
epochs, acknowledgements, native prerequisites and source-control cutover.

## Implementation and deletion gates

| Step | Deliverable | Remove at cutover | Acceptance |
| --- | --- | --- | --- |
| Completed: local dispatch | Required request event, bounded spin before sleep and independent Publish reply eventfd | Fixed Manager 50 us idle poll, Publish 100 us reply sleep and forwarding descriptor helpers | Microbenchmarks with CPU/latency, process-boundary wake races, Publish faults and independent notification counters |
| Completed: local completion | Shared terminal records and direct eventfd signal | Terminal Poll command, dispatcher restore scan, retired timeline decoder | Cross-process results, stale IDs, missing notifications, real GPU faults |
| Completed: payload backing | Shared memfd for every pool shard | Private anonymous and `cudaHostAlloc` pool paths, `cpu_readable` plumbing | FD transfer, independent registration and GPU bytes after producer mapping teardown |
| Completed: peer lookahead | One active READ plus one next authorization | Sequential runtime selector | Prefix integrity, cancelled/lost grants, release pressure, real multi-segment TENT bytes |
| Next: raw engine restore | Payload FD attachment, owned source grants and a native engine executor | Manager submission for the same exact/raw host plan | Partial submission, destination retention, both process deaths, source budget and reconnect fencing |
| Next: execution overlap | Layer-group dependencies with one final retirement fence | Whole-restore waits from engine consumption sites covered by qualified group dependencies | Pinned engine releases, eager/graph replay, page reuse, TTFT/ITL and CPU cost |
| Next: native metadata | Bounded binary notification API and per-peer transport selection | Unsafe string framing and first-transport notification dispatch | Size/queue limits, unreachable peer, mixed transports and native shutdown |
| Next: peer session | Batched lookup/grant/completion with application ACK and credits | Corresponding hot gRPC methods, retry owner and protobuf messages | Loss, duplication, reorder, restart, corruption, slow peer and multi-host qualification |

Each cutover replaces its old implementation and updates all callers in the
same change. Capability-specific SSD preparation or codec work remains with
its resource owner; it is not a fallback copy of the exact/raw executor.

## Qualification

For local completions, cover real process boundaries, capacity pressure,
generation reuse, reconnects, concurrent readers, complete error payloads,
lost eventfd notifications, and timeouts that retain pending ownership. Run
the existing Manager/native-client GPU integration and fault gates after the
ABI update.

For peer lookahead, use deterministic synchronization to prove authorization
actually overlaps a blocked READ. Compare with the sequential baseline revision,
including partial prefixes, rejection, failed reads and cancellation. On real
peers compare segment counts, authorization/READ timing, source bytes held,
outstanding completion records and end-to-end restore latency, then serving
TTFT/ITL and goodput. Test both DRAM and owner-prepared SSD sources.

Use the [single-node](single-node-performance.md) and
[shared-cache qualification](shared-cache-qualification.md) workloads. Record
the engine, CUDA, transport and graph configurations; account for polling CPU
cores and speculative source holds. Passing protocol tests does not establish
an end-to-end speedup or cross-host RDMA qualification.
