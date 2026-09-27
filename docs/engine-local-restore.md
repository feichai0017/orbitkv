# Engine-local GPU restore

## Status and objective

This is the target design for moving GPU restore submission into the inference
process. The Manager currently submits restores through its GPU workers and
publishes terminal results into the shared completion mapping. The payload pool
now uses size-sealed memfd backing for regular and reserved huge-page policies;
private mappings and the `cudaHostAlloc` allocation branch have been removed.
Source preparation and pointer-free geometry now have separate owners, consumed
by the current Manager worker path. Raw resident plans compile directly into
copy descriptors and retain each selected leased source once through GPU drain;
the worker no longer needs layer/block source expansion for that route.
The current Manager executor also consumes client-reserved operation identities:
shared claim/cancel admission preserves a handle after a lost submission ACK,
and completion remains readable when the descriptor channel closes. This is
already used by all native Restore callers. Its claim authorizes the Manager,
not the engine; the engine-local grant/drain lifecycle below is still required.
An engine-local executor, payload-arena
grants, and the protocol described below are **not implemented**. The implementation sequence is tracked in
[the communication plan](communication-plan.md).

The shared-backing GPU gate has passed with separate exec processes on the same
GPU using regular pages. The importer independently maps and registers the FD,
then verifies H2D/D2H bytes after the producer normally drops its CUDA
registration, mapping, and FD. This does not qualify SIGKILL during DMA,
huge-page imports, or multiple GPUs. Production FD import and restore-grant
ownership still need implementation.

The first executable slice is exact/raw DRAM-to-HBM restore on one GPU, with a
whole-operation completion fence. The target removes the Manager's involvement
in CUDA submission for this path while retaining its authority over source
allocations, representation selection, and admission. It does not remove the
physical host-to-device transfer or make generation checks a substitute for
holding memory alive.

The existing ownership behavior provides the starting point:

- [Engine restore](../crates/orbitkv-core/src/engine/restore.rs) consumes query
  leases atomically in a batch, prepares topology/group/slot ownership without
  GPU addresses, then binds prevalidated local destinations. Raw resident
  sources produce an owned copy-descriptor batch; encoded/SSD/mixed routes keep
  the layer metadata their workers need. Invalid preparation does not consume
  any valid lease share.
- [Query leases](../crates/orbitkv-core/src/query/lease.rs) transfer source
  references and `QueryReservation` into the task. Lease expiry and session
  cleanup apply to leases still in the lease table.
- [GPU workers](../crates/orbitkv-core/src/transfer/worker/mod.rs) retain the
  source payloads and reservations through completion. `finish_load` drops
  them only after the transfer path has drained.
- [Transfer completion](../crates/orbitkv-core/src/transfer/mod.rs) synchronizes
  even after partial submission fails. An inability to establish completion
  aborts the Manager rather than returning a reusable-page result.
- [The endpoint](../crates/orbitkv-server/src/endpoint/restore.rs) publishes the
  worker's outcome; [the native client](../crates/orbitkv-channel/src/cache_client.rs)
  reads it without a terminal Poll RPC. This result acknowledgement says the
  client consumed a result. It is not an engine-to-Manager DMA fence.

## Resource owners

| Resource | Target owner | Condition for release or reuse |
| --- | --- | --- |
| Payload arena backing | Manager plus each importing engine's independent FD/mapping owner | Each process releases its own mapping only after its submitted accesses drain |
| CUDA host registration | The process using that mapping for CUDA | All GPU accesses through that registration have drained |
| Allocated source range | Manager restore grant holding the exact sealed block/allocation | Grant was revoked before claim, or matching engine drain evidence was accepted |
| Query byte reservation | Manager restore grant after lease consumption | Same source-grant terminal condition, including quarantine |
| Engine tensor allocation | Engine/native destination binding | Every submitted copy and engine consumer using it has drained |
| Logical GPU page assignment | Framework scheduler and native operation owner | Restore has drained and the engine's own page-use rules permit reassignment |
| Prepared plan bytes | Manager-owned bounded plan slot | The claiming engine has copied them locally, or an unclaimed grant has been revoked |
| Grant record | Session mapping, retained by outstanding owners | Source owner reaped and the engine observed retirement; a dead session's entire mapping can instead be retired |
| User-visible restore result | Native operation in the engine process | All result consumers have consumed or dropped it |

Two independent lifetimes matter. A mapped memfd keeps the backing object alive
across Manager death. It does not stop a live Manager from recycling an offset
inside that object. The Manager's source allocation reference prevents that
reuse. Conversely, an `Arc` in the Manager does not keep an engine's CUDA
registration or destination tensor alive. Both sides need real owners.

Holding a tensor allocation also does not pin a logical page assignment inside
the tensor. The connector must keep the request's destination page IDs reserved
until the native operation reaches its declared terminal condition. A page
generation detects reassignment; it cannot repair a scheduler that reused a page
while a copy was in flight.

## Shared payload arenas

The descriptor and completion memfds hold metadata only. The current
[pinned-memory pool](../crates/orbitkv-core/src/memory/pinned.rs) separately owns
shared payload backing. That backing supplies the first part of this contract;
FD distribution, importing registrations, identities, and allocation grants are
the remaining work. The complete arena path must satisfy all of the following:

1. The Manager creates a sized memfd, maps it with `MAP_SHARED`, seals growth and
   shrinkage, and owns allocation and reclamation. Size seals do not make
   payload contents immutable. Immutable sealed-block ownership supplies that
   property while a restore grant is live.
2. NUMA placement and huge-page policy remain explicit pool policies. The
   Manager first-touches the backing before engine registration. A missing
   huge-page reservation must not silently select a private, unexportable
   allocator. Ordinary and huge-page pools use the same sharing contract.
3. Authenticated UDS bootstrap or an arena-registration control message passes
   the FD with `SCM_RIGHTS`. Registration is once per arena and engine session,
   not per restored block. File numbers and virtual addresses are process-local.
4. The engine validates the FD size/seals, arena identity, Manager epoch, and
   negotiated format before mapping and registering it in its CUDA context.
   Registration failure rejects the route before any GPU write.
5. Each process owns its CUDA registration separately and obtains its own
   device-visible host pointer. Never derive the engine's host/device pointer
   from the Manager's pointer. The distinction already exists in `CopyDesc` and
   `MappedPinnedPtr`.
6. Arena teardown stops new grants, drains local accesses, deregisters CUDA,
   then unmaps and closes the local owner. A Manager restart creates fresh
   arena identities and backing objects; it must not reopen and allocate from
   an old arena still mapped by a surviving engine.

The first implementation can use read/write mappings within the existing
same-UID, trusted-engine boundary. Passing an arena FD exposes that arena, not
only the ranges in a grant; a receiver can retain the FD indefinitely. Offset
and allocation-generation checks provide protocol validation, not process or
tenant isolation. Untrusted tenants require separate arenas and an explicit
OS-permission/isolation design. Read-only CUDA registration is an optional
qualification, not an assumed property of the first slice.

Every source reference contains an arena UUID/epoch, allocation identity and
generation, allocation bounds, and a checked subrange. Generation changes when
an allocator slot is reused; wraparound retires the identity. Checked arithmetic
must prove both `subrange ⊆ allocation` and `allocation ⊆ arena`. A mapping-size
check alone would allow one block to read another live allocation.

The authoritative Manager grant captures these identities while owning the
allocation. The engine validates the captured descriptor against its arena
binding and the operation generation. It does not consult an eventually
consistent allocation directory to justify DMA. The held allocation prevents
the identity from changing between validation and execution.

## Planning and native API boundary

Split the existing restore operation at the ownership boundary, retaining one
implementation of topology, representation, and layout validation:

- **Manager preparation:** authenticate the session, validate the recovery
  sources and group/TP/PP geometry, consume the relevant query leases, acquire
  any storage materialization, and construct an immutable bounded plan. Move
  the resulting `RestoreSource` references and `QueryReservation` into a grant
  owner before publishing the plan. For repeated TP consumers, preserve the
  current reference-sharing and reservation accounting behavior.
- **Engine execution:** retain the engine's destination binding, claim the
  matching grant, copy and validate the plan, resolve local pointers, submit
  CUDA work, and publish its actual drain outcome. The operation remains owned
  if the Python caller drops a handle or a wait deadline expires.

The wire plan contains arena/file references, representation and codec
parameters, actual byte lengths, padded host strides, storage-group and slot
identity, layer/group destination IDs, and destination layout generation. It
never contains a dereferenceable Manager virtual address. Destination GPU
pointers are bound from an engine-local registration, not accepted from a
Manager response.

[The affine layout implementation](../crates/orbitkv-core/src/transfer/layout.rs)
now separates validated `KVCacheGeometry` from `KVCacheLayout` pointer binding.
An engine executor must reuse its contiguous,
split K/V, fused-stride, and page-first rules when producing `CopyDesc` batches.
Do not duplicate these formulas in Python or create a second cache planner in
the native executor.

The PyO3 owner retains the actual registered tensor/exporter objects and the
native destination registration. Python still discovers engine layouts,
allocates GPU storage, and invokes framework callbacks. Rust owns grant state,
batching, CUDA submission, waiting, cancellation, and reclamation. Blocking
native work releases the GIL. An integer `data_ptr()` without the allocation
owner is insufficient.

Use the CUDA context that owns the engine's allocations and bind it on native
worker threads. Do not create an unrelated context merely from `device_id`.
The executor owns its copy streams and events and participates in orderly
engine shutdown. Re-registering layouts increments their generation and waits
for operations using the old binding; changing a dictionary entry must not
drop old tensor owners.

The existing copy backends already enqueue without synchronizing. Move or
reuse their concrete implementation behind the engine executor without adding
a forwarding-only crate. `python/Cargo.toml` already depends on
`orbitkv-core`; a new dependency layer is not required just to call them.

## Grant and operation state machine

Use a versioned shared grant protocol. The engine reserves a bounded record
and chooses the session-scoped operation identity **before** sending preparation
to the Manager. A lost preparation response therefore does not lose the
identity needed for cancellation or diagnosis. Repeated preparation for the
same identity is deduplicated before consuming a query lease again.

The following are protocol states, not new APIs that already exist:

| Transition | Writer | Required action |
| --- | --- | --- |
| `Acked/Free → Preparing` | Engine | Reserve a new generation, retain destinations, and send the preparation request |
| `Preparing → Granted` | Manager | Install source owner, write immutable plan, then publish its reference with release ordering |
| `Preparing → CancelRequested` | Engine | Prevent a later grant from being published; Manager still drains any submitted preparation I/O |
| `Preparing → Rejected` | Manager | Publish a bounded preparation error; no destination GPU work was submitted |
| `CancelRequested → Cancelled` | Manager | Finish preparation cleanup; all submitted preparation I/O has drained |
| `Granted → Active` | Engine | Win the generation-tagged claim CAS before reading plan bytes or submitting any CUDA work |
| `Granted → Revoked` | Manager | Win the competing CAS; no engine may subsequently claim this generation |
| `Active → Drained` | Engine | Publish result only after all submitted accesses drain, or after proving nothing was submitted |
| `Drained/Rejected/Cancelled/Revoked → Reaped` | Manager | Release the appropriate source owner and admission credits, then publish retirement |
| `Reaped → Acked` | Engine | Observe retirement; permit record reuse with a new generation |

The competing `Granted` transitions provide the cancellation boundary. A TTL
or disconnect can revoke an unclaimed grant by winning that CAS. Once `Active`
is visible, the Manager cannot infer whether the engine has enqueued work and
must retain the sources. This deliberately includes a crash between claim and
the first CUDA call.

Preparation publication also uses generation-tagged CAS: `Preparing → Granted`
or `Rejected` races with `Preparing → CancelRequested`. The Manager constructs
the plan and source owner before attempting publication. If cancellation wins,
it cleans up that owner and any submitted I/O; it must not overwrite
`CancelRequested` with a late grant or rejection.

Claim before reading plan bytes so revocation cannot recycle the plan while a
reader copies it. After copying the plan into owned native memory, the engine
publishes a separate generation-tagged `plan_consumed` flag. The Manager can
then return that plan slot while retaining the source grant. Validation failure
after claim produces a no-submission drain result. Shared error/descriptor
payloads remain owned until consumed; a generation check alone does not make
concurrent non-atomic reads and writes safe.

`Active` is a conservative submission window, not evidence that work was
submitted. Native operation state records whether a CUDA call was attempted,
which streams were used, and whether their terminal events were successfully
recorded. Only that owner may publish `Drained`; dropping a handle, Python
exception handling, or a timeout must not do so.

Normal completion has two different observations:

1. The engine proves GPU terminal completion and makes its local result
   available to the connector. Successful restored pages may now be consumed;
   a drained failed restore may be handed back for recomputation.
2. The engine release-publishes `Drained` and signals the Manager. The Manager
   consumes it, drops the source owner, and publishes `Reaped`. This retirement
   acknowledgement is not on the page-consumption critical path.

A native retirement queue retains the shared mapping and operation identity
until `Reaped` is observed, even if the Python caller has consumed or dropped
the local result. Retirement does not retain already-drained GPU destinations
solely to wait for the Manager acknowledgement. Manager source release is
generation-specific and idempotent; no completion message bypasses the
claim/revoke state machine.

The existing Manager-to-client result acknowledgement cannot implement this
handoff unchanged. At cutover, replace it with the grant lifecycle and an
engine-local result owner. Do not keep two terminal state machines or treat a
client's acknowledgement of the old result as proof of newly submitted DMA.

Use eventfd plus a bounded dirty-record queue for wakeups in each direction.
Notifications are hints; generation-tagged records are authoritative. Overflow
sets a rescan flag for the bounded record table. Avoid a periodic scan of every
restore as the normal completion path. An engine-local notifier wakes
`wait_restore`/framework polling without a Manager terminal RPC.

### Admission and metadata bounds

Retain the current upper bound of 1024 outstanding records per session and 64
live-or-retained session mappings unless measurements justify smaller limits.
Disconnected mappings with unresolved operations count against that limit.
An initial grant layout should budget at most 128 bytes per record and a
separate 1 MiB plan/error bank per session: approximately 72 MiB across 64
sessions, plus headers and the existing channel descriptor memory. These are
target limits for the new protocol, not measurements of current memory use.

Plan slots have a maximum encoded length; oversized operations are partitioned
into bounded suboperations whose parent retains the whole-operation fence.
The plan bank is reclaimed after copying, so 1024 records do not each require
a permanently reserved maximum-size plan or error buffer. Admission acquires
record, plan, source-byte, and executor-queue credits before publishing a grant.
It rejects or defers work when those credits are unavailable.

Quarantined grants retain source-byte and record credits. Metrics distinguish
active, awaiting retirement, and quarantined bytes. No cleanup path can make
these bytes disappear from accounting while retaining their physical owners.

## CUDA readiness and event publication

The destination binding must establish that previous engine users of the
selected pages have finished before the restore stream overwrites them. Use
an engine stream dependency captured at the scheduling boundary; do not infer
readiness from completion of a Python callback. The local copy stream records
its terminal event after the last copy/decode using the sources. A native
progress worker observes the event or synchronizes the owned stream and only
then publishes the drain result.

For a multi-stream operation, one event on one stream is insufficient. Join all
participating work onto the terminal stream, or retain and drain all terminal
events before publishing whole-operation completion. On partial submission
failure, stop enqueueing new work and drain every stream that may have accepted
work. If event recording failed, synchronizing the owned stream remains the
required completion attempt. If that also fails, poison the executor and retain
the unresolved owners; do not translate the error into a reusable-page result.

Local restore needs local CUDA events, not IPC events: submission and
consumption now belong to the engine. Where a real cross-process dependency
remains, such as Manager-side Publish, an IPC event has this stricter protocol:

1. Create it with interprocess support and timing disabled. Keep the exporting
   event owner alive while an importer may access it.
2. Record on the actual producing stream, after the work being fenced.
3. Only after successful recording, release-publish the matching operation and
   event generation as armed. An importer acquire-checks that record before
   waiting or querying.
4. Do not re-record, destroy, or recycle the event until all importers have
   acknowledged they are finished with that generation.

An unrecorded event can report success, and re-recording changes the work a
later query observes. An already-issued wait does not automatically follow a
later recording. These rules follow the
[CUDA event contract](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EVENT.html).
CUDA also declares use of an imported event after the exporting event has
been destroyed undefined; an IPC event must not be treated as a crash-surviving
source lease. See the
[CUDA IPC event contract](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html).

## Failure and shutdown rules

| Failure point | Required behavior |
| --- | --- |
| Preparation rejected or cancelled before claim | No destination GPU write is possible; Manager drains its own preparation I/O and releases sources |
| Lost preparation reply | Recover state using the already-known operation identity; never submit without successfully claiming the matching grant |
| Cancellation after claim but before any CUDA call | Native owner publishes a no-submission drain result; Manager must not assume this case from elapsed time |
| Error after a partial enqueue | Retain all source ranges, registrations, staging buffers, and destination owners until the entire submitted prefix drains |
| User wait timeout or dropped result handle | Detach the waiter; native operation and source grant continue to own the work |
| Live engine loses its UDS session | Stop admission, continue draining local operations, and publish drain records while their mapping remains valid; do not recreate operations in a new session |
| Engine dies with an unclaimed grant | Manager wins revocation and releases its grant; retire the old session mapping without reusing it for a new session |
| Engine dies after claim or while recording a terminal event | Manager quarantines sources, registrations, and credits unless valid drain evidence was already published |
| Manager dies during a claimed restore | Engine keeps its independently mapped and registered arena and destination owners, drains locally, and reports only its local terminal outcome |
| CUDA cannot establish completion | Mark the executor unavailable and fail closed; neither source retirement nor destination reuse follows from an ordinary error result |

Engine process death, pidfd readiness, lease expiry, and UDS HUP are liveness
evidence. This design does not assume they establish CUDA quiescence. An IPC
event imported from the dead engine is not a safe replacement proof. A Manager
may accept a correctly published `Drained` record even after disconnect; a
claimed operation without such evidence remains quarantined.

Releasing quarantine requires a separately qualified fence that covers the
exact device/context and all relevant submissions, or controlled node/device
recovery that establishes those accesses have ended. A timeout or ordinary
Manager restart is not that fence. The first slice therefore has a deliberate
availability limit: repeated ambiguous engine deaths can exhaust bounded
credits and require recovery. It must not hide this limit by prematurely
returning allocations to the pool.

When the Manager dies, a surviving engine's independent registration keeps the
shared source available for its own DMA. Stop accepting new grants from that
epoch, finish the local work, and discard the old protocol mapping after its
local owners drain. A replacement Manager creates a fresh arena; reconnect
does not replay old grants against new allocations. If the engine chooses to
continue serving a successfully completed restore, its higher-level cache
session still needs explicit reconnection before future external operations.
Restart capacity must include old arenas still registered by surviving engines;
allocating a full replacement pool does not erase their pinned-memory cost.
Release the old engine registrations after local drain and account for any
temporary overlap, or wait for that retirement before admitting the replacement
capacity.

Orderly engine shutdown closes admission, prevents further submissions, drains
operations, publishes their outcomes, and then destroys local registrations and
tensor bindings. Orderly Manager shutdown revokes unclaimed grants and waits
for claimed grants to drain or enter controlled recovery. It cannot drop the
quarantine owner merely because normal session cleanup finished. Simultaneous
process/device failures require the node recovery contract; this proposal does
not claim arbitrary CUDA failure recovery from an OS process notification.

## Execution roles for raw, codec, and SSD sources

One preparation contract selects a physical plan before submission. Different
routes have concrete resource owners, not a catch-all path that retries the
old restore RPC after a native failure.

| Selected source/route | Manager responsibility | Engine responsibility |
| --- | --- | --- |
| Exact/raw DRAM | Own immutable shared allocation and validate slot/range geometry | Copy into its own GPU pages using the selected memcpy or mapped-memory kernel backend |
| Encoded host representation | Own encoded allocation; validate representation version, lengths, and allowed reconstruction semantics | Stage/decode with qualified native GPU codecs, retain scratch through terminal completion |
| Host materialization/decode route | Own bounded SSD read or CPU transform, wait for its I/O/writes, then publish a shared HostReady source | Execute the resulting raw or encoded host plan |
| Direct SSD/cuFile route | Retain exact SSD record/read lease, provide immutable file/extent identity and bounded ranges | Own file import/cuFile registration, GPU destinations, I/O completion, and any GPU decode |
| Peer source | Complete and validate the peer READ into Manager-owned local residency under the existing remote export protocol | Consume a subsequent local grant only after its required source is ready |

The existing
[SSD host materialization](../crates/orbitkv-core/src/transfer/worker/restore.rs)
already waits for every submitted read, including after one fails. Move that
behavior into Manager preparation without binding Manager-side destination GPU
pointers. Its resulting shared host allocation remains owned by the grant.

A file FD keeps a file object alive; it does not prevent compaction or overwrite
of leased extents. Direct SSD plans need the existing storage read owner as well
as file identity, aligned ranges, and an engine-local cuFile lifetime. Short
reads, decode errors, and partial I/O all participate in the same terminal
ownership protocol. Direct SSD is ineligible until that implementation exists.

Lossy codec restoration is eligible only when permitted by the state contract;
moving execution must not broaden its equivalence class. The planner advertises
only routes supported by the registered executor. A raw-only rollout explicitly
limits its supported configuration. Supporting a HostReady route for SSD or
codec sources is a selected physical execution role, not recovery by retrying
an obsolete GPU restore API after an ambiguous submission.

Manager-side GPU Publish/encoding and storage work retain their current owners
until their own migration is complete. Engine-local restore alone does not
justify deleting CUDA IPC tensor registration needed by those operations or
claiming the Manager no longer uses CUDA.

## Layer groups and CUDA graphs

The first slice retains the current whole-operation gate. In the baseline,
vLLM's `wait_for_layer_load` is empty and completion is reported through
`get_finished`; SGLang's counter fences the whole restore before forward
construction because graph replay can bypass Python pool accessors.

Layer-group readiness is a later capability of the same native operation:

- Compile groups from actual engine dependencies and storage-group geometry,
  not an assumption that numerical layer order describes every hybrid model.
- Record each group event only after all copies/decode required by that group.
  A consumer stream waits on the matching recorded generation before first
  use. All groups still belong to one operation for cancellation and errors.
- Initially retain sources until the whole operation drains. Earlier source
  release requires separate reference counts for every allocation's last
  consuming group; page-first allocations can contain several groups.
- Do not report the entire request ready after its first group. Failure after
  an early group was consumed must follow the engine's execution error path;
  it cannot retroactively request transparent recomputation of consumed state.

Graph support is qualified per engine version, graph mode, and cache layout.
The safe initial mode observes whole-operation completion before launching the
graph. Group overlap requires dependencies that execute on every replay,
including a valid way to update/arm their event generations. A Python callback
that runs only during capture does not establish replay-time readiness. Until
that mode passes the ownership and byte-correctness gates, capability selection
must choose the whole-operation gate or reject that configuration explicitly.

Publish readiness is a separate direction: record after the engine producer
stream has written KV, including graph replay. Do not record that producer event
from a background save thread and assume it covers the original stream.

## Implementation and deletion sequence

Each step lands with a current consumer and its tests. Do not merge unused wire
types or an executor that no connector can invoke.

1. **Unify payload backing — implemented.** The pool uses shared memfd ownership
   while preserving NUMA, page policy, mapped-device addressing, and allocation
   accounting. Private/cudaHostAlloc production branches and the obsolete
   `cpu_readable` selector have been removed. The cross-exec GPU gate covers
   regular pages, the same GPU, and normal producer-owner drop as described
   above. This is the backing prerequisite; it does not add a production engine
   import or executor.
2. **Separate preparation from GPU binding — implemented for the current worker
   consumer.** `PreparedRestore` owns the leased sources, reservations, group
   targets and physical route before binding GPU addresses. Batched lease
   validation/consumption and pointer-free geometry each have one implementation.
   Stable arena/allocation identities remain part of the next grant increment,
   where they acquire a real cross-process consumer. SSD materialization still
   belongs to the current worker and must migrate with that route.
3. **Connect the raw native executor end to end.** Change channel ABI, UDS FD
   registration, Manager grant ownership, PyO3 tensor registration, and native
   restore submission in one executable slice. Connect both vLLM and SGLang to
   it. Preparation remains one batched command; claim, GPU completion, and
   retirement use the shared records/local operation, without another terminal
   RPC. Remove the old raw Manager restore dispatch when this slice replaces it.
4. **Complete supported physical routes.** Connect host materialization, codec
   execution, and qualified direct SSD through the same preparation and grant
   contract. Route eligibility is explicit. Do not keep a generic old-protocol
   compatibility branch to conceal missing capabilities.
5. **Remove displaced restore machinery.** Delete the old restore command and
   response encoding, Manager restore-result publisher and bootstrap FD, and
   worker load lanes once their last supported route has migrated. Remove or
   move `LoadTask`, `LoadOutcome`, and restore-only worker code according to
   their remaining consumers. Preserve worker/save/IPC owners still used by
   Publish or storage. Update native methods, Python stubs, fault tests, and
   protocol version together; older clients fail the version handshake.
6. **Qualify overlap.** Add native group readiness and per-replay graph
   dependencies after the whole-operation path is correct. Remove obsolete
   framework waiting code when its replacement is connected, rather than
   retaining two independent schedulers or completion owners.

Steps 3–5 can be split by supported feature configuration, but a release must
state which configurations work. A capability means a real physical execution
role; it must not select an old API solely because the new implementation failed.
The pool-only increment does not warrant changing the advertised execution
location or performance claims.

## Gates and measurements

The required tests exercise ownership boundaries, not only serialization:

| Gate | Required evidence |
| --- | --- |
| Shared backing process test | Distinct virtual mappings of the same FD, validated bounds/seals, byte visibility, and survival of exporter exit while an importer owns it |
| Shared backing GPU integration | Separate CUDA registration in Manager and engine, correct engine-local host-device pointer resolution, byte-accurate H2D, registration rollback, NUMA/page-policy configurations |
| Grant state tests | Claim/revoke race; cancellation during preparation I/O; duplicate preparation; lost reply; stale generation and reconnect; full record/plan/source budgets; notification loss and dirty-queue overflow |
| Allocation ownership test | Lease TTL sweep, query release, instance unregister, eviction, and allocator pressure cannot recycle an active or quarantined source |
| GPU layout test | Contiguous, split K/V, MLA, fused stride, page-first, TP/group slot selection, padding, and destination range/generation rejection |
| Partial submission test | Inject a failure after an accepted copy; prove source and destination owners remain retained until the accepted work drains |
| Engine fault test | Exit before claim, after claim, during enqueue, after event record, and after drain publication; prove only the revocable or proven-drained cases release source credits |
| Manager fault test | Kill the Manager during an engine-owned H2D; the surviving engine's mapping/registration supports safe drain, and reconnect uses a different arena identity |
| Shutdown test | Dropped handles and timeouts keep native owners; layout re-registration and orderly shutdown drain; unproven CUDA completion poisons admission and never reports reusable pages |
| Engine correctness E2E | Supported vLLM and SGLang releases, restart reuse, hybrid groups, error/recompute semantics, eager and advertised CUDA graph modes |
| Route qualification | Encoded representations preserve their state contract; SSD short/error reads and compaction retain leases; peer materialization does not expose a source before READ completion |

Use deterministic hooks around claim, enqueue, event recording, and terminal
publication for the crash tests. Sleeps alone cannot show which ownership state
was exercised. Cross-process CUDA tests must use the real driver and payload
mapping; fake completion records cannot establish crash-time DMA safety.

Measure preparation/queue delay, native submission time, GPU completion,
connector observation, first engine use, and source-retirement lag separately.
Same-host monotonic timestamps can measure cross-process intervals with an
explicit clock contract. CUDA events measure GPU work; a CPU publication time
is not a GPU completion timestamp. Record source/registration bytes retained,
quarantined bytes, pinned-memory limits, CPU polling use, and plan-bank pressure.

Compare the matched raw whole-operation path first, then group overlap and
codec/SSD routes. The acceptance result is serving TTFT/ITL and goodput at equal
memory, CPU, model, and quality budgets, together with correct lifecycle tests.
A cheaper metadata round trip alone does not establish a serving improvement.
