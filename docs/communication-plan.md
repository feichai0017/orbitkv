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
| Local Restore grant record | Retained process channel session | Manager reaped the sources and engine acknowledged retirement |
| Host allocation and query reservation | Cache Manager grant or physical worker | Submitted copies have drained, including partial failures; unresolved local grants remain quarantined |
| Remote export | Source Manager | The requester proves its submitted READ batch has drained |

Descriptor/result generations protect communication slots. They do not replace
allocation ownership or GPU page generations. A timeout, lost notification or
membership expiry is not evidence of DMA completion.

## Current increment

There is one supported implementation of each completed path. Protocol changes
require clients and Managers from the same revision; retired wire decoders and
runtime implementation selectors are removed. Performance controls run the
baseline revision in a separate checkout with matched workloads and budgets.

### Engine-local raw Restore

Unencoded resident DRAM Restore now runs in the inference process. The Manager
validates topology, storage groups, TP/page-first placement, destination bounds,
and the source representation once. It compiles a pointer-free raw plan during
batched lease validation, then transfers the selected sealed blocks and
`QueryReservation` owners into `RawRestoreGrant`.

The native engine worker retains the actual registered tensors, independently
imports the payload arenas, resolves local addresses, and uses the existing
memcpy or mapped-memory kernel backend. Destination ranges are checked and
sorted before submission; adjacent copies merge only within matching source
and destination allocations. CUDA submission and whole-operation drain now
belong to this native worker. The Manager raw-descriptor worker branch and its
`LoadPayload` enum have been removed.

Encoded, SSD, and mixed-source plans keep the Manager physical workers that own
materialization, decoding, and SSD I/O. Publish also keeps its Manager worker
and CUDA IPC registrations. Route choice follows source capabilities; failure
of a local raw operation does not invoke another executor or an old protocol.

`register_context_batch` requires the actual `tensors` in addition to IPC
metadata. Successful GPU registration attaches payload FDs once over UDS.
`start_restore` requires `ready_stream`, captured from the actual engine stream
whose previous destination-page users must finish. The native implementation
queries that stream before preparation: idle means prior users completed;
busy records and waits on a reusable readiness event. It then uses one copy
stream. Both vLLM and SGLang call this path. Repeated registration of
the same binding is rejected; unregister and close drain accepted operations
before releasing their tensor and CUDA owners. Python keeps framework callback
and layout work, while Rust owns the operation state machines and waiting.

This implements the first single-GPU, unencoded DRAM, whole-operation slice.
It does not establish layer/group overlap, replay-time CUDA graph dependencies,
or broad serving qualification. The communication-branch production build
passed the scoped Qwen3-8B DRAM correctness gates in both pinned engines. The exact ownership and
remaining gates are described in [engine-local Restore](engine-local-restore.md).

The initial executor cutover regressed serial Restore. The subsequent idle
readiness and plan-compaction changes recover dense-transfer performance, while
small-payload handoff overhead remains. The [measurement report](communication-performance.md)
compares the Manager executor, initial local executor, and optimized local
executor with the same workloads. These measurements do not qualify serving
performance or compute overlap.

### Operation identity and source retirement

The client reserves a session-local identity before sending its descriptor.
The Manager authenticates and atomically claims it before decoding or consuming
leases. Cancellation competes with preparation admission. Once preparation is
claimed, a lost or malformed ACK retains the original handle; no request retry
can consume the same lease again. Native issuer, epoch, session token, and
operation generation fence handles from replacement clients and mappings.

For local execution, the Manager installs source ownership before publishing
`Granted`. Engine `Granted → Active` competes with Manager revocation. After
claim, the engine copies the immutable plan, validates it, submits CUDA, and
publishes `Drained` only after no-submit proof or actual copy completion. The
local result becomes visible without waiting for Manager source retirement.
A native retirement owner retains the record until Manager `Reaped` and engine
acknowledgement allow record reuse.

A session disconnect cannot release an active grant. Unknown completion after
engine death retains both the exact source allocations and their query credits
in bounded quarantine. Manager death does not fence local DMA: the surviving
engine's independent mapping, CUDA registration, and tensor owners permit its
own drain. SSD/codec Manager execution has a distinct `Managed` state within
the same versioned records and publishes only its worker's drained outcome.

### Shared records and bounded metadata

The grant mapping has 1024 records of 192 bytes plus a 1 MiB shared plan bank;
bounded errors use at most 88 bytes inside each record. At most 64 session
mappings may be live or retained, including disconnected unresolved grants.
The native worker admits at most 1024 pending operations, and a native client
imports at most 64 payload arenas.

The engine acknowledges plan consumption separately from DMA completion, so
plan-bank storage can be recycled while sources remain held. The Manager
uses an eventfd and a bounded dirty bitset to process grant updates; notifications
are hints and atomic records remain authoritative. Source retirement has its
own eventfd, and engine-local result notification does not wait on a Manager
terminal RPC. The old terminal Poll RPC and old result wire codec remain deleted.

A full shared plan bank defers prepared grants until space returns. Individual
encoded plans larger than 1 MiB after compaction are rejected before consuming
leases. Preparation sorts destinations and merges only consecutive source and
destination ranges within the same layer and allocation identity/bounds. The
existing `cpu_path/load_submit_wait/32768` case now passes its GPU submission/
drain smoke. Automatic bounded partitioning remains missing for fragmented
plans; no Manager raw fallback is retained.

### Shared payload arenas

Every pinned-pool shard uses a size-sealed memfd and `MAP_SHARED`. NUMA first
touch precedes CUDA registration. Regular and reserved huge pages share this
backing implementation; private mappings, `cudaHostAlloc` pool allocation,
and the `cpu_readable` selector are removed.

Successful GPU registration replies pass arena identities, sizes, and FDs.
The engine validates size seals and independently maps and CUDA-registers the
backing in its tensor context. Source plans carry arena ID, allocation ID,
allocation bounds, and checked subranges; allocation IDs are monotonic within
an arena and never reused with a recycled offset. Destination bindings use
registered layer names and byte offsets. No Manager virtual pointer crosses
this boundary.

A mapping keeps the backing object alive, while a source grant prevents
allocator reuse. Both owners are required. Huge-page import and multiple-GPU
qualification remain explicit gates even though the pool sharing API supports
the corresponding backing policy.

### Request dispatch and protocol cutover

The iceoryx2 request event wakes the Manager after a command is enqueued. The
Manager clears stale events, briefly spins on the queue, then waits for an event
or its maintenance deadline. Publish keeps an independent reply eventfd and
Manager pidfd wait. Fixed Manager 50 us idle polling and Publish 100 us reply
sleep remain removed.

Bootstrap version **7**, channel ABI **11**, and lifecycle version **4** require
matched client and Manager builds. The five bootstrap FDs are the descriptor
memfd, grant memfd, Manager-to-engine Restore eventfd, engine-to-Manager
retirement eventfd, and Publish reply eventfd. GPU payload FDs are attached to
registration replies; the engine-local result eventfd is created by the native
worker. No previous ABI decoder or compatibility runtime selector is retained.

Request encoding uses exact payload sizes. Oversized Publish requests are
partitioned by encoded length from borrowed block ranges, without repeatedly
cloning and encoding binary-search candidates. This existing Publish behavior
does not yet provide partitioning for an oversized raw Restore plan. Each
channel's descriptor slot retains its request/response lock.

The [measured local comparison](communication-performance.md) records matched
Query, Publish, Restore and IPC latency with CPU accounting. Historical
Manager-owned Restore measurements must be distinguished from this executor
cutover; protocol and ownership tests alone do not demonstrate a speedup.

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

## Next increments after consolidating PR #188

Development now continues in the main checkout on `refactor/route-cost-evidence`.
The communication branch's measured runtime is `eb61d166`; its last report
commit is `5c029ed9`. Those measurements are reference evidence, not timings of
the later merged binaries. The [serial](single-node-performance.md#matched-vllm-end-to-end-comparison)
and [fixed-cohort](sustained-performance.md#engine-local-restore-fixed-cohort-comparison)
results set the next priorities: 8K pressure TTFT was 6.79 ms behind native CPU offload,
while the one C4 cohort was 1.9% behind native CPU and 4.8% ahead of LMCache MP.
Neither result establishes a universal performance advantage.

1. **Measure the engine-visible completion boundary.** Extend the existing
   timeline/observation owners to separate preparation, grant wait, native queue,
   submission, actual GPU drain, connector observation and first engine use.
   Record source retirement separately. Carry bounded, generation-fenced drain
   evidence before enabling `CacheRestore` training for engine-local
   raw grants; Manager reap time cannot stand in for DecodeReady. Keep tracing
   opt-in and measure its overhead. This also closes the gap introduced when
   combining Manager-side completion observations with the local executor.
2. **Remove the measured 8K overhead.** Use the decomposition to choose between
   repeated destination validation, descriptor work, queue handoffs and copy
   submission. Reuse immutable registration geometry and bounded scratch where
   evidence supports it. Preserve allocation boundaries, ready-stream ordering,
   the shared 128-operation device admission and quarantine lifetime. Change
   one measured bottleneck per increment; remove its superseded path outright.
3. **Repeat matched end-to-end acceptance.** Run native HBM, native CPU, OrbitKV
   and LMCache in the same engine/runtime and capacity budget, retaining cold,
   resident and pressure phases. Alternate backend order across repeated runs.
   Add deterministic C4 output parity before treating the non-batch-invariant
   cohort as quality-preserving evidence. Refresh SGLang performance separately;
   its correctness gate does not substitute for that comparison. Accept a change
   only with a repeatable TTFT/E2E or throughput gain and no material cold-path,
   CPU-cost or correctness regression.
4. **Then qualify partitioning and overlap.** Add bounded fragmented-plan
   partitions with one parent completion owner before layer/group consumption.
   For overlap, validate actual eager and graph-replay dependencies, cancellation,
   partial enqueue and page reuse. A retained whole-operation source fence is
   still required even when the engine can consume an earlier group.
5. **Move remote metadata only with a measured hot path.** Keep etcd membership
   outside request execution. Complete bounded binary TENT notifications and
   peer credits/ACKs before replacing hot metadata RPCs. Separately, the route
   selector must consume both source leases and P/D handoff authority before it
   can act on the unified cost evidence. Do not enable selection from stale
   hints or merge control ACKs with payload drain evidence.

### Engine-local completion evidence

The native worker carries six cumulative nanosecond offsets from one engine
`Instant`: readiness finished, dispatched, dequeued, grant claimed, CUDA enqueue
returned, and GPU drain observed. The first interval includes native argument
conversion, client/executor locking and destination readiness. Dispatch ends at
native job enqueue; grant wait includes worker scheduling and plan consumption.
Submission includes validation and descriptor compilation. These are host
observations, not GPU kernel timestamps.

The existing session/operation-fenced completion record transports the bounded
report with `Active -> Drained`. The Manager accepts it when consuming the source
owner, once, before reaping. Invalid, absent or over-one-day reports are ignored
without blocking drain, reaping or record reuse. Failed and never-submitted
operations do not train success estimates. Late Manager observation cannot
extend engine-ready latency, and no cross-process timestamp subtraction is used.

`ORBITKV_COST_OBSERVATIONS=1` on both processes records the independent
`engine_local_restore` cost key. It is deliberately excluded from route selection:
Manager `cache_restore` starts at preparation, and P/D starts at handoff enqueue.
A shared target GPU does not make these start boundaries comparable.

`ORBITKV_TRACE_TRANSFERS=1` additionally emits `local_restore_complete` from the
Manager and `local_restore_observed` from native result consumption. The common
Rust tracer serves both processes. The benchmark parser joins those records to
existing connector request links, counts each physical batch once, and reports
native stage and consumer-wait quantiles separately from framework first-use
callbacks. Local completions do not require a Manager-to-engine notification.
Tracing and cost collection are opt-in; the disabled path takes no stage clocks.

Qualification includes stale/recycled and duplicate completions, malformed
reports, lost notification, cancellation and actual process exit. The merged
serving and instrumentation-overhead gates remain distinct from code completion.

The following performance increment selects one demonstrated bottleneck and
compares the change with that baseline. Use at least three paired repetitions
with backend order reversal for native HBM, native CPU offload, OrbitKV and
LMCache under the same engine, capacity, prompt and output-quality controls.
Record TTFT, end-to-end latency, tails, CPU cost and transfer bytes. The older
6.79 ms gap is a profiling lead, not a guaranteed amount recoverable in Restore.

After that, implement bounded large-plan partitioning and then layer/group
overlap as separate changes. Cost-driven direct/P-D execution follows only when
both candidates have the same measured completion boundary and real source,
destination and capacity authority. Retain deterministic selection while that
contract is incomplete. Real two-host DP qualification remains a separate gate;
local performance work does not establish RDMA or catalog availability.

## Next: bounded large restores and execution overlap

Complete automatic plan partitioning with a parent whole-operation fence before
claiming arbitrary large-prefix support. Suboperations must retain source and
destination ownership through accepted CUDA work, preserve batch/lease semantics,
and respect plan, record, source-byte, and native queue budgets.

Then add layer/group dependencies so early groups can be consumed while later
groups restore. Compile actual framework dependencies, preserve one final drain,
and qualify both eager execution and replay-time graph dependencies. Page-first
allocations require shared last-use ownership across consuming groups.

SSD materialization, codec execution, and direct SSD routes may migrate only
with their concrete storage, scratch, registration, and completion owners.
Their current Manager execution remains a supported physical route. Moving
CUDA submission does not remove the physical host-to-HBM transfer or establish
an advantage over an engine's resident HBM hit.

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
| Implemented, single-GPU process gates passed: raw engine restore | Payload FD attachment, retained source grants, tensor ownership, native CUDA execution and local results | Manager raw-descriptor submission and `LoadPayload` enum | [Process fault evidence](fault-qualification.md#engine-local-raw-restore-gates); scoped DRAM serving gates also passed on the communication-branch build; merged-artifact, graph replay and extended-environment qualification remain separate |
| Completed: raw plan and idle readiness | Allocation-aware run compaction before encoding; query idle streams and reuse the busy-stream event | Per-page descriptors for contiguous runs and redundant GPU event submission on idle streams | [Matched measurements](communication-performance.md), large dense plan bytes, lease preservation on oversized fragmented plans, and reused-event readiness |
| Next: residual raw overhead | Profile native scheduling, fragmented plans and scratch reuse | Measured redundant work in the remaining path | Small-payload latency, unchanged source/destination drain guarantees and failure gates |
| Next: large raw plans | Bounded suboperations with a parent whole-operation fence | Current rejection above the 1 MiB per-plan limit | Fragmented large-prefix bytes, cancellation between partitions and bounded plan/source credits |
| Next: execution overlap | Layer-group dependencies with one final retirement fence | Whole-restore waits from engine consumption sites covered by qualified group dependencies | Pinned engine releases, eager/graph replay, page reuse, TTFT/ITL and CPU cost |
| Next: native metadata | Bounded binary notification API and per-peer transport selection | Unsafe string framing and first-transport notification dispatch | Size/queue limits, unreachable peer, mixed transports and native shutdown |
| Next: peer session | Batched lookup/grant/completion with application ACK and credits | Corresponding hot gRPC methods, retry owner and protobuf messages | Loss, duplication, reorder, restart, corruption, slow peer and multi-host qualification |

Each cutover replaces its old implementation and updates all callers in the
same change. Capability-specific SSD preparation or codec work remains with
its resource owner; it is not a fallback copy of the exact/raw executor.

## Qualification

For local completions, cover real process boundaries, capacity pressure,
generation reuse, reconnects, concurrent readers, bounded UTF-8 error payloads,
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
