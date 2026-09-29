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
sorted before submission. The direct backend merges contiguous ranges and
explicit equal-width, constant-pitch rows within one host CUDA registration and
one GPU allocation. Every logical source allocation is validated and retained
independently. CUDA submission and whole-operation drain now
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

The native executor now publishes per-layer events while retaining one final
source-retirement fence. vLLM uses synchronous admission, per-layer callbacks
for eager/piecewise execution and a dependency on all restored layers at full-graph
entry. SGLang installs persistent external events before first capture and orders
each pool's copies by their consuming layer. Dense/hybrid DRAM/SSD correctness
and graph gates pass on H20. The [three-round comparison](communication-performance.md#repeated-serving-comparison-after-layer-readiness)
shows a vLLM serving gain; SGLang throughput is unchanged within the observed
ranges and remains behind native CPU offload. The exact ownership and
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

A full shared plan bank defers prepared grants until space returns. Preparation
sorts destinations and merges consecutive source and destination ranges only
within the same layer and allocation identity/bounds. It validates all source
ranges and destination overlaps before consuming any lease, then partitions the
remaining descriptors into parts of at most 1 MiB. The whole operation is
limited to 32 MiB of encoded metadata; the Manager reserves at most 64 MiB of
pending plan metadata per session. Admission failure does not submit any DMA.
No second raw executor or compatibility protocol is retained.

One operation ID, source grant, query reservation and destination owner span all
parts. Nonfinal parts use `GrantedMore → ActiveMore → PartDrained`; the Manager
acknowledges each drain once, returns the operation to `Preparing`, and appends
its next part to the pending queue. Only the final part or a drained failure
publishes `Drained`. Intermediate completion cannot expose a successful Python
result or release any source credits. Disconnect revokes unclaimed parts;
unknown active DMA retains its owners and metadata budget in quarantine.
The shared-grant schema is version 5 and requires matched builds.

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
cloning and encoding binary-search candidates. Raw Restore partitions its
compiled pointer-free plan separately, with one whole-operation completion owner. Each channel's descriptor slot retains its
request/response lock.

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

1. **Completed: measure the engine-visible completion boundary.** Bounded,
   generation-fenced native reports separate readiness, dispatch, queue, grant
   wait, plan/enqueue, drain and result consumption from Manager retirement.
   Source/backend-scoped caller-to-drain estimates remain separate from Manager
   and P/D route-selection intervals. Instrumentation overhead, actual serving
   stages and both engines' deterministic C4 output parity are recorded in
   [communication measurements](communication-performance.md) and
   [the C4 gate](sustained-performance.md#deterministic-c4-qualification-after-completion-evidence).
2. **Remove the measured 8K overhead.** Use the decomposition to choose between
   repeated destination validation, descriptor work, queue handoffs and copy
   submission. Reuse immutable registration geometry and bounded scratch where
   evidence supports it. Preserve allocation boundaries, ready-stream ordering,
   the shared 128-operation device admission and quarantine lifetime. Change
   one measured bottleneck per increment; remove its superseded path outright.
3. **Repeat matched end-to-end acceptance.** Run native HBM, native CPU, OrbitKV
   and LMCache in the same engine/runtime and capacity budget, retaining cold,
   resident and pressure phases. Alternate backend order across repeated runs.
   The deterministic C4 gate now passes for both engines; do not transfer that
   claim to ordinary-mode cohorts. Refresh SGLang performance separately;
   its correctness gate does not substitute for repeated comparisons. Accept a change
   only with a repeatable TTFT/E2E or throughput gain and no material cold-path,
   CPU-cost or correctness regression.
4. **Qualify partitioning, then overlap.** Bounded fragmented-plan parts now
   share one parent completion owner. Retain their byte and process-fault gates
   before adding layer/group consumption.
   For overlap, validate actual eager and graph-replay dependencies, cancellation,
   partial enqueue and page reuse. A retained whole-operation source fence is
   still required even when the engine can consume an earlier group.
5. **Remove directory round trips with a local global index.** Publish block
   metadata to etcd in bounded background batches and maintain a complete view
   through a fixed-revision snapshot and Watch. Remove Catalog lookup RPCs at
   cutover; retain source grants/completions and upstream TENT READ/WRITE.
   Measure remaining source-control cost before replacing its transport. The
   route selector still needs source leases and P/D handoff authority; index
   freshness and control ACKs never replace payload drain evidence.

### Engine-local completion evidence

The native worker carries six cumulative nanosecond offsets from one engine
`Instant`: readiness finished, dispatched, dequeued, grant claimed, CUDA enqueue
returned, and GPU drain observed. The first interval includes native argument
conversion, client/executor locking and destination readiness. Dispatch reaches
native job construction; the queue interval includes enqueue handoff and worker
delay. Grant wait includes worker scheduling and plan consumption.
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
reports, lost notification, cancellation and actual process exit. The final
single-GPU serving and instrumentation-overhead gates pass; repeated performance
acceptance remains separate from code completion.

The following performance increment selects one demonstrated bottleneck and
compares the change with that baseline. Use at least three paired repetitions
with backend order reversal for native HBM, native CPU offload, OrbitKV and
LMCache under the same engine, capacity, prompt and output-quality controls.
Record TTFT, end-to-end latency, tails, CPU cost and transfer bytes. The older
6.79 ms gap is a profiling lead, not a guaranteed amount recoverable in Restore.

Bounded large-plan partitioning now retains a whole-operation fence. Add
layer/group overlap as a separate change. Cost-driven direct/P-D execution
follows only when both candidates have the same measured completion boundary and real source,
destination and capacity authority. Retain deterministic selection while that
contract is incomplete. Real two-host DP qualification remains a separate gate;
local performance work does not establish RDMA or catalog availability.

## Strided DMA for raw transfers

The direct backend now compiles regular rows into `cuMemcpy2DAsync` submissions
for both H2D and D2H. This replaces the old contiguous-only compiler; there is
no compatibility backend or new runtime selector. Engine-local restore and
Manager publication use the same compiler. Single rows still use directional
1D DMA. Model-serving and repeated performance acceptance are separate gates.

The implementation preserves these constraints:

1. `transfer/memcpy.rs` compiles each already validated descriptor list
   into contiguous runs or constant-pitch rows. A row group grows only from
   explicit input descriptors of equal width, with checked pointer arithmetic
   and source/destination pitches between row width and the device's maximum
   pitch. Do not infer missing rows or copy inter-row gaps.
2. The imported arena's CUDA registration identity is separate from the source
   allocation's lifetime identity in the local executor. Continue validating
   every logical allocation ID, generation and bounds before compilation; the
   Manager grant must retain every contributing allocation until final drain.
   Combining rows in one CUDA registration must never turn that registration
   into permission to access an unlisted allocation or a gap.
3. Preserve the final stream drain, partial-enqueue error handling,
   destination readiness, shared device permits and quarantine. Do not retry a
   failed batch or introduce a second runtime backend selection switch.
4. Count actual compiled submissions using the same compiler and device pitch
   limit in the existing direct-backend cost shape.
   Keep historical direct/kernel observations isolated; do not enable dynamic
   selection from the synthetic probe.

Unit gates expand the compiled rows back to the original byte pairs, including
2,000 irregular layouts, registration boundaries, uneven strides, overlap,
overflow and maximum pitch. GPU acceptance requires
H2D/D2H sentinel-gap tests, engine/Manager death and partial-enqueue fault gates,
both deterministic model gates, and at least three matched serving repetitions
with order reversal. A synthetic copy gain alone cannot close this item.

### Functional qualification, 2026-09-28

The integrated compiler passes full release workspace tests and strict workspace
Clippy. Four H20 GPU cases cover bidirectional sentinel gaps, shared-arena
suballocation bounds, context restoration and kernel/direct byte parity.
Twelve process-fault cases pass, including Manager death after claim,
engine-death quarantine, lost notifications and partial-submit drain.
Qwen3-8B passes the vLLM deterministic gate (six checks; one hybrid-only skip)
and SGLang DRAM restart recovery. Raw logs and artifact hashes are retained under
`benches/results/runs/strided-dma-20260928/`.

The [three-round serving comparison](communication-performance.md#repeated-serving-comparison-after-strided-dma)
now covers both engines and all four backends, with 24 accepted cohorts and
1,152 matching non-native output controls. OrbitKV throughput exceeds LMCache
by 5.18%/5.27% for vLLM/SGLang, but remains 4.08%/2.25% below native CPU
offload. SGLang tail latency also remains higher. These comparisons do not
isolate the effect of 2D DMA: a repeated before/after implementation control
is still required before attributing a serving speedup to this change.

The subsequent [layer-readiness comparison](communication-performance.md#repeated-serving-comparison-after-layer-readiness)
adds an explicit pre-change implementation control. It supersedes these serving
numbers for the current code; it still does not isolate the earlier 2D DMA change.

## Bounded large restores and remaining execution overlap

Raw plans are automatically partitioned under one whole-operation fence. The
32 MiB operation metadata and 64 MiB session metadata limits still bound
admission; this is not unbounded large-prefix support. Source, destination,
query-byte, record and native queue owners remain held across every part.

On 2026-09-28, the H20 gate restores 2,048 permuted pages across five layers
(20,480 separate K/V ranges, 40 MiB payload) through multiple parts. It checks
all destination bytes, no early result or load-byte accounting, retained query
credits between parts, second-part partial-enqueue drain, and Manager death
between parts. All three cases pass; seven existing local ownership/fault cases
also pass. Raw artifacts and test-build hashes are under
`benches/results/runs/partitioned-restore-20260928/`. This is correctness and
fault evidence, not a measured serving speedup.

The normal release build also passes the vLLM Qwen3-8B correctness gate
(six passes; one hybrid-only case skipped for this dense model), SGLang DRAM
and SSD process-restart gates (two passes), and the same-A100 SGLang P/D
restart/reuse gate with exact outputs and 576 cached tokens. A fresh two-host
H20↔A100 byte gate passes 8 MiB in each direction, including re-serving the
received replica after original-source eviction; all checked counters drain.
The matching native hashes, launch logs and results are retained in the same
artifact directory. These gates do not establish layer overlap or RDMA.

Per-layer event publication now allows raw single-part restores to overlap
consumer execution. Bounded parts retain their intermediate drains; vLLM
recurrent state and packed cross-layer bindings retain coarser dependencies.
The [implemented engine consumption contracts](engine-local-restore.md#layer-readiness-and-framework-consumption)
separate event publication, final DMA completion and source retirement. Serving
and matched performance gates must qualify each advertised mode.

SSD materialization, codec execution, and direct SSD routes may migrate only
with their concrete storage, scratch, registration, and completion owners.
Their current Manager execution remains a supported physical route. Moving
CUDA submission does not remove the physical host-to-HBM transfer or establish
an advantage over an engine's resident HBM hit.

## Next: local global index and background metadata

The selected [distributed design](distributed-cache.md#selected-target-local-global-index-and-etcd-metadata)
replaces sharded Catalog discovery with a complete local global index at every
Manager. etcd stores block locations and membership; publication, snapshot and
Watch run in background. Both warm and previously unqueried keys use local
discovery. No etcd call or remote Catalog fallback enters the request path.

Retain bounded batched OrbitKV source grants/completions and Mooncake TENT
READ/WRITE. Remove Catalog serving, placement, lookup coalescing and TTL hints
together at cutover. The custom native binary prototype is outside this plan;
its application protocol was never deployed. The [control boundary and upstream
audit](peer-control.md) distinguish it from independent upstream defects/fix PRs.

Qualify ordered publication, snapshot/Watch repair, index capacity, etcd quota/
churn, leader/quorum loss and both engines' remote serving. Source lifetimes are
independent: an expired metadata record cannot release memory still reachable
by an undrained READ. Measure source RPC cost after local-index cutover before
proposing another transport, and keep application protocols in OrbitKV.

## Implementation and deletion gates

| Step | Deliverable | Remove at cutover | Acceptance |
| --- | --- | --- | --- |
| Completed: local dispatch | Required request event, bounded spin before sleep and independent Publish reply eventfd | Fixed Manager 50 us idle poll, Publish 100 us reply sleep and forwarding descriptor helpers | Microbenchmarks with CPU/latency, process-boundary wake races, Publish faults and independent notification counters |
| Completed: local completion | Shared terminal records and direct eventfd signal | Terminal Poll command, dispatcher restore scan, retired timeline decoder | Cross-process results, stale IDs, missing notifications, real GPU faults |
| Completed: payload backing | Shared memfd for every pool shard | Private anonymous and `cudaHostAlloc` pool paths, `cpu_readable` plumbing | FD transfer, independent registration and GPU bytes after producer mapping teardown |
| Completed: peer lookahead | One active READ plus one next authorization | Sequential runtime selector | Prefix integrity, cancelled/lost grants, release pressure, real multi-segment TENT bytes |
| Implemented, single-GPU process gates passed: raw engine restore | Payload FD attachment, retained source grants, tensor ownership, native CUDA execution and local results | Manager raw-descriptor submission and `LoadPayload` enum | [Process fault evidence](fault-qualification.md#engine-local-raw-restore-gates); scoped DRAM serving gates also passed on the communication-branch build; merged-artifact, graph replay and extended-environment qualification remain separate |
| Completed: raw plan and idle readiness | Allocation-aware run compaction before encoding; query idle streams and reuse the busy-stream event | Per-page descriptors for contiguous runs and redundant GPU event submission on idle streams | [Matched measurements](communication-performance.md), large dense plan bytes, lease preservation on invalid or over-budget plans, and reused-event readiness |
| Next: residual raw overhead | Profile native scheduling, fragmented plans and scratch reuse | Measured redundant work in the remaining path | Small-payload latency, unchanged source/destination drain guarantees and failure gates |
| Implemented: large raw plans | Bounded parts under one operation ID, with a final completion fence and per-session metadata credits | Rejection solely because a compacted plan exceeds the 1 MiB shared bank | Fragmented large-prefix bytes, cancellation between partitions and bounded plan/source credits |
| Implemented, H20 dense/hybrid serving qualified: raw execution overlap | Native layer events with one final retirement fence; vLLM layer callbacks/full-graph entry waits and SGLang external graph waits | vLLM asynchronous load notification bookkeeping, SGLang per-request Restore window and whole-operation first-use wait | Pinned engine releases, eager/graph replay, page reuse, TTFT/ITL and CPU cost |
| Next: global synchronization | Fenced etcd publication, snapshot/Watch and complete local index | Sharded inventory replay and Catalog hosting | Delete/recreate, uncertain writes, compaction, restart, memory and etcd quota limits |
| Next: local discovery | Warm/cold-key local lookup; existing source grants and TENT payloads | Catalog lookup RPCs, fixed placement, TTL hints and remote lookup coalescing | Zero foreground discovery RPCs, two-host output controls, three-member etcd failures and metadata cost |

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
