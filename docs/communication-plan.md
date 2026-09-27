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

The existing completion timeline distinguishes worker completion from result
publication/notification. There is no server-side terminal Poll delivery event
on this path. Missing delivery observations are not zero delivery latency.

### One-segment peer authorization lookahead

`ORBITKV_PEER_PIPELINE=1` allows the authorization of the next planned segment
to overlap the current segment's READ. The default remains sequential for
matched comparisons. There is at most one speculative grant per fetch plan;
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
bounded pressure recovery applies to sequential operation.
The three-second bound is for release waiting, not the entire fetch; each
authorization retains its own RPC deadline. `release_wait` records this wait,
and fetch-plan attempt counts describe logical segment attempts rather than
individual RPC retries.

This increment uses the existing source-control RPC and Mooncake READ backend.
It does not introduce speculative payload reads, pre-authorized persistent
hotspot replicas, or a new metadata transport. Prepared-grant residence time
must not train the complete-route service-cost estimator.

## Next: engine-side GPU executor

The smallest useful slice is exact/raw DRAM-to-HBM restoration on one GPU.
The Manager remains the allocator and source-lease owner; a native executor in
the engine process owns CUDA submission using the engine's current GPU pages.

This requires a shared backing for KV payload allocations. The existing
descriptor memfd is not a payload pool, and current private anonymous/pinned
allocations cannot be handed to another process merely by sending an offset.
The shared pool must establish per-process mappings, CUDA registration,
NUMA placement, and allocation-generation validation before executing plans.

A prepared plan holds its sources and describes bounded arena ranges. Engine
destination pages are bound and validated when allocated. The executor first
preserves the whole-operation completion fence, then adds layer-group
dependencies so early groups can be consumed while later groups are restored.
Both eager execution and the supported CUDA graph replay modes must pass the
same page-reuse and cancellation gates. Moving CUDA submission into the engine
does not remove the physical HBM/host-memory transfer.

## Next: batched remote metadata messages

Keep etcd membership, epochs and placement outside the per-request lookup
path. Retain ordinary RPC for bootstrap and low-frequency management. Define
batched grant, completion, acknowledgement and credit messages around the
existing source authority, then compare TENT notifications with the existing
RPC path and a UCX Active Messages implementation if needed.

The pinned TENT source contains an RDMA SEND/RECV notification backend. Its
TCP backend uses a control RPC. A production adapter needs bounded receive
queues, binary-safe framing, message-size limits, reconnect epochs and
application acknowledgements. A successful notification send is not proof
that the receiver consumed it or that a separate payload READ drained.

Only after revocation/drain is qualified should hotspot grants be issued ahead
of demand. Each grant must hold the precise source allocation and consume a
bounded budget. A local directory hint is not an authorization. Unused and
in-flight grants need distinct retirement paths; lease expiry cannot release
memory still accessible to submitted READs.

## Qualification

For local completions, cover real process boundaries, capacity pressure,
generation reuse, reconnects, concurrent readers, complete error payloads,
lost eventfd notifications, and timeouts that retain pending ownership. Run
the existing Manager/native-client GPU integration and fault gates after the
ABI update.

For peer lookahead, use deterministic synchronization to prove authorization
actually overlaps a blocked READ. Compare sequential and pipelined results,
including partial prefixes, rejection, failed reads and cancellation. On real
peers compare segment counts, authorization/READ timing, source bytes held,
outstanding completion records and end-to-end restore latency, then serving
TTFT/ITL and goodput. Test both DRAM and owner-prepared SSD sources.

Use the [single-node](single-node-performance.md) and
[shared-cache qualification](shared-cache-qualification.md) workloads. Record
the engine, CUDA, transport and graph configurations; account for polling CPU
cores and speculative source holds. Passing protocol tests does not establish
an end-to-end speedup or cross-host RDMA qualification.
