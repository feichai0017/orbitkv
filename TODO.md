# OrbitKV implementation TODO

This is the repository-wide execution checklist. Completed items must have code
and a passing gate; design text alone does not close an item.

Follow the [current delivery priorities](docs/roadmap.md#current-delivery-priorities):
close the single-node hybrid-layout gates, maintain deterministic demand and
model-serving fault coverage, and start real two-host DP qualification.
Warming gains are not a DP prerequisite. P/D with cache reuse follows; replicated
catalogs are required before production distributed deployment. Milestone
numbers below group work areas rather than imposing a strict serial schedule.

## GPU storage

- [x] Implement Rust cuFile demand reads and complete-group GPU writes with SSD extent
  leases, bounded GPU staging, split/page-first layouts, oversized checkpoints,
  cancellation, failed writes and short reads; keep both engines on the existing cache API.
- [ ] Qualify native GDS on a supported NVMe mount with CPU fallback disabled;
  distinguish first writes from overwrites and compare io_uring using matched
  working sets, TTFT, throughput, CPU use and bytes.
- [x] Provide a bare-metal acceptance script with fallback rejection and matched
  io_uring/auto/cuFile workloads for both engines (`python -m benches.gds`).
- [x] Default to automatic native cuFile initialization with io_uring fallback;
  stop new GPU I/O after an operation failure without revoking in-flight ownership.
- [x] Reserve physical GPU-storage file space before admission; report allocation
  failures and release partial startup reservations. Keep file ownership in `backing/ssd/files.rs`.
- [x] Implement bounded Rust asynchronous cuFile submissions with reusable staging
  slots, stream/event completion, per-operation byte/error checks and cancellation drain.
- [x] Coalesce reads by file across leased sources without broadening required ranges;
  verify call counts, separate files, unrequested gaps and cancellation ownership.
- [x] Bound queued GPU-storage writes and staging so demand reads make progress
  between write batches: two 4 MiB slots, one in-flight write, eight admitted write
  jobs and bounded read bursts; saturation uses host publication/io_uring.
- [ ] Measure selective DRAM admission and shorter source-HBM holds for GPU writeback;
  retained staging and unpublished SSD reservations must survive disk completion.
- [ ] Qualify direct registered engine-page I/O, multi-writer GPU assembly,
  and workload-based path selection. Include registration cost,
  engine-page hold times, I/O fragmentation and additional HBM in the decision.

See [GPU storage recovery](docs/gds.md) for deployment and reproduction, and the
[LMCache v0.5.5 review](docs/gds.md#review-against-lmcache) for optimization evidence and gates.
The [upstream design mapping](docs/architecture.md#upstream-designs-and-orbitkv-owners)
records LMCache, FlexKV and Mooncake mechanisms, owners and implementation status.

## M0 — framework-neutral foundation

- [x] Establish the OrbitKV data plane and workspace.
- [x] Move Rust packages under `crates/`.
- [x] Keep NUMA topology/affinity in Core and HLL reuse statistics in Server;
  limit `orbitkv-common` to shared process logging and peer-connection defaults.
- [x] Remove the unused repository-root `src/main.rs`.
- [x] Add `orbitkv-state` with state identity, format, page generation, and
  recovery-bundle types.
- [x] Name the bundle's current component-presence check honestly; it is not
  yet a restorable-state proof.
- [x] Move the canonical vLLM package to `orbitkv.vllm`.
- [x] Group the vLLM cache connector and P/D adapter under `orbitkv.vllm`.
- [x] Add `orbitkv.client` and `orbitkv.sglang` package boundaries.
- [x] Add `orbitkv-channel` with a versioned 64-byte iceoryx2 request/response ABI.
- [x] Add a real two-process channel test.
- [x] Integrate the iceoryx2 lifecycle endpoint into `orbitkv-server`.
- [x] Add Python `ChannelProbeClient` bindings with epoch fencing.
- [x] Replace the copied native RDMA stacks with a pinned stable Mooncake
  Transfer Engine sys crate and one clean transfer API.
- [ ] Add Python representations/serialization for `orbitkv-state`.
- [x] Run the full M0 validation matrix and record results in the commit.

## M1 — SGLang direct GPU-page linker

- [x] Register SGLang full-attention MHA/MLA GPU buffers through CUDA IPC.
- [x] Document SGLang direct-linker configuration and supported layouts.
- [x] Add UDS bootstrap for the memfd-backed descriptor arena.
- [x] Bind `orbitkv-channel` QueryBundle to the shared core query path.
- [x] Bind `orbitkv-channel` Release to the shared core lease path.
- [x] Bind `orbitkv-channel` Publish to the shared core save path.
- [x] Bind `orbitkv-channel` Restore to core oneshot completion and eventfd wakeup.
- [x] Add Python bindings for the iceoryx2 local client.
- [x] Reject SGLang representations without a complete recovery contract.
- [ ] Add cold-miss, partial-prefix, warm-hit, cancellation, and restart tests.
- [x] Run one real SGLang model E2E, including restore after radix-cache flush.
- [x] Register a SGLang RadixCache plugin that transfers full-attention GPU KV
  through CUDA IPC and iceoryx2, with a real Cache Manager load after SGLang
  process restart and cold-inference output comparison.
- [x] Add direct GPU recovery contracts for Full + SWA and Full + recurrent/conv,
  including sparse SSD membership, exact-boundary validation and cancellation.
- [x] Compose Full + SWA + recurrent/conv in both adapters; give vLLM windows
  independent storage and GPU save ownership, and register SGLang same-layer
  convolution state without empty temporal buffers. Verify required GPU bytes
  and incomplete-state rejection from DRAM and SSD.
- [x] Qualify native Full + SWA serving in both engines with Mellum and
  SGLang Full + SWA + convolution with Inkling; retain Qwen3.5 regression gates.
  SGLang covers DRAM/SSD, concurrent restore, HBM flush and engine restart;
  vLLM compares matched native-cache execution plans and restart loads.
- [ ] Qualify native model serving with Full + SWA + temporal recurrent state
  in both engines; current combined temporal coverage is exact GPU recovery.
- [x] Qualify Qwen3.8-27B-FP8 on both engines with DRAM and forced SSD recovery;
  qualify GLM-4.7-Flash, DeepSeek-V2-Lite and Kimi Linear FP8 with SSD-enabled
  reuse, native output controls and engine restart. Keep exact artifacts,
  settings and larger-model blockers in [model qualification](docs/models.md).
- [ ] Add recovery contracts for DSA, draft-model and further auxiliary state.

## M2 — common bundle and local IPC

- [x] Fingerprint immutable model artifacts and bind computation/configuration in
  both adapters; reject dynamic LoRA until adapter-content identities are available.
- [x] Use the shared versioned `StateKey` across DRAM/SSD and remote directory
  records; include actual registered storage geometry and invalidate old keys.
- [x] Carry SGLang's engine-held prefix origin and leased group positions into
  shared recovery validation; intersect legal boundary sets across ranks.
- [x] Carry vLLM hybrid spans and leased component evidence into shared recovery
  validation; intersect absolute legal boundaries across shards.
- [ ] Support adapter identities and invalidate caches on live weight updates.
- [x] Compile vLLM cache-group requirements and assemble hybrid `StateBundle`
  evidence through the shared native binding.
- [x] Compile prefix/window/checkpoint rules and validate complete token coverage
  in the registered model/format namespace before SGLang advertises a hit.
- [x] Compile state demand: normalize those rules to page requirements
  and expose `required_ranges(namespace, start, end)` as absolute aligned
  group intervals from a valid HBM origin; use the same rules for leased evidence.
- [x] Gate SGLang exact transferred-plus-retained coverage and vLLM hybrid
  allocation against compiled ranges.
- [x] Separate metadata candidate discovery from byte materialization. Rust
  computes rank-common legal boundaries and fetches only selected required
  ranges; actual leases are revalidated before admission. vLLM clamps before
  reading; SGLang synchronizes readiness before allocation. A stale range
  releases partial leases and falls back to the valid HBM origin.
  Exact SSD-byte gates cover selected prefixes, windows and checkpoints;
  additional discovery rounds are not claimed as a TTFT improvement.
- [x] Move hybrid-boundary validation out of `orbitkv.vllm`; retain engine-owned
  allocation and checkpoint handoff, and apply the token limit before reading.
- [x] Handle asynchronous vLLM checkpoint queries from SSD, retain completed
  groups during preparation, and retire pending groups on cancel/drift/expiry.
  Verify native validation and exact DRAM/SSD GPU restoration in
  `python/tests/integration/test_vllm_recovery.py`.
- [ ] Define framework-neutral region registration RPCs.
- [x] Pass the descriptor-arena memfd and notification eventfd over UDS.
- [x] Add bounded restore operations that replace per-load `PyLoadState`
  for the native Cache Manager client.
- [x] Implement direct SGLang full-attention GPU restore through the process
  channel to Cache Manager.
- [x] Switch vLLM Query/Publish/Restore/Release to the Cache Manager client.
- [x] Move registration, health, session watching, and unregister to UDS;
  remove the inference gRPC endpoint.
- [x] Require the node-local process endpoint; fail fast if its socket is missing.
- [x] Group the Cache Manager's cache operations and process endpoint separately;
  convert protobuf registration messages before entering the cache lifecycle.
- [ ] Replace framework CUDA IPC wrapper pickle in the Cache Manager with an explicit
  region registration contract after the existing vLLM path is qualified.
- [x] Serialize lifecycle operations and drain GPU queues before unmapping CUDA IPC.
- [x] Remove per-load `PyLoadState` from the vLLM path.
- [x] Keep local control messages descriptor-only; prohibit KV payload bytes in
  UDS or iceoryx2 messages.
- [x] Require UDS and iceoryx2 for local inference control.
- [x] Qualify the revised vLLM correctness E2E on a
  GPU/vLLM host. It compares the same prompt/reuse plan against native prefix
  caching, checks the native prefix hit, and requires `long_warm` to load KV
  bytes after process restart;
  the earlier cold-vs-warm comparison reproduced native vLLM divergence.
- [x] Requalify the vLLM E2E against release 0.29.0, including a hybrid model
  that exercises scheduler boundary-state hand-offs.
- [ ] Add generation validation to every local page reference.
- [x] Keep restore destinations held on lost acknowledgements, poll failures,
  and deadlines; fail the engine instead of claiming DMA was cancelled.
- [x] Drain partially submitted H2D/D2H work before returning a backend error;
  terminate the manager if CUDA cannot establish completion.
- [x] Split Publish metadata to the negotiated descriptor capacity while
  preserving per-page layer completeness and retaining sources through all chunks.
- [ ] Benchmark the M2 path against the current CUDA IPC baseline.
- [x] Record Qwen3-8B serial cold, resident, and post-eviction latency against
  vLLM CPU offload and SGLang HiCache, with equal payload budgets and verified
  cache sources; retain raw measurements and commands in
  `docs/single-node-performance.md`.
- [x] Measure forced SSD restores with live `O_DIRECT` evidence on both engines;
  retain SGLang's unsuccessful prefetches as misses in `docs/ssd-performance.md`.
- [x] Verify both stored page layouts through SSD write, DRAM eviction, and
  exact GPU-byte restoration with explicitly polled readiness.
- [x] Use SGLang 0.5.20's general plugin admission hook to defer pending requests,
  and qualify single-rank DRAM/SSD serving recovery (P0/P2).
- [x] Unify query ownership in the endpoint; remove core request-string tasks,
  bind immutable arguments, cancel superseded work, bound active operations,
  and drain late results after cancel/disconnect without another poll (P1 foundation).
- [x] Add explicit query operation/revision tickets and global/per-instance
  byte admission retained through result leases and GPU completion.
- [x] Move shared client query/warming ownership, independent publish sessions
  and restore waiting into Rust. Remove the Python manager facade and raw
  Python ChannelClient API; reuse immutable Rust hash batches and shared prefix
  views on repeated polls and bind restore handles to their issuing client.
  Keep engine allocation in Python. Avoid cloning pending Manager query inputs
  until byte admission succeeds.
- [ ] Profile remaining adapter hashing, per-page metadata and PyO3 conversion
  under matched workloads before claiming a latency improvement from the Rust client.
- [x] Share identical backing reads with independent cancellation and leases;
  make SSD read queue pressure wait for capacity.
- [x] Record shared/mixed 1/4/8-request bursts on both engines with a 2 GiB
  query budget; preserve output differences and native/deterministic controls
  in `docs/concurrent-performance.md`.
- [x] Add bounded sustained mixed reuse/cold traffic, admission and completion
  timing, post-run drain checks and incomplete-report rejection in `benches/`.
- [x] Prioritize admitted vLLM restores through their first compute step so
  deferred lookups cannot strand them behind a GPU allocation failure.
- [x] Record sustained single-node native/DRAM/SSD controls for both engines;
  separate throughput, transfer evidence and output diagnostics.
- [x] Retire SGLang queries when HBM covers the legal recovery boundary;
  enforce admission expiry without another lookup and test simultaneous recovery.
- [ ] Profile vLLM duplicate H2D restores for shared prefixes; any reuse must
  respect engine-owned GPU destinations, mutable tails, and completion fences.
- [x] Qualify deterministic SSD delay/cancel, lost restore notifications,
  Manager restart with live old clients, and stalled/malformed Publish replies
  using real CUDA registrations and bounded test-only barriers. Assert other
  queries progress and reservations drain; fence ambiguous Publish replies
  until peer death and create a fresh channel incarnation on every restart.
  See `docs/fault-qualification.md`.
- [x] Qualify Qwen3-8B TP=1 concurrent serving under delayed-read cancellation,
  dropped notifications and engine/Manager restart in both engines, with ordinary
  demand and owned preparation. Keep deterministic output and exact-byte gates.
- [x] Profile ordinary Qwen3-8B cold/shared/mixed recovery with stage timing,
  TTFT, throughput, read/copy bytes and resource-drain evidence. Record output
  differences separately (`docs/recovery-performance.md`).
- [x] Fence vLLM saves on producer-stream events outside CUDA graph capture;
  verify exact restored bytes while unrelated GPU work remains in flight.
- [x] Separate SSD read/write submission queues and distribute single-file
  reads across existing io_uring workers. Preserve in-flight limits and qualify
  SSD round trips, cancellation, lost notifications and stalled Publish.
- [x] Reclaim SSD save/restore memory per page segment and NUMA node, so retained
  prefix pages do not hold unrelated evicted pages. Verify real reclamation
  and exact GPU bytes; measure total/speculative reservations independently
  of phase transitions.
- [x] Measure natural DRAM/SSD contention with a 27 GiB Qwen3-8B prefix set,
  9 GiB GPU KV and 4 GiB Manager DRAM in both engines. Retain final code and
  prefill-batch controls, allocation failures, write drops and cleanup. No
  overall throughput gain or hardware limit is established (`docs/ssd-performance.md`).
- [x] Add optional byte-bounded demand protection and bounded-history SSD write
  admission in Rust; exclude speculative interest, validate resident generations,
  preserve transfer ownership and measure policy counters (`docs/cache-policies.md`).
- [x] Compare retention-only, admission-only and combined policies in three
  matched DRAM/SSD windows per engine. Keep final aggregates and unchanged
  defaults: short-window selective admission loses throughput; protection
  reduces writes with little throughput change (`docs/cache-policies.md`).
- [x] Add one 768-request write-admission pair per engine with multiple SSD
  turnovers. Selective writes improve throughput by 15.8%/13.7% here while
  reducing writes; keep this distinct from the short-window regression and
  require repeated workload-specific evidence before changing defaults.
- [ ] Add per-request deadline/priority hints and long-running serving fault/soak
  runs; qualify multi-rank SGLang TP independently of TP=1 admission tests.
- [x] Add bounded queued-prefix DRAM warming for both pinned engine releases;
  keep foreground headroom, revalidate demand, and retire warmups without a lease
  or polling. Expose optional request-correlated transfer timeline logs.
- [x] Record matched Qwen3-8B warming on/off pressure controls and native output
  diagnostics. Keep automatic warming opt-in: initial controls increased SSD
  bytes per request without improving throughput (`docs/queued-warming.md`).
- [x] Attribute warmup page footprints to first successful H2D, last-owner
  unused release and live pending bytes; record completed byte-seconds. Keep
  enqueue peeks cold, reclaim unused warming before retained pages, and skip
  new hints while foreground query ownership is active.
- [x] Bound each pressure-reclaim batch by the allocation's requested bytes,
  then recheck real contiguous capacity; cover small-pool preservation and
  fragmented free space instead of unconditionally evicting up to 512 pages.
- [x] Repeat matched warming controls with page outcomes and byte-bounded
  reclamation: vLLM admits little warming; SGLang leaves 92.9% of prepared
  footprints unused. Keep warming opt-in; no throughput improvement is established.
- [x] Review pinned LMCache/SGLang/FlexKV/Dynamo implementations, distinguish
  request-owned prefetch from unlocked warming, and separate open RFC/PR ideas
  from release behavior (`docs/queued-warming.md`).
- [x] Qualify bounded preparation for near-admission requests using existing
  query/lease ownership; keep prepared residency budgeted through consumer
  handoff, cancellation or expiry. The automatic TP=1 dense path uses at most
  four arrival-order candidates; hybrid forecasts and priority prediction remain
  unqualified. Ordinary demand remains the control (`docs/request-preparation.md`).
- [x] Add best-effort/relative-timeout stopping and bounded read submission;
  drain submitted work. Best-effort returns completed dense prefixes; strict
  recovery rejects incomplete coverage and deadline fallback returns a miss.
  Cover shared reads, changed ranges, cancellation and expiry without polling.
- [x] Complete three matched preparation pairs per engine, stopping controls and
  DRAM-only recovery. Retain final variation, read bytes, output diagnostics and
  cleanup. Keep preparation off: SGLang throughput improves but P95 regresses.
- [ ] Calibrate expected use time and priority from engine HBM hits, prepared
  consumption and restorable-prefix/bundle coverage. Qualify bounded writer
  staging, source-expiration accounting and multi-rank behavior.
- [ ] Calibrate restore-versus-recompute and write admission using
  `docs/state-planning.md` (P4); speculative workflow hints remain optional.
- [ ] Profile the measured restore latency gap to both built-in CPU caches;
  measure transfer batching, completion observation, and inference overlap.
- [ ] Record vLLM/SGLang cold, warm, partial, and restart TTFT/TPOT,
  throughput, P50/P95 query/save/restore, and pinned-memory use against
  native-engine and no-cache baselines.
- [x] Make waiting local QueryBundle operations asynchronous; support
  `orbitkv.wait_for_full_prefix` without blocking other descriptor requests.
- [x] Move Publish D2H completion off the shared dispatcher while retaining
  its reply until framework-owned source pages may be reused.
- [x] Give Publish a separate on-demand local descriptor session so a blocked
  save does not serialize Query/Restore calls from the same worker.
- [ ] Profile the deferred Publish path under concurrent Query/Publish load.
- [x] Make Publish wait fail closed: keep vLLM source pages pinned until D2H
  completion or confirmed Cache Manager process death, even past the normal IPC timeout.
- [ ] Add an operational watchdog for a live Cache Manager that never finishes a
  Publish; correctness currently takes priority over save-worker availability.

## M2.5 — distributed cache reliability

Implementation order and failure contracts: `docs/distributed-cache.md`.

- [x] D0: sequence all owner residency transitions and keep bounded replay
  history; detect lost notifications and require resynchronization.
- [x] D0: implement paginated inventory snapshots with a complete delta cut,
  including concurrent eviction, duplicates and replay overflow.
- [x] D1: use etcd for member incarnations and configuration; recover Watches
  after disconnection or compaction without per-block etcd operations.
- [x] D1 discovery: bounded positive candidate indexing, coalesced batched
  lookup, Manager-side planning and source runtime/residency checks.
- [x] D1 cancellation: hold destination buffers and source-release guard in the
  blocking transfer until it finishes; caller cancellation cannot drop them.
- [x] D1: embed fixed catalog shards with per-shard replay and etcd membership/configuration.
- [x] D1 source ownership: retain overdue source pins, account entire allocations
  under a byte/session budget, retry completion releases and export native stage timing.
- [x] D1 same-host serving: both engines pass independent TP=1 replica sharing,
  catalog replay and source restart gates over TCP (`docs/shared-cache-qualification.md`).
- [x] D1 completion recovery: reserve bounded per-requester/per-peer release capacity
  before window setup; retain idempotent retries until acknowledged and drain
  uncertain native batches.
- [x] D1 discovery RPC reduction: batch shards by catalog host, share connections,
  bound host concurrency and include coalescing in the common lookup deadline.
- [x] D1 authorization reconciliation: source-issued windows and generation-fenced
  slots identify holds before authorization; reconcile lost grant replies and
  cancellation, reject delayed authorizations, and bound idle replay metadata.
- [ ] D1 revocation: reclaim orphaned source reservations only after transport
  termination is established; a timeout or membership expiry cannot free them.
- [ ] D1: qualify source incarnation checks, transfer completion/revocation,
  cancellation and sender/receiver budgets on two real hosts for both engines.
- [x] D1: replace the standalone directory with `orbitkv-catalog` and remove
  obsolete executables, Python launcher and fixed-directory APIs.
- [ ] D2: implement versioned shard placement, replicated evidence, handoff and
  bounded subscriptions; qualify partitions and coordinator/catalog failure.
- [ ] D3: support source-local SSD staging and measured source selection without
  recursive peer fetches or unbounded staging.
- [x] Measure cold discovery RPCs and source authorization, READ and completion
  stages independently in the shared-cache serving gate.
- [ ] Measure background synchronization and etcd traffic, index bytes and recovery
  lag under multi-host load and failure.

## M3 — routing and replica planning

- [ ] Normalize vLLM and SGLang KV events.
- [ ] Build a worker/tier replica catalog with sequence recovery.
- [ ] Delegate cross-host TP query fan-out to node-local Cache Managers.
- [ ] Integrate pinned `dynamo-kv-router` worker selection and production service
  lifecycle; verify hash/event mapping and request-load reservations (R1 in
  `docs/state-planning.md`).
- [ ] Add measured HBM/DRAM/SSD/RDMA restore cost.
- [ ] Add recompute and queue-delay estimates.
- [ ] Add eviction externality and replica-risk terms.
- [ ] Select a worker through the router, then revalidate and lease its
  transfer/restore plan at the Cache Manager.
- [ ] Evaluate load-only, overlap-only, and joint planning on the same trace.

- [x] Add the pinned Mooncake Transfer Engine native sys/build boundary.
- [x] Map OrbitKV remote-cache authorization to Mooncake Segment addresses.
- [ ] Qualify RDMA READ demand fetch and RDMA WRITE replication.
- [ ] Import topology-aware slicing, endpoint pooling, and alternate-rail retry.
- [ ] Keep rkeys and raw addresses out of the global replica directory.
- [x] Delete native v1 and vendored v2 RDMA implementations.

## M4 — generation-safe page references

- [ ] Introduce manager-authored external `PageHandle { pool, page, generation }`
  and validate engine-owned GPU page generations at transfer boundaries.
- [ ] Track the semantic frontier independently from execution completion.
- [ ] Unify CUDA event, RDMA completion, and SSD completion fences.
- [ ] Reject stale page generations at every adapter boundary.
- [ ] Integrate handles into SGLang Radix lifecycle events.
- [ ] Migrate the vLLM adapter without regressing its E2E path.
- [ ] Add cancellation, preemption, crash, and delayed-completion stress tests.

## M5 — semantic state compiler

The existing compiler validates declared prefix/window/checkpoint recovery.
The M2 increment adds deterministic page demand and adapter consumption of the
same rules; its gate is tracked above. This is a limited implementation toward
M5, with no measured latency claim. The general compiler work remains open:

- [ ] Define the `may_read(query, state)` IR.
- [ ] Compile full-attention retention.
- [ ] Compile sliding-window and sink-local retention.
- [ ] Derive recurrent checkpoint placement and retention from the general IR;
  declared exact-boundary checkpoint recovery is already implemented.
- [ ] Solve Minimum Persistent State Realization for hybrid bundles.
- [ ] Emit placement, checkpoint, prefetch, and reclamation plans.
- [ ] Measure Retention Amplification and semantic reclaim latency.

## Hygiene and release

- [x] Document independent per-node Managers, shared-instance capacity and automatic
  SSD selection; keep backend overrides in diagnosis/qualification instructions.
- [ ] Replace the engine-coupled Docker build with independently versioned Manager
  and engine images built from the validated wheel artifacts.
- [ ] Qualify concurrent engines sharing one Manager: matched runtime/device
  identities, bounded query ownership, engine/Manager restart and resource drain.
- [ ] Qualify container GPU access, shared UDS/iceoryx2/PyTorch IPC and pidfd
  visibility before publishing DaemonSet/Deployment manifests. Test native SSD
  mounts separately from container functional recovery.
- [ ] Keep all public capability claims tied to a reproducible test.
- [ ] Separate client and Cache Manager release artifacts when their contracts are
  stable; catalog shards remain embedded in the Manager.
- [ ] Keep heavy GPU/RDMA gates explicitly marked.
- [ ] Preserve license and upstream provenance requirements.
- [ ] Keep SGLang support claims aligned with the direct-linker E2E gate.
