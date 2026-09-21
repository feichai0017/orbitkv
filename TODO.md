# OrbitKV implementation TODO

This is the repository-wide execution checklist. Completed items must have code
and a passing gate; design text alone does not close an item.

## M0 — framework-neutral foundation

- [x] Import and rename the PegaFlow 0.24.5 data plane.
- [x] Move Rust packages under `crates/`.
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
- [x] Reject SGLang hybrid, draft, DSA, and auxiliary GPU state at startup
  until their complete recovery contracts are implemented.
- [ ] Add cold-miss, partial-prefix, warm-hit, cancellation, and restart tests.
- [x] Run one real SGLang model E2E, including restore after radix-cache flush.
- [x] Register a SGLang RadixCache plugin that transfers full-attention GPU KV
  through CUDA IPC and iceoryx2, with a real Cache Manager load after SGLang
  process restart and cold-inference output comparison.
- [ ] Add direct GPU recovery contracts for hybrid SWA/Mamba, DSA, draft-model,
  and auxiliary state.

## M2 — common bundle and local IPC

- [x] Fingerprint immutable model artifacts and bind computation/configuration in
  both adapters; reject dynamic LoRA until adapter-content identities are available.
- [x] Use the shared versioned `StateKey` across DRAM/SSD and remote directory
  records; include actual registered storage geometry and invalidate old keys.
- [ ] Carry absolute token spans and component evidence from both engines into
  shared recovery validation; SGLang PoolTransfer currently supplies only hashes.
- [ ] Support adapter identities and invalidate caches on live weight updates.
- [ ] Convert the vLLM cache-group layout to `StateBundle`.
- [ ] Define a recovery validator for matching token coverage, model/format,
  and complete hybrid component sets before using bundles for cache hits.
- [ ] Move hybrid-boundary reconciliation out of `orbitkv.vllm`.
- [ ] Define framework-neutral region registration RPCs.
- [x] Pass the descriptor-arena memfd and notification eventfd over UDS.
- [x] Add bounded restore operations that replace per-load `PyLoadState`
  for `ChannelClient`.
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
- [ ] Complete delivery-loss/restart fault qualification and deadline/priority
  demand hints; current gates cover revisions, cancellation, and session cleanup.
- [ ] Qualify delayed-read cancellation under concurrent serving and multi-rank
  SGLang TP; controlled admission tests do not replace those workload gates.
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
- [ ] Qualify queued warming under delayed reads, cancellation and sustained
  pressure; calibrate admission from page outcomes and engine consumption, add
  priority/deadline and per-device/staging budgets (remaining P3).
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
- [ ] D1: qualify source incarnation checks, transfer completion/revocation,
  cancellation and sender/receiver budgets on two real hosts for both engines.
- [x] D1: replace the standalone directory with `orbitkv-catalog` and remove
  obsolete executables, Python launcher and fixed-directory APIs.
- [ ] D2: implement versioned shard placement, replicated evidence, handoff and
  bounded subscriptions; qualify partitions and coordinator/catalog failure.
- [ ] D3: support source-local SSD staging and measured source selection without
  recursive peer fetches or unbounded staging.
- [ ] Measure discovery RPCs separately from background synchronization, source
  authorization and etcd activity; record index bytes and recovery lag.

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

- [ ] Define the `may_read(query, state)` IR.
- [ ] Compile full-attention retention.
- [ ] Compile sliding-window and sink-local retention.
- [ ] Compile recurrent checkpoint contracts.
- [ ] Solve Minimum Persistent State Realization for hybrid bundles.
- [ ] Emit placement, checkpoint, prefetch, and reclamation plans.
- [ ] Measure Retention Amplification and semantic reclaim latency.

## Hygiene and release

- [ ] Keep all public capability claims tied to a reproducible test.
- [ ] Separate client, Cache Manager, and directory release artifacts once the local
  and multi-node contracts are stable; retain one source workspace.
- [ ] Keep heavy GPU/RDMA gates explicitly marked.
- [ ] Preserve license and upstream provenance requirements.
- [ ] Keep SGLang support claims aligned with the direct-linker E2E gate.
