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

- [ ] Replay resident inventories with a catalog epoch after MetaServer restart.
- [ ] Batch and bound registration/lookups and cache candidates at each manager.
- [ ] Revalidate source residency and leases after owner churn and stale hints.
- [ ] Qualify Mooncake remote fetch, retry, and node-loss behavior on multiple hosts.
- [ ] Prototype embedded replicated catalog shards, compare against a dedicated
  fallback, and measure metadata request rate without per-block consensus.

## M3 — routing and replica planning

- [ ] Normalize vLLM and SGLang KV events.
- [ ] Build a worker/tier replica catalog with sequence recovery.
- [ ] Delegate cross-host TP query fan-out to node-local Cache Managers.
- [ ] Reproduce Dynamo's weighted-overlap selector.
- [ ] Add measured HBM/DRAM/SSD/RDMA restore cost.
- [ ] Add recompute and queue-delay estimates.
- [ ] Add eviction externality and replica-risk terms.
- [ ] Return a worker plus a transfer/restore plan.
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
