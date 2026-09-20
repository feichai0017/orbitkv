# OrbitKV implementation TODO

This is the repository-wide execution checklist. Completed items must have code
and a passing gate; design text alone does not close an item.

## M0 — framework-neutral foundation

- [x] Import and rename the PegaFlow 0.24.5 data plane.
- [x] Move Rust packages under `crates/`.
- [x] Remove the unused repository-root `src/main.rs`.
- [x] Add `orbitkv-contract` with state identity, format, page generation, and
  recovery-bundle types.
- [x] Name the bundle's current component-presence check honestly; it is not
  yet a restorable-state proof.
- [x] Move the canonical vLLM package to `orbitkv.vllm`.
- [x] Preserve `orbitkv.connector` as a compatibility alias.
- [x] Add `orbitkv.client` and `orbitkv.sglang` package boundaries.
- [x] Add `orbitkv-local` with a versioned 64-byte iceoryx2 request/response ABI.
- [x] Add a real two-process local-control test.
- [x] Integrate the iceoryx2 lifecycle endpoint into `orbitkv-server`.
- [x] Add Python `LocalControlClient` bindings with epoch fencing.
- [x] Replace the copied native RDMA stacks with a pinned stable Mooncake
  Transfer Engine sys crate and one clean transfer API.
- [ ] Add Python representations/serialization for `orbitkv-contract`.
- [x] Add compatibility tests for `orbitkv.connector` and `orbitkv.vllm`.
- [x] Run the full M0 validation matrix and record results in the commit.

## M1 — SGLang HiCache backend

- [x] Implement `OrbitKVHiCacheStorage` using bounded UDS host-page transfers.
- [x] Document the SGLang dynamic-backend config.
- [x] Require `allocator=shm` for SGLang's HiCache host pool.
- [x] Add UDS bootstrap for the memfd-backed descriptor arena.
- [ ] Add UDS registration for framework-owned shared host page regions.
- [x] Bind `orbitkv-local` QueryBundle to the shared core query path.
- [x] Bind `orbitkv-local` Release to the shared core lease path.
- [x] Bind `orbitkv-local` Publish to the shared core save path.
- [x] Bind `orbitkv-local` Restore to core oneshot completion and eventfd wakeup.
- [x] Add Python bindings for the iceoryx2 local client.
- [x] Map SGLang `PoolName` values to `StateComponent`.
- [ ] Map `ALL_PAGES` and `TRAILING_PAGES` into recovery contracts.
- [x] Return SGLang `PoolTransferResult.restorable_prefix_pages` for hybrid
  checkpoints; a largest-hit count alone cannot express legal trailing pages.
- [x] Implement `batch_exists_v2` with all-pages and trailing-pages policies.
- [ ] Implement zero-copy `batch_get_v2` and `batch_set_v2`.
- [ ] Add fail-open behavior for non-hybrid requests.
- [ ] Add explicit fail-closed behavior where incomplete hybrid state cannot be
  recomputed safely.
- [ ] Add cold-miss, partial-prefix, warm-hit, cancellation, and restart tests.
- [x] Run one real SGLang model E2E on H20, including restore after L1/L2 flush.
- [x] Register a SGLang RadixCache plugin that transfers full-attention GPU KV
  through CUDA IPC and iceoryx2, with a real Cache Manager load after SGLang
  process restart and cold-inference output comparison.
- [ ] Add direct GPU recovery contracts for hybrid SWA/Mamba, DSA, draft-model,
  and auxiliary state; retain the HiCache backend where it has a complete
  recovery contract until then.

## M2 — common bundle and local IPC

- [ ] Convert the vLLM cache-group layout to `StateBundle`.
- [ ] Define a recovery validator for matching token coverage, model/format,
  and complete hybrid component sets before using bundles for cache hits.
- [ ] Move hybrid-boundary reconciliation out of `orbitkv.vllm`.
- [ ] Define framework-neutral region registration RPCs.
- [x] Pass the descriptor-arena memfd and notification eventfd over UDS.
- [ ] Pass framework-owned shared-page file descriptors over UDS.
- [x] Add bounded local restore operations that replace per-load `PyLoadState`
  for `LocalQueryClient`.
- [x] Implement direct SGLang full-attention GPU restore through the local
  Cache Manager endpoint; the HiCache L3 compatibility path still uses UDS
  host-page payloads.
- [x] Switch vLLM Query/Publish/Restore/Release to the local data client.
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
- [x] Move vLLM hot local control off gRPC automatically when the local socket
  is available; retain explicit gRPC fallback.
- [x] Qualify the revised vLLM correctness E2E with `--orbitkv-local-data` on a
  GPU/vLLM host. It compares the same prompt/reuse plan against native prefix
  caching, checks the native prefix hit, and requires `long_warm` to load KV
  bytes after process restart;
  the earlier cold-vs-warm comparison reproduced native vLLM divergence.
- [x] Requalify the vLLM E2E against release 0.29.0, including a hybrid model
  that exercises scheduler boundary-state hand-offs.
- [ ] Add generation validation to every local page reference.
- [ ] Benchmark the M2 path against the current CUDA IPC baseline.
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

## M3 — routing and replica planning

- [ ] Normalize vLLM and SGLang KV events.
- [ ] Build a worker/tier replica catalog with sequence recovery.
- [ ] Delegate cross-host TP query fan-out to node-local agents before removing
  compatibility Query/Save/Load gRPC methods from the network service.
- [ ] Reproduce Dynamo's weighted-overlap selector.
- [ ] Add measured HBM/DRAM/SSD/RDMA restore cost.
- [ ] Add recompute and queue-delay estimates.
- [ ] Add eviction externality and replica-risk terms.
- [ ] Return a worker plus a transfer/restore plan.
- [ ] Evaluate load-only, overlap-only, and joint planning on the same trace.
- [ ] Add catalog epoch plus resident-inventory resynchronization after directory restart.

- [x] Add the pinned Mooncake Transfer Engine native sys/build boundary.
- [x] Map OrbitKV remote-cache authorization to Mooncake Segment addresses.
- [ ] Qualify RDMA READ demand fetch and RDMA WRITE replication.
- [ ] Import topology-aware slicing, endpoint pooling, and alternate-rail retry.
- [ ] Keep rkeys and raw addresses out of the global replica directory.
- [x] Delete native v1 and vendored v2 RDMA implementations.

## M4 — page authority and safety

- [ ] Introduce manager-authored `PageHandle { pool, page, generation }`.
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
- [ ] Publish SGLang support only after the M1 E2E gate.
