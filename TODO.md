# OrbitKV implementation TODO

This is the repository-wide execution checklist. Completed items must have code
and a passing gate; design text alone does not close an item.

## M0 — framework-neutral foundation

- [x] Import and rename the PegaFlow 0.24.5 data plane.
- [x] Move Rust packages under `crates/`.
- [x] Remove the unused repository-root `src/main.rs`.
- [x] Add `orbitkv-contract` with state identity, format, page generation, and
  recovery-bundle types.
- [x] Move the canonical vLLM package to `orbitkv.vllm`.
- [x] Preserve `orbitkv.connector` as a compatibility alias.
- [x] Add `orbitkv.client` and `orbitkv.sglang` package boundaries.
- [x] Add `orbitkv-local` with a versioned 64-byte iceoryx2 request/response ABI.
- [x] Add a real two-process local-control test.
- [x] Introduce `RemoteMover` and adapt the native RDMA engine.
- [ ] Add Python representations/serialization for `orbitkv-contract`.
- [x] Add compatibility tests for `orbitkv.connector` and `orbitkv.vllm`.
- [x] Run the full M0 validation matrix and record results in the commit.

## M1 — SGLang HiCache backend

- [ ] Implement `OrbitKVHiCacheStorage`.
- [ ] Add a SGLang entry point or documented dynamic-backend config.
- [ ] Require `allocator=shm` for the zero-copy host path.
- [ ] Add UDS registration for memfd-backed host regions.
- [ ] Bind `orbitkv-local` QueryBundle/Restore/Publish handlers to the sidecar.
- [ ] Add Python bindings for the iceoryx2 local client.
- [ ] Map SGLang `PoolName` values to `StateComponent`.
- [ ] Map `ALL_PAGES` and `TRAILING_PAGES` into recovery contracts.
- [ ] Implement `batch_exists_v2`.
- [ ] Implement zero-copy `batch_get_v2` and `batch_set_v2`.
- [ ] Add fail-open behavior for non-hybrid requests.
- [ ] Add explicit fail-closed behavior where incomplete hybrid state cannot be
  recomputed safely.
- [ ] Add cold-miss, partial-prefix, warm-hit, cancellation, and restart tests.
- [ ] Run one real SGLang model E2E on H20.

## M2 — common bundle and local IPC

- [ ] Convert the vLLM cache-group layout to `StateBundle`.
- [ ] Move hybrid-boundary reconciliation out of `orbitkv.vllm`.
- [ ] Define framework-neutral region registration RPCs.
- [ ] Pass shared-memory file descriptors over UDS.
- [ ] Replace per-load `PyLoadState` files with a bounded shared completion ring.
- [ ] Keep control messages descriptor-only; prohibit KV payload bytes in gRPC,
  UDS, or iceoryx2 messages.
- [ ] Move hot local control off gRPC; retain gRPC only as compatibility fallback.
- [ ] Add generation validation to every local page reference.
- [ ] Benchmark the M2 path against the current CUDA IPC baseline.

## M3 — routing and replica planning

- [ ] Normalize vLLM and SGLang KV events.
- [ ] Build a worker/tier replica catalog with sequence recovery.
- [ ] Reproduce Dynamo's weighted-overlap selector.
- [ ] Add measured HBM/DRAM/SSD/RDMA restore cost.
- [ ] Add recompute and queue-delay estimates.
- [ ] Add eviction externality and replica-risk terms.
- [ ] Return a worker plus a transfer/restore plan.
- [ ] Evaluate load-only, overlap-only, and joint planning on the same trace.
- [ ] Add catalog epoch plus resident-inventory resynchronization after directory restart.

- [ ] Add an optional Mooncake Transfer Engine build/runtime backend.
- [ ] Map OrbitKV authorized regions to Mooncake Segment offsets.
- [ ] Qualify RDMA READ demand fetch and RDMA WRITE replication.
- [ ] Import topology-aware slicing, endpoint pooling, and alternate-rail retry.
- [ ] Keep rkeys and raw addresses out of the global replica directory.
- [ ] Compare Mooncake and native RDMA with identical transfer-plan tests.

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
- [ ] Keep heavy GPU/RDMA gates explicitly marked.
- [ ] Preserve license and upstream provenance requirements.
- [ ] Publish SGLang support only after the M1 E2E gate.
