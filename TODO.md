# OrbitKV implementation TODO

This is the repository-wide execution checklist. Completed items must have code
and a passing gate; design text alone does not close an item.
The [implementation plan and agent handoff](docs/implementation-plan.md) maps
upstream references, deployment profiles and the P4.1 implementation contract to these gates.

Use the [staged completion and acceptance plan](docs/completion-plan.md) as the
current delivery order. It separates missing implementation, unqualified paths
and obsolete checklist entries; each stage has an independent review gate.

After S1 acceptance, S2 qualifies metadata reliability and capacity; S3 then
extends transfer termination and page generations. S4–S8 consume those contracts
according to the completion plan. The [roadmap](docs/roadmap.md#current-delivery-priorities)
groups capability areas; it is not a competing execution queue. The
metadata implementation has replaced single-copy Catalog shards with a complete
local global index on every Manager, synchronized through etcd block metadata.
Same-host and physical H20/A100 TCP sharing/restart/SSD gates pass on both engines;
scale, host-failure-domain and transfer-revocation gates remain open.
Milestone numbers group work areas rather than imposing a strict
serial schedule. Every open item below names its S2–S8 owner. **Implementation
open** means the consumed extension is missing; **implementation partial** means
existing owners need extending; **qualification open** means code exists but the
listed hardware, failure, model or performance gate remains open. **Release gate**
is a recurring S8 obligation. **Deferred research decision** requires an explicit
scope decision before implementation and does not block the current supported
path. Retired entries are explanatory text, never a claim that an unrun gate passed.

S1 separates historical evidence through [the artifact policy](docs/benchmark-evidence.md).
Its implementation and acceptance evidence are supplied with the delivery commit;
S2 starts only after independent S1 acceptance.

## GPU storage

- [x] Implement Rust cuFile demand reads and complete-group GPU writes with SSD extent
  leases, bounded GPU staging, split/page-first layouts, oversized checkpoints,
  cancellation, failed writes and short reads; keep both engines on the existing cache API.
- [ ] [S4][s4] — **qualification open**: Qualify native GDS on a supported NVMe mount with CPU fallback disabled;
  distinguish first writes from overwrites and compare io_uring using matched
  working sets, TTFT, throughput, CPU use and bytes.
- [x] Provide a bare-metal acceptance script with fallback rejection and matched
  io_uring/auto/cuFile workloads for both engines (`python -m benches.gds`).
- [x] Default to automatic native cuFile initialization with io_uring fallback;
  stop new GPU I/O after an operation failure without revoking in-flight ownership.
- [x] Reserve physical GPU-storage file space before admission; report allocation
  failures and release partial startup reservations. Keep file ownership in `storage/ssd/files.rs`.
- [x] Implement bounded Rust asynchronous cuFile submissions with reusable staging
  slots, stream/event completion, per-operation byte/error checks and cancellation drain.
- [x] Coalesce reads by file across leased sources without broadening required ranges;
  verify call counts, separate files, unrequested gaps and cancellation ownership.
- [x] Bound queued GPU-storage writes and staging so demand reads make progress
  between write batches: two 4 MiB slots, one in-flight write, eight admitted write
  jobs and bounded read bursts; saturation uses host publication/io_uring.
- [x] Verify event-tracked host copies overlap SSD submission while Publish and
  unregister retain ownership until both DMA paths complete.
- [x] Implement GPU ANS, FP8 and 3/4-bit TurboQuant, encoded DRAM/SSD/peer payloads,
  explicit head/K/V registration, raw fallback and CPU SIMD.
- [x] Add runtime AVX-512F/AVX2/scalar FP8 CPU encoding and decoding, with
  exhaustive BF16/FP16 value checks and a reproducible single-core benchmark.
- [x] Verify optional FP8 storage, bounded scratch, mixed raw/encoded
  prefixes, cancellation, corruption rejection/repair and raw fallback.
- [ ] [S4][s4] — **qualification open**: Qualify lossy storage on representative model-quality workloads; Qwen3-8B
  encoded slots shrink, but vLLM greedy-output checks still fail with TurboQuant.
- [x] Qualify engine-native FP8 KV on Qwen3-8B in both engines; isolate external
  SGLang scale artifacts and compare against same-dtype cold controls.
- [ ] [S4][s4] — **qualification open**: Extend quality and latency qualification of GPU codecs across hybrid models; retain
  exact recurrent state until model-quality and recovery gates pass.
- [x] Batch codec segments and reuse GPU scratch to reduce per-segment launches,
  allocations and synchronization; compare complete-request latency under inference load.
- [x] Add compressed cuFile reads and complete-group writes with GPU CRC validation,
  bounded decode workspace, corruption isolation and cancellation ownership.
  Qualify both engines in cuFile compatibility mode; native GDS remains a separate gate.
- [ ] [S4][s4] — **qualification open**: Measure selective DRAM admission and shorter source-HBM holds for GPU writeback;
  retained staging and unpublished SSD reservations must survive disk completion.
- [ ] [S4][s4] — **implementation partial**: Extend the existing staged GPU path to direct registered
  engine-page I/O and multi-writer GPU assembly, then qualify those paths
  and workload-based path selection. Include registration cost,
  engine-page hold times, I/O fragmentation and additional HBM in the decision.

See [GPU storage recovery](docs/gds.md) for deployment and reproduction, and the
[LMCache v0.5.5 review](docs/gds.md#review-against-lmcache) for optimization evidence and gates.
The [upstream design mapping](docs/architecture.md#upstream-designs-and-orbitkv-owners)
records LMCache, FlexKV and Mooncake mechanisms, owners and implementation status.

## Measured transfer planning

Structural work: [unified replicas, routes and owned plans](docs/state-planning.md#unified-replicas-routes-and-execution-ownership).

- [x] Replace dedicated local DRAM/SSD/peer-DRAM fields with bounded replica
  records separating medium from acquisition evidence; preserve current
  namespace compatibility, exact source versions and unknown peer metadata.
  Move SSD route eligibility/acquisition and peer source segmentation into
  `planning/`, preserving existing execution defaults and completion owners.
- [ ] [S6][s6] — **implementation partial**: Complete consumed endpoint descriptors for owner/resource identity,
  representation and bytes; bind actual TE transport capability to GPU routes
  without treating GPUDirect RDMA as a tier or assuming peer HBM authorization.
- [x] Make Mooncake memory registration an RAII token that retains the
  `TransferEngine` and unregisters before its backing owner is released. Migrate
  the pinned pool to these tokens; future HBM grants must pair the token with an
  engine page owner and `cuda:N` location.
- [x] Carry versioned replica medium, representation family and known stored
  bytes through owner inventory, Catalog storage and bounded discovery rows.
  Populate current DRAM evidence, feed peer authorization cost shape, and keep
  advertised SSD/HBM evidence ineligible until their executors and grants exist.
- [x] Move the residency journal owner from `storage/dram` to `storage/` while
  preserving atomic DRAM transitions, inventory replay and source version
  checks. This establishes one sequence owner for later DRAM/SSD transitions.
- [x] Publish SSD evidence only after terminal commit, prefer live DRAM for the
  same owner/key, fall back to surviving SSD after DRAM eviction, and remove
  evidence on ring overwrite or encoded corruption.
- [x] Add source-local peer SSD materialization through io_uring. Revalidate and
  pin the exact advertised extent generation, reserve rounded pinned-allocation
  bytes before staging, atomically replace the estimate with actual allocation
  footprint, and retain reservation/extent ownership through cancellation and
  detached I/O drain. Requester planning now uses this executor only after peer
  DRAM and an eligible local SSD route; two-host qualification remains open.
- [x] Retain unresolved candidates within each admitted query batch; distinguish
  host preparation from engine restoration, borrow SSD/peer plans over the same
  records, and hand exact source versions to existing query/completion owners.
  Move shared-read coordination into `query/` and remove by-key SSD rescans and
  forwarding/argument wrappers, preserving default source priority.
- [x] Move host-ready peer/SSD route choice into `planning/`; keep source
  acquisition in its execution owner and avoid allocating peer authorization
  records for every rejected source candidate. Preserve peer-before-SSD defaults,
  full-prefix coverage requirements and immutable source versions.
- [ ] [S6][s6] — **implementation partial**: Extend batch plans into complete-route comparisons and joint demand
  coverage across groups/ranks, with actual destination and staging admission.
- [x] Add explicit completion intent and target resource to every comparable
  cost key; recompute it when paths/resources change and reject evidence for a
  different GPU before future DecodeReady route comparison.
- [x] Build a consumed `RestorePlan` at the Rust engine boundary from only
  sources with real destinations. Bind its EngineRestore target to the CUDA
  device, deduplicate source geometry, preserve one SSD route, and make worker
  lane/cost construction validate and consume the plan.
- [x] Share the existing bounded GPU SSD-write admission across every instance
  registered on the same CUDA device. Acquire without blocking, retain the
  permit inside `SaveTask` through terminal completion, and preserve host
  publication fallback when the shared device budget is full.
- [x] Distinguish automatic and explicit SSD read routes, and assign one
  persistent cuFile/codec staging owner per CUDA device. Automatic restores
  fall back to the same extent's io_uring route before submission; explicit
  cuFile remains fail-closed. Retain ownership until every instance lane drains.
- [ ] [S6][s6] — **implementation partial**: Separate operation observations from complete-route estimates and attach
  live resource evidence, without double-counting queue time or composite stages.
- [x] Make operation/route sample boundaries explicit and distinguish GPU,
  SSD store/file and peer-incarnation cost resources. Guard existing same-source
  shadow comparisons by completion family, resource and shape buckets; require
  a gain beyond both empirical errors and a declared shadow margin. This does
  not complete live resource admission or enable execution selection.
- [x] Establish the [ownership layout](docs/state-planning.md#code-ownership-and-migration):
  DRAM/SSD residency under `storage/`, peer workflows under `peer/`, inbound RPC
  adaptation in Server, shared reads in `query/`, publication in its worker,
  and separate cost observation/estimate/shadow modules. Remove the former
  `backing/`, `internode/`, weak insert dependency wrapper and storage forwarding
  methods; hold registered pinned pools through Mooncake unregister.

Follow [P4](docs/state-planning.md#p4-calibrate-costs-and-choose-useful-writes)
and its [deployment contracts](docs/state-planning.md#policies-by-deployment-mode).
P4.1 instrumentation and raw-copy/SSD-route shadow are implemented as an opt-in.
The first SSD source/path separation is implemented; dynamic selection and its
qualification remain open. See the
[observation contract](docs/state-planning.md#p41-observation-contract) and
[implementation contract](docs/implementation-plan.md#p41-implementation-contract).

- [x] Add bounded Rust queue/service/completion estimates keyed by path,
  representation, size/fragmentation and resource/peer incarnation. Share event
  inputs with metrics; preserve uncertainty and censored timeout observations.
- [x] Run shadow DMA/kernel candidates on actual raw-copy metadata; report
  prediction error without claiming unexecuted alternatives as measured savings.
- [x] Distinguish descriptor count from DMA-coalesced range count using the
  executor's merge logic, and support fixed direct/kernel comparisons in both
  engine harnesses while preserving registration defaults.
- [x] Compile direct DMA into contiguous ranges and explicit constant-pitch rows;
  distinguish physical host registration from retained allocation generations.
  Pass full Rust, H20 bidirectional/gap/bounds, process-fault and both-engine
  Qwen3-8B correctness gates. Count actual submissions with the executor compiler.
- [x] Qualify integrated strided DMA with three order-reversed, matched serving
  repetitions against native HBM, native CPU and LMCache on each engine.
  Both deterministic C4 gates pass; a separate before/after A/B is still needed
  to attribute gains specifically to 2D DMA.
- [x] Partition large raw Restore plans under one operation/source/destination
  fence; bound per-part, operation and session metadata. Pass actual H20 bytes,
  second-part enqueue failure and Manager exit between parts.
- [x] Qualify raw layer/group restore-compute overlap on H20, including eager,
  external-event graph replay, vLLM full graphs and recurrent migration, and both
  engines' dense/hybrid DRAM/SSD serving gates. Preserve the final drain fence;
  Manager codec/SSD execution and multipart early publication remain separate
  execution work. See [the layer gate](docs/engine-local-restore.md#layer-readiness-qualification-2026-09-29).
- [x] Measure the complete layer-readiness/scheduling increment against its
  previous implementation, native HBM, native CPU and LMCache in three reversed
  orders per engine. All 30 cohorts pass exact outputs; vLLM gains repeat, while
  SGLang throughput remains unchanged and its native CPU gap stays open. See
  [the full comparison](docs/communication-performance.md#repeated-serving-comparison-after-layer-readiness).
- [x] Run the [three-pair DRAM/raw comparison](docs/implementation-plan.md#dmakernel-comparison-final-evidence)
  on both engines. Fixed kernel exceeds throughput and TTFT p50 regression
  budgets in both cells; keep the default direct backend for these layouts.
- [x] Separate immutable SSD extent leases from route eligibility. Execute the
  same generation through io_uring host materialization or cuFile GPU staging;
  use a separate host-restore lane and preserve terminal ownership.
- [x] Add explicit demand-route controls and shadow full-restore estimates for
  both eligible SSD routes. Keep default selection and DRAM preparation unchanged.
- [x] Separate metadata residency candidates from acquisition: preserve local
  DRAM/SSD and cached peer DRAM/SSD evidence, and revalidate the exact SSD generation
  before pinning. Discovery does not read or reserve payloads.
- [x] Complete [current route correctness and lifecycle validation](docs/implementation-plan.md#ssd-sourcepath-separation-final-evidence)
  across both engines, same-generation raw/ANS recovery, cancellation and remote
  misses. The previous 36-run observation matrix does not qualify these executor changes' overhead.
- [ ] [S4][s4] — **qualification open**: Close the SGLang ANS SSD TTFT p50 overhead gate before default enablement:
  the matched three-pair result is +5.915%, above the predeclared 3% budget.
- [ ] [S6][s6] — **implementation partial**: Extend shadow decisions to legal boundaries and additional qualified paths;
  retain unknown alternatives when existing metadata or measurements are absent.
- [ ] [S6][s6] — **implementation partial**: Qualify per-batch DMA/kernel selection, then io_uring/native cuFile choice,
  with both SSD routes independently eligible over one index/extent lifetime;
  separate capability and failure handling from measured choice. Include shared
  device budgets, switching margins, fixed-format data and inference contention.
- [ ] [S6][s6] — **implementation partial**: Compare legal `required_ranges` boundaries and engine-owned recomputation;
  calibrate preparation deadlines and retention/write admission. Bound DRAM,
  GPU workspace, SSD/TE work and per-instance shares through terminal completion.
- [ ] [S6][s6] — **implementation partial**: Extend qualified peer source selection with discovery, authorization,
  TE and decode/H2D costs. Keep same-host TCP and physical two-host/RDMA evidence
  separate; DP qualification does not wait for local policy gains.
- [x] Add guarded execution selection between equal-coverage owners of the same
  peer medium. Train only complete HostReady observations scoped by peer
  incarnation; require every estimate to be fresh and compatible plus a gain
  beyond empirical error and 5%. Keep it behind both cost opt-ins and replan on
  stale or temporarily resource-exhausted source authorization.
- [x] Normalize io_uring host materialization as `local_ssd_host_ready`, using
  enqueue-to-reconstruction time plus stored-byte/block shape. Compare it in
  shadow with equal-coverage single-owner peer DRAM/SSD routes; keep fixed
  cross-medium execution until external H20 evidence qualifies switching.
- [x] Distinguish complete peer fetch, exhausted pre-payload authorization and
  submitted-payload failure. Replan retained local/peer evidence only for the
  authorization case; never retry another medium after a Mooncake payload error.
- [x] Add a separate `ORBITKV_CROSS_MEDIUM_SELECTION=1` experiment gate. Require
  both existing cost opt-ins, equal coverage, complete single-owner resource
  identity and fresh compatible estimates; preserve fixed defaults otherwise.
- [ ] [S6][s6] — **qualification open**: Qualify cross-medium execution on the external H20 TCP/RDMA matrix before
  recommending or enabling it in ordinary deployments.
- [ ] [S6][s6] — **implementation partial**: Implement the [Manager-owned cluster decision loop](docs/state-planning.md#cache-manager-decisions-below-the-engine):
  bounded residence/resource evidence with freshness, joint local/peer route
  ranking, source credit admission and bounded replanning. Qualify without a
  request router and with concurrent destinations contending for one peer.
- [ ] [S6][s6] — **implementation partial**: Integrate P/D completion/admission evidence separately from cache misses;
  qualify rank-common TP and stage-dependent PP plans as later topology gates.
- [x] Add the first bounded authenticated P/D completion observation: vLLM's
  decode-side TENT waiter reports its registered target device, hashed prefill
  endpoint identity, nonzero transfer generation, raw logical/wire bytes,
  fragments and terminal outcome through `orbitkv-channel`. Train only
  admitted completions; keep standalone P/D unchanged and do not enable route
  selection.
- [x] Add the matching direct-to-decode completion boundary from Restore
  submission through terminal GPU completion. Validate exact registered target
  ranges into a device-bound `RestoreTargetShape` retained by the consumed plan;
  keep this observation-only.
- [x] Add bounded per-device direct-restore admission, vLLM handoff queue
  depth/parallelism, admission-time TENT NIC pressure, and an authoritative
  SGLang decode callback after metadata and HiCache restore commit. Keep this
  resource evidence out of metric labels and expire it after two seconds.
- [ ] [S6][s6] — **implementation partial**: Make one consumed planner own both the direct source lease and P/D
  handoff authorization, then enable the guarded decode-route selector.

## M0 — framework-neutral foundation

- [x] Establish the OrbitKV data plane and workspace.
- [x] Move Rust packages under `crates/`.
- [x] Keep NUMA topology/affinity in Core and HLL reuse statistics in Server;
  limit `orbitkv-common` to shared process logging and peer-connection defaults.
- [x] Remove the unused repository-root `src/main.rs`.
- [x] Add `orbitkv-state` with state identity, format and recovery-bundle types.
  Remove unused page-generation types until an actual execution contract owns them.
- [x] Name the bundle's current component-presence check honestly; it is not
  yet a restorable-state proof.
- [x] Move the canonical vLLM package to `orbitkv.vllm`.
- [x] Group the vLLM cache connector and P/D adapter under `orbitkv.vllm`.
- [x] Remove the vLLM role-selecting `PdConnector` compatibility facade,
  test-only worker attribute proxies, the runtime-packaged no-op test connector
  and pre-0.29 preemption/metrics branches.
- [x] Consume vLLM 0.29 BHNC raw views with storage/stride/spec validation;
  remove the old split-K/V and three-dimensional registration adapters.
  Prepare Prefill sends before forward and publish receive completion only
  through the Decode owner; remove duplicate merging and completion queues.
- [x] Validate Qwen3-8B P/D greedy output on two A100 replicas and 1044 actual
  GPU KV ranges over H20→A100 TCP. Keep the heterogeneous strict-output gate
  failed (1/3 identical); byte equality is not a model-quality waiver.
- [x] Move P/D notification polling, counting and close/reopen generation
  fencing into the Rust TENT owner so Python waiters release the GIL.
- [x] Add `orbitkv.client` and `orbitkv.sglang` package boundaries.
- [x] Add `orbitkv-channel` with a versioned 64-byte iceoryx2 request/response ABI.
- [x] Add a real two-process channel test.
- [x] Integrate the iceoryx2 lifecycle endpoint into `orbitkv-server`.
- [x] Add Python `ChannelProbeClient` bindings with epoch fencing.
- [x] Replace the copied native RDMA stacks with the pinned Mooncake TENT C ABI
  and one clean transfer API; do not build or load the legacy TE runtime.
- Retired: unused Python state representations. `python/src/recovery.rs` already consumes shared Rust recovery contracts; add only a binding required by an adapter.
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
- [x] Bind SGLang 0.5.20's native P/D room and page-grant lifecycle to the
  shared Rust TENT registration/completion owner without copying its state
  machine into Python.
- [x] Define the first SGLang P/D plus external-cache composition: matching P/D
  workers share one namespace, P restores before handoff, and D publishes a
  longer completed prefix for a later P request; add a restart E2E gate.
- [x] Qualify same-A100 two-replica SGLang P/D plus external-cache restart/reuse
  with strict output controls; keep Decode external hits out of the HiCache-only
  restore path. H20→A100 reuse passes but its 64-token equality gate fails.
- [ ] [S5][s5] — **qualification open**: Qualify strict SGLang P/D output across distinct GPUs, then two hosts with
  RDMA counters; cover abort, worker restart and partial failure.
- [ ] [S3][s3] — **implementation open**: Expose TENT peer-liveness probing through its stable C ABI before enabling
  SGLang's optional failed-session recovery probe.
- [x] Reject SGLang representations without a complete recovery contract.
- Retired: the undifferentiated SGLang test request. `python/tests/e2e/test_sglang_direct_e2e.py`, integration admission/recovery tests and the P/D restart gate cover existing paths; S5 owns the remaining P/D fault and deployment cells.
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
- [ ] [S5][s5] — **qualification open**: Qualify native model serving with Full + SWA + temporal recurrent state
  in both engines; current combined temporal coverage is exact GPU recovery.
- [x] Qualify Qwen3.8-27B-FP8 on both engines with DRAM and forced SSD recovery;
  qualify GLM-4.7-Flash, DeepSeek-V2-Lite and Kimi Linear FP8 with SSD-enabled
  reuse, native output controls and engine restart. Keep exact artifacts,
  settings and larger-model blockers in [model qualification](docs/models.md).
- [ ] [S7][s7] — **deferred research decision**: Add recovery contracts for DSA, draft-model and further auxiliary state.

## M2 — common bundle and local IPC

- [x] Fingerprint immutable model artifacts and bind computation/configuration in
  both adapters; reject dynamic LoRA until adapter-content identities are available.
- [x] Use the shared versioned `StateKey` across DRAM/SSD and remote directory
  records; include actual registered storage geometry and invalidate old keys.
- [x] Carry SGLang's engine-held prefix origin and leased group positions into
  shared recovery validation; intersect legal boundary sets across ranks.
- [x] Carry vLLM hybrid spans and leased component evidence into shared recovery
  validation; intersect absolute legal boundaries across shards.
- [ ] [S5][s5] — **implementation open**: Support adapter identities and invalidate caches on live weight updates.
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
- [x] Send complete selected-boundary `RecoveryDemand` to the Manager with each
  group read. Validate registered groups before admission, include all ranges
  in query revisions and reject incomplete selected-group leases.
- [ ] [S6][s6] — **implementation partial**: Add joint multi-group physical planning and admission; retain engine-owned
  rank agreement, HBM allocation and legal recovery boundaries.
- [x] Handle asynchronous vLLM checkpoint queries from SSD, retain completed
  groups during preparation, and retire pending groups on cancel/drift/expiry.
  Verify native validation and exact DRAM/SSD GPU restoration in
  `python/tests/integration/test_vllm_recovery.py`.
- Retired: a second generic registration RPC. The UDS lifecycle endpoint already owns registration; S3 replaces `wrapper_bytes` in that consumed path.
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
- [ ] [S3][s3] — **implementation open**: Replace framework CUDA IPC wrapper pickle in the Cache Manager with an explicit
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
- [ ] [S3][s3] — **implementation open**: Add generation validation to every local page reference.
- [x] Keep restore destinations held on lost acknowledgements, poll failures,
  and deadlines; fail the engine instead of claiming DMA was cancelled.
- [x] Drain partially submitted H2D/D2H work before returning a backend error;
  terminate the manager if CUDA cannot establish completion.
- [x] Split Publish metadata to the negotiated descriptor capacity while
  preserving per-page layer completeness and retaining sources through all chunks.
- Retired: the pre-cutover CUDA IPC baseline task. The implemented process channel and engine-local Restore have matched controls in `docs/communication-performance.md`; new increments use S4 gates.
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
- [x] Qualify generation-fenced engine-local completion timing, isolated
  caller-to-drain cost estimates and native result-consumption traces. Twelve
  H20 process/GPU fault cases pass, including readiness during paused Manager
  retirement and exactly-once timing consumption.
- [x] Measure native timing/tracing overhead in three order-reversed repetitions
  per mode and decompose real vLLM Restore. Keep the slower kernel control and
  opt-in tracing; see `docs/communication-performance.md`.
- [x] Add explicit deterministic engine controls to the benchmark launcher and
  reject output comparisons between different computation modes.
- [x] Qualify deterministic C4 native/CPU/OrbitKV/LMCache output parity on both
  engines with the final merged production build: 512 completed requests,
  384 cross-backend comparisons without differences. Keep this single-cohort
  result separate from repeated performance acceptance.
- Retired: the stale baseline refresh. Three reversed-order layer-readiness comparisons are recorded in `docs/communication-performance.md#repeated-serving-comparison-after-layer-readiness`; S4 requires new matched controls for the next optimization.
- [ ] [S4][s4] — **qualification open**: Profile remaining adapter hashing, per-page metadata and PyO3 conversion
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
- [ ] [S4][s4] — **qualification open**: Profile vLLM duplicate H2D restores for shared prefixes; any reuse must
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
- [ ] [S4][s4] — **implementation partial**: Add per-request deadline/priority hints and long-running serving fault/soak
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
- [ ] [S6][s6] — **implementation partial**: Calibrate expected use time and priority from engine HBM hits, prepared
  consumption and restorable-prefix/bundle coverage. Qualify bounded writer
  staging, source-expiration accounting and multi-rank behavior.
- [ ] [S4][s4] — **qualification open**: Profile the measured restore latency gap to both built-in CPU caches;
  measure transfer batching, completion observation, and inference overlap.
- [ ] [S4][s4] — **qualification open**: Record vLLM/SGLang cold, warm, partial, and restart TTFT/TPOT,
  throughput, P50/P95 query/save/restore, and pinned-memory use against
  native-engine and no-cache baselines.
- [x] Make waiting local QueryBundle operations asynchronous; support
  `orbitkv.wait_for_full_prefix` without blocking other descriptor requests.
- [x] Move Publish D2H completion off the shared dispatcher while retaining
  its reply until framework-owned source pages may be reused.
- [x] Give Publish a separate on-demand local descriptor session so a blocked
  save does not serialize Query/Restore calls from the same worker.
- [ ] [S4][s4] — **qualification open**: Profile the deferred Publish path under concurrent Query/Publish load.
- [x] Make Publish wait fail closed: keep vLLM source pages pinned until D2H
  completion or confirmed Cache Manager process death, even past the normal IPC timeout.
- [ ] [S3][s3] — **implementation open**: Add an operational watchdog for a live Cache Manager that never finishes a
  Publish; correctness currently takes priority over save-worker availability.

## M2.5 — distributed cache reliability

Implementation order and failure contracts: `docs/distributed-cache.md`.

- [x] Audit current Mooncake main and existing notification/lifecycle issues;
  reproduce and submit the two remaining C-string termination and terminal
  teardown defects as upstream issues/PRs. Record evidence and related fixes
  in [peer control](docs/peer-control.md#upstream-audit-2026-09-29).
- [x] Implement the [local global-index design](docs/distributed-cache.md#local-global-index-and-etcd-metadata):
  background etcd block publication with lease/incarnation fences and ordered
  retry reconciliation; complete fixed-revision snapshots followed by Watch.
- [x] Replace Catalog lookup and inventory RPCs with local discovery and etcd
  synchronization in one cutover. Remove fixed placement, `--catalog-nodes`,
  the TTL hint cache and remote lookup coalescing; retain source grant/completion
  RPCs. The isolated native binary experiment is outside the delivery plan.
- [x] D0: sequence all owner residency transitions and keep bounded replay
  history; detect lost notifications and require resynchronization.
- [x] D0: implement paginated inventory snapshots with a complete delta cut,
  including concurrent eviction, duplicates and replay overflow. D2 uses one
  owner stream with independent DRAM/SSD records.
- [x] D1: use etcd for member incarnations and configuration; recover Watches
  after disconnection or compaction. D2 extends background etcd publication to
  block locations; request-time lookups remain local.
- [x] D2 discovery: replace D1 positive hints and coalesced Catalog lookup with
  local global-index reads; retain planning and exact source checks.
- [x] D1 cancellation: hold destination buffers and source-release guard in the
  blocking transfer until it finishes; caller cancellation cannot drop them.
- [x] D2: replace the D1 fixed Catalog shards with complete local indexes and
  replicated etcd location metadata; remove placement, serving and lookup RPCs.
- [x] D1 source ownership: retain overdue source pins, account entire allocations
  under a byte/session budget, retry completion releases and export native stage timing.
- [x] D1 receiver placement: derive per-slot NUMA allocation from the receiving
  GPU registration, keep it outside storage identity, and include it in shared-read
  coalescing. H20/A100 byte-exact forward and re-serving gates pass.
- [x] D1 two-host TCP serving: both engines pass the Qwen3-8B natural-text sharing,
  catalog restart and source-loss gates, with 288 MiB READ/H2D per engine.
  Random-token cross-GPU equality remains unqualified; see
  `docs/shared-cache-qualification.md` for the accepted and rejected scope.
- [x] D1 same-host serving: both engines pass independent TP=1 replica sharing,
  catalog replay and source restart gates over TCP (`docs/shared-cache-qualification.md`).
- [x] D1 completion recovery: reserve bounded per-requester/per-peer release capacity
  before window setup; retain idempotent retries until acknowledged and drain
  uncertain native batches.
- [x] Historical D1 discovery RPC reduction (removed by D2): batch shards by catalog host, share connections,
  bound host concurrency and include coalescing in the common lookup deadline.
- [x] Historical D1: coalesce matching directory batches without serializing unrelated queries.
  Bound pending metadata and per-owner/global RPC concurrency; cancellation and
  membership changes cannot leave stale evidence or detached lookup owners.
- [x] D1 authorization reconciliation: source-issued windows and generation-fenced
  slots identify holds before authorization; reconcile lost grant replies and
  cancellation, reject delayed authorizations, and bound idle replay metadata.
- [ ] [S3][s3] — **implementation open**: D1 revocation: reclaim orphaned source reservations only after transport
  termination is established; a timeout or membership expiry cannot free them.
- [ ] [S5][s5] — **qualification open**: D1: qualify source incarnation checks, transfer completion/revocation,
  cancellation and sender/receiver budgets on two real hosts for both engines.
- [x] D1: replace the standalone directory with `orbitkv-catalog` and remove
  obsolete executables, Python launcher and fixed-directory APIs.
- [x] D2: qualify leader loss and quorum-loss fencing with three etcd processes;
  cover snapshot/Watch compaction, lease deletion/incarnation restart, uncertain
  publication, bounded-index failure and snapshots larger than 4 MiB.
- [x] D2 serving: repeat same-host and physical H20/A100 TCP sharing, index restart
  and source-loss recovery on vLLM/SGLang; qualify forced source SSD recovery on
  both engines. See `docs/shared-cache-qualification.md` for the exact scope.
- [ ] [S2][s2] — **qualification open**: D2 scale: measure background churn, index memory and recovery lag; qualify
  etcd quota exhaustion and separate host-failure domains. Foreground discovery
  now reads the local index and has no directory RPC implementation.
- [ ] [S6][s6] — **qualification open**: D3: qualify requester peer-SSD routes and add measured source selection
  without recursive peer fetches or unbounded staging. Fixed-priority peer-SSD
  planning, source-local io_uring staging and two-phase byte/session admission
  are implemented. Ordinary two-host H20→A100 TCP SSD recovery passes on both
  engines; mixed-load selection, cancellation and RDMA qualification remain open.
- [x] Extend `benches.shared_cache` with `--source-medium ssd`: require committed
  source SSD bytes, evict only source DRAM, resynchronize inventory, prove source
  SSD reads plus target Mooncake/GPU restore, and drain both sides.
- [x] D3 prerequisite: distinguish owner/resource and HBM/DRAM/SSD residence in
  candidate/inventory records; preserve surviving SSD evidence after DRAM
  eviction. Keep temporary staging private unless explicitly admitted. Engine
  leases remain required before advertising general peer-HBM sources.
- [x] Measure source authorization, READ and completion independently in the
  shared-cache serving gate; retire the removed directory-RPC metric after D2.
- [ ] [S2][s2] — **qualification open**: Measure background synchronization and etcd traffic, index bytes and recovery
  lag under multi-host load and failure.

## M3 — routing and replica planning

- [ ] [S6][s6] — **deferred research decision**: Normalize vLLM and SGLang KV events.
- Retired: a second replica catalog. `crates/orbitkv-catalog/src/index.rs` and Server cluster synchronization already own sequenced DRAM/SSD discovery; capacity and recovery qualification belongs to S2.
- [ ] [S5][s5] — **implementation open**: Delegate cross-host TP query fan-out to node-local Cache Managers.
- [ ] [S6][s6] — **deferred research decision**: Integrate pinned `dynamo-kv-router` worker selection and production service
  lifecycle; verify hash/event mapping and request-load reservations (R1 in
  `docs/state-planning.md`).
- [ ] [S6][s6] — **deferred research decision**: Evaluate NIXL's Preview Mooncake backend before expanding transfer
  abstraction: qualify registration/completion, pinned TE compatibility and
  overhead; treat missing backend cost estimates as unknown. Keep direct TE
  until evidence justifies migration; do not add deprecated KVBM.
- [ ] [S6][s6] — **deferred research decision**: Feed qualified local/peer restore, recompute and queue estimates from
  [measured transfer planning](#measured-transfer-planning) into router summaries.
- [ ] [S6][s6] — **deferred research decision**: Add eviction externality and replica-risk terms.
- [ ] [S6][s6] — **deferred research decision**: Select a worker through the router, then revalidate and lease its
  transfer/restore plan at the Cache Manager.
- [ ] [S6][s6] — **deferred research decision**: Evaluate load-only, overlap-only, and joint planning on the same trace.

- [x] Add the pinned Mooncake TENT native sys/build boundary, dynamic ABI,
  cancellation drain, notifications and relocatable runtime packaging.
- [x] Remove the legacy Transfer Engine runtime: build and package only
  `tent_shared`, bind `tent_*` symbols, use terminal task status plus best-effort
  cancellation before free, and expose TENT NIC pressure to P/D diagnostics.
- [x] Map OrbitKV remote-cache authorization to Mooncake Segment addresses.
- [ ] [S5][s5] — **qualification open**: Qualify the existing RDMA READ demand-fetch path
  with visible NICs and physical transport counters.
- [ ] [S6][s6] — **deferred research decision**: Decide whether to add RDMA WRITE cache
  replication and its admission/ownership contract. Existing P/D WRITE support
  does not implement an active cache-replication owner.
- [ ] [S6][s6] — **deferred research decision**: Import topology-aware slicing, endpoint pooling, and alternate-rail retry.
- Maintained invariant: rkeys/raw addresses stay out of the global index (`crates/orbitkv-catalog/src/index.rs`, `crates/orbitkv-proto/proto/engine.proto`). Discovery records contain state/location metadata; authoritative exports remain source-owned.
- [x] Delete native v1 and vendored v2 RDMA implementations.

## M4 — generation-safe page references

- [ ] [S3][s3] — **implementation open**: Introduce manager-authored external `PageHandle { pool, page, generation }`
  and validate engine-owned GPU page generations at transfer boundaries.
- [ ] [S7][s7] — **implementation open**: Track the semantic frontier independently from execution completion.
- Retired: an unused generic fence framework. S3 extends the existing CUDA, TENT and SSD operation owners and keeps their distinct termination proofs.
- [ ] [S3][s3] — **implementation open**: Reject stale page generations at every adapter boundary.
- [ ] [S5][s5] — **implementation open**: Integrate handles into SGLang Radix lifecycle events.
- [ ] [S3][s3] — **implementation open**: Migrate the vLLM adapter without regressing its E2E path.
- [ ] [S3][s3] — **qualification open**: Add cancellation, preemption, crash, and delayed-completion stress tests.

## M5 — semantic state compiler

The existing compiler validates declared prefix/window/checkpoint recovery.
The M2 increment adds deterministic page demand and adapter consumption of the
same rules; its gate is tracked above. This is a limited implementation toward
M5, with no measured latency claim. The general compiler work remains open:

- [ ] [S7][s7] — **implementation open**: Define the `may_read(query, state)` IR.
- [ ] [S7][s7] — **implementation open**: Compile full-attention retention.
- [ ] [S7][s7] — **implementation open**: Compile sliding-window and sink-local retention.
- [ ] [S7][s7] — **implementation open**: Derive recurrent checkpoint placement and retention from the general IR;
  declared exact-boundary checkpoint recovery is already implemented.
- [ ] [S7][s7] — **implementation open**: Solve Minimum Persistent State Realization for hybrid bundles.
- [ ] [S7][s7] — **implementation open**: Emit placement, checkpoint, prefetch, and reclamation plans.
- [ ] [S7][s7] — **implementation open**: Measure Retention Amplification and semantic reclaim latency.

## Hygiene and release

- [x] Document independent per-node Managers, shared-instance capacity and automatic
  SSD selection; keep backend overrides in diagnosis/qualification instructions.
- [ ] [S8][s8] — **implementation open**: Replace the engine-coupled Docker build with independently versioned Manager
  and engine images built from the validated wheel artifacts.
- [ ] [S5][s5] — **qualification open**: Qualify concurrent engines sharing one Manager: matched runtime/device
  identities, bounded query ownership, engine/Manager restart and resource drain.
- [ ] [S5][s5] — **qualification open**: Qualify container GPU access, shared UDS/iceoryx2/PyTorch IPC and pidfd
  visibility before publishing DaemonSet/Deployment manifests. Test native SSD
  mounts separately from container functional recovery.
- [ ] [S8][s8] — **release gate**: Keep all public capability claims tied to a reproducible test.
- [ ] [S8][s8] — **implementation open**: Separate client and Cache Manager release artifacts when their contracts are
  stable; the complete global index remains embedded in the Manager.
- [ ] [S8][s8] — **release gate**: Keep heavy GPU/RDMA gates explicitly marked.
- [ ] [S8][s8] — **release gate**: Preserve license and upstream provenance requirements.
- [ ] [S8][s8] — **release gate**: Keep SGLang support claims aligned with the direct-linker E2E gate.

[s2]: docs/completion-plan.md#s2--metadata-reliability-and-measured-capacity

[s3]: docs/completion-plan.md#s3--transfer-lifetime-and-generation-safe-ownership

[s4]: docs/completion-plan.md#s4--finish-communication-execution-and-demonstrate-gains

[s5]: docs/completion-plan.md#s5--complete-engine-deployment-and-pd-contracts

[s6]: docs/completion-plan.md#s6--one-consumed-route-and-admission-planner

[s7]: docs/completion-plan.md#s7--semantic-retention-and-checkpoint-compiler

[s8]: docs/completion-plan.md#s8--product-organization-final-artifacts-and-release
