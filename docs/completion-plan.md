# Completion stages and independent acceptance

This is the implementation-agent handoff and reviewer acceptance sequence, audited
on 2026-09-29 against `29d95b77` (merged PR #189; implementation head `32f5f916`).
PR #189's required checks passed. Recheck the current base and working tree before
starting; this plan does not freeze concurrent user development.

The implementation agent delivers one stage or coherent substage per reviewable
change. The reviewing agent checks the actual consumed path and independently
reproduces its acceptance gates. Follow dependencies below; an infrastructure
gap may leave a qualification cell open while independent work proceeds.

## Baseline: preserve these implementations

| Area | Present implementation | Remaining boundary |
| --- | --- | --- |
| Local transport | UDS bootstrap, iceoryx2 descriptors, shared arenas and bounded operation ownership | Page-generation coverage, explicit registration data and live-process stall handling |
| Raw copies | Contiguous/strided DMA, bounded multipart Restore, engine-local execution | Residual overhead and earlier multipart consumption |
| Compute overlap | Qualified raw single-part layer/group readiness; coarse dependencies where required | SSD/codec pipelines and legal multipart overlap |
| Distributed discovery | Complete local indexes, fenced etcd publication, snapshot/Watch; no Catalog directory RPCs | Sustainable churn/capacity, quota recovery and separate host-failure domains |
| Remote recovery | Source authorization, TENT READ, release reconciliation, peer DRAM/SSD | Permanent requester loss and transfer/partition fault qualification |
| Engine support | vLLM 0.29.0 and SGLang 0.5.20 local recovery and TP=1 replica sharing | Explicit multi-instance, container, P/D and TP/PP qualification cells |
| Cost decisions | Resource-scoped observations, shadow estimates and guarded experimental peer choices | One consumed planner for complete routes, legal boundaries and P/D authority |
| Hybrid semantics | Declared recovery contracts and compiled page demands | General read-set/retention IR, checkpoint placement and semantic reclamation |
| Packaging | Wheel construction and installed-artifact gates already exist | New final artifact requalification, independent service images and publication |

The current global-index path passes same-host and H20/A100 TCP sharing, restart
and source-SSD gates on both engines. Same-A100 SGLang P/D evidence does not close
its heterogeneous-GPU strict-output gap. Raw-copy and discovery improvements do
not imply an end-to-end win over an engine's resident HBM hit.

S0 is this handoff change: correct the agent guide, migrate the four project
skills to `.agents/skills/`, remove stale Claude copies, and make this document
the current execution sequence. Runtime stages below are not marked complete
by writing this plan.

## S1 — Evidence separation and one truthful work queue

**Scope:** `benches/`, `docs/`, `TODO.md`, `.gitignore`, website content and the
smallest relevant CI check. No performance policy changes.

- Inventory every open TODO. Link it to S2–S8, mark code-present/qualification-open
  separately, and retire obsolete tasks with a reason. Do not implement unused
  Python state wrappers, a second replica catalog, a new generic region RPC or
  transport abstraction merely because an old checkbox requests one.
- Archive the tracked `benches/results/` collections before removing them from
  the current tree. An immutable Git revision can retain historical tracked
  evidence; untracked run data needs a verified external archive. Preserve hashes,
  commands, controls and failed runs. Do not rewrite history or delete originals
  before archive verification.
- Move generated output defaults out of the checkout. Use an explicit external
  output directory in benchmark/qualification examples and CI artifacts for CI.
  Keep reusable programs, workload definitions and small deterministic fixtures.
- Replace links into removed result directories with maintained methodology and
  stable evidence links. Keep only concise reviewed conclusions in public docs;
  large tables, per-run reports and generated plots remain external artifacts.
- Add a focused guard against tracking generated experimental output. Validate
  actual tracked files without banning legitimate test fixtures or arbitrary
  source files just because their names contain `result`.

**Acceptance:** tracked generated results are gone; archives are readable; normal
benchmark execution writes outside Git; benchmark tests pass; website build,
search and link tests pass. Every remaining open TODO has an owning stage or an
explicit deferred research decision. Do not rename Rust crates to mimic Python
package layout.

## S2 — Metadata reliability and measured capacity

**Depends on:** S1. **Owners:** `orbitkv-server/src/cluster/`, `orbitkv-catalog/`,
core inventory and metadata diagnostics.

- Extend real-etcd coverage to quota exhaustion/recovery, slow/disconnected
  Watches, concurrent rebuild and eviction, sustained journal overflow, member
  churn, index exhaustion and network partition/rejoin. Preserve incarnation and
  publication-cursor fences; never advertise a partial index as complete.
- Measure publication latency, Watch lag, snapshot rebuild time, update CPU,
  logical index bytes/RSS and etcd growth across increasing node/key/update loads.
  Define a tested operating envelope instead of claiming unlimited replication.
- Fix concrete bottlenecks and recovery gaps revealed by those runs. Keep local
  discovery free of directory/coordinator RPCs. Separate membership renewal from
  bulk synchronization and apply bounded work/backoff throughout recovery.
- Verify leader loss with a quorum and host loss across actual failure domains.
  Three etcd processes on one machine prove process faults only. Include startup
  readiness, operational diagnosis and recovery instructions.

**Acceptance:** no lost final puts/deletes after convergence, no stale incarnation
  authorization, bounded memory/queues, local cache still usable during metadata
  loss, and a repeatable capacity/recovery report. A fenced runtime restarts with
  a new incarnation; it cannot silently resume using the expired one.

## S3 — Transfer lifetime and generation-safe ownership

**Depends on:** S2's metadata contracts. **Owners:** core peer/query/storage owners,
transfer/sys C ABI, process channel and the consumed adapter registrations.

- Establish transport termination evidence before reclaiming exports belonging
  to a permanently lost requester. Inspect the current pinned TENT and latest
  upstream main/issues before changing the ABI; reuse or submit a focused
  transport fix if needed. Keep OrbitKV metadata/control state in OrbitKV.
- Bound and diagnose retained/quarantined source memory while proof is absent.
  Timeout, etcd expiry, cancellation, remote process disappearance and a control
  ACK are not interchangeable with native drain. An unavailable upstream primitive
  is an explicit blocker to safe reclamation, not permission to free memory.
- Carry owner-authored page/allocation generations through actual registration,
  source grants, destination use and adapter boundaries. Keep semantic last-use
  separate from completion of CUDA/RDMA/SSD work. Extend existing owners instead
  of adding an unused generic fence framework.
- Replace the remaining framework CUDA IPC pickle interpretation in the Manager
  with explicit validated registration data; remove the retired wire/API path.
- Add a bounded operational response to a live Manager that never completes a
  Publish, preserving held pages until drain or established process death.

**Acceptance:** real tests for requester/source crash, delayed/lost grants and
  release ACKs, partial submission, repeated page reuse and restart. No use-after-
  free or premature recycle; resource retention is bounded and observable; safe
  reclamation eventually happens when its proof exists. Both adapter lifecycles
  consume the new generation checks. No TTL-based reclamation substitute.

## S4 — Finish communication execution and demonstrate gains

**Depends on:** S3 for new reuse/early-consumption paths. **Owners:** existing raw
transfer compiler/workers, channel completion owners, SSD/codec owners and engine
layer hooks. See `communication-plan.md` and `engine-local-restore.md`.

- Profile adapter hashing/conversions, repeated destination checks, fragmented
  descriptors, dispatch, allocation/scratch work and duplicate shared-prefix H2D.
  Remove measured redundant work one bottleneck at a time.
- Extend readiness to legal multipart and SSD/codec pipelines where it can
  shorten the engine critical path. Retain the parent source/destination fence
  and final drain; a ready layer does not retire the whole operation.
- Add only consumed deadline/priority hints and bounded preparation needed by
  these paths. Qualify concurrent Query/Publish, fairness and long-running stalls.
- Address the measured SGLang native-CPU latency gap and ANS SSD overhead before
  enabling new defaults. Qualify GPU writeback/admission and codec quality over
  actual dense/hybrid workloads; native GDS needs a real supported mount with
  fallback disabled and physical I/O evidence.

**Acceptance:** byte-exact eager/graph, fragmented/packed/hybrid and partial-failure
  gates; repeated matched native HBM, native CPU offload, OrbitKV and LMCache
  comparisons for each engine. At least three order-alternated runs, fixed
  workloads/budgets and thresholds declared before execution. Report TTFT/TPOT,
  throughput, tails, CPU/pinned-memory costs and physical bytes; include misses
  and failed preparations. Reject overhead-only optimizations or keep them
  experimental. Do not compare an external miss to a resident HBM hit as equal work.

## S5 — Complete engine deployment and P/D contracts

**Depends on:** S2/S3; consume S4 changes when ready. **Owners:** vLLM scheduler/
worker/P/D adapters, SGLang linker/P/D integration and node-local Managers.

- Define one support matrix per engine: single instance, independent matching
  replicas, multiple instances sharing a Manager, dense/hybrid recovery, P/D,
  TP/PP, same-host TCP, two-host TCP and RDMA. Mark implemented, qualified,
  experimental and unsupported cells separately.
- Complete P/D plus cache reuse for cold, partial and full hits; verify Prefill
  and Decode restart, cancellation, preemption, in-flight faults and bounded
  source/destination ownership. Close SGLang's heterogeneous-GPU output gap with
  matched native controls, not relaxed assertions or a different prompt silently.
- Move cross-host rank fan-out/coordinated recovery into node-local Manager owners
  where appropriate. Agree a common TP recovery boundary and stage-specific PP
  layout/ownership. Reject unsupported heterogeneous layouts explicitly.
- Integrate SGLang Radix lifecycle events and model/adapter or live-weight
  invalidation where their cache semantics require them. Extend model-state
  support only with real consumed recovery contracts.
- Qualify multiple engines sharing one Manager and container UDS/iceoryx2/CUDA IPC,
  pidfd visibility, budgets and failure isolation. A shared service does not
  imply cross-engine interchangeable cache bytes.

**Acceptance:** an executable scenario and output/physical-transfer/resource-drain
  evidence for every claimed matrix cell. Cross-host TP/PP needs enough devices;
  RDMA needs visible NICs and transport counters. Unsupported hardware cells stay
  open. Preserve thin, distinct engine callbacks and shared Rust ownership.

## S6 — One consumed route and admission planner

**Depends on:** S2/S3, measured S4 paths and S5 P/D authority. **Owners:** existing
`planning/`, `cost/`, `query/`, residency/peer owners and actual engine feedback.

- Extend complete-route estimates to compatible local DRAM/SSD, peer DRAM/SSD,
  supported direct transfers and engine-owned recomputation. Compare only equal
  state coverage and completion targets; include queue/admission/authorization,
  transfer, decode and H2D once, with freshness and uncertainty.
- Make the selected plan acquire the actual source/extent lease, byte credits
  and destination ownership. For P/D vs direct recovery, the same consumed plan
  must own handoff authorization; observation-only messages are insufficient.
- Qualify bounded replanning and contending destinations, device/SSD/NIC budgets,
  per-instance shares, preparation deadlines and retention/write admission.
  Train only observed outcomes. An unknown estimate must not masquerade as zero.
- Compare DMA/kernel and io_uring/native cuFile only when both are physically
  eligible and qualified. Introduce guarded switching with measured margins;
  retain a clear deterministic policy for unknown estimates, without a duplicate
  legacy planner or generic forwarding layer.
- Keep Manager route decisions independent of request routing. After that works,
  integrate the pinned upstream router through a narrow consumed boundary if the
  routing milestone is retained; include worker evidence, replica risk and
  revalidation. NIXL/backend expansion is an evaluation decision, not an assumed
  prerequisite to finishing TENT recovery.

**Acceptance:** selected routes explain their coverage, ownership and predicted
  complete cost; stale evidence/resource rejection replans safely; submitted
  payload failure does not blindly retry another medium. Fixed-policy, shadow
  and executed-policy comparisons demonstrate repeatable benefit under contention
  without violating correctness, memory budgets or cold-path regression limits.

## S7 — Semantic retention and checkpoint compiler

**Depends on:** S3 generations/lifetimes, S5 consumed hybrid contracts and S6
physical planning. **Owners:** `orbitkv-state`, core planning/storage and adapter
state declarations; retain framework scheduling ownership.

- Define a finite, explicit `may_read(query, state)` IR for supported state
  families. Compile full attention, sliding windows/sinks and recurrent state
  into required ranges, legal checkpoints, retention and reclamation decisions.
- Implement minimum persistent state for the defined model and supported
  families, with an exact oracle on bounded cases and explicit approximation
  limits elsewhere. Do not claim a general optimum for undeclared semantics.
- Feed checkpoint placement, prefetch/placement and semantic release decisions
  into real storage/planning owners. Safe reclamation requires both semantic
  non-use and physical completion; compile-only output is not delivery.
- Extend DSA, draft-model or auxiliary state only with a matching semantic
  contract and representative model gate. Unknown state stays unsupported.

**Acceptance:** real dense, SWA/sink and recurrent/hybrid model recovery, plus
  boundary/preemption/restart adversarial cases. Check required-state soundness,
  generation safety, bounded-case minimality, retained bytes, retention
  amplification and semantic reclaim latency. Retain a workload with measured
  state reduction; do not describe recovery validation alone as this compiler.

## S8 — Product organization, final artifacts and release

**Depends on:** accepted scope from S1–S7; retain explicit exclusions for unqualified
hardware/research work. **Owners:** docs/README/website, examples, containers and
existing build/release workflows.

- Organize README/docs like LMCache: product purpose, supported deployment modes,
  installation, short quickstart, engine/model recipes, architecture, operation
  and benchmark methodology. Use its separation of library, tests, examples,
  benchmark programs and tools as guidance, preserving OrbitKV's Rust owners.
- Preserve OrbitKV's visual identity. Keep one canonical technical document per
  subject; remove chronological development diaries and duplicate result tables
  from user paths. Show full local/distributed/P/D ownership and data/control
  paths in diagrams, with status and limits next to capability claims.
- Publish curated experiments from external versioned artifacts, linking model,
  engine/source revisions, transport, budgets, controls and reproducible scripts.
  Verify links/search/navigation after moving legacy result content.
- Build independent Manager and engine images from validated artifacts. Separate
  Manager/client packaging only at a clear dependency/version contract; qualify
  shared-process resources, GPU access and SSD mounts before publishing manifests.
- Build final wheels/images, install outside the checkout on both hosts and
  repeat the claimed sharing/P/D/restart gates with those exact hashes. Source
  test evidence cannot replace final artifact qualification.
- Use the existing release workflow for publication within the user's release
  authorization. Publishing is a distinct delivery action; manual candidate
  builds and successful CI do not mean a PyPI release has happened.

**Acceptance:** clean install and documented launch work for each supported
  profile; no undocumented source-tree/native-library dependency; CI and website
  checks pass; no generated experiment results are tracked; package/image hashes,
  external evidence, actual publication status and remaining limits are reviewable.

## Review contract for every delivery

The implementing agent provides:

1. Stage/substage, base and final commits, and the changed user-visible behavior.
2. The concrete state/resource owner and invariants affected; removed APIs and
   duplicate paths; any remaining compatibility or forwarding layer justified.
3. Exact checks and exit status. Run only the gates relevant to the change, plus
   required repo checks. Never build native libraries during live runtime tests.
4. External evidence locations and artifact hashes, with failed controls and
   environmental limitations. No private credentials or results in the commit.
5. Updated TODO/support status and the next dependency. Passing unit tests does
   not close a hardware, model-serving or performance cell.

The reviewer inspects implementation and call sites, repeats the smallest tests
that establish the changed contract, audits lifecycle failures and benchmark
fairness, and checks public claims against exact artifacts. Acceptance is
**accepted**, **changes required**, or **implemented / qualification blocked**,
with concrete reasons. The last status does not mark a gate complete.

## References and scope

These sources guide organization and workflows, not OrbitKV capability claims:

- [LMCache project layout and README](https://github.com/LMCache/LMCache).
- [LMCache documentation and recipes](https://docs.lmcache.ai/).
- [LMCache benchmarking guide](https://docs.lmcache.ai/getting_started/benchmarking.html).
- [Codex repository skill discovery](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills).

Architecture contracts remain in [distributed cache](distributed-cache.md),
[communication](communication-plan.md), [state planning](state-planning.md),
[deployment](deployment.md), [P/D](pd.md) and [release](releases.md) documentation.
Existing historical reports describe their recorded revisions only.
