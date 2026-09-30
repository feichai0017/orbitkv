# Completion and upstream integration plan

This is the single execution queue for OrbitKV. Architecture documents describe
contracts; deployment documents describe supported configurations. Neither is a
second checklist. The implementing agent delivers one stage or coherent substage;
the reviewer independently checks its consumed path, tests and evidence.

## Release baseline and reference policy

Official latest non-prerelease versions were checked on **2026-09-30**:

| Project | Reference release / commit | OrbitKV qualification |
| --- | --- | --- |
| [vLLM](https://github.com/vllm-project/vllm/releases/tag/v0.30.0) | `v0.30.0` / `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | Dependency/submodule upgraded. A100 Qwen3-8B DRAM/SSD and eager/graph gates pass locally; independent upgrade acceptance and broader model/topology qualification remain open. |
| [SGLang](https://github.com/sgl-project/sglang/releases/tag/v0.5.20) | `v0.5.20` / `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` | Current dependency/submodule and recorded serving baseline. |
| [LMCache](https://github.com/LMCache/LMCache/releases/tag/v0.5.5) | `v0.5.5` / `05a013b29da78cf2321b9b46ec5039dde2fb0bb0` | Integration and matched-comparison reference, not evidence of OrbitKV support. |

At the start of an engine upgrade, recheck official releases, record the selected
tag and commit, then freeze that version for the review. Inspect upstream main
and related issues/PRs for existing fixes; main-only APIs are not release APIs.
Upgrade dependency pins, lockfiles, submodules, examples and qualification together.
If a required fix is unreleased, identify the upstream patch and keep that profile
experimental. Remove superseded version branches at cutover rather than building
a multi-version compatibility layer. A documentation update alone never upgrades
the runtime or establishes output/performance parity.

The [adapter contract](adapters.md) compares the released LMCache integrations
and defines engine/cache ownership. The first upstream contribution is a small,
maintainable cache backend with explicit support limits; hybrid lifecycle and P/D
extensions can land separately. Maintainer acceptance is external to this plan:
record submitted, merged and released as different states.

## Delivery status and dependencies

| Stage | Status / next action |
| --- | --- |
| S0 | Agent handoff and Codex skill migration merged in PR #190. |
| S1 | Evidence separation independently verified at `ec3add9b`; this delivery consolidates plans and release-based integration guidance. Acceptance covers S1 only; native CUDA qualification remains blocked on this host. |
| S2 | Partial: S2.1–S2.7 are independently accepted; bounded publication coalescing keeps a zero-wait default with a qualified 2 ms opt-in. S2.8 owner inventory streams are next. Three-host metadata remains blocked by CPU-node SSH authorization and mutually unreachable A/B container data addresses; live-store soak and separate-host cells remain open under S2.10. |
| S3 | Open: native termination proof, page generations and explicit registration. |
| S4 | Partial: optimize measured execution gaps; qualify mixed communication. |
| S5 | Partial: [S5.1 release/interface audit](engine-release-audit.md) independently accepted at `38f8dbb2`; the 0.30.0 adapter upgrade consumes native transfer results with local A100 cache/P/D evidence. Independent upgrade acceptance and public lifecycle/deployment gates remain open. |
| S6 | Partial: observations/limited choices exist; unified executed decisions remain open. |
| S7 | Open: consumed retention/checkpoint compiler beyond recovery validation. |
| S8 | Partial: existing wheel workflow; final images, artifact gates and publication remain open. |

S5.1 interface research and S8 documentation/package preparation may proceed while
S2/S3 run. New reuse, reclamation and early-consumption behavior must wait for its
listed lifetime dependencies. Hardware gaps leave their cells open and do not stop
independent work. Use **implementation open**, **implementation partial**,
**qualification open**, **deferred research**, and **release gate** precisely.

## Baseline: preserve these implementations

| Area | Present implementation | Remaining boundary |
| --- | --- | --- |
| Local transport | UDS bootstrap, iceoryx2 descriptors, shared arenas and bounded operation ownership | Page-generation coverage, explicit registration data and live-process stall handling |
| Raw copies | Contiguous/strided DMA, bounded multipart Restore, engine-local execution | Residual overhead and earlier multipart consumption |
| Compute overlap | Qualified raw single-part layer/group readiness; coarse dependencies where required | SSD/codec pipelines and legal multipart overlap |
| Distributed discovery | Complete local indexes, fenced etcd publication, snapshot/Watch; no Catalog directory RPCs | Sustainable churn/capacity, quota recovery and separate host-failure domains |
| Remote recovery | Source authorization, TENT READ, release reconciliation, peer DRAM/SSD | Permanent requester loss and transfer/partition fault qualification |
| Engine support | vLLM 0.30.0 local recovery under qualification; SGLang 0.5.20 unchanged; historical 0.29.0 local recovery and replica-sharing evidence retained | New-release model/topology requalification; multi-instance, container, P/D and TP/PP cells |
| Cost decisions | Resource-scoped observations, shadow estimates and guarded experimental peer choices | One consumed planner for complete routes, legal boundaries and P/D authority |
| Hybrid semantics | Declared recovery contracts and compiled page demands | General read-set/retention IR, checkpoint placement and semantic reclamation |
| Packaging | Wheel construction and installed-artifact gates already exist | New final artifact requalification, independent service images and publication |

The current global-index path passes same-host and H20/A100 TCP sharing, restart
and source-SSD gates on both engines. Same-A100 SGLang P/D evidence does not close
its heterogeneous-GPU strict-output gap. Raw-copy and discovery improvements do
not imply an end-to-end win over an engine's resident HBM hit.

The baseline above describes implemented paths, not universal model, topology or
performance guarantees. Keep historical measurements tied to their exact builds.

## S1 — Evidence separation and one truthful work queue

**Scope:** `benches/`, `docs/`, agent skills, `.gitignore`, website content and the
smallest relevant CI check. No performance policy changes.

- Consolidate the former TODO and competing plans into S2–S8 below. Preserve
  missing-code versus qualification status and explicit research deferrals. Remove
  obsolete checklists after migrating their obligations. Do not implement unused
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
search and link tests pass. Every migrated obligation has an owning stage or an
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

### S2.1 — Quota and index-budget recovery

The real-etcd metadata-owner gate now covers backend quota exhaustion, a stalled
publication cursor, compaction/defragmentation/alarm disarm and resumed publication
without resurrecting a deleted replica. A separate small-index case rejects all
partial results, keeps lease renewal alive beyond its initial conservative
validity window and rebuilds only after eviction permits a complete snapshot.
The existing lost-reply, paginated-snapshot, incarnation, leader-loss and
quorum-loss gates remain part of this substage's validation. The leader-loss
rerun exposed startup failure on an unavailable balanced endpoint after a lease
was granted. Metadata bootstrap now retries transport failures with the same
lease/incarnation under a finite startup budget; format/identity rejection is
terminal. A repeated registration reconciles only its exact committed owner and
lease without advancing the node epoch or stealing another registration.

This scope uses etcd 3.5.21 processes on one container/host and the actual Rust
publisher, Watch, membership and index owners. Synthetic residency records enter
the publisher directly; it does not qualify GPU storage publication, journal
churn through live stores, data-plane availability, cross-host HA or capacity.
Those S2 obligations remain open. The [metadata gate recipe](distributed-cache.md#quota-and-index-budget-recovery)
records commands, external evidence and operational recovery boundaries.

### S2.2 — Watch delay, partition and compaction recovery

A test-owned TCP gate now isolates one Manager's actual etcd connection while the
real etcd service remains writable. It delays Watch responses, severs existing
HTTP/2 streams, permits source publication and compaction through an independent
connection, then heals the Manager connection. The reader retains its last
complete index during the transient partition, hides the index while compacted
history is rebuilt and exposes only the final complete snapshot afterward.

The gate exposed a real classification defect: tonic reports an HTTP/2 Watch-body
reset as gRPC `Unknown`. Treat only `Unknown` messages that identify HTTP/2 or
transport failure as resumable disconnections; other unknown statuses still
force a rebuild. Resume starts at the last applied revision, and etcd compaction
or missing previous metadata then triggers the normal reset/snapshot path. The
[Watch fault recipe](distributed-cache.md#watch-delay-partition-and-compaction)
records the observed lag/rebuild measurements and remaining scope.

This substage uses synthetic publication records and one etcd process on one
host. It does not close sustained inventory-journal overflow, concurrent live
DRAM/SSD publication, local-cache availability through Manager metadata loss,
multi-host partitions or the measured operating envelope.

### S2.3 — Measured metadata scaling smoke

The real-etcd capacity gate registers 1, 4 and 16 source owners with 512 keys per
owner, publishes complete owner states, starts a fresh reader to measure snapshot
rebuild and deletes half of every owner's records. It reports transaction
latency, observed Watch lag, rebuild time, logical index bytes, process CPU ticks
and RSS, and etcd backend growth. A fixed 500 ms delay lets etcd backend-size
statistics settle outside the timed publication and Watch samples.

Three repetitions pass on both the current H20 container and the separate A100
host using identical frozen artifacts. At the largest checked point (16 owners,
8,192 records), the two hosts report publication p50 0.64/0.71 ms, observed Watch
p50 about 2.08 ms, snapshot rebuild medians 56.2/60.1 ms and 2,315,392 logical
index bytes. The 1 ms observer makes Watch values quantized diagnostics rather
than transport-only latency. See the [capacity smoke recipe](distributed-cache.md#measured-metadata-scaling-smoke).

The final correction verifies every retained key, owner/incarnation, sequence and
complete metadata on both the original and rebuilt readers. Codex independently
reran that frozen gate and accepted S2.3 at `680eaa1a`; the earlier sandbox bind
failure is retained separately as an environmental control.

This is a tested smoke envelope, not a maximum: one etcd process per run,
synthetic Publisher records, no inference load and at most 16 registered sources.
Larger node/key/update loads, sustained live-store journal churn, concurrent
Managers, three-host etcd and failure-domain capacity remain open.

### S2.4 — Bounded live-DRAM journal and metadata-loss correctness

An A100 gate now sends real GPU blocks through the production Engine DRAM store,
evicts them and saves replacements while a test-owned TCP gate disconnects the
source Cluster from real etcd. A 1 KiB inventory journal advances from sequence
64 to 192 and explicitly reports a history gap. The observer retains its last
complete view during the partition; after reconnection the publisher snapshot
removes every old key and publishes every replacement under the exact source
owner. Zeroed GPU memory restores exactly 65,536 local bytes both during the
transient partition and after the source's real lease disappears from an
available observer. Healing cannot revive the expired incarnation.

Codex independently reran the frozen A100 artifact and accepted the final
observer-availability barrier at `f7d953a8`. This is one bounded live-DRAM
correctness scenario using Engine and Cluster owners in one test process. It is
not sustained churn or a capacity envelope, and does not qualify a complete
Manager process, SSD, cross-host cache, P/D, RDMA, native GDS or three-host etcd.
See the [live metadata-loss recipe](distributed-cache.md#live-dram-journal-overflow-and-metadata-loss).

### S2.5 — Full-Manager DRAM and io_uring metadata-fault correctness

The Python distributed gate now runs an independent client process against two
real Cache Manager processes and one real etcd process. DRAM and explicit
io_uring SSD cases each cover a no-fault remote restore, three rounds of source
publication/eviction/re-save while only the source's etcd transport is severed,
snapshot recovery after healing, and a second partition through actual lease
expiry. Every round and block uses a distinct deterministic payload; every
restore zeroes the GPU destination and compares all bytes.

The test reads the isolated etcd namespace as an independent oracle, validates
the exact source incarnation and location-key digest, and requires the complete
set of retained hashes under the expected DRAM or SSD medium. Deleted hashes
must be absent. For SSD, each save waits for io_uring completion, evicts DRAM and
requires the SSD-read counter to advance before accepting restored bytes. Before
testing the expired source from the second Manager, its own DRAM is evicted so a
local hit cannot mask remote rejection. Healing cannot re-register the old
source incarnation.

The implementation run passes both media on one A100 host with frozen Manager,
wheel, extension, TENT and etcd artifacts. It covers five eight-block rounds per
medium and forces repeated 1 KiB journal overflow through `e4cfc810`. Codex
independently reran both media and accepted the complete S2.5 delivery at
`f3a44ce1`.
This is same-host multi-process correctness, not cross-host HA, native GDS, a
large sustained-capacity envelope or an engine-serving qualification. See the
[Manager process recipe](distributed-cache.md#manager-process-dram-and-io_uring-metadata-faults).

### S2.6 — Sustained metadata churn and capacity

**Independently accepted at `ab306965`.** A 60-second ignored release gate uses
16 registered production Publishers and real etcd to rotate 24,576 active DRAM/
SSD metadata records through 245,760 changes. It starts fresh complete-index
readers during eviction rounds, checks every final key, source incarnation,
sequence and medium, and measures publication latency, Watch lag, rebuild time,
logical index bytes, bounded publication batches, CPU/RSS and etcd growth against
thresholds committed before execution. Manager metadata status also exposes the
live residency count and retained journal current/peak bytes, capacity and
history-gap count; the accepted S2.5 gate consumes these fields on the actual
storage path.

The committed workload contract and command are in the
[sustained-capacity recipe](distributed-cache.md#sustained-metadata-churn-and-capacity).
The frozen candidate at `85f0f656` passes once locally and three times on the
A100 host. All A100 runs sustain 4,032 changes/second; publication p95 is
1.02–1.04 ms, Watch p95 is 3.46–3.47 ms, fresh-reader snapshots finish within
374 ms, and target convergence finishes within 454 ms. The exact final 24,576
records, 7,015,680-byte index peak and every predeclared resource bound pass.
The same frozen Manager/wheel also reruns the S2.5 DRAM/io_uring gate successfully
with the new journal diagnostics. Codex independently repeated the capacity gate
at 4,032.52 changes/second and both Manager media cases in 38.34 seconds, verified
the frozen evidence and accepted the substage with no blocking findings. This
one-host/one-etcd-member gate cannot close three-host or independent-failure-domain
qualification.

### S2.7 — Bounded coalescing and frozen comparison baseline

**Independently accepted at `afa72863`.** The candidate coalesces at
most 1,024 records / 512 KiB from a contiguous journal interval, preserves the
latest key/medium generation and final deletes, and advances the input cursor
only after every etcd transaction commits. Flush bypasses the bounded wait.
The [coalescing benchmark](distributed-cache.md#bounded-etcd-publication-coalescing)
compares frozen S2.6 and candidate 0/2/5 ms profiles with exact byte/generation
checks, visibility, effective hits and local save/query costs. All 20 rotated A100
runs and the selected-default S2.5/S2.6 regressions pass the predeclared limits.
The default remains 0; 2 ms is a qualified opt-in tradeoff. Evidence is frozen
outside the checkout. Codex independently checked all 20 runs and 4,200 raw
samples, reran the real-etcd cursor and 60-second capacity gates plus both full-
Manager media cases, verified frozen hashes and accepted the same-host substage
without blocking findings. Cross-host and serving cells remain open.

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
- **Implementation open:** expose a stable native peer-liveness contract before
  enabling SGLang failed-session probing. Liveness is not transfer termination.
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
layer hooks. The current contracts live in [transport](transport.md) and
[engine-local restore](engine-local-restore.md).

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

- Measure native completion observation and Python waiting/thread handoffs before
  replacing them. Prefer bounded batching and demand-driven progress; completion
  must return network credits without waiting for parsing, H2D or model callbacks.
  Reuse upstream TENT priority/progress capabilities after checking their actual
  release/configuration behavior. Reuse the node-local Manager and existing
  native execution owners; keep short protocol work near I/O progress and heavy
  codec/SSD work on bounded workers. Do not add a second RPC Agent.
  Do not build a custom TCP stack or modify TENT
  to host OrbitKV's metadata protocol.
- Add mixed inference/network controls: cache READ, P/D WRITE and engine NCCL/
  expert traffic on shared versus separate NICs. Measure decode ITL tails, TTFT,
  SLO goodput, CPU cores, PCIe/NUMA pressure and physical NIC counters. TENT's own
  load counters do not measure non-TENT traffic; RDMA is not bandwidth isolation.
- **Implementation partial:** direct registered engine-page SSD I/O and multi-writer
  GPU assembly remain beyond the existing staged GPU path. Include registration,
  source-page hold time, fragmentation and extra HBM in their admission decision.
- **Qualification open:** native GDS, selective DRAM admission/GPU writeback,
  hybrid codec quality, and the SGLang ANS SSD latency gate. Historical TurboQuant
  greedy-output failures remain visible; keep recurrent state exact until its
  quality and correctness gates pass. Current hardware evidence cannot close them.

**Acceptance:** byte-exact eager/graph, fragmented/packed/hybrid and partial-failure
  gates; repeated matched native HBM, native CPU offload, OrbitKV and LMCache
  comparisons for each engine. At least three order-alternated runs, fixed
  workloads/budgets and thresholds declared before execution. Report TTFT/TPOT,
  throughput, tails, CPU/pinned-memory costs and physical bytes; include misses
  and failed preparations. Reject overhead-only optimizations or keep them
  experimental. Do not compare an external miss to a resident HBM hit as equal work.

## S5 — Released-engine integration and upstream contributions

**Depends on:** S2/S3 for new lifetime behavior; consume S4 when ready. S5.1 may
start independently. **Owners:** engine adapters, existing native client owners
and narrowly scoped upstream engine interfaces. Reference released LMCache
integration contracts from [the adapter guide](adapters.md#lmcache-and-flexkv-reference).

### S5.1 — Release and interface audit

**vLLM 0.30.0 implementation delivered with local qualification; independent
review open.** Dependency and submodule use the exact release above. Cache and
P/D workers return native `KVConnectorTransferResults`; failed receive and
finished receive share one snapshot, replacing `PdWorkerMetadata` and its
duplicate queue. Best-effort cache publication explicitly returns false for
`requires_kv_delivery`. The native MultiConnector consumer is covered by a
released-engine test. No 0.29/0.30 compatibility alias remains.

The complete CUDA 13 wheel passes build/repair/isolated import. Its installed
production package passes A100 Qwen3-8B DRAM eager, DRAM graph and forced-io_uring
SSD eager gates (six checks each; the recurrent-only check is inapplicable).
Same-GPU TCP P/D and save-only MultiConnector each pass one gate; the latter
uses DecodeBench as the load owner and does not qualify live P/D/cache overlap.
Fourteen exact native recovery tests and three released connector-contract tests
pass. Fixed Qwen3.5-0.8B passes seven checks each for DRAM eager, DRAM graph and
forced-io_uring SSD eager modes. The full A100 selection totals 58 passes and
three inapplicable dense-model recurrent skips.
These profiles use TP=1/PP=1; multi-GPU, RDMA and historical model/topology cells
remain separately unqualified on the new release.
Source-only Python passes 413 tests, with one skip and 160 deselected. The
generated `python/uv.lock` is ignored by repository policy; its resolved 0.30.0
snapshot is archived with the wheel rather than force-added.
All 44 Python production modules match the tested wheel byte for byte. The first
six gates preserve all 146 installed-artifact hashes and 72 test-file hashes;
the extended hybrid/recovery selection also preserves its installed/test hashes
and returns GPU usage to zero. Local full Rust tests reach 47 CUDA-device failures
in `orbitkv-core` on this GPU-inaccessible host; they are not recorded as passed.
The PR's CI covers both CUDA builds, Clippy, Python and wheel packaging separately.
Evidence: `/root/orbitkv-artifacts/engine-native-lifecycle-20260930/vllm-0.30/`.
Previous 0.29.0 model, topology and performance measurements remain historical.
This delivery does not close S5.3/S5.4 or establish a performance advantage.

- Audit vLLM 0.30.0 and SGLang 0.5.20 by exact release commits above, and LMCache
  0.5.5 against those interfaces. Check recipe prerequisites in release source;
  do not assume a documented upstream PR is included in either engine release.
- Inventory every public callback, internal Hook, monkey patch, local request
  state machine and configuration entry. Name its resource owner and replacement,
  distinguishing already released APIs from main-only or proposed extensions.
- Upgrade vLLM from 0.29.0 as its own tested change, synchronizing package pins,
  lockfile and engine submodule. Retain 0.29.0 evidence as historical; run new
  release correctness, restart, eager/graph and overhead gates before changing
  public support claims. Keep one maintained release implementation per engine.
- Audit reuse of native vLLM P/D: layout/state coverage, rank mapping, source
  retention, cancellation, error reporting and composition with cache restore.
  Record concrete gaps before expanding or deleting `vllm/pd/`.

**Acceptance:** release/API inventory with exact callers and an upstream issue/PR
or a bounded local responsibility for each remaining internal dependency.

### S5.2 — Minimal official cache backends

**Local adapter cleanup implemented and checked on A100; independent review open.**
The public vLLM entry constructs only `SchedulerAdapter` or `WorkerAdapter`,
closes native connections on initialization failure, and inherits unchanged
optional callbacks. The unused service availability owner and its health thread
are removed; restore exceptions still retain destinations until native drain.
SGLang event ownership lives in `events.py`, and disabled-backend plugin/admission
paths leave native and GPU modules unloaded. Registration conflicts are explicit.
Prefill and Decode now directly own their P/D worker callbacks and request state;
the intermediate Handlers, class-callback mixin and generic executor facade are
removed. SGLang cancellation consumes the released cache finish/linker release
chain instead of a duplicate Scheduler abort Hook. Ordinary cache registration
has two internal Hooks; enqueue preparation and P/D observation Hooks are opt-in.
No native P/D replacement is claimed.

The initial cleanup's source-only gate passed 414 tests. With frozen native artifacts, the pinned
SGLang 0.5.20 admission/event gate passes 22 tests and the vLLM 0.29.0 native
recovery-contract gate passes seven. These integration tests use controlled
completion and CUDA-event doubles, not real GPU DMA. The vLLM gate also corrects
a pre-existing async-load expectation reproduced on the unchanged baseline: a
forward-consumed recovery reports a synchronous scheduler hit.
The A100 Qwen3-8B TP=1/PP=1 serving gates pass: vLLM reports six passed and one
hybrid-only assertion skipped; SGLang passes DRAM and io_uring SSD recovery.
The gates compare model outputs with controls, check native cache reuse and
verify external loads after process restart. The dense model leaves hybrid
serving qualification open.
The SGLang event/linker gate passes 19 tests, including real GPU page overwrite,
full/window/checkpoint recovery on DRAM/SSD, and controlled failure/drain cases.

Two initial SGLang SSD starts failed at TCPStore binding before model loading.
A minimal reproduction showed that loopback availability did not imply wildcard
availability. The test port helper now checks wildcard bind/listen; a real-socket
regression protects this boundary, and the fixture retains Manager logs.
Both failures and the successful rerun remain in external evidence. The original
fixture deleted its temporary Manager logs during the first run; its pytest and
engine logs remain available.

The P/D ownership follow-up passes 413 source-only tests, including 139 P/D
contracts; the assertion-only test for the removed Handler layer is deleted.
SGLang's pinned admission/event gate passes 24 tests, now including native
cache-finish cancellation and successful-finish preservation. A reusable vLLM
P/D E2E gate runs two TP=1/PP=1 eager workers on one A100 over forced TCP:
129/257/769-token prompts, including chunked prefill, produce the same text and
output tokens as native execution; 198,180,864 payload bytes are reported and
all sender/waiter gauges drain. This profile does not qualify multi-GPU/RDMA,
hybrid P/D, live handoff/cache composition or the proxy HTTP service itself.
SGLang DRAM/SSD serving recovery also passes again (two cases), as do four
GPU recurrent/window recovery cases including cancellation of published pages.
The seven frozen native artifacts are unchanged; postflight finds no engine or
Manager processes and GPU usage returns to zero.
The obsolete Python P/D launcher is removed: it selected ordinary cache
connectors and invoked the removed `orbitkv-router` binary. The maintained
deployment entry is `scripts/run_pd_local.sh`.
Follow-up evidence: `/root/orbitkv-artifacts/adapter-cleanup-20260930/pd-followup/`.

Overall S5.2 acceptance, engine upgrades, upstream registration, and additional
multi-GPU/P/D qualification remain open.
Evidence: `/root/orbitkv-artifacts/adapter-cleanup-20260930/`.

- vLLM: retain `OrbitKVConnector` and distinct scheduler/worker responsibilities.
  Implement released lookup/allocation, registration, load/save and terminal
  callbacks; consume success and failure through the selected release's API.
  Keep cache transfers and engine page-allocation ownership separate.
- SGLang: retain `UnifiedRadixCache` plus `OrbitKVLinker`, with native tree and
  component ownership. Reuse the released construction and layer-counter flow;
  propose a small external-linker registration extension only where needed.
- Keep runtime/native imports lazy and dependencies optional. Unselected OrbitKV
  must not initialize CUDA, TENT or a Manager connection. A complete native HBM
  hit must not add synchronous external lookup or a duplicate H2D copy.
- Remove duplicate plugin registration at the official-registry cutover; do not
  suppress all registration errors. Remove internal forwarding facades while
  retaining the one adapter that actually implements each engine contract.
- Submit baseline registration/configuration, focused tests and installation
  docs as small upstream PRs. Core cache policy, etcd and storage stay in OrbitKV.

**Acceptance:** clean supported engine checkout plus an installed OrbitKV wheel;
no source editing needed for the claimed profile. Cold/partial/full hits,
pressure/restart reuse and disabled-backend import behavior pass. Upstream PR
status is reported separately from local adapter qualification.

### S5.3 — Public lifecycle and hybrid-state contracts

The 2026-09-30 source check confirms that released 0.30.0 and current vLLM main
still invoke preemption drain after page initialization, and synchronous restore
after recurrent preprocessing. Generic preemption-ordering patch `98b917d5` is
prepared against vLLM main `fb91712b`: six CPU tests pass; unchanged main fails
both new real-step regressions. Its complete upstream lint selection passes.
Evidence and portable patch: `/root/orbitkv-artifacts/engine-native-lifecycle-20260930/vllm-upstream/`.
It is not submitted, merged, released or qualified with a main-branch GPU model;
upstream requires human review and test execution before submission.
Moving only that fence does not remove `runtime.py`'s restore
boundary. SGLang PR #40595 (external-linker construction) and #40896 (load-failure
lifecycle) are open; #40759 (Mamba lifecycle proof of concept) is closed unmerged.
Do not delete consumed safety behavior based on these proposals.

- vLLM: replace the `runtime.py` runner patch only after the released interface
  guarantees preemption/save drain before page reuse and restore after page
  initialization/COW but before recurrent state preprocessing. Prefer a generic
  ordering fix to a new callback when the existing contract suffices.
- Replace `scheduler.py`'s blanket multi-group native-prefix bypass with correct
  atomic state-group availability/recovery. Audit existing divergent-hit APIs;
  changing a capability flag alone does not prove all states ready.
- SGLang: move generic recurrent checkpoint allocation/commit/abort and external
  linker lifecycle support into engine components where accepted. Shrink
  `RecoveryLinkerWrapper`; preserve its safety checks until equivalents are
  actually consumed. Public Full/SWA support does not imply recurrent support.
- Replace private enqueue/abort/release and decode-ready Hooks with explicit
  lifecycle contracts. Keep telemetry out of ownership transitions and model
  computation in the engine. Incorporate generation checks from S3.
- **Implementation open:** adapter/LoRA identity and live-weight invalidation.
  Preserve cache salt, computation identity, multimodal inputs and representation
  boundaries; reject unsupported combinations instead of sharing unsafe keys.
- **Qualification open:** Full + SWA + temporal recurrent native serving, page
  reuse under preemption, cancellation, engine restart and each advertised graph
  mode. Exact synthetic GPU recovery does not establish native model support.

**Acceptance:** no private runtime replacement remains in the claimed upstream
profile; all-state readiness, GPU ownership and resident-prefix behavior pass.
Unreleased required fixes keep only their dependent profiles experimental.

### S5.4 — Native P/D lifecycle with TENT payloads

The 0.30.0 Mooncake receive timeout/error path reports completion without a
remote-write drain acknowledgement; its thread-pool shutdown is nonblocking.
An explicit TENT constructor is insufficient by itself. Native replacement must
first consume destination generation, write authorization/revocation and final
drain evidence from S3. Existing custom owners therefore remain implementation
dependencies, not obsolete code ready for mechanical deletion.

- Keep cache offload/reuse and live P/D handoff independently selectable. Prefer
  native engine bootstrap, request states and DecodeReady authority; add a
  released, explicit TENT transport construction boundary without replacing a
  module's global class. TENT API calls do not establish GPUDirect RDMA use.
- SGLang: replace `install_sglang_tent_backend` class substitution and private
  completion/release Hooks. Keep native bootstrap/rank logic. Optional failed-peer
  probing waits for S3's stable native liveness contract.
- vLLM: reuse native P/D after S5.1 proves lifecycle/layout equivalence. Fill real
  upstream gaps in focused PRs; then remove superseded custom handshake, request
  states and proxy code. Keep only examples needed to launch the chosen upstream
  router, with one tested production handoff path.
- Qualify cold/partial/full P/D plus cache reuse. Exactly one owner writes each
  destination range, commits DecodeReady and authorizes release. `MultiConnector`
  registration/order alone does not establish safe composition.
- Preserve producer events, destination generations and physical drain on abort,
  preemption, restart and partial submission. Close heterogeneous-GPU strict-output
  failures with matched native controls, not relaxed output assertions.

**Acceptance:** real transfer/output/drain evidence for both P and D, standalone
cache and composed modes. Retire old paths only in the same change that proves
the replacement; do not keep TE/TENT compatibility fallbacks or duplicate owners.

### S5.5 — Deployment matrix and upstream maintenance

- Track each engine separately: single instance; independent replicas; multiple
  instances sharing a Manager; dense/hybrid; P/D; attention DP, TP, PP, MoE EP and
  relevant combinations; homogeneous/heterogeneous P/D parallelism; same-host TCP,
  physical two-host TCP and RDMA. Label implemented, qualified, experimental and
  unsupported independently. Do not infer KV partitioning from EP world size.
- **Implementation open:** node-local Manager query fan-out for cross-host TP.
  The engine still owns rank agreement, legal common recovery boundaries, GPU
  allocation and execution collectives. Define stage-specific PP state/layout;
  reject unsupported resharding and state mappings explicitly.
- Qualify source incarnations, cancellation, sender/receiver budgets and transport
  failures on real hosts. RDMA needs visible NICs and physical counters; multiple
  ranks/processes on one GPU do not qualify multi-GPU or host-failure behavior.
- Qualify simultaneous engines, Manager/engine restart, shared DRAM/SSD budgets,
  containers, UDS/iceoryx2/CUDA IPC, PID visibility and resource drain. Shared
  service support does not establish cross-engine interchangeable cache bytes.
- Maintain upstream registration/docs/tests and a release upgrade gate. Submit
  lifecycle fixes and TENT support separately from the baseline cache entry.
  Record review links, maintainers, merge commit and first containing release.

**Acceptance:** an executable scenario with output, physical-transfer and drain
evidence for every claimed cell. The minimal upstream cache PR need not wait for
all topology/research cells; no untested cell becomes supported by that merge.

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

- Add receiver-driven admission using a bounded in-flight byte window **and**
  pacing. One node/physical-NIC/direction budget must include Manager READ and
  engine-process P/D WRITE; Manager-only admission misses the latter. Lease
  bounded credits in batches while keeping payload movement direct through TENT.
- Keep network credits, destination buffer lifetime and source export authority
  distinct. Return credits on verified receive/transport progress; retain memory
  until its final user drains. Credit timeout is never a memory-release proof.
- Schedule ready data using deadlines, remaining critical state/rank/layer work,
  per-instance fairness and aging. Pair receiver admission with source egress
  limits; a receiver cannot control other flows or all fabric bottlenecks. Reserve
  headroom for engine communication using measured physical interference.
- **Qualification open:** cross-medium selection and peer SSD under contention,
  cancellation and RDMA. Two-host TCP byte recovery does not qualify policy gain.
- **Deferred research:** router event normalization and pinned upstream router
  service integration; eviction externalities/replica risk; load-only versus
  overlap-only versus joint routing on one trace; active RDMA WRITE cache
  replication; topology slicing, endpoint pooling and alternate-rail retry;
  NIXL Mooncake-backend evaluation. These need a concrete scope decision and
  measured benefit. Current P/D WRITE is not cache replication. Do not introduce
  a new transport abstraction or deprecated KVBM to close the current plan.

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
- Preserve licenses and bundled upstream provenance. Keep heavy GPU/RDMA gates
  explicitly marked and distinguish engine support from runtime build success.
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
5. Updated completion-plan/support status and the next dependency. Passing unit tests does
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
[transport](transport.md), [state planning](state-planning.md),
[deployment](deployment.md), [P/D](pd.md) and [release](releases.md) documentation.
Existing historical reports describe their recorded revisions only.

## Consolidated documentation and retired work

The former root TODO, implementation handoff, communication sequence and roadmap
have been consolidated here. Their historical revision remains available at
[the S1 source snapshot](https://github.com/feichai0017/orbitkv/tree/ec3add9bcaf91a0171685ba42b5ae8330076c397).
All 70 open obligations were assigned to S2–S8: storage/codec execution and
performance to S4; transfer generations/revocation to S3; engine identity,
Radix/P-D/topology/container contracts to S5; cost/network admission and explicit
router/backend research to S6; retention semantics to S7; package/image/provenance
and recurring release qualification to S8. S2 retains metadata capacity/faults.

Retired proposals remain retired: a second shard Catalog, binary OrbitKV metadata
inside Mooncake, unused Python state facades, a generic region RPC without a
consumer, and duplicated backend/cost wrappers. Existing raw 2D DMA, bounded raw
parts, single-part layer events and local global indexes need extension or
qualification, not another implementation. The former direct engine-page SSD
and multi-writer assembly item is **implementation partial**, not just untested.

Historical result tables stay at immutable revisions or external evidence storage.
Current resource contracts remain in their owner documents. No compatibility
stub documents or duplicate execution queues replace the deleted files.
