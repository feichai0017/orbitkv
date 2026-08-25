# Token Virtualization and Attention Expansion Roadmap

This roadmap starts from the live ABI8 architecture. Qualification status is
normative only in the [Capability Matrix](capability-matrix.md).

## Current checkpoint

The modular Rust core and exact 40-symbol C ABI8 wire are host-qualified L2.
They provide immutable snapshots, request fork, page-aligned Prefix ownership,
joint Full+SWA COW, detach actions, page-owned reclamation, and an independent
fixed-state checkpoint pool.

The ABI8 Python runtime, state-pool client, and SGLang Prefix adapter are
host-qualified L2. The current sealed ABI8 H20 record provides scoped Prefix
correctness only for Qwen2.5 Full and GPT-OSS Full+SWA; it has
`performance_go=false`, reports an observed configured arena reservation
difference of **0%** rather than an end-to-end memory saving, and explicitly
excludes fixed state. The frozen ABI5-v5 record remains historical scoped L4
correctness only and does not qualify this source.
The separate Qwen3.5 archive completes scoped fixed-state pair verification
from recorded H20 runtime snapshots, including outputs, lifecycle counters,
stream/event completion, and final drain. It remains independently unattested,
`qualified=false`, `hardware_attested=false`, and
`performance_go=false`, so it does not add L4 or performance qualification.
A newer Qwen3.8-27B-FP8 diagnostic adds eight passing, token-exact pairs on a
recorded H20, but its dirty source, missing qualification preflight, and
`qualified=false`/`hardware_attested=false` flags keep it below L4. OrbitKV's
target remains an attention-state compiler plus transactional ownership
runtime, not a full SGLang replacement or a mature L5 system.
For token relocation specifically, the ABI8-preserving runtime now executes one
multi-request scheduler batch with one mark, prepare, submit, and complete call
followed by one aggregate registry commit and one aggregate request-head
replacement. The plugin flattens all moves into one backend move and one event,
validates every mirror plan before any write, applies non-rollback-atomic mirror
writes, and performs one batch ACK; the scalar entry point is a singleton compatibility
wrapper. The ABI8 core and Python runtime also repeat
append-mark-full-evacuation-ACK for one request-private Full class, and the
SGLang periodic trigger is host-tested across two boundaries. Post-mark failures
are fail-stop and non-rollback. Event completion is currently eager and
host-blocking, with no asynchronous overlap. Packed Prefix, fork, and shared
partial-tail COW remain fail-closed. A new engine-neutral CUDA opaque-byte
harness now passes all seven H20 component cases, including two real append,
copy, and consumer-stream cycles at B1/B4/B32. A separate pinned SGLang
Qwen2.5-0.5B diagnostic passes all 8/8 Naive/Relocate B1/B4 pairs with exact
tokens, expected counters, complete drain, and zero failures. That engine record
is unsealed, dirty-source, independently unattested, `diagnostic_only`,
`qualified=false`, and `performance_go=false`; formal L3/L4, performance, and
capacity qualification remain pending.

## Why the module split is a roadmap prerequisite

Relocation and Graph add two new forms of concurrency: physical placement can
change without logical identity changing, and captured work can outlive the
host call that described it. Those rules must not be mixed into one manager or
adapter file.

The live ownership boundaries are:

```text
identity/arena          who and which physical generation
persistent snapshot    immutable logical view
append transaction     private candidate and publication
Prefix                 shared snapshot residency and attachment
reclamation            final-reference proof and reuse
state checkpoint       request-owned fixed-width replace / retire / ACK
test model/oracles      independent full-scan correctness model
```

Python mirrors the same separation across `ffi/`, `runtime/`, and `plugin/`.
CI limits production Rust/Python modules to 1,500 lines so relocation and Graph
cannot silently recreate the former monoliths.

## Correctness vocabulary

Token virtualization must keep semantic liveness separate from physical
placement and from policy-driven quality changes:

```text
TokenDisposition =
    SemanticallyDead { compiler_proof }
  | PolicyEvicted { policy_id, policy_version, quality_contract }
  | Retained
```

`SemanticallyDead` is lossless for the qualified attention relation.
`PolicyEvicted` is explicitly lossy and requires its own model-quality
contract. Relocation must preserve every byte of every `Retained` token.

The term **compaction** below means token-exact K/V relocation and
defragmentation. It is not quantization, a codec, low-rank compression, or a
same-capacity memory result.

## M1: Freeze the ABI8 Python/runtime boundary

Status: **L2 GO**.

- complete ctypes parity with `orbitkv.h` and ABI version 8;
- load exactly the 40 allowed symbols and reject all compatibility aliases;
- freeze 73 ctypes layouts, including the independent state-pool records;
- keep FFI layout/workspace code separate from lifecycle journals;
- make snapshot heads, materialized views, detach actions, and reclamation
  receipts generation checked in Python;
- preflight complete batches and all mirror mutations before commit; and
- qualify malformed spans, stale leases, short buffers, fail-stop, and
  quarantine paths.

Exit gate: the ABI8 Python runtime is L2 against the exact release library.

## M2: Integrate SGLang Prefix ownership

Status: **host L2 GO; scoped exact-source H20 correctness for the sealed
Qwen2.5 Full and GPT-OSS Full+SWA boundary; performance pending**.

Register an `OrbitKVPrefixCache` at the official SGLang `v0.5.17` cache seam.
Radix remains a token/digest/LRU index. It stores an opaque `PrefixLease`, not
page IDs, generations, free-list state, or CUDA tensors.

The first profile is deliberately narrow:

- eager, single GPU, page16 BF16 NHD;
- Qwen2.5 Full and GPT-OSS ordered Full+SWA;
- page-aligned publish and attach only;
- shared partial-tail divergence through exact COW; and
- overlap, Graph, speculation, disaggregation, remote/hierarchical cache, and
  multi-GPU disabled.

The sealed ABI8 H20 record compares cold and warm paths, verifies matching
request outputs, and proves Prefix activity and final drain for this exact
boundary. Its timings are diagnostic and `performance_go=false`; it does not
qualify Qwen3.5, fixed state, or a general SGLang replacement.

Expected benefits are fewer duplicated physical KV pages and less repeated
prefill work for warm prefixes. No performance benefit is qualified. The
compared runs reserve equal configured KV tensor arenas, so the observed
configured arena reservation difference is **0%**; this is not an end-to-end
memory-saving result.

## M3: Token table and exact relocation

Status: **core, C wire, Python wire, and eager SGLang adapter host L2 GO; H20
component conformance and a recorded-device SGLang diagnostic pass; sealed
L3/L4 and performance qualification pending**.

The canonical-manager surface retains stable logical token IDs, canonical
disposition batches, and class-specific physical placement without changing
logical token identity:

```text
TokenPlacement {
    token_id,
    disposition,
    location: Option<{ page: PageLease, backend_index, offset }>,
}
```

The host scheduler-batch transaction now:

1. freeze the ordered set of immutable snapshot heads and candidate mirrors;
2. mark dispositions once for the complete request set;
3. prepare once, reserve exact destination `PageLease` values from aggregate
   bounded headroom, and pin exact source generations;
4. flatten ordered token-copy intents into one backend move and one completion
   event while preserving per-request receipt groups;
5. submit once and complete once after validating backend copy receipts;
6. validate the complete returned batch, then perform one aggregate page-registry
   commit followed by one aggregate request-head replacement;
7. validate all SGLang mirror plans before the first write, then apply them
   within the scheduler-batch publication; a write-time failure fail-stops the
   runtime and does not roll back earlier mirror writes; and
8. send one batch ACK, retiring sources only after every old snapshot and
   reader pin is gone.

The scalar relocation entry point delegates to this ABI8 batch path as a
singleton compatibility wrapper. This is collective fail-stop containment, not
rollback: once the mark succeeds, later prepare/copy/submit/complete/publication/
mirror/ACK failures or uncertain returns stop the runtime without restoring the
old disposition snapshot or heads.

The proven repeated subset is one private Full class with full evacuation:
append to a dense or previously packed request, mark dispositions, relocate,
publish, exact-ACK the retired generations, and repeat. The SGLang adapter
stores an integer next-reclamation boundary rather than a one-shot flag, and
host tests exercise two boundaries for both Naive and Relocate policy modes.
It rejects a missed boundary and a nonempty Prefix mirror before manager
mutation. Prefix publication and request fork after packed publication, plus
shared partial-tail COW append on a packed root, remain unsupported and fail
closed.

Global invariants are token conservation, unique placement, completion
visibility, snapshot isolation, generation safety, and deferred source reuse.
Physical generation reuse requires semantic unreachability (the Semantic
Frontier), execution completion (the Execution Frontier), and exact backend
ACK; none of the three alone authorizes reuse.
Unknown or mismatched copy receipts quarantine destinations and fail-stop the
runtime. The eager adapter now contains Full KV copy, CUDA stream/event
ordering, compact ReqToToken publication, and class-specific Full-to-SWA LUT
handling. Its single completion event is synchronized by the host before
publication; asynchronous consumer-stream waiting and copy/compute overlap are
not implemented. An engine-neutral CUDA harness now passes seven H20 cases and
encodes B1/B4/B32, two cycles on the same requests/cursors, an independent live-token/payload oracle,
257-byte coordinate-bearing records, distinct non-default append/copy/consumer
streams, exact 3-page-to-2-page evacuation with 24 moves per request/cycle,
event-ordered byte checks, ACK-gated same-page/higher-generation reuse, and
final drain. The component result is not sealed L3/L4, performance, or capacity
qualification.

The model-level diagnostic pins official SGLang `v0.5.17`, Qwen2.5-0.5B,
page16 BF16 NHD Full attention, eager single-GPU execution, and FlashInfer in
both Naive and Relocate modes. Four alternating-order epochs at B1/B4 produce
8/8 token-exact pairs, five iterations/process, two reclamation rounds per
iteration, expected relocation/reclaimed-page counters, complete drain, and
zero failures. Iteration 0 is excluded, leaving 16 hot samples per mode/group.

| Case | Relocate throughput delta | Mean latency delta | p95 latency delta |
| --- | ---: | ---: | ---: |
| B1 | +7.629% | -7.088% | -22.844% |
| B4 | -1.232% | +1.247% | +2.880% |

B1 has material inter-epoch jitter and B4 is slightly slower. The archive
therefore remains `performance_go=false`; it is also `diagnostic_only`,
unsealed, dirty-source, independently unattested, and `qualified=false`. It
makes no capacity or end-to-end memory claim. A relocate-only FA3 smoke passed,
but sparse Naive+FA3 is invalid and fails closed, so FlashInfer is the paired
same-policy oracle. The remaining gates are clean, preflight-bound, sealed,
independently attested L3/L4 qualification and asynchronous overlap.

[Qwen2.5-0.5B relocation diagnostic archive](../results/h20-sglang-v0517-token-relocation-diagnostic-20260825/README.md)

Relocation should run only when `source_pages > destination_pages` after
accounting for temporary destination headroom. It should not scan every token
on every decode step; live-slot counts and compaction candidates must be
incremental.

The likely benefit is low for an already dense contiguous sliding window,
which has only boundary slack. Evaluation should target non-contiguous
liveness: sink-plus-window, sparse/heavy-hitter policies, lifetime-normalized
classes, and private suffixes around protected Prefix pages. Published vToken
block-reduction numbers must not be projected onto OrbitKV before matched
experiments exist.

## M3b: Heterogeneous state backends

Status: **compiler L1 and ABI8 checkpoint core/wire L2 GO; strict normalized
official Qwen3.5-0.8B request-private GDN+convolution has scoped host
implementation/evidence and scoped pair verification from recorded H20 runtime
snapshots; replacement trigger, other families, independent hardware
attestation, and L4/performance
qualification pending**.

The heterogeneous compiler separates token KV, MLA latent plus RoPE
components, recurrent Mamba/GDN/KDA/linear state, and finite convolution state.
Its token-manager projection includes only token-addressable classes. MLA keeps
component geometry for independent copies; recurrent and convolution state use
a generation-checked fixed-width checkpoint transaction with no TokenMove
surface.
The runtime policy admits GDN/convolution structurally from the compiled plan
and live tensor geometry. Each evidence claim remains model-specific and must
pin the exact checkpoint and backend contract.

The independent ABI8 state-pool surface provides atomic prepare, submit,
completion-gated publish, replace retirement, release retirement, ACK, abort,
current-owner lookup, census, and fail-stop quarantine. Its identities and
wire records are separate from token IDs, page leases, and TokenMove. That host
protocol is connected to a restricted SGLang request-owned seam: the native
Mamba free list is replaced by a census-only facade, state leases map to
physical slot `slot_id + 1`, request allocation and initial
`MambaPool.clear_slots` execute before the first forward, the forward completion
event is propagated, and release waits before retire/clear/exact-ACK. Same-owner
`MambaPool.copy_from` replacement is
covered only by coordinator and real-CPU-tensor host tests; its production
trigger remains pending.
The token manager and fixed-state pool remain independent handles: a failure
between their commits has fail-stop containment only, with neither atomic joint
commit nor cross-handle rollback. A future unified transaction is required
before claiming cross-state atomicity.

The pure-MLA host seam now validates compiled latent/RoPE byte widths against
SGLang's real `MLATokenToKVPool` and exercises its combined-row copy API. It is
limited to BF16 Full retention, page16, eager, single GPU, without DSA, FP4, or
DCP. The same-owner fixed-state copy production trigger remains implementation
work. The first bound family is
the strict normalized official `Qwen/Qwen3.5-0.8B` profile: six Full layers use
token KV and the other 18 layers use request-private FP32 GDN recurrent state
plus BF16 convolution history. Its adapter accepts only fresh-prompt,
Radix-disabled operation and fails closed unless Full attention uses FA3, all
general/prefill/decode linear-attention selectors use Triton, temporal state is
actually FP32, and the Mamba Radix strategy is `no_buffer`. Prefix-state
sharing and a fixed-state copy trigger are not admitted by this production
profile.

Other GDN profiles, KDA, ShortConv, and other linear-attention family bindings
are still implementation work. The Qwen3.5 profile is restricted to eager,
single-GPU operation without ReplaySSM, int8 checkpoints, extra/ping-pong
buffers, speculation, overlap, Graph, or unified memory. The MLA path, this
Qwen3.5 path, and every later fixed-state family need their own qualification
and exact-byte or numerical-state oracle. Passing Full KV relocation or host
fixed-state tests does not qualify any of them.

The Qwen3.5 scoped pair-verification step is now complete from recorded H20
runtime snapshots for official
`Qwen/Qwen3.5-0.8B` on official SGLang `v0.5.17` commit
`29481685462732237d80d86076d6563e1f658102`. It records page16 BF16 NHD,
eager Full FA3 plus Triton linear backends, fresh prompts, disabled Radix, and
`cached_tokens=0` on one H20 snapshot. B1 uses one request, state capacity two,
and one iteration; B4 uses four requests, state capacity four, and five
iterations. Three epochs produce six passing stock/manager pairs with equal
output-token totals, exact token/fixed-state lifecycle and CUDA event
completion, and final drain. Fixed-state prepare/clear/retire/ACK counts are one
per B1 epoch
and 20 per B4 epoch; copies are zero.

The timing result is negative: B1 manager 1.5898847853 s versus stock
1.5818591726 s is +0.507353%, and B4 manager 0.9498731474 s versus stock
0.9156339097 s is +3.739403%. Therefore `performance_go=false`. The observed
configured arena reservation difference is **0%**; this is not a
qualified end-to-end memory-saving result. The archive also records
`qualified=false` and `hardware_attested=false`; the H20 runtime snapshots,
with UUID prefix `GPU-3a35…`, are independently unattested. The remaining gate is L4
qualification, not another claim of implementation or scoped pair execution.

[Qwen3.5 H20 fixed-state pair-verification record](../results/h20-sglang-v0517-abi8-qwen35-fixed-state-pair-verification-20260823/README.md)

The Qwen3.8-27B-FP8 diagnostic pair execution is no longer pending. The run
declares `Qwen/Qwen3.8-27B-FP8` repository revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`; its raw records bind the
downloaded config, index, and all 66 shards by hash and byte count
(30,866,866,928 bytes and 1,606 tensors), while repository provenance is not
independently online-attested. It records official SGLang `v0.5.17` revision
`29481685462732237d80d86076d6563e1f658102`; and NVIDIA H20 UUID
`GPU-3a35e57b-fc54-5620-56ee-deaf5a9c40d3`. Both modes explicitly set
`fp8_gemm_runner_backend=triton` and run eager single-GPU page16 BF16 NHD KV.

Four epochs balance ordering completely: 1/3 manager to stock and 2/4 stock
to manager. Each epoch contains one B1 and one B4 pair, yielding eight pairs
total; every process runs five iterations. All eight pairs pass the verifier
and match tokens exactly; final manager census and failure/fail-stop
counts are zero. Hot statistics discard iteration 0 per process and therefore
contain 16 samples per mode/batch. B1 stock/manager mean is
2.6296723178/2.8031814888 s, median 2.5374011379/2.6451683380 s, and p95
3.0721712420/3.3369778013 s: latency is +6.5981% and throughput -6.1897%.
Its epoch deltas (+2.7822%, +0.7129%, +26.3407%, -1.9835%) show clear jitter
and an outlier, so no positive claim follows. B4 mean is
2.9576119229/3.0395950049 s, median 2.9602836296/3.0345056280 s, and p95
3.0001941137/3.0800264925 s: latency is +2.7719%, throughput -2.6972%, and
epoch deltas are +1.8441%, +2.8850%, +5.0070%, +1.3885%.

This is diagnostic only. Manager and stock have the same configured
tensor-arena capacity and reported KV-cache reservation, an observed configured
arena reservation difference of **0%**; this is not a qualified end-to-end
memory-saving result. The source is dirty and the record is `sealed=false`,
`preflight_bound=false`, `hardware_attested=false`, `qualified=false`, and
`performance_go=false`. The
default-auto DeepGEMM path did load all 66/66 shards, but lengthy precompilation
was externally terminated before E2E completion. The next gate is a clean,
preflighted, independently attested qualification run, not model download or a
first E2E diagnostic.

[Qwen3.8-27B-FP8 diagnostic archive](../results/h20-sglang-v0517-abi8-qwen38-fp8-diagnostic-20260824/README.md)

## M4: Multiple completion domains and CUDA Graph

Status: **pending after relocation correctness**.

Graph compatibility requires more than stable tensor addresses:

- fixed-address device descriptor or slot-table storage;
- generation-checked descriptor patches outside captured kernels;
- per-stream completion domains for forward, copy, and replay;
- replay-scoped reader pins separate from `GraphExec` lifetime;
- cancellation and unknown-launch handling; and
- a wait on the actual consuming stream before replay sees a new placement.

The current relocation backend remains eager and host-blocking: it synchronizes
the copy event before publication. It does not yet install an asynchronous wait
on the consumer stream or overlap copies with sampling, CPU scheduling, or
compute. Only after that path is implemented and qualified should overlap be
considered; copies should not overlap memory-bandwidth-bound attention by
default, and profiling must determine the policy.

## M5: Speculation and branching

Status: **pending**.

Use immutable snapshot heads as branch roots. Each branch receives private
write deltas and either atomically publishes or aborts. A branch may share
sealed pages but cannot evict, relocate, or mutate a sibling's placement.

Qualification must cover accepted/rejected token boundaries, partial-tail
COW, rollback, cancellation, beam fork/release storms, stale expected heads,
and delayed copy/completion events.

## M6: Multi-GPU and disaggregation

Status: **pending**.

Each TP shard keeps generation-checked local placement for common logical token
IDs. Remote transfer is a separate transaction with source/destination leases,
codec identity, copy and network completion receipts, failure recovery, and
admission cost. Same-GPU relocation evidence cannot qualify fabric transfer.

## Qualification required at every milestone

- property and fault traces for all new transitions;
- exact-source startup and unsupported-mode rejection;
- released-model logits or deterministic-token comparison;
- manager and engine memory census, including padding and temporary headroom;
- fresh-process paired TTFT, ITL, throughput, p95/p99, and CPU/GPU profiles;
- long-running dynamic arrival/departure and pressure tests; and
- an append-only manifest binding source, ABI, engine, dependencies, hardware,
  commands, outputs, and hashes.

No memory or speed claim may compare different retention decisions,
advertised capacities, model/kernel profiles, or Full attention against SWA.
