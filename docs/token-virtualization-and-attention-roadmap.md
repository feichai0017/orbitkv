# Token Virtualization and Attention Expansion Roadmap

This roadmap starts from the live ABI8 architecture. Qualification status is
normative only in the [Capability Matrix](capability-matrix.md).

## Current checkpoint

The modular Rust core and exact 40-symbol C ABI8 wire are host-qualified L2.
They provide immutable snapshots, request fork, page-aligned Prefix ownership,
joint Full+SWA COW, detach actions, page-owned reclamation, and an independent
fixed-state checkpoint pool.

The ABI8 Python runtime, state-pool client, and SGLang Prefix adapter are
host-qualified L2. The sealed Prefix record provides scoped correctness only
for its manifest-bound engine profile and explicitly excludes fixed state. The
frozen ABI5 record remains historical and does not qualify this source.
Fixed-state pair verification and a later weight-backed diagnostic remain
independently unattested and below L4/performance qualification. OrbitKV's
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
host-blocking, with no asynchronous overlap. Generation-safe packed fork and
packed shared-tail COW append are host-tested through Rust, raw ABI8, and
Python FFI. Packed Prefix remains fail-closed. The packed COW work postdates
and does not inherit the sealed engine scope. An engine-neutral CUDA harness
provides component conformance for payloads, ordering, reuse, and drain. A
separate pinned engine record is preflight-bound, sealed, and `qualified=true`
only for scoped request-private Full relocation correctness and lifecycle. It
remains independently unattested and `performance_go=false`; capacity, general
speedup, and production qualification remain pending.

Opt-in request-private pressure telemetry is host-tested for consumed,
resident, request-reachable, and semantic-live bytes plus retention
amplification. No append-only, sealed, or qualified asynchronous GPU pressure
record is published; fixed-state bytes and shared Prefix/request-fork retention
amplification are excluded. The
separate `orbitkv-runtime` and `orbitkv-reference` wheels also provide an
engine-neutral SPI and reference external-arena adapter. The reference is not a
complete serving engine, and SGLang has not migrated to the SPI.

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

Status: **host L2 GO; scoped exact-source correctness for the sealed
manifest-bound Prefix profile; performance pending**.

Register an `OrbitKVPrefixCache` at the official SGLang `v0.5.17` cache seam.
Radix remains a token/digest/LRU index. It stores an opaque `PrefixLease`, not
page IDs, generations, free-list state, or CUDA tensors.

The first profile is deliberately narrow:

- eager, single-device token KV with a manifest-bound backend;
- Full and ordered Full+SWA retention;
- page-aligned publish and attach only;
- shared partial-tail divergence through exact COW; and
- overlap, Graph, speculation, disaggregation, remote/hierarchical cache, and
  multi-GPU disabled.

The sealed ABI8 record compares cold and warm paths, verifies matching
request outputs, and proves Prefix activity and final drain for this exact
boundary. Its timings are diagnostic and `performance_go=false`; it does not
qualify fixed state or a general SGLang replacement. See the
[sealed Prefix record](../results/h20-sglang-v0517-abi8-full-hybrid-20260823/README.md).

Expected benefits are fewer duplicated physical KV pages and less repeated
prefill work for warm prefixes. No performance or end-to-end memory benefit is
qualified.

## M3: Token table and exact relocation

Status: **core, C wire, Python wire, and eager SGLang adapter host L2 GO;
component conformance plus sealed clean-source scoped Full relocation
correctness/lifecycle qualification pass; independent hardware attestation and
performance qualification pending**.

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
mutation. Generation-safe packed request fork and repeated shared partial-tail
COW append on a packed root are now host/raw-ABI8/Python-FFI tested. Packed
Prefix operations remain unsupported and fail closed; packed COW still needs
separate engine qualification.

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
not implemented. An engine-neutral CUDA harness verifies an independent
payload oracle, real stream ordering, exact evacuation, ACK-gated generation
reuse, and final drain. The component result is not sealed L3/L4, performance,
or capacity qualification.

The pinned engine record is clean, preflight-bound, and sealed for scoped
request-private Full relocation correctness and lifecycle. It remains
`hardware_attested=false` and `performance_go=false`; capacity, end-to-end
memory, production, broader engine features, and asynchronous overlap remain
unqualified. See the
[sealed relocation qualification](../results/h20-sglang-v0517-abi8-token-relocation-20260825/README.md).

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

Status: **compiler L1 and ABI8 checkpoint core/wire L2 GO; a manifest-bound
request-private GDN+convolution frontend profile has scoped host implementation
and pair verification; replacement trigger, other families, independent
hardware attestation, and L4/performance
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

The pure-MLA host seam validates compiled latent/RoPE byte widths against
SGLang's real `MLATokenToKVPool` and exercises its combined-row copy API. DSA,
reduced-precision variants, distributed execution, and Hybrid Linear remain
outside that seam. The same-owner fixed-state copy production trigger remains
implementation work. The first bound frontend family compiles Full token KV
separately from request-private GDN recurrent and convolution state. Its adapter
accepts only fresh-prompt, Radix-disabled operation and fails closed unless the
backend, dtype, state layout, and cache settings match the compiled structural
contract. Prefix-state sharing and a fixed-state copy trigger are not admitted.

Other GDN profiles, KDA, ShortConv, and other linear-attention family bindings
are still implementation work. The bound profile excludes Prefix sharing,
speculation, overlap, Graph, and unified memory. Every fixed-state family needs
its own qualification and exact-byte or numerical-state oracle; Full KV
relocation or host fixed-state tests do not transfer qualification.

The scoped pair-verification record matches outputs, lifecycle, completion, and
final drain, but remains `qualified=false`, `hardware_attested=false`, and
`performance_go=false`. See the
[fixed-state pair-verification record](../results/h20-sglang-v0517-abi8-qwen35-fixed-state-pair-verification-20260823/README.md).

A later weight-backed execution is diagnostic-only because its source and
qualification preflight do not meet the release gate. See the
[diagnostic archive](../results/h20-sglang-v0517-abi8-qwen38-fp8-diagnostic-20260824/README.md).

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
