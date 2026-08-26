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
amplification. A real single-device diagnostic has run, but no append-only,
sealed, or qualified pressure record is published and no allocator, performance,
or capacity claim follows. Fixed-state bytes and shared Prefix/request-fork
retention amplification are excluded. The
separate `orbitkv-runtime` and `orbitkv-reference` wheels also provide an
engine-neutral SPI and reference external-arena adapter. The reference is not a
complete serving engine. SGLang can opt into the two-phase external-write
protocol only for its scoped eager BF16/NHD `token_kv` subset; broader
migration and qualification remain roadmap work.

## Definition of done: a compiled hybrid-attention manager

OrbitKV becomes a compiled manager when the compiler output, rather than an
engine adapter's model- or layout-specific branches, is the executable source
of truth for allocation, addressing, mutation, completion, and reclamation.
Code that merely recognizes more model names does not satisfy this definition.

The required milestones are:

| Phase | Required result | Exit gate |
| --- | --- | --- |
| P1: executable manifest | Emit one versioned artifact containing token and fixed-state classes, component geometry, physical layouts, address programs, lifecycle operations, capability requirements, and a stable fingerprint | SGLang consumes the compiled artifact directly; it no longer reparses separate manager and uncompiled state inputs or reconstructs their relationship |
| P2: complete physical executor | Execute every admitted retention domain and storage kind, including whole-token Full/SWA, component-aware MLA, head/region partitions, pinned plus sliding regions, and resettable/chunked state | The compiler cannot emit a plan that the selected manager/adapter later rejects; unsupported opcodes fail before arena allocation |
| P3: unified lifecycle schedule | Compile append, COW, relocation, fixed-state replacement, semantic-death proofs, stream dependencies, completion, mirror publication, and reuse into one transaction graph | External append and relocation can coexist, token and fixed state share an explicit commit group, and injected failures prove no premature reuse or split publication |
| P4: cost-based physical planning | Select append-only, bounded ring, packed, shared Prefix, or relocation plans from state geometry, workload pressure, copy cost, and backend capabilities | Two semantically equivalent plans can be compared by a documented cost model, and the selected plan is recorded in telemetry |
| P5: matched benefit qualification | Run released checkpoints against the same engine, kernels, capacity, prompts, batching, and sampling configuration | A sealed multi-epoch record passes the correctness, memory/capacity, latency, throughput, pressure, and long-running-reuse gates below |

P1 through P3 are the architectural threshold for calling OrbitKV a compiled
hybrid-attention manager. P4 and P5 are the additional threshold for claiming
that compilation produces a real deployment benefit. The live tree has the
semantic compiler and several individually executable mechanisms, but has not
yet crossed either complete threshold.

### Family-specific lowering targets

| State family | Compiled physical strategy | Current gap | Expected source of benefit |
| --- | --- | --- | --- |
| Full MHA/GQA/MQA | Append-only paged KV with Prefix sharing and COW | Structured append exists, but remains a scoped SGLang path | Prefix reuse and low control overhead; Full attention alone has no semantic-death memory reduction |
| SWA/local | Bounded cyclic or generation-indexed slots derived from the window and page size | Full+SWA is implemented, but multi-class execution is still adapter-shaped | Bounded resident KV and prompt-independent decode capacity |
| Sink+window, dilated, chunked, and per-head windows | Separate lifetime-normal-form regions with independent address and retirement programs | Compiler representations exist, while the canonical manager rejects several non-whole-domain forms | Lower retention amplification by avoiding widest-window allocation for every head or region |
| MLA | Component-aware latent and RoPE token rows | Compiler geometry and a host relocation seam exist; structured append and engine qualification are pending | Smaller token records and exact relocation without pretending the components are ordinary K/V |
| Mamba/GDN/KDA/linear plus convolution | Request-owned recurrent checkpoints and finite convolution rings | One GDN/convolution profile is connected; component-specific pools, replacement triggers, and joint commit are pending | Bound state independently of context length without applying token relocation to non-token state |
| Sparse or policy-selected retention | Token dispositions with exact semantic proof or an explicit lossy quality contract | No general selector/backend contract is qualified | Reclaim non-contiguous holes where relocation can reduce physical pages |

### Benefit GO gates

A result is a real systems benefit only when a matched, sealed experiment
satisfies all applicable gates:

- lossless plans match stock output tokens and, where practical, logits; lossy
  policies carry and evaluate a separate quality contract;
- the unoptimized Full baseline keeps throughput and inter-token-latency tax at
  or below 2%, so ownership machinery does not consume the expected gain;
- an optimized hybrid workload demonstrates either at least 15% lower measured
  peak attention-state bytes or at least 15% more admitted requests/tokens at a
  fixed memory limit, with no more than 3% throughput regression;
- alternatively, at equal memory and semantics, throughput improves by at least
  5% without more than 3% regression in TTFT or p95 inter-token latency;
- pressure telemetry reports physical resident bytes, semantically live bytes,
  temporary relocation headroom, and retention amplification instead of only
  configured tensor capacity; and
- a dynamic arrival/departure run crosses repeated reuse generations with zero
  stale identity, leak, quarantine, or fail-stop events and bounded host/event
  metadata.

These are project promotion gates, not claims about the current implementation.
They deliberately target non-contiguous or heterogeneous liveness, where a
compiler can improve placement; dense Full attention and an already compact
sliding window are control cases rather than expected memory wins.

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
