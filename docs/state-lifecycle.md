# State Lifetime and Reclamation

This document defines how the OrbitKV product reasons about attention-state
lifetime. The [Capability Matrix](capability-matrix.md) remains the normative
implemented and qualified boundary.

OrbitKV is one complete pinned SGLang product with a Rust `RuntimeSession`, not
a collection of interchangeable data planes. SGLang owns scheduling, tensors,
kernels, and CUDA execution. OrbitKV owns the admitted lifecycle, physical-page
selection, semantic and execution frontiers, retirement, acknowledgement, and
safe reuse.

## Scope

Attention state is not one homogeneous byte array. Different state classes have
different address programs, future-read sets, and retirement conditions. The
compiler preserves those differences so the runtime can reclaim state without
pretending that every component is token-addressable K/V.

This architecture distinguishes:

- the semantics that determine whether state can be read again;
- the physical plan that maps live state to pages or fixed-width slots;
- SGLang execution that reads and writes the actual tensors; and
- the evidence protocol that makes a detached resource reusable.

The former neutral Python runtime/reference adapter and structured data-plane
path are removed history. They are not current reclamation backends.

## State taxonomy

| State class | Logical lifetime | Physical strategy | Relocation meaning |
| --- | --- | --- | --- |
| Full token KV | All prior retained tokens remain visible | Append-only paged state, optionally shared through Prefix snapshots | Byte-exact token relocation is meaningful, but the product path is still migrating |
| Sliding token KV | Only the admitted recent window remains visible | Periodic or generation-indexed slots derived from window and page geometry | Ordinary semantic retirement is primary; relocation must not widen the window |
| Chunked token KV | State is visible within an admitted epoch/chunk | Request-private resettable arena | Epoch reset, not generic compaction, determines semantic death |
| Latent KV and RoPE components | Token-addressable component rows with explicit geometry | Component-aware paged state | A move must preserve the combined component contract exactly |
| Recurrent state | Request-owned evolving checkpoint | Generation-checked fixed-width slot | Not a token move; replacement publishes a new checkpoint generation |
| Convolution state | Request-owned finite history | Fixed-width state or bounded ring | Not a token move; retirement follows checkpoint semantics |

Head partitions, block domains, pinned ranges, periodic ranges, and resettable
arenas must remain explicit in the compiled plan. Lowering every state family to
the widest Full-token lifetime would retain semantically dead data and erase the
reason to compile attention state.

## Logical token contract

For token-addressable state, a logical token identity is stable across physical
movement. A canonical view maps each logical token to one of three
dispositions:

- `Retained`: the admitted semantics may read it again and an exact physical
  location must exist;
- `SemanticallyDead`: a checked semantic proof shows it cannot be read again;
  or
- `PolicyEvicted`: an explicit lossy policy and quality contract authorize its
  removal.

Physical compaction never changes this set. It may move `Retained` bytes, but it
cannot reclassify a token or make a dead token visible. Unknown or incomplete
semantic evidence fails closed.

SGLang's sequence length and model position remain execution semantics; they
must not be rewritten to match a compacted physical layout. The bridge changes
only the checked physical mappings consumed by SGLang kernels.

## Semantic Frontier

The Semantic Frontier answers: **which state is unreachable by every future
query admitted by this plan?** It can advance because of:

- ordinary request release;
- page-aligned Prefix eviction after all references leave;
- a Sliding boundary that excludes an old logical range;
- an exact Chunked epoch boundary;
- a proved token disposition; or
- replacement of a request-owned fixed-state checkpoint.

The frontier is plan-specific. A host counter, free-list pressure, elapsed time,
or successful copy is not semantic-death evidence. Full attention ordinarily
does not advance a token's semantic frontier before request/Prefix release.

## Execution Frontier

The Execution Frontier answers: **which submitted SGLang operations have
finished reading or writing the resource?** SGLang owns the current CUDA
streams and events. The bridge binds a real event to the exact native session
ticket, observes it, and advances a monotonic completion value.

Rust validates session ownership, batch association, completion-domain shape,
and monotonicity. It does not create or authenticate the CUDA fence. This keeps
execution provenance with the component that actually launches kernels while
allowing the native lifecycle to enforce safe reuse.

## Proof-carrying reclamation

A physical page or fixed-state slot becomes reusable only when both frontiers
permit it:

```text
Reusable(resource) = SemanticDead(resource, Fs)
                  && ExecutionComplete(resource, Fe)
                  && MirrorCleanupConfirmed(resource)
                  && RetirementAcknowledged(resource)
```

The runtime emits generation-checked retirement identities. The bridge must
confirm the exact ordered receipts after applying the specified SGLang table or
tensor cleanup. Rust then ACKs retirement and increments the generation when a
physical slot is reused.

This protocol prevents four common errors:

- freeing a page because it left a logical snapshot while a kernel still reads
  it;
- treating a completed copy as proof that the source is semantically dead;
- reusing a page while a stale SGLang mirror still addresses it; and
- accepting a receipt for the wrong page generation or operation.

## Full and shared Prefix lifetime

Full token KV is append-only during an ordinary request. Pages may also carry
request, Prefix, reader, and writer references. A shared Prefix holds its native
snapshot independently of any one request. Eviction establishes semantic death
only after the relevant Prefix and reader references are gone.

Request fork and shared partial-tail extension use copy-on-write. The old root
remains immutable; Rust prepares destination pages and copy intents; SGLang
performs the actual tensor copy; and Rust publishes a new root after completion.
Atomic publish-and-release transfers the final request reference into the Prefix
before request recycling.

Full retention has no bounded-retention memory advantage to demonstrate on a
normal decode workload. The latest earlier-wire Full accelerator record
establishes narrow exact-source correctness and clean drain but shows overhead
rather than a speed or memory benefit. It does not qualify the live wire.

## Ordered Full+Sliding lifetime

The ordered two-class plan keeps Full and Sliding state in distinct arenas. The
class order is part of admission. Rust selects both physical bindings; the
bridge maintains the `ReqToToken` effects and SGLang's
`full_to_swa_index_mapping` as a checked Full-to-Sliding LUT.

Shared Prefix operations and joint COW must cover both classes consistently. A
Sliding page can become semantically dead as the window advances, while the
corresponding Full page remains live. Reclamation therefore uses class-specific
certificates and cannot infer one class's lifetime from the other.

The profile is implemented and host-tested. Its current real-device correctness,
actual Sliding retirement/reuse trace, capacity, and performance qualification
remain pending.

## Pure Sliding lifetime

Pure Sliding uses a request-private session and rejects every Prefix operation.
The compiler derives the periodic slot count and retirement program from the
window and page geometry. `ReqToToken` remains a physical-address mirror; there
is no Full-to-Sliding LUT.

Host tests cover periodic allocation, semantic retirement, Sliding-leaf COW,
mirror cleanup, release, ACK-gated generation reuse, and final drain. Those
tests do not establish actual device execution. A current product qualification
must observe the Sliding activity as applicable, cross a retirement boundary,
tie retired page generations to completion and ACK, and prove later safe reuse.
That run is still pending.

## Exact Chunked lifetime

The admitted Chunked profile consumes canonical Retention IR and lowers one
whole-domain token class to a request-private resettable arena. Logical columns
remain absolute across epochs. At `EpochEnd`, the bridge clears exactly the old
epoch mappings; Rust retires their pages only after confirmed execution and
reuses them only after exact ACK.

The current pre-allocation validator checks pinned configuration indicators for
the required chunk-local attention and safe scheduling mode and fails closed on
drift. These host checks do not observe real kernel or scheduler execution.
There is no current real-accelerator, released-model, performance, or
complete-engine qualification for this profile.

## Latent and fixed-state lifetime

Full latent KV is token-addressable but uses explicit latent/RoPE geometry. It
has a request-private native session and host-tested lifecycle. Prefix sharing,
broader latent profiles, relocation engine execution, and device qualification
remain pending.

Recurrent and convolution state uses an independent checkpoint pool rather than
the token manager. Prepare, submit, completion, replacement retirement, release,
ACK, abort, and quarantine are host-tested. Token and fixed-state handles do not
yet share one atomic transaction, so failure handling is containment rather than
cross-handle rollback. The admitted native-session product profiles currently
exclude fixed state.

## Token relocation transaction

Relocation is exact physical movement of `Retained` token state. It is not
compression and does not prove a capacity benefit merely because source pages
are reclaimed. A correct batch transaction requires:

1. read canonical token views and versions;
2. commit explicit semantic/policy disposition updates;
3. prepare one ordered relocation plan for the complete scheduler batch;
4. validate every source, destination, token, class, and mirror effect;
5. let SGLang execute the byte-exact CUDA copies;
6. bind completion evidence to the submitted operation;
7. publish all request heads and engine mirrors;
8. retire the exact source generations; and
9. ACK the batch before any source generation is reused.

The batch is collective but not end-to-end rollback-capable. After a disposition
mark succeeds, a later observed or uncertain failure must fail-stop or
quarantine; it cannot restore an invented pre-mark snapshot. An explicitly
unobserved prepared copy may be aborted safely.

Rust transaction code, opaque `RuntimeSession` wire operations, and a SGLang
CUDA copy path exist, and a separate device harness exercises component
conformance. Product integration is still migrating to that single session
route. The current claim is host-verified migration state only, not native-
session E2E, capacity, memory saving, performance, or production qualification.
Archived relocation records qualify only their own source closures.

## Pressure and benefit metrics

The relevant memory numerator is physical resident state, including padding,
sharing, temporary copy headroom, and state trapped by still-live neighbors. The
semantic denominator is the state that the admitted future queries can still
read. Their ratio is retention amplification.

An implementation may reclaim pages without reducing configured tensor
reservation, or may reduce resident state while increasing latency. Therefore a
benefit claim needs matched stock/manager workloads and, separately:

- token/logit correctness for the same semantic policy;
- physical and semantic memory census;
- capacity under the same allocation budget;
- copy and reclamation overhead;
- latency and throughput distributions; and
- long-running reuse and pressure behavior.

No current OrbitKV result establishes a general memory, capacity, or speed
benefit.

## Qualification boundary

| Capability | Current product status |
| --- | --- |
| Full/shared-Prefix lifecycle | Host-tested; latest earlier-wire accelerator record has narrow exact-source correctness and no benefit; live-wire device and production qualification pending |
| Ordered Full+Sliding lifecycle | Host-tested; current real-device qualification pending |
| Pure Sliding lifecycle | Host-tested; current real-device retirement/reuse qualification pending |
| Exact Chunked lifecycle | Host-tested; real kernel/scheduler and device qualification pending |
| Full latent KV lifecycle | Host-tested; device/engine qualification pending |
| Token relocation | Host-verified migration state; device component harness only; no current product E2E |
| Fixed-state checkpoints | Host-tested independent pool; not one atomic native-session product profile |
| Overlap, graph replay, distributed or remote state | Not qualified |

Historical exact model, device, engine, and protocol identities remain in the
[Results Index](../results/README.md) as provenance and do not widen this table.
