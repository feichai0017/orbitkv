# Token Reclamation and Hybrid-State Architecture

The normative shipped capability boundary remains capability-matrix.md. This
document specifies the ABI8 implementation and qualification contract.
Code existence is not hardware or performance evidence.

## Scope

OrbitKV adopts the virtualization boundary from vToken (arXiv:2608.13263):
logical token liveness is independent of physical page placement, and retained
K/V can be repacked byte-for-byte before emptied pages are reclaimed. OrbitKV
does not inherit the paper's vLLM implementation or reported performance.
At the higher level, OrbitKV is an attention-state compiler: it lowers state
semantics and lifetimes into checked token-page or fixed-width physical plans,
paired with a transactional ownership runtime that enforces those plans. Ring
layouts and cache mechanisms are possible compiled plans, not the top-level
abstraction or an automatic claim of model support. OrbitKV does not replace
SGLang's complete model-execution stack and is not yet a mature L5 system.

The mechanism is useful only after a semantic compiler or explicitly lossy
policy creates token-granular holes. It does not create a same-capacity memory
win for ordinary dense Full or contiguous SWA by itself.

## State taxonomy

Hybrid attention is not one storage type:

| Class | Examples | Token relocation |
| --- | --- | --- |
| Token-addressable KV | Full MHA/GQA/MQA, SWA/local, MLA latent KV | Eligible with exact per-token copy and slot mapping |
| Recurrent state | Mamba2, GDN, KDA, Lightning/linear attention | Not token-relocatable; use generation-checked recurrent checkpoints |
| Convolution state | LFM2 ShortConv and finite convolution buffers | Not token-relocatable; use fixed-width state snapshots |
| Sparse auxiliary state | DSA/HiSparse index and compressed host tiers | Separate backend contract; selection is not ownership |

The `orbitkv.attention-state-plan.v1` compiler makes this taxonomy executable:
`token_kv` lowers to key/value token slots, `latent_kv` lowers to distinct
latent and RoPE components, `recurrent` lowers to fixed-width generation-checked
checkpoints, and `convolution` lowers to a fixed-width state ring. The
`compile-state-manager-plan` projection feeds only TokenKV and MLA classes to
the existing token manager, using the summed per-token width for capacity while
retaining component geometry for independent latent/RoPE copies. Recurrent and
convolution state never enter that projection. The
standalone recurrent/convolution checkpoint pool and its independent ABI8
C/Python wire are host L2. The restricted production request-owned fixed-state
seam maps zero-based leases to physical slots `slot_id + 1` and connects
request allocation, initial `MambaPool.clear_slots`, the forward completion
event contract, and release-time wait/retire/clear/exact-ACK. Same-owner
replacement through `MambaPool.copy_from` is covered by coordinator and
real-CPU-tensor host tests only; its production trigger remains pending. Scoped
fixed-state pair verification is not L4 qualification or a performance result.
The pure-MLA
SGLang seam validates the compiled component widths against
`MLATokenToKVPool` and copies its combined
latent+RoPE row through the same completion-gated relocation transaction; this
is host L2 only and excludes DSA, FP4, DCP, and Hybrid Linear models. Frontend
fixtures validate plan and runner geometry; they are not hardware evidence.
The restricted fixed-state seam requires `ORBITKV_STATE_PLAN` in addition to
the token-only `ORBITKV_PLAN`; their canonical token projection, page size,
layer coverage, and byte geometry are checked together before pool creation.
The token manager and state pool remain separate handles: a partial commit is
contained only by fail-stop, with no cross-handle atomic commit or rollback.

The first bound GDN frontend profile is deliberately narrow, not generic family
support. Its compiled schedule sends Full token KV to `CanonicalKvManager` and
request-private GDN recurrent and convolution state to the generation-checked
checkpoint pool.

This profile is request-private: Prefix/Radix state sharing, attach, and publish
are disabled, and no same-owner copy is admitted. The SGLang cache object stays
installed so canonical release hooks still run, but it always misses. Startup
fails closed unless the compiled schedule, backend, dtype, state layout, and
cache configuration satisfy the accepted structural contract.
This runtime admission rule is a generic structural GDN/convolution policy.
Hardware evidence remains model-specific and must pin the exact checkpoint.

The host-qualified token-KV profiles cover Full KV and ordered Full+SWA with
class-specific placements; the latter fails closed when SWA visibility
diverges. The currently bound GDN profile is the first host-qualified
fixed-state family binding. MLA still needs engine qualification. Recurrent and convolution state
must use request-level fixed-width checkpoints, never token pages or
TokenMove. The current production fixed-state subset is restricted to eager,
single GPU, with no Prefix-state sharing, ReplaySSM, int8 checkpoint pool,
extra/ping-pong buffer, speculation, overlap, Graph, or unified memory. Its
same-owner replacement trigger is still absent. The bound GDN+convolution
profile has host qualification plus scoped, independently unattested pair
verification; KDA, ShortConv, and other linear-attention family bindings are
not implemented. L4 and performance qualification remain pending, and every
additional fixed-state profile needs its own engine qualification. Distributed and cross-device
execution also remain pending.

The relocation claim is narrower than the overall dense ABI8 capability set.
Without changing ABI8, the Python/runtime path preserves one multi-request
scheduler batch: native mark, prepare, submit, and complete are each called
once, followed by one aggregate page-registry commit and one aggregate
request-head replacement. The SGLang
plugin flattens all request moves into one backend move and one completion
event and validates all mirror plans before any mirror write. Those mirror
writes are not rollback-atomic; after they succeed, the plugin sends one batch
ACK. The scalar API remains a singleton batch compatibility wrapper. Host tests
cover successful runtime batches, multi-request plugin orchestration, and
stale-member batch preflight. Full-copy coverage remains encoded in the CUDA
conformance harness.

For one request-private Full class with full evacuation, the Rust core and ABI8
Python runtime are also host-tested through append, disposition mark, relocate,
exact ACK, further append into the packed layout, and a second
mark-relocate-ACK cycle. The SGLang periodic trigger is host-tested at two
successive active-length thresholds and recomputes the next absolute boundary
after each reclamation. Dense Prefix ownership, request fork, and shared COW
remain separate capabilities: generation-safe request fork and shared
partial-tail COW from a packed publication are host-tested through Rust, raw
ABI8, and Python FFI, including repeated COW. Packed Prefix operations remain
unsupported and fail closed; packed COW is outside the sealed engine
qualification.

The request-private pressure observer is also host-tested behind explicit
opt-in. It samples lifecycle events and separates consumed capacity, resident
data, request-reachable bytes, semantic-live bytes, free-space minima,
high-water marks, and retention amplification. This is telemetry plumbing, not
a memory result: no append-only, sealed, or qualified asynchronous GPU pressure
record is published, fixed-state bytes are excluded, and shared
Prefix/request-fork retention amplification is rejected.

The engine-neutral `orbitkv-runtime` SPI and `orbitkv-reference` external
tensor-arena adapter have separate buildable, clean-installable wheels. The
reference adapter is a reusable effect implementation and contract oracle, not
a complete engine. SGLang still uses its existing adapter and has not migrated
to the SPI.

## Ownership and reuse frontiers

SGLang owns tensor allocation, the Full/GDN/convolution kernels, scheduling,
and model execution. OrbitKV owns the admitted attention-state plan, logical
identities, page/state ownership, transactions, and reclamation protocol. The
adapter mirrors SGLang slot indices, but those mirrors are never the ownership
authority.

Reuse requires two independent facts:

- the **Semantic Frontier** has passed when no live compiled snapshot can read
  the page or state generation; and
- the **Execution Frontier** has passed when the recorded GPU completion event
  proves that all earlier device use has finished.

Semantic death alone cannot make in-flight storage reusable, and a completed
event cannot revoke a live semantic reference. Recycle follows only after both
frontiers, exact retirement receipts, and ACK.

## Logical contract

TokenDisposition has three variants:

- Retained;
- SemanticallyDead with compiler proof identity and version; or
- PolicyEvicted with policy identity, policy version, and quality contract.

SemanticallyDead is lossless. PolicyEvicted is approximate and must name a
quality contract; H2O, Random, and Scissorhands results are not interchangeable.
Relocation never changes the retained token set or its K/V bytes.

Every executable request snapshot has two lengths:

- absolute_seq_len: token generation boundary and RoPE/query position;
- active_kv_len: retained entries visible to attention.

SGLang currently uses one seq_lens field for both meanings. The adapter must
split them before token eviction is enabled: query positions continue to use
absolute_seq_len, while attention metadata and retained slot arrays use
active_kv_len. Shortening seq_lens without preserving positions is incorrect.

## Relocation transaction

The ordered scheduler-batch transaction is:

1. Freeze all candidate views, victim sets, and engine mirrors, then mark all
   dispositions with one native batch call.
2. Select private, unpinned, non-Prefix source generations for every request.
3. Require fragmentation at or above the configured threshold.
4. Reserve bounded aggregate destination headroom and require source pages to
   exceed destination pages.
5. Prepare once and emit ordered, generation-bearing TokenMoves for the batch.
6. Flatten all moves into one byte-exact backend copy and record one completion
   event; the current eager path immediately synchronizes that event on the
   host.
7. Submit once with grouped exact receipts and complete once with the shared
   completion point.
8. Validate the entire returned publication/readback/retirement set, then
   publish the aggregate page registry and request heads.
9. Construct and validate every mirror plan before writing any ReqToToken row,
   LUT entry, or request mirror; commit them only after all validation passes.
10. Host-synchronize the publication stream, send one batch ACK, and only then
    permit source generation reuse.

There is currently no asynchronous consumer-stream wait or copy/attention
overlap. Those are future execution-model capabilities, not implied by the
single event used by this eager, host-blocking path.

The default fragmentation threshold is 0.25, represented as 250 thousandths,
matching the paper's evaluated default. It is an explicit evidence field, not
a universal optimum. Headroom lives inside the same admission ledger;
reclamation cannot wait until free capacity reaches zero.

Shared Prefix generations are excluded initially. A request must first obtain
private ownership through Snapshot/COW. No plan may overlap append, COW,
relocation, or publication for the same request and class.
The current append transaction performs that private-ownership transition for
a shared packed partial tail through exact COW, and repeated packed COW is
host-tested. Packed Prefix publication/attach is still not admitted, so this
does not make packed roots Prefix-shareable or extend the sealed engine scope.

## Correctness invariants

1. Token conservation: every retained token and byte-exact K/V payload remains.
2. Unique placement: one placement per retained token and one owner per slot.
3. Generation safety: stale engine, pool, page, or view identity fails first.
4. Pre-attention visibility: copy completion precedes every destination read.
5. Deferred reuse: old readers and references discharge before source reuse.
6. Positive reclamation: admitted plans strictly reduce physical page count.
7. Fail-stop containment: before mark, rejected admission is zero-mutation;
   after mark succeeds, any later failure or uncertainty is non-rollback and
   fail-stops the runtime. An explicitly unobserved copy may abort reservations,
   but neither that abort nor quarantine restores the pre-mark dispositions or
   an older head.

## Qualification matrix

The engine-neutral CUDA harness verifies payload uniqueness, stale-member
atomicity, real stream ordering, exact evacuation, retirement/completion
metadata, ACK-gated generation reuse, and final drain. This is component
conformance only, not sealed L3/L4, performance, capacity, or a complete engine
qualification.

The pinned engine record is clean, preflight-bound, and sealed for scoped
request-private Full relocation correctness and lifecycle. It remains
`hardware_attested=false` and `performance_go=false`; capacity, end-to-end
memory saving, production readiness, and broader engine features are not
qualified. See the
[sealed relocation qualification](../results/h20-sglang-v0517-abi8-token-relocation-20260825/README.md).

The fixed-state record verifies plan identity, outputs, completion, lifecycle,
and final drain for its manifest-bound frontend profile. It is verification,
not qualification: `qualified=false`, `hardware_attested=false`, and
`performance_go=false`. See the
[fixed-state pair-verification record](../results/h20-sglang-v0517-abi8-qwen35-fixed-state-pair-verification-20260823/README.md).

A later weight-backed run exercises the same ownership seam but remains below
the release gate because its source and preflight status are diagnostic-only.
See the [diagnostic archive](../results/h20-sglang-v0517-abi8-qwen38-fp8-diagnostic-20260824/README.md).

The engine release gate must include:

- a dense Full model and an ordered Full+SWA model;
- deterministic victim sets shared by Naive-Evict and relocation modes;
- validation-build byte hashes for retained K/V before and after every move;
- token/logit comparison with the same-policy non-relocating reference;
- CUDA stream/event evidence for copy, wait, publication, and reuse order;
- fragmentation, reclaimed-page, temporary-headroom, and all-free census;
- repeated fresh-process paired TTFT, ITL, output-token throughput, request
  throughput, p50/p95/p99, and GPU-copy-time statistics; and
- append-only exact source, dependencies, commands, raw outputs, provenance,
  manifest, and hashes.

The primary comparison holds model, prompts, victim set, quality contract, and
capacity constant: Naive-Evict leaves logical holes without repacking, while
OrbitKV relocates the same retained tokens byte-exactly. Dense Full versus an
approximate policy, different capacities, or different victim sets cannot
support a memory or throughput claim.

Token relocation retains L2 qualification plus narrow component conformance
and a sealed, clean-source, scoped engine correctness/lifecycle qualification.
No independent hardware attestation, performance GO, same-capacity benefit, or
complete-SGLang-replacement claim is made.
The qualified ABI8 Prefix seal covers only its manifest-bound scope and
explicitly excludes fixed state. The fixed-state archives add scoped pair
verification and diagnostic execution only. None establishes broader L4
status, a performance gain, mature L5 operation, or complete SGLang replacement.
