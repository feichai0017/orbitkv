# Token Reclamation and Hybrid-State Architecture

The normative shipped capability boundary remains capability-matrix.md. This
document specifies the ABI8 implementation and qualification contract.
Code existence is not an H20 or performance claim.

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
real-CPU-tensor host tests only; its production trigger remains pending. The
production seam now also has scoped Qwen3.5 pair-verification evidence from
recorded H20 runtime snapshots, without independent hardware attestation; it
is not L4 qualification or a performance result. The pure-MLA
SGLang seam validates the compiled component widths against
`MLATokenToKVPool` and copies its combined
latent+RoPE row through the same completion-gated relocation transaction; this
is host L2 only and excludes DSA, FP4, DCP, and Hybrid Linear models.
The checked qualification fixture for `deepseek-ai/DeepSeek-V2-Lite` uses 27
layers, BF16 `kv_lora_rank=512`, and `qk_rope_head_dim=64`, or 1024 latent plus
128 RoPE bytes per token per layer. It is a plan/runner fixture, not H20 evidence.
The restricted fixed-state seam requires `ORBITKV_STATE_PLAN` in addition to
the token-only `ORBITKV_PLAN`; their canonical token projection, page size,
layer coverage, and byte geometry are checked together before pool creation.
The token manager and state pool remain separate handles: a partial commit is
contained only by fail-stop, with no cross-handle atomic commit or rollback.

The first bound GDN profile is the strict normalized official
`Qwen/Qwen3.5-0.8B` manifest, not generic Qwen3.5 support. Its 24-layer plan
contains Full attention at layers `3, 7, 11, 15, 19, 23` and GDN plus
convolution at the remaining 18 layers. Per Full layer it compiles 1,024 BF16
bytes each for K and V per token. Per GDN layer it compiles 1,048,576 FP32
recurrent bytes plus 36,864 BF16 convolution-history bytes; kernel width four
means three history positions persist between decode steps. The Full projection
goes to `CanonicalKvManager`, while both fixed-width components go to the
generation-checked checkpoint pool.

This profile is request-private: Prefix/Radix state sharing, attach, and publish
are disabled, and no same-owner copy is admitted. The SGLang cache object stays
installed so canonical release hooks still run, but it always misses. The
pair-verification workload uses fresh prompts and requires zero cached tokens and
zero fixed-state copies. Production startup fails closed unless Full attention
uses FA3, all general/prefill/decode linear-attention selectors use Triton,
configured and actual temporal state are FP32,
`mamba_radix_cache_strategy=no_buffer`, and `disable_radix_cache=true`.
This runtime admission rule is a generic structural GDN/convolution policy.
Hardware evidence remains model-specific and must pin the exact checkpoint.

The first host-qualified token-KV profile is single-GPU eager Full KV. The
second is ordered Full+SWA with one logical victim set and class-specific
placements; it fails closed when SWA visibility diverges. The strict Qwen3.5
profile is the first host-qualified fixed-state family binding.
MLA still needs real-model H20 qualification. Recurrent and convolution state
must use request-level fixed-width checkpoints, never token pages or
TokenMove. The current production fixed-state subset is restricted to eager,
single GPU, with no Prefix-state sharing, ReplaySSM, int8 checkpoint pool,
extra/ping-pong buffer, speculation, overlap, Graph, or unified memory. Its
same-owner replacement trigger is still absent. The Qwen3.5 GDN+convolution
binding has host qualification plus scoped pair verification from recorded H20
runtime snapshots, without independent hardware attestation; KDA, ShortConv,
and other linear-attention family bindings are not implemented.
Qwen3.5 L4 qualification and performance qualification still require more
evidence; every other fixed-state profile still needs its own real
CUDA/model/H20/performance qualification. Distributed and cross-device
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
cover successful runtime batches at B1/B4, multi-request plugin orchestration at
B1/B2/B4, and stale-member batch preflight at B1/B4/B32. Full-copy B32 remains
encoded in the CUDA conformance harness and now passes on H20 alongside B1/B4.

For one request-private Full class with full evacuation, the Rust core and ABI8
Python runtime are also host-tested through append, disposition mark, relocate,
exact ACK, further append into the packed layout, and a second
mark-relocate-ACK cycle. The SGLang periodic trigger is host-tested at two
successive active-length thresholds and recomputes the next absolute boundary
after each reclamation. Dense Prefix ownership, request fork, and shared COW
remain separate capabilities: after a packed publication, Prefix publication,
fork, and shared partial-tail COW are unsupported and fail closed before
mutation.

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
The current repeated path does not perform that private-ownership transition
for a packed shared root; packed Prefix, fork, and shared COW combinations are
therefore not admitted.

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

The engine-neutral CUDA conformance harness passes seven H20 cases: payload
uniqueness, stale-member atomicity at B1/B4/B32, and real CUDA execution at
B1/B4/B32. It encodes two cycles on the same requests and cursor objects,
with an independent live-token/payload oracle and 257-byte records that embed
cycle, request, and token coordinates. Across distinct non-default append,
copy, and consumer streams it checks emitted append COW copies, exact
3-page-to-2-page full evacuation with 24 moves per request/cycle, exact
retirement spans and completion metadata, event-ordered byte readback, blocked
reuse before ACK, same-page/higher-generation reuse after ACK, and final drain.
This proves the component's bounded copy, ordering, generation-reuse, and drain
contract on the recorded H20. It does not by itself qualify L3/L4, performance,
capacity, or a complete engine path.

The separate pinned SGLang diagnostic uses Qwen2.5-0.5B Full attention, page16
BF16 NHD storage, eager single-GPU execution, and FlashInfer for both the Naive
token-indexed oracle and Relocate mode. Four alternating-order epochs at B1 and
B4 yield 8/8 exact-token pairs. Every process runs five iterations and each
iteration reaches two reclamation rounds; relocation/reclaimed-page counters
match exactly, manager census drains, and failure/quarantine/fail-stop counters
remain zero. Excluding iteration 0 leaves 16 hot samples per mode and group.
Relocate throughput is +7.629% at B1 and -1.232% at B4; mean/p95 latency deltas
are -7.088%/-22.844% and +1.247%/+2.880%, respectively. B1 varies materially
across epochs, so these observations do not support a speedup claim.

This SGLang result is `diagnostic_only`, unsealed, dirty-source, based on
recorded H20 observations rather than independent attestation,
`qualified=false`, and `performance_go=false`. It does not establish capacity
or end-to-end memory savings. A relocate-only FA3 E2E smoke passed, but sparse
Naive+FA3 is not representable and now fails closed, so the valid paired oracle
uses FlashInfer.

[Qwen2.5-0.5B relocation diagnostic archive](../results/h20-sglang-v0517-token-relocation-diagnostic-20260825/README.md)

For the Qwen3.5 fixed-state profile, the scoped fresh-prompt stock/manager
pair-verification step was executed with runtime records observing one H20.
The archive does not independently attest that hardware. It uses the
official checkpoint and SGLang `v0.5.17` commit
`29481685462732237d80d86076d6563e1f658102`, validates exact token/state plan
identity, and records CUDA stream/event completion, exact lifecycle counters,
equal output-token totals, and final drain. B1 runs one request with state
capacity two for one iteration; B4 runs four requests with state capacity four
for five
iterations. Across three epochs, six stock/manager pairs pass. Per epoch,
prepare/clear/retire/ACK counts are one for B1 and 20 for B4; copies are zero.

This gate is verification, not qualification. The archive is
`qualified=false`, `hardware_attested=false`, and `performance_go=false`. It
records H20 runtime snapshots with UUID prefix `GPU-3a35…`, but no independent
hardware attestation. The three-epoch aggregate manager-over-stock differences
are +0.507353% for B1 and +3.739403% for B4, so neither case shows a speedup.
The observed configured arena reservation difference is **0%**; this
is not a qualified end-to-end memory-saving result. L4 qualification remains pending.

[Qwen3.5 H20 fixed-state pair-verification record](../results/h20-sglang-v0517-abi8-qwen35-fixed-state-pair-verification-20260823/README.md)

The subsequent Qwen3.8-27B-FP8 diagnostic is weight-backed but remains below
the release gate. It declares `Qwen/Qwen3.8-27B-FP8` repository revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. Its raw records bind the
downloaded config, index, and all 66 shards by hash and byte count
(30,866,866,928 bytes and 1,606 tensors), while repository provenance is not
independently online-attested. It records official SGLang `v0.5.17` revision
`29481685462732237d80d86076d6563e1f658102`, and recorded NVIDIA H20 UUID
`GPU-3a35e57b-fc54-5620-56ee-deaf5a9c40d3`. The compared processes use
explicit `fp8_gemm_runner_backend=triton`, eager single-GPU execution, and
page16 BF16 NHD KV storage.

There are four balanced-order epochs: 1/3 manager to stock and 2/4 stock to
manager. Each epoch contains one B1 and one B4 pair, yielding eight pairs
total; every process runs five iterations. All eight pairs pass the verifier
and match tokens exactly. Every manager run ends with zero live or
pending ownership and zero failure/fail-stop counters. Excluding iteration 0
from every process leaves 16 hot samples per mode and batch size.

| Case | Stock mean / median / p95 | Manager mean / median / p95 | Latency / throughput delta | Epoch latency deltas |
| --- | --- | --- | --- | --- |
| B1 | 2.6296723178 / 2.5374011379 / 3.0721712420 s | 2.8031814888 / 2.6451683380 / 3.3369778013 s | +6.5981% / -6.1897% | +2.7822%, +0.7129%, +26.3407%, -1.9835% |
| B4 | 2.9576119229 / 2.9602836296 / 3.0001941137 s | 3.0395950049 / 3.0345056280 / 3.0800264925 s | +2.7719% / -2.6972% | +1.8441%, +2.8850%, +5.0070%, +1.3885% |

The B1 epochs show substantial jitter and a +26.3407% outlier; neither case
supports a positive performance claim. Stock and manager use identical
configured tensor-arena capacity and report identical KV-cache reservation,
an observed configured arena reservation difference of **0%**; this is not a
qualified end-to-end memory-saving result. This is diagnostic
only: the source is dirty and the record is `sealed=false`,
`preflight_bound=false`, `hardware_attested=false`, `qualified=false`, and
`performance_go=false`. The default-auto DeepGEMM
path loaded 66/66 shards but its E2E run remained incomplete after lengthy
precompilation was externally terminated.

[Qwen3.8-27B-FP8 diagnostic archive](../results/h20-sglang-v0517-abi8-qwen38-fp8-diagnostic-20260824/README.md)

The first H20 release gate must include:

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

Token relocation retains L2 qualification plus narrow H20 component conformance
and an unqualified recorded-device SGLang diagnostic. No sealed L3/L4,
same-capacity, speedup, or complete-SGLang-replacement claim is made.
The existing qualified ABI8 H20 Prefix seal covers only Qwen2.5-7B Full and
GPT-OSS-20B Full+SWA and explicitly excludes fixed state. The Qwen3.5 archive
fills only its scoped pair-verification step, and the Qwen3.8 diagnostic adds
only dirty-source model execution and pair evidence. None establishes broader
L4 status, a performance gain, mature L5 operation, or complete SGLang
replacement.
