# Token Reclamation and Hybrid-State Architecture

The normative shipped capability boundary remains capability-matrix.md. This
document specifies the post-ABI6 implementation and qualification contract.
Code existence is not an H20 or performance claim.

## Scope

OrbitKV adopts the virtualization boundary from vToken (arXiv:2608.13263):
logical token liveness is independent of physical page placement, and retained
K/V can be repacked byte-for-byte before emptied pages are reclaimed. OrbitKV
does not inherit the paper's vLLM implementation or reported performance.

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

The first implementation profile is single-GPU eager Full KV. The second is
ordered Full+SWA with one logical victim set and class-specific placements.
MLA needs separate backend geometry qualification. Recurrent, convolution,
Graph, speculation, distributed, and cross-device paths fail closed until
their distinct protocols are implemented and qualified.

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

The ordered transaction is:

1. Mark dispositions against an immutable TokenView version.
2. Select private, unpinned, non-Prefix source generations.
3. Require fragmentation at or above the configured threshold.
4. Reserve bounded destination headroom and require source pages to exceed
   destination pages.
5. Emit exact generation-bearing TokenMoves.
6. Copy K/V byte-for-byte and submit exact receipts.
7. Record a completion event on the relocation stream.
8. Make the next attention stream wait before using destination slots.
9. Atomically publish the target TokenView and retained slot mapping.
10. Retire old pages, clean mirrors, accept exact backend ACK, then reuse.

The default fragmentation threshold is 0.25, represented as 250 thousandths,
matching the paper's evaluated default. It is an explicit evidence field, not
a universal optimum. Headroom lives inside the same admission ledger;
reclamation cannot wait until free capacity reaches zero.

Shared Prefix generations are excluded initially. A request must first obtain
private ownership through Snapshot/COW. No plan may overlap append, COW,
relocation, or publication for the same request and class.

## Correctness invariants

1. Token conservation: every retained token and byte-exact K/V payload remains.
2. Unique placement: one placement per retained token and one owner per slot.
3. Generation safety: stale engine, pool, page, or view identity fails first.
4. Pre-attention visibility: copy completion precedes every destination read.
5. Deferred reuse: old readers and references discharge before source reuse.
6. Positive reclamation: admitted plans strictly reduce physical page count.
7. Failure atomicity: unknown copy/event state quarantines destinations and
   preserves source authority.

## Qualification matrix

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

Until this matrix passes, token relocation is host L2 only and no new H20,
same-capacity, speedup, or complete-SGLang-replacement claim is made.
