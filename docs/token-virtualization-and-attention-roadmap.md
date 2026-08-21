# Token Virtualization and Attention Expansion Roadmap

This roadmap starts from the live ABI6 architecture. Qualification status is
normative only in the [Capability Matrix](capability-matrix.md).

## Current checkpoint

The modular Rust core and exact 23-symbol C ABI6 wire are host-qualified L2.
They provide immutable snapshots, request fork, page-aligned Prefix ownership,
joint Full+SWA COW, detach actions, and page-owned reclamation.

The ABI6 Python runtime and SGLang Prefix adapter are host-qualified L2. There
is no ABI6 H20 Prefix evidence. The frozen ABI5-v5 H20 record is historical
scoped L4 correctness only and does not qualify this source.

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

## M1: Freeze the ABI6 Python/runtime boundary

Status: **L2 GO**.

- complete ctypes parity with `orbitkv.h` and ABI version 6;
- load exactly the 23 allowed symbols and reject all compatibility aliases;
- keep FFI layout/workspace code separate from lifecycle journals;
- make snapshot heads, materialized views, detach actions, and reclamation
  receipts generation checked in Python;
- preflight complete batches and all mirror mutations before commit; and
- qualify malformed spans, stale leases, short buffers, fail-stop, and
  quarantine paths.

Exit gate: the ABI6 Python runtime is L2 against the exact release library.

## M2: Integrate SGLang Prefix ownership

Status: **host L2 GO; H20 pending**.

Register an `OrbitKVPrefixCache` at the official SGLang `v0.5.17` cache seam.
Radix remains a token/digest/LRU index. It stores an opaque `PrefixLease`, not
page IDs, generations, free-list state, or CUDA tensors.

The first profile is deliberately narrow:

- eager, single GPU, page16 BF16 NHD;
- Qwen Full and GPT-OSS ordered Full+SWA;
- page-aligned publish and attach only;
- shared partial-tail divergence through exact COW; and
- overlap, Graph, speculation, disaggregation, remote/hierarchical cache, and
  multi-GPU disabled.

Host qualification precedes H20. The H20 record must compare cold and warm
paths, verify token/logit equivalence, prove shared-page ref counts and final
drain, and report TTFT/ITL/throughput with repeated paired processes.

Expected benefits are fewer duplicated physical KV pages and less repeated
prefill work for warm prefixes. Those benefits are not yet measured on H20 and do not
reduce the bytes of a same-sized preallocated KV tensor arena.

## M3: Token table and exact relocation

Status: **pending after Prefix**.

Add stable logical token IDs and class-specific placement generations without
changing snapshot identity:

```text
TokenPlacement {
    token_id,
    class_id,
    page: PageLease,
    offset,
    placement_generation,
}
```

A relocation transaction must:

1. plan against one immutable snapshot head;
2. reserve exact destination `PageLease` values from bounded headroom;
3. pin exact source generations;
4. emit ordered token-copy intents;
5. validate backend copy receipts and completion;
6. atomically publish the new placement view; and
7. retire sources only after every old snapshot and reader pin is gone.

Global invariants are token conservation, unique placement, completion
visibility, snapshot isolation, generation safety, and deferred source reuse.
Unknown copy launch or completion quarantines destinations and preserves
source pins.

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

## M4: Multiple completion domains and CUDA Graph

Status: **pending after relocation correctness**.

Graph compatibility requires more than stable tensor addresses:

- fixed-address device descriptor or slot-table storage;
- generation-checked descriptor patches outside captured kernels;
- per-stream completion domains for forward, copy, and replay;
- replay-scoped reader pins separate from `GraphExec` lifetime;
- cancellation and unknown-launch handling; and
- a wait on the actual consuming stream before replay sees a new placement.

The first relocation backend remains eager and publishes after copy completion.
Only after that path is qualified may copies overlap sampling, CPU scheduling,
or compute-bound work. Copies should not overlap memory-bandwidth-bound
attention by default; profiling determines the policy.

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
