# SGLang End-to-End Validation

## Boundary

OrbitKV is the project under test. SGLang remains an unmodified external
inference engine, workload driver, correctness oracle, and strong baseline.
OrbitKV does not import or fork SGLang source.

The validation target is initially pinned to:

```text
095ec6c997bfdd25d3864cb0ce77a6562a934b96
```

## Retention frontend

OrbitKV now accepts a declarative `orbitkv.retention-ir.v1` program whose
`may_read` predicate is an affine AST over query and key positions. Legacy
Full/SWA JSON is syntax sugar and compiles through the same IR.

The first analyzer recognizes exact difference constraints on
`query_position - key_position`. It derives:

```text
true
    -> unbounded lifetime
    -> append-only address program

query_position - key_position < W
    -> maximum delta W-1
    -> fixed window W
    -> block death and periodic cell count

key_position < S OR query_position - key_position < W
    -> pinned sink block domain
    -> periodic local block domain

floor(query_position / C) == floor(key_position / C)
    -> resettable arena
    -> epoch-end retirement
```

Unsupported or insufficiently constrained relations lower to unbounded
retention instead of guessing a finite death time.

The partitioned Sink+Sliding plan is currently qualified in the Rust simulator
and reference manager. `emit-sglang-policy` rejects partitioned block domains
until the adapter owns separate bindings for the pinned and periodic
components; it does not silently flatten them into one SWA policy.

The same-chunk plan is also qualified in the Rust simulator and reference
manager only. The SGLang policy rejects it rather than treating it as SWA.
SGLang's `attention_chunk_size` can describe chunk-relative sliding behavior,
so the HF frontend does not infer same-chunk retention from that field alone.

Per-head lifetime plans are likewise kept above the current SGLang compatibility
backend. OrbitKV can compile and manage disjoint KV-head ranges in the same
layer, but SGLang currently exposes one shared SWA allocation class. The
adapter therefore rejects head-aware plans rather than silently widening every
head to the maximum window.

## Stage A: Shadow measurement

The `orbitkv_shadow` plugin hooks the existing
`SWATokenToKVPoolAllocator` methods:

```text
alloc
alloc_extend
alloc_decode
free
free_swa
```

It records allocator transitions without changing return values. Stage A
answers:

1. How far is SGLang's observed peak SWA residency from OrbitKV's block-level
   semantic lower bound?
2. How much extra residency is caused by page rounding, eviction cadence,
   overlap headroom, or retained prefix locks?
3. Which SGLang paths must a future owning adapter reproduce?

The repository CI already validates registration and event ordering against
SGLang's pinned real `HookRegistry`, then feeds the emitted JSONL into the Rust
trace analyzer. GPU allocator behavior and serving performance remain separate
gates below.

### A0: Isolated lifecycle

Configuration:

- released Full+SWA model;
- radix cache disabled;
- speculative decoding disabled;
- hierarchical cache disabled;
- overlap disabled first, then enabled;
- one fixed page size;
- deterministic prompts;
- no P/D disaggregation.

Matrix:

```text
prefix: 1K, 4K, 16K, 32K
decode: 256, 2K
concurrency: 1, 8, 32
chunked prefill: off, on
overlap: off, on
```

Metrics:

- full and SWA peak used token slots;
- semantic live SWA slots;
- physical-to-semantic residency ratio;
- allocation and free event counts;
- SGLang TTFT, TPOT, throughput, and p99;
- correctness against the same SGLang run without the plugin.

The plugin must have statistically negligible overhead before any owning
experiment is accepted.

### A1: Prefix lifecycle

Enable radix cache and separately measure:

- prefix lock residency;
- early SWA lock release;
- tombstoned SWA components;
- trailing-window restore;
- repeated shared-prefix workloads.

This stage is deliberately separate because cached shared state is retained for
future requests and is not request-live state.

## Stage B: Owning runtime invariants

The Rust owning block manager now verifies:

```text
logical identity includes request + class + ordinal
immutable views carry physical slot + generation
semantic death does not imply immediate physical reuse
out-of-order completion advances only a contiguous execution frontier
certified slots remain unavailable until backend commit
failed batch commit leaves every certificate pending
```

The core tests cover multi-request identity, stale-generation rejection,
out-of-order submission completion, request release, and two-phase
certificate commit.

## Stage C: Owning reclamation adapter

The first owning adapter is implemented for the strict SWA chunk-cache path.
OrbitKV owns the retirement frontier and proof; SGLang remains the physical
page backend. The protocol is:

```text
Rust plan_reclamation
    -> retirement certificate
    -> SGLang free_swa group
    -> Rust commit_reclamations
```

The adapter rejects radix cache, overlap scheduling, and speculative decoding
instead of silently applying an unqualified execution proof. SGLang continues
to own scheduling, model execution, attention kernels, request protocol, and
the current physical tensor pools.

The released-checkpoint gate is now complete for `openai/gpt-oss-20b`.
Stock SGLang and OrbitKV loaded the same indexed MXFP4 shards with FA3
attention. Under a fixed 1.979 GiB KV budget, OrbitKV increased Full capacity
by 25.81%. A balanced four-way ablation showed that manually configured
Stock32 reproduces the capacity and most of the makespan gain:

```text
Stock32 / Stock128 = 0.7828x
Policy32 / Stock32 = 0.9969x
Owner32 / Policy32 = 1.0218x
Owner32 / Stock128 = 0.7970x
```

The plan is generated directly from the checkpoint config. A separate
fixed-capacity experiment measured a `0.9992x` median Owner/Stock ratio. See
`docs/h20-gpt-oss-20b-real-validation-20260817.md`.

The generated Retention IR now feeds a physical-plan optimizer. Given the
recorded 1.979 GiB per-rank KV budget and eight-request pressure workload, it
evaluated intervals 16/32/64/128, selected 32, and emitted the selected SGLang
policy plus an engine compatibility contract. Fresh SGLang processes matched
all four predicted pool capacities exactly.

The H20 comparison uses:

1. generated OrbitKV policy with SGLang-owned reclamation decisions;
2. the same capacity policy with Rust-owned certificate decisions.

Three fresh-process pairs measured a median Owner/Policy ratio of `1.0062x`
with identical capacity and output digests. See
`docs/h20-owning-vmm-validation-20260817.md`.

## Stage D: NVIDIA physical backend

The isolated `orbitkv-cuda` backend implements CUDA Driver API VMM slots. H20
qualification proved 64 fresh physical backing remaps at one stable virtual
address, with verified data and no before/after GPU-memory delta.

The H20 minimum VMM granularity is 2 MiB. The optimizer therefore keeps the
paged backend for small regions and selects VMM only when stable addresses are
required and rounding amplification is within budget.

VMM is not yet SGLang tensor storage. That integration and CUDA Graph replay
remain separate gates.

## Stage E: Remaining value gates

### Correctness gate

- identical greedy tokens;
- bounded final-logit error under the same numerical backend;
- `W-1`, `W`, `W+1`, page boundaries, and multiple wraps;
- cancellation and overlap stress;
- no allocator invariant failure or leaked slot.

### Engineering gate

At least one:

- 15% lower physical resident KV than stock SGLang on a target irregular
  lifetime workload;
- 10% more admitted concurrency at fixed KV budget and SLO;
- 20% fewer continuation or P/D transfer bytes;
- materially lower retraction or offload frequency.

The first owning adapter met the project target of less than 2% median
control-plane overhead on both the controlled fixture and the real-checkpoint
fixed-capacity workload. Radix, overlap, speculative decoding, and CUDA Graph
remain separate correctness and performance gates.

### Research gate

OrbitKV must beat a hand-written Full+SWA manager on at least one workload with
more than two lifetime classes, or demonstrate that the same compiler adds a
new retention pattern without changing the runtime state machine.

## Qualified environment

Recorded GPU evidence uses:

- NVIDIA H20, compute capability 9.0;
- driver 535.161.08;
- PyTorch 2.13.0+cu130;
- FlashInfer 0.6.17;
- SGLang `095ec6c997bfdd25d3864cb0ce77a6562a934b96`.

Every performance claim remains scoped to its recorded fixture, disabled
features, source hashes, and result manifest.
