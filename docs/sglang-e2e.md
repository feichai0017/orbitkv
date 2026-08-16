# SGLang End-to-End Validation

## Boundary

OrbitKV is the project under test. SGLang remains an unmodified external
inference engine, workload driver, correctness oracle, and strong baseline.
OrbitKV does not import or fork SGLang source.

The validation target is initially pinned to:

```text
095ec6c997bfdd25d3864cb0ce77a6562a934b96
```

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
control-plane overhead on the recorded three-pair workload. More samples and
additional request distributions remain required.

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
