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

## Stage B: Offline replacement replay

Replay the Stage A JSONL trace through OrbitKV's block manager simulator.
For each event boundary verify:

```text
required read block is resident
retired block is not reused while pinned
resident blocks do not exceed the compiled bound plus declared headroom
release returns all unshared blocks
```

The first owning adapter is blocked until replay is exact for all A0 traces.

## Stage C: Owning allocator adapter

The owning adapter will implement SGLang's
`BaseTokenToKVPoolAllocator` contract but use OrbitKV for block identity,
allocation, and retirement. SGLang continues to own:

- scheduling;
- radix-tree policy;
- model execution;
- attention kernels;
- CUDA Graph execution;
- network and connector transport.

The first owning implementation uses SGLang's existing physical KV pools and
attention kernels. It does not introduce a ring kernel or VMM.

Comparators:

1. stock SGLang hybrid allocator;
2. OrbitKV with the same page size and kernels;
3. all-full fallback as a diagnostic only.

## Stage D: Value gates

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

The owning adapter's manager overhead must remain below 2% of decode wall time.

### Research gate

OrbitKV must beat a hand-written Full+SWA manager on at least one workload with
more than two lifetime classes, or demonstrate that the same compiler adds a
new retention pattern without changing the runtime state machine.

## Required environment

The current development host has no functioning NVIDIA driver and does not
contain SGLang's complete Python dependencies. Therefore Stage A GPU results
are intentionally not reported from this host. Host compiler and lifecycle
tests remain valid, while all SGLang performance claims require a recorded GPU
run with source revision, model revision, command, environment, and raw trace.
