# Counter-Based Sampling Design

## Admission Result

vLLM 0.24's CUDA sampler delegates top-k/top-p requests without per-request
generators to FlashInfer. Loom should not replace that path without evidence.
The remaining gap is narrower: `TopKTopPSampler.forward_cuda` falls back to
the PyTorch-native path whenever seeded per-request generators are present.
That path materializes an F32 probability matrix and an equally large
exponential-noise matrix, then launches `exponential_` once per seeded row
before division and argmax.

That boundary passed its source-pinned H20 admission gate. Two full row-matrix
runs and an isolated 32-row confirmation measured:

| Rows | All-seeded sampling only | Unseeded control | CUDA kernels | Peak increment |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `32.64–47.18 us` | `29.94–43.83 us` | `3 / 3` | `0.61 MB` |
| 8 | `131.13–132.95 us` | `43.56–44.29 us` | `10 / 3` | `4.87 MB` |
| 32 | `265.79–422.32 us` | `55.18–58.10 us` | `34 / 3` | `19.45 MB` |

The kernel column is `all-seeded / unseeded`. Every seeded row contributes a
separate exponential-noise kernel, and peak incremental bytes match one full
F32 probability-shaped noise tensor plus the small output/allocation boundary.
The 32-row latency is order-sensitive, so the range is retained; the lower,
isolated endpoint is still `4.82x` the unseeded control.

The candidate was therefore admitted for implementation. This historical gate
is not itself a Loom performance result, and the larger complete
native-fallback-versus-FlashInfer gap is not an entitlement for a standalone
categorical sampler. See the
[machine-readable admission evidence](../results/h20-vllm-seeded-sampling-admission-20260727.json).

## Direct Implementation Result

ABI8-A is now implemented through the full direct path: Rust contract and CPU
oracle, handwritten CUDA, safe dispatch, checked bridge, boxed Stable ABI
PyTorch schema, and public Python API. One CUDA block samples each row, while
all rows share one kernel launch. Caller-owned state persists and the operator
allocates only its int64 token output at the PyTorch convenience boundary.

The H20 gate compares exactly the normalized-probability boundary above. Each
repeat reverses provider order, and generator/state construction is outside the
CUDA-event interval:

| Rows | vLLM seeded | Loom ABI8 | Ratio | Kernels | Peak increment |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `32.97 us` | `49.46 us` | `0.67x` | `3 / 1` | `0.61 MB / 512 B` |
| 2 | `55.79 us` | `62.71 us` | `0.89x` | `4 / 1` | `1.22 MB / 512 B` |
| 4 | `80.07 us` | `69.86 us` | `1.15x` | `6 / 1` | `2.43 MB / 512 B` |
| 7 | `118.13 us` | `73.01 us` | `1.62x` | `9 / 1` | `4.26 MB / 512 B` |
| 8 | `130.70 us` | `72.48 us` | `1.80x` | `10 / 1` | `4.87 MB / 512 B` |
| 32 | `411.13 us` | `76.20 us` | `5.40x` | `34 / 1` | `19.45 MB / 512 B` |

The kernel column is `vLLM / Loom`. Exact replay and counter advancement pass;
65,536 draws have maximum absolute frequency error `0.00292` against an
`0.008` bound and never select a zero-mass token. Non-default stream,
`torch.compile`, FakeTensor/opcheck, and CUDA Graph replay also pass. Rows 1–2
are measured losses, not optimization wins. The persistent engine adapter
cannot switch an in-flight request back to vLLM at those batch sizes without
changing its RNG stream, so this crossover is reported rather than hidden
behind a per-step fallback. See the
[direct H20 evidence](../results/h20-categorical-sample-20260727.json).

## Engine Integration Result

The vLLM 0.24/0.25 adapter is implemented as an explicit engine-lifetime
registration. It leaves temperature, processors, top-k/top-p filtering,
softmax, and processed-logprob return modes in vLLM, then replaces only the
native `random_sample` call. Every random request must carry an explicit
signed-int64 seed. Unseeded random admission and speculative-engine
construction fail before sampling rather than silently splitting a request
between two random streams.

Each `CachedRequestState` owns a persistent two-int64 device tensor. The active
`InputBatch` owns one contiguous `[max_requests, 2]` tensor used by the kernel.
Add, remove, swap, and condense hooks copy or move rows on device; preempted or
temporarily unscheduled requests retain their state on the cached request and
restore it when rescheduled. Stable decode steps perform no Python state
reconstruction or host counter update.

The lifecycle test removes a middle request, condenses the batch, samples,
restores the request, swaps slots, samples again, and finally copies state back
to all three cached requests. Expected counters and seeds remain attached to
the correct request. The same test passes on:

- PyTorch `2.11.0+cu130`, vLLM `0.24.0`, bridge ABI 8;
- PyTorch `2.11.0+cu130`, vLLM `0.25.1`, bridge ABI 8.

A default Linux vLLM EngineCore multiprocessing smoke also reproduces the
in-process Loom token stream, which is distinct from the native vLLM stream.
The instrumented performance gate disables EngineCore multiprocessing only so
the parent process can read the bridge's in-memory launch telemetry.

The real Qwen2.5-0.5B-Instruct H20 gate uses BF16 model execution, 151,936
sampling logits, input/output length 32, `top_k=50`, `top_p=0.9`, and a
different explicit seed per request. Three measured generations follow one
warmup in each provider process:

| Batch | Baseline-first ratio | Loom-first ratio | Result |
| ---: | ---: | ---: | --- |
| 1 | `0.979x` | `0.981x` | measured small-batch cost |
| 2 | `0.981x` | `0.984x` | measured small-batch cost |
| 4 | `0.985x` | `0.976x` | measured small-batch cost |
| 8 | `1.003x` | `1.013x` | near crossover |
| 32 | `1.057x` | `1.081x` | order-stable engine win |

Every case exactly replays within its provider. Loom records exactly one
categorical launch per decode step, 640 calls per provider-order run, zero
initialization calls, and no contract rejection. Native and Loom token streams
are intentionally different because ABI8-A declares its own Philox/CDF
mapping. See the
[baseline-first](../results/h20-vllm-engine-categorical-sample-20260727.json)
and
[Loom-first](../results/h20-vllm-engine-categorical-sample-loom-first-20260727.json)
engine evidence.

## Product Boundary

Loom may own deterministic random-state advancement and categorical selection.
It does not own:

- temperature, penalties, masks, or sampling-policy construction;
- top-k/top-p filtering already handled by a qualified Loom or vendor path;
- FlashInfer sampling for unseeded requests;
- scheduler request identity or lifetime;
- sampled-token logprob calculation, which remains a separate supported Loom
  tail.

The engine supplies normalized probabilities and caller-owned state. Loom
selects one token per admitted row on the current stream.

## ABI8-A Contract

The first operator is intentionally standalone:

```text
categorical_sample(
    probabilities: read F32[rows, vocab],
    rng_state: read/write I64[rows, 2],
    token_ids: write I64[rows],
)
```

`rng_state[row]` is `(seed, counter)`. The values are non-negative signed
64-bit integers so the same representation is available in Rust, PyTorch, and
the checked bridge. A successful sample consumes exactly one counter value per
row and increments that row's counter by one on device. A counter already at
`INT64_MAX` cannot be submitted because it cannot complete that advancement.

The probability contract is:

- rank two, same-device, unit vocabulary stride, and non-overlapping rows;
- finite, non-negative F32 values;
- at least one positive value per row;
- each row normalized to `1.0` within the fixed absolute tolerance `1e-5`,
  evaluated with an F64 sum by the Rust oracle.

Selection uses canonical Philox4x32-10. The seed's low and high 32-bit words
form the two-word Philox key; the counter's low and high 32-bit words form the
low half of the four-word Philox counter, whose upper half is zero. If `x` is
the first 32-bit Philox output, the row uniform is exactly:

```text
u = (x + 0.5) / 2^32
```

This places `u` strictly inside `(0, 1)`. Inverse CDF uses 1,024 fixed logical
lanes arranged as 32 logical warps. Each warp owns one contiguous token
interval; its 32 lanes accumulate strided F32 partials and use the fixed
`16,8,4,2,1` reduction tree. Warp totals are accumulated in warp order. Only
the selected warp replays its interval in 32-token chunks with the fixed
`1,2,4,8,16` inclusive-scan tree. The Rust oracle emulates this exact logical
tree, so the deterministic stream does not depend on host scheduling or a
vendor RNG implementation.

The selected token is the first ascending token whose tree-defined cumulative
mass is greater than `u`. The last positive token absorbs accepted
normalization rounding when the cumulative sum ends below `u`. Equal
probabilities therefore have no separate tie rule: token order and F32
accumulation order are both explicit parts of the ABI8-A stream.

The Rust reference validates every buffer, RNG row, and probability row before
writing any token or counter. A rejected reference call therefore leaves all
outputs and RNG state byte-for-byte unchanged.

This is a new deterministic stream contract, not a promise to reproduce
PyTorch's exponential-race token for the same seed. The direct gate requires
exact replay for the same `(probabilities, seed, counter)` and a declared
statistical agreement test against categorical probabilities. Engine
integration must remain opt-in because native-vLLM seed-to-token identity
changes.

## One Execution Pattern

Every shape uses the same counter mapping, state advancement, 1,024
logical lanes, token order, and selection algorithm. The Rust/CUDA surface has
no seedless mode, implicit global generator, host fallback, or shape-specific
public variant. Without explicit registration, vLLM keeps its original
sampler. Once an engine registers Loom, unsupported random admission fails
before dispatch; the adapter never switches an active request between native
and Loom RNG according to the instantaneous batch size.

The implementation owns no temporary tensor proportional to
`rows * vocab_size`. PyTorch may allocate the output token vector, while
caller-owned RNG state persists across steps and CUDA Graph replays.

## Request-State Lifecycle

A production vLLM adapter cannot reconstruct state from a Python dictionary on
every decode step. ABI8 binds explicit state to both the cached request and the
engine's persistent request slots:

1. initialize one request-owned state tensor at explicit seeded admission;
2. copy that tensor into the active contiguous batch slot;
3. preserve the active row when `InputBatch` compacts or swaps request slots;
4. let the CUDA operator advance counters in place;
5. copy the row back to `CachedRequestState` when a request is unscheduled or
   preempted, then restore it when the request returns;
6. release the request-owned tensor with the finished cached request.

The first adapter is deliberately explicit-seed-only. Greedy requests may
share the batch because they consume no random result. An unseeded random
request is rejected at admission, and a speculative engine is rejected at
construction. Loom does not splice two RNG streams into one batch or invent
implicit state.

This lifecycle is part of the engine gate. A direct PyTorch microbenchmark or
a monkeypatch that rebuilds state on the host does not qualify integration.

## Admission And Exit Gates

The admission gate is complete:

- rows `1, 2, 4, 7, 8, 32` were measured at vocabulary 151,936;
- unseeded, one-seeded, and all-seeded sampling-only and full-fallback paths
  were timed in alternating provider order;
- CUDA launch count and peak temporary bytes were captured per variant;
- deterministic sequence replay and non-default current-stream execution
  passed for every row count;
- native capture failed until every per-request `torch.Generator` was
  explicitly registered with the CUDA Graph, while registered capture passed;
- vLLM 0.24 sampler and request-slot source hashes are recorded in the result.

Implementation and integration gates:

1. [x] Rust Philox, fixed-CDF-tree, and state-validation oracle tests;
2. [x] handwritten CUDA agreement at small and 151,936-token vocabulary,
   invalid-buffer rejection, and no host sync;
3. [x] checked bridge ABI8 and Stable ABI PyTorch mutation schema;
4. [x] FakeTensor/opcheck, `torch.compile`, current-stream, and CUDA Graph
   replay;
5. [x] distribution tests over multiple counters, including zero-probability
   tokens, plus the direct H20 baseline;
6. [x] a vLLM seeded-request adapter with persistent cached-request and
   request-slot state, including remove/condense/resume/swap tests on vLLM
   0.24 and 0.25;
7. [x] provider-order-reversed Qwen2.5-0.5B engine evidence: exact replay, one
   Loom launch per decode step, no rejection, and an order-stable
   `1.057–1.081x` batch-32 latency/throughput ratio.

The declared explicit-seed, non-speculative sampling subsystem is complete in
source. ABI8 matrix-wheel clean-install qualification remains a separate
binary-distribution gate.
