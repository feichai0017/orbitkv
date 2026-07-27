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

The candidate is therefore admitted for implementation. This is not a Loom
performance result: no ABI8-A operator exists yet, and the larger complete
native-fallback-versus-FlashInfer gap is not an entitlement for a standalone
categorical sampler. See the
[machine-readable admission evidence](../results/h20-vllm-seeded-sampling-admission-20260727.json).

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
row and increments that row's counter by one on device.

The probability contract is:

- rank two, same-device, unit vocabulary stride, and non-overlapping rows;
- finite, non-negative F32 values;
- at least one positive value per row;
- each row normalized within a declared tolerance.

Selection uses a counter-based Philox mapping to one uniform variate per row,
then chooses the first token whose ascending-token-ID cumulative mass exceeds
that variate. The last positive token absorbs bounded F32 normalization
rounding. Equal probabilities therefore have no separate tie rule: token order
is the explicit CDF order.

This is a new deterministic stream contract, not a promise to reproduce
PyTorch's exponential-race token for the same seed. The direct gate requires
exact replay for the same `(probabilities, seed, counter)` and a declared
statistical agreement test against categorical probabilities. Engine
integration must remain opt-in because native-vLLM seed-to-token identity
changes.

## One Execution Pattern

Every admitted shape uses the same counter mapping, state advancement, token
order, and selection algorithm. The Rust/CUDA surface has no seedless mode,
implicit global generator, host fallback, or shape-specific public variant.
The framework adapter either satisfies the explicit-state contract or calls
the engine's original sampler.

The implementation owns no temporary tensor proportional to
`rows * vocab_size`. PyTorch may allocate the output token vector, while
caller-owned RNG state persists across steps and CUDA Graph replays.

## Request-State Lifecycle

A production vLLM adapter cannot reconstruct state from a Python dictionary on
every decode step. It must bind the explicit `(seed, counter)` state to the
engine's persistent request slots:

1. initialize one state row when an explicitly seeded request is admitted;
2. preserve that state when `GPUInputBatch` compacts or swaps request slots;
3. let the CUDA operator advance counters in place;
4. delete state when the request leaves the batch;
5. fall back before dispatch if any random row lacks a Loom state.

The first adapter is therefore deliberately all-seeded only. A mixed batch
containing an unseeded random row stays entirely on vLLM's native sampler; Loom
does not splice two RNG streams into one batch or invent implicit state for the
unseeded row.

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

Implementation then requires:

1. Rust Philox and state-validation oracle tests;
2. handwritten CUDA agreement, invalid-buffer rejection, and no host sync;
3. checked bridge and Stable ABI PyTorch mutation schema;
4. FakeTensor/opcheck, `torch.compile`, current-stream, and CUDA Graph replay;
5. distribution tests over multiple seeds, including zero-probability tokens;
6. a vLLM seeded-request adapter with persistent request-slot state;
7. provider-order-reversed engine evidence for launch, temporary-memory, and
   latency improvement under the declared opt-in seed semantics.

Only the final gate justifies calling this a completed sampling subsystem.
