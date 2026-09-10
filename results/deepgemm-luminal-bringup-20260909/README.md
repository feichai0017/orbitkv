# Qwen3.5 27B block-FP8 Luminal closure — 2026-09-10

This is a bounded correctness and short-trace serving result on one NVIDIA H20
(SM90, 96 GB). The checkpoint directory is named `qwen3.8-27b-fp8`, while its
metadata identifies `Qwen3_5ForConditionalGeneration` / `qwen3_5_text`. It is
not evidence for a model named Qwen 3.8.

## Compiler integration

- DeepGEMM is pinned at `559d79fb6994a58b8a15b4b93bf13ccc16edf247`.
- Luminal exposes a provider-neutral 128x128 block-scaled linear semantic op.
  An independent CUDA reference and four DeepGEMM SM90 1D2D schedules are
  unioned in egglog, JIT-compiled before timing, device-profiled per dynamic
  bucket, and persisted with provider revision and tile identity. Selection is
  not dispatched by checkpoint name or GPU name.
- The selected DeepGEMM launch sequences participate in Luminal CUDA Graphs.
  FlashInfer supplies the current paged-attention implementation; ordinary
  BF16 matrix products use the existing cuBLASLt path. Attention does not yet
  have multiple searchable backend candidates.
- Loop rolling can independently construct read and commit streams with
  different nominal stream IDs but identical loop ID, dtype, and ordered
  per-iteration values. A general egglog equivalence rule now unions those
  tensor values. Together with required in-place recurrent/convolution writes,
  this lets candidate validation reject a final state chain that does not
  alias the manager-owned input arena.

The first attempted optimization instead shared one index SSA node in the
model graph. Although that removed the large copies, its decode logits differed
from the independent oracle by `2.1640625`; forcing copying scatter and
disabling CUDA Graph replay produced the same error. That experiment was
reverted rather than weakening the `max_abs <= 1.0` gate.

## Complete-checkpoint correctness and profile

The final two-bucket artifact cold-searched in `330.257 s`. Strict replay took
about `38 s`. For prompt IDs `[1, 2, 3, 4]`, OrbitKV produced token `5` at
prefill and token `0` at the first decode step. An independent Transformers
5.12.1 `Qwen3_5ForConditionalGeneration` oracle, using DeepGEMM 2.6.1 after
removing 65 erroneous `.mlp.gate` skip-list entries that prefix-matched
`gate_proj`, produced the same tokens. Maximum absolute logit error was
`0.421875` for prefill and `0.375` for decode.

The following synchronized device-step profile is diagnostic, not HTTP serving
latency:

| Bucket | CUDA Graph | copying `Scatter` | `ScatterNoCopy` |
| --- | ---: | ---: | ---: |
| prefill, `s=4` | 86.229 ms | 1.817 ms | 0.954 ms |
| decode, `s=1` | 53.785 ms | 0.972 ms | 0.898 ms |

Before the compiler-side equivalence fix, a recurrent F32 arena with
`150,994,944` elements and a convolution BF16 arena with `5,898,240` elements
were materialized by large copying scatters on every step, making the same
bounded steps about 2.7 seconds. The corrected path is therefore roughly 31x
faster for this prefill device step and 50x for this decode device step. Those
ratios are not product-serving speedups.

## Matched short-trace serving diagnostic

Two alternating epochs used the same vLLM 0.29.0 benchmark client, checkpoint,
tokenizer, greedy sampling, fixed four-token request length, eight generated
tokens, eight requests, concurrency one, 1024-token model limit, and one H20.
The baseline and candidate servers ran sequentially. Prefix/radix reuse was
disabled. The exact memory allocators and engine internals differ, so this is a
matched workload diagnostic rather than an ISO-resource claim.

| Engine | Output tok/s | Median TTFT | Median TPOT | Median E2E |
| --- | ---: | ---: | ---: | ---: |
| OrbitKV | 14.07 | 226.14 ms | 50.45 ms | 579.67 ms |
| SGLang 0.5.17 | 31.61 | 118.06 ms | 19.07 ms | 251.51 ms |
| OrbitKV | 14.40 | 219.97 ms | 49.47 ms | 564.85 ms |
| vLLM 0.29.0 | 36.98 | 87.17 ms | 18.18 ms | 214.37 ms |

The paired median candidate/baseline ratios were:

| Baseline | Throughput | TTFT | TPOT | E2E | Random-trace digest |
| --- | ---: | ---: | ---: | ---: | --- |
| SGLang | 0.445x | 1.917x | 2.646x | 2.305x | equal |
| vLLM | 0.390x | 2.524x | 2.721x | 2.635x | different |

Lower latency ratios and higher throughput ratios favor OrbitKV. These results
show a clear serving deficit, not an OrbitKV win. The vLLM random-trace digest
differs in one of eight prompts; the other seven generated texts match.

A separate fixed pre-tokenized SSE probe sent `[1, 2, 3, 4]`, requested eight
tokens with greedy sampling and ignored EOS, and ran eight sequential requests
per engine:

| Engine | Generated token IDs | Median TTFT | Median token-span TPOT | Median E2E |
| --- | --- | ---: | ---: | ---: |
| OrbitKV | `5,0,31,0,31,0,31,0` | 221.20 ms | 49.91 ms | 569.36 ms |
| SGLang | `5,0,31,46474,4,5,0,31` | 115.02 ms | 18.97 ms | 245.66 ms |
| vLLM | `5,0,31,0,31,0,31,0` | 85.04 ms | 18.17 ms | 212.33 ms |
| Transformers oracle | `5,0,31,46474,4,5,0,31` | not timed | not timed | not timed |

All eight requests within each server were deterministic. SGLang matches the
independent oracle across all eight tokens. OrbitKV and vLLM match it through
the first three generated tokens and choose the runner-up at token four. A
direct eight-step OrbitKV logits run with the general test artifact matches the
oracle at every step with maximum absolute error at most `0.42285156`; the
serving artifact's token-four logits also stay within `0.5`, but its top two are
near-tied (`0=11.1875`, `46474=11.125` versus oracle `46474=11.25`,
`0=11.125`). Thus the service mismatch is a legitimate BF16/FP8 greedy boundary
effect, not state corruption. It still prevents a strict matched-output claim
against vLLM, whose choice follows OrbitKV at that boundary.

## Current blocker

The compiler-selected DeepGEMM path and the large-state in-place fix are real.
The next milestone is a robust multi-token correctness contract that accepts
bounded logit error without pretending near-tied argmax is bitwise portable.
Then attribute the remaining TTFT/TPOT gap to block-scaled linear choices,
unfused graph regions, gather/scatter metadata, scheduler/frontend overhead,
and attention; widen the search space and rerun longer, higher-concurrency
matched suites.
