# Official SGLang v0.5.17 ABI5 Full and Hybrid H20 Qualification

This append-only evidence set qualifies compact batch-only ABI5 correctness in
a deliberately narrow real-engine profile. It does **not** grant performance
GO or general SGLang replacement status.

The run used one NVIDIA H20, official SGLang `v0.5.17` at peeled commit
`29481685462732237d80d86076d6563e1f658102`, page size 16, BF16 NHD KV,
eager execution, ChunkCache, TP/PP/DP/DCP=1, disabled Radix, overlap, CUDA
Graph, speculation, and disaggregation. Qwen2.5-7B used Full attention with
FlashInfer. GPT-OSS-20B used ordered Full+SWA attention with FA3 and the
explicit SGLang built-in `moe_runner_backend=triton` profile. The GPT result
does not qualify the default external `triton_kernel` MoE path.

## Result

Four manager records are PRIMARY correctness evidence. Their four stock
records are paired references, not additional PRIMARY records. For B1 and B4,
the pair contract, deterministic inputs, request IDs, complete 33-token output
traces, per-request digests, capacity readback, checkpoint identity, and
official-release identity match exactly. B4 repeats each request position five
times within one engine process and the complete token sequences remain stable.

All manager records end with every page free and all request/step/reclamation
state drained. Full-only records report zero SWA activity. Hybrid records show
real SWA retirement: B1 reports 26 certificates/pages reclaimed and 3 wraps;
B4 reports 520 certificates/pages reclaimed and 60 wraps. Failure,
quarantine, hot-workspace-allocation, capacity-memset, root-crossing, and
materialized-page counters are all zero.

The stock and manager runs use the same SGLang tensor-arena sizing cap in each
pair. Intrinsic same-capacity KV memory reduction is therefore **0%**; this
record makes no compression or memory-saving claim.

## Performance boundary

The diagnostic metric is the B4 steady median after excluding iteration zero
from each five-iteration sequence:

| Model/profile | Stock median | Manager median | Manager overhead | Gate |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-7B Full | 1.0345796533 s | 1.0404137820 s | +0.5639129563% | non-regression pass |
| GPT-OSS-20B Full+SWA, built-in Triton MoE | 1.4032781962 s | 1.4735001586 s | +5.0041369271% | fails strict `<=5%` |

These measurements contain five in-process iterations but only one run epoch;
there are no repeated-epoch statistics. Consequently `performance_go=false`:
Qwen passes the non-regression boundary, while Hybrid is marginally above the
strict threshold and neither profile has repeat-run confidence.

## Evidence layout

- `raw/`: the eight v4 JSON records and their eight stderr logs. Only the four
  `*-manager.json` files listed in `summary.json` are PRIMARY.
- `source-closure/`: all 23 source files in the runner's frozen closure, with
  aggregate SHA-256
  `e61045f6fe3731f2bf76fb141f385c2898b3254b056a6818acdc216ea555b567`.
- `qualification/`: the exact runner, requirements lock, plans, and ABI5
  release library used by the epoch.
- `sglang-v0.5.17/`: pristine stock and patched manager loader bytes.
- `diagnostics/`: three earlier superseded epochs, retained byte-for-byte only
  for diagnosis. They are explicitly excluded from PRIMARY qualification and
  performance statistics.
- `provenance.json`: source, loader, runner, library, and input identities.
- `summary.json`: qualification result and its scope.
- `manifest.json`: SHA-256 for every other file in this directory tree.

Prefix/Radix/COW, overlap, CUDA Graph, speculation, disaggregation, multi-GPU,
TP/PP/DP/DCP greater than one, other attention families, intrinsic
compression, and production performance remain unqualified.
