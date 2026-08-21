# Official SGLang v0.5.17 ABI5-v5 Grouped-Release H20 Evidence

This append-only archive qualifies a narrow correctness profile for the
compact batch-only ABI5 manager with grouped request release. It does **not**
claim production readiness, general SGLang replacement, speedup, KV
compression, or memory savings.

The run used one NVIDIA H20 and official SGLang `v0.5.17` at peeled commit
`29481685462732237d80d86076d6563e1f658102`. Both profiles used page size 16,
BF16 NHD KV, eager execution, ChunkCache, and TP/PP/DP/DCP=1, with Radix,
overlap, CUDA Graph, speculation, and disaggregation disabled. Qwen2.5-7B
used Full attention with FlashInfer. GPT-OSS-20B used ordered Full+SWA
attention with FA3 and SGLang's explicit built-in `moe_runner_backend=triton`
path; the default external `triton_kernel` path is outside this scope.

## Correctness result

The four manager records are PRIMARY evidence and the four stock records are
paired references. The frozen runner's own pair checker passed all four pairs.
It checks the pair contract, deterministic inputs, submitted and returned
request IDs, every returned token, per-request and aggregate token digests,
within-lane stability, capacity and checkpoint identity, exact official
release identity, final manager census, SWA behavior, and grouped lifecycle
counters.

For B4, each record contains four concurrent requests repeated for five
iterations in one engine process. Every stock/manager token trace matches
exactly. Each B4 manager record performs 5 grouped `release_batch` calls, 5
request-recycle calls, and releases 20 requests total. Qwen finishes with
256/256 Full pages free. GPT-OSS finishes with 256/256 Full and 205/205 SWA
pages free; it records 520 SWA retirement certificates, 520 reclaimed pages,
and 60 wraps. All manager records finish with no active requests or pages and
zero failure, quarantine, hot-workspace-allocation, capacity-memset,
root-crossing, or materialized-page counters.

## Claim boundary

The stock and manager records use identical SGLang tensor-arena sizing within
each pair. Same-cap intrinsic KV reduction is therefore 0%; there is no
compression or memory-saving claim.

Performance is diagnostic only and `performance_go=false`. The archive has
one run epoch. Qwen B1's one-sample manager overhead is
`+5.0008848205%`, marginally above the strict `<=5%` red-team guard. For
context, B4 steady medians (iteration zero excluded) are `+4.1932114820%` for
Qwen and `-5.2048208958%` for GPT-OSS. These numbers do not support a speedup
or production-performance claim.

Prefix/Radix/COW, overlap, CUDA Graph, speculation, disaggregation,
multi-GPU, TP/PP/DP/DCP greater than one, other attention families, and any
future source or artifact bytes remain unqualified.

## Evidence layout

- `raw/`: eight v5 JSON records and eight stderr logs.
- `source-closure/`: the runner's complete ordered 23-file source closure;
  aggregate SHA-256
  `9233c06d40ffa19eb08b88cc1cb6fa3b72cceffcaecc96c830950f93bd5c70bc`.
- `qualification/`: exact runner, requirements lock, two plans, and ABI5
  release library used for this epoch.
- `sglang-v0.5.17/`: exact stock and manager loader bytes.
- `summary.json`: scoped result, counters, and explicit non-claims.
- `provenance.json`: source, runner, library, loader, and evidence identities.
- `manifest.json`: repository-relative hashes for all evidence files except
  `manifest.json` and `SHA256SUMS`.
- `SHA256SUMS`: hashes every archive file except itself, including
  `manifest.json`, breaking the otherwise circular seal dependency.
