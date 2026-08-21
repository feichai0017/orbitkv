# Official SGLang v0.5.17 Full and Hybrid H20 Qualification

This append-only record qualifies two narrow, exact-source OrbitKV manager
paths on one NVIDIA H20 against stock official SGLang `v0.5.17`, peeled commit
`29481685462732237d80d86076d6563e1f658102`:

- Qwen2.5-7B Full attention with FlashInfer, B1 and B4;
- GPT-OSS-20B ordered Full+SWA attention with FA3 and a 128-token window,
  B1 and B4.

The primary records use the post-hardening manager (`03dc15b…` plugin and
`2c1d383…` runtime). Every stock/manager pair has an identical pair key, checkpoint identity,
engine contract, input digest, capacity readback, completion count, and output
token digest. All four manager runs end fully reclaimed: every arena page is
free and every live, prepared, submitted, reserved, writing, retiring,
quarantined, exhausted, and pending-reclamation counter is zero. GPT B1/B4
also expose ordinary-path SWA activity of 26/104 retirement certificates,
26/104 reclaimed pages, and 3/12 ring wraps. Qwen correctly reports SWA as not
applicable with zero counters.

## Claim boundary

This is same-capacity correctness and lifecycle evidence. Stock and manager
use the same SGLang tensor-arena sizing within each pair and report the same KV
cache size. The intrinsic same-capacity memory reduction is **0%**. These runs
do not test token compaction, compression, or a smaller admission budget.

The B4 five-iteration diagnostic has equal pair keys and output digests. Its
median manager overhead is 13.9113% for Qwen (1.1933137 s versus 1.0475819 s)
and 6.7148% for GPT-OSS (2.3761110 s versus 2.2265993 s). This is a
same-capacity smoke result, not production performance qualification. It
motivates ABI5 batch operations to reduce per-request FFI/control overhead;
there is **no performance GO and no speedup claim**.

The qualified engine boundary is page16 BF16 NHD KV, eager execution,
ChunkCache, one GPU, TP/PP/DP/DCP=1, prompt 513, decode 33, chunked prefill
256, and B1/B4 only. Radix/Prefix/COW, overlap, CUDA Graph, speculation,
disaggregation, multi-GPU, churn/cancellation, other attention families, and
production operation remain unqualified.

## Source boundary

The primary manager and i5 records use plugin/runtime/harness hashes
`03dc15b…` / `2c1d383…` / `3e28e933…`; their byte-exact adapter tree is under
`source-snapshot/posthardening/`. Three primary stock baselines were captured
before hardening with the same harness and explicit stock sentinel, so they do
not execute the captured OrbitKV plugin/runtime. The post-hardening Qwen B1
stock record and both i5 stock records bind the post-hardening tree directly.

The initial manager runs are retained only as superseded diagnostics. Their
`162d1b9c…` / `d787bcfc…` generation and Qwen B1's still-earlier
`9511df8a…` / `4b716083…` plugin/harness bytes are also preserved; they are
not the primary qualification. `provenance.json` records this distinction.
The native ABI4 library, both plans, the stock and patched SGLang loader bytes,
and every record-declared adapter source are archived.

The mutable repository worktree changed after these runs and is not silently
attributed to them. The records identify the native library by exact binary
hash but do not contain an exhaustive Rust rebuild-source inventory.

See `summary.json` for the checked claims, `provenance.json` for source
identities, `raw/` for the eight JSON records and eight stderr logs, and
`manifest.json` for the archive-wide hash inventory.
