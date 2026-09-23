# Benchmark results

Keep maintained final summaries and reproduction instructions here. Benchmark
code and its tests live in `benches/` and `benches/tests/`. New result output is
ignored by default; raw samples, manifests, logs and failed attempts belong in
`runs/` or CI artifacts. Refresh an existing report after a matched rerun rather
than adding another copy in CSV and JSON.

| Maintained report | Scope |
| --- | --- |
| [Single-node offload](20260923-offload/README.md) | Large working set, concurrent SSD reads/writes and compute-batch controls |
| [Request preparation](20260922-preparation/README.md) | Repeated preparation controls and DRAM supplement |
| [Shared-cache serving](20260923-shared-cache/README.md) | Both engines' same-host TCP sharing and restart gates |

Historical single-node comparisons, SSD readiness, warming, client-control and
catalog-cleanup datasets are available in the
[immutable Git snapshot](https://github.com/feichai0017/orbitkv/tree/44c1e5f9a253aa7378c6187b2aeea9bff93df304/benches/results).
Their conclusions, limitations and reproduction commands remain in `docs/`.
Those data describe their original source revisions, not current performance.

This follows the separation used by LMCache's
[benchmark scripts](https://github.com/LMCache/LMCache/tree/b53efadb3812fc8ce520d2a9cda7c0a0f128b566/benchmarks)
and [generated-output rules](https://github.com/LMCache/LMCache/blob/b53efadb3812fc8ce520d2a9cda7c0a0f128b566/.gitignore).
