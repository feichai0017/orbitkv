# Recorded measurements

| Files | Experiment |
| --- | --- |
| `catalog-cleanup-d0.json` | D0 owner cleanup: 10,000 owned keys in a million-key directory, 10 CPU samples; setup and directory destruction excluded |
| `qwen3-8b-h20.csv`, `qwen3-8b-h20.json` | 270 requests: native HBM, built-in CPU cache, and OrbitKV on both engines |
| `qwen3-8b-comparisons.csv`, `qwen3-8b-comparisons.json` | 360 follow-up requests, matched LMCache controls, completion notification and copy-backend experiments, plus failed FlexKV attempts |
| `qwen3-8b-ssd.csv`, `qwen3-8b-ssd-summary.csv`, `qwen3-8b-ssd-summary.json` | 120 requests with SSD enabled: vLLM restores and SGLang's unused SSD prefetches, with DRAM and cold controls |
| `qwen3-8b-query-readiness.csv`, `qwen3-8b-query-readiness-summary.csv`, `qwen3-8b-query-readiness-summary.json` | 120 requests after query ownership and SGLang admission changes: 15/15 SSD restores per engine, with DRAM and cold controls |
| `qwen3-8b-query-budgets.csv`, `qwen3-8b-query-budgets-batches.csv`, `qwen3-8b-query-budgets-summary.csv`, `qwen3-8b-query-budgets-summary.json` | 468 requests in 108 shared/mixed 1/4/8-concurrent bursts with a 2 GiB query budget; counters are recorded per burst |
| `qwen3-8b-query-budgets-output-controls.json` | Exact varying prefix, native batching controls, and deterministic OrbitKV DRAM/SSD output checks |

See the [analysis and reproduction instructions](../../docs/single-node-performance.md).
The [SSD report](../../docs/ssd-performance.md) explains the forced-tier workload,
the original SGLang readiness limitation, and the qualified recovery follow-up.
Its baseline raw runs live in `runs/qwen3-8b-ssd-20260921/`; follow-up runs live
in `runs/query-readiness-{vllm,sglang}/` on the measurement host.
These are historical measurements, not claims about the latest working tree.
The [concurrent report](../../docs/concurrent-performance.md) explains byte
accounting, shared reads versus GPU destinations, the SGLang admission fix, and
the retained ordinary-mode output differences. Its raw runs are
`runs/query-budgets-{vllm,sglang}/`; controls and reproduction scripts are in
`runs/query-budgets-output-controls/` and `runs/query-budgets-output-controls-vllm/`.
The incomplete pre-fix run remains in `runs/query-budgets-sglang-failed-shared-prefix/`.
Their recorded launch commands, paths, versions, and source commits remain
unchanged when files are reorganized.

On the measurement host, original manifests, raw samples and service logs now
live in `runs/orbitkv-qwen3-8b/` and `runs/orbitkv-qwen3-8b-comparisons/`.
`runs/` is ignored by Git. New runs default to timestamped subdirectories there;
models, engine environments, and third-party build dependencies stay outside
the results directory.
