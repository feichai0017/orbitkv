# Ordinary recovery evidence

Qwen3-8B, H20, TP=1, vLLM 0.29.0 and SGLang 0.5.20. Source:
`90bb88255597513be9c92ef32ed3f96ebee2c0e0`. Model revision:
`b968826d9c46dd6066d109eabc6255188de91218`.

See [the interpretation and limitations](../../../docs/recovery-performance.md).
Each run directory contains its launch arguments, commands, pinned packages,
storage evidence, latency samples, complete counter windows/batches and stage
percentiles. The aggregate JSON and CSV are generated with `benches.report`.
The source revision predates consumer preparation; this is the demand-only
profile, not a matched preparation on/off comparison.

The curated export removes output strings, duplicated per-request counter
snapshots and unrelated working-tree changes. It preserves all mismatch flags,
timing samples and batch/window counters. Full logs, prompts and outputs remain
under `benches/results/runs/20260922-recovery-*` on the qualification machine.
Only completed runs are included. The container's overlay storage qualifies
the functional SSD path, not a particular NVMe device.
