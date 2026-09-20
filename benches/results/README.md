# Recorded measurements

| Files | Experiment |
| --- | --- |
| `qwen3-8b-h20.csv`, `qwen3-8b-h20.json` | 270 requests: native HBM, built-in CPU cache, and OrbitKV on both engines |
| `qwen3-8b-comparisons.csv`, `qwen3-8b-comparisons.json` | 360 follow-up requests, matched LMCache controls, completion notification and copy-backend experiments, plus failed FlexKV attempts |

See the [analysis and reproduction instructions](../../docs/single-node-performance.md).
These are historical measurements, not claims about the latest working tree.
Their recorded launch commands, paths, versions, and source commits remain
unchanged when files are reorganized.

On the measurement host, original manifests, raw samples and service logs now
live in `runs/orbitkv-qwen3-8b/` and `runs/orbitkv-qwen3-8b-comparisons/`.
`runs/` is ignored by Git. New runs default to timestamped subdirectories there;
models, engine environments, and third-party build dependencies stay outside
the results directory.
