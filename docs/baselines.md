# Pinned baselines

This file is the M0 source-of-truth for inputs that affect compiler semantics or
performance claims. A revision change is deliberate and reviewed; `main` is not
a reproducible dependency.

## Execution substrate

| Component | Revision or contract | Role |
| --- | --- | --- |
| `pegainfer-project/kern` | `05df6d9cf8233b2438a7a584ce4ed7a0666abf53` | Direct `kern-manifest` dependency and unmodified execution substrate |
| Manifest | schema v5 | Sole emitted runtime artifact contract |

## Model contracts

| Target | Checkpoint revision | Initial hardware boundary |
| --- | --- | --- |
| `Qwen3.8-27B-FP8` | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | Complete target-only decoder on one H20 / SM90 |
| `GLM-5.3-Flash` | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | Layer fixtures on H20; complete model needs explicit residency |
| `DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` | Blackwell or another qualified native-FP4 multi-GPU path |

## Semantic and performance oracles

The initial upstream code snapshots inspected during architecture design were:

- vLLM `6ca2b23e22aab8534e00a85e8ec5222508a6ecf3`;
- SGLang `b02e16a895add01a0cfe24bb74922de92ab4d895`;
- PegaInfer `72cbbe8a72e06329b2b4d6fa1e8e906acf2acc85`.

They are references, not build dependencies. Before the M1 performance gate,
the benchmark environment must additionally pin engine release commits, CUDA,
driver, provider revisions, clocks, request traces, and complete command lines.
No comparison from the archived implementation is a performance claim for this
line.
