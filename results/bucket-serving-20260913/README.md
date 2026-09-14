# Bucket residency in serving — H20, 2026-09-13

The verified `orbitkv-serve` executable serves the same Qwen3.8 27B FP8 artifact
with graph-cache capacity one and two. The comparison changes only that capacity.
It keeps the checkpoint, quantization tuning, page/state pools, request trace,
client, sampling and selected programs fixed. Each profile has two timing
processes per capacity in ABBA order, with 16 requests per process at C1.

| HTTP workload and metric | Capacity 1 | Capacity 2 | Paired change |
| --- | ---: | ---: | ---: |
| Short output: output throughput | 18.59 token/s | 34.73 token/s | +86.8% |
| Short output: median TTFT | 161.64 ms | 49.30 ms | -69.5% |
| Short output: median TPOT | 38.98 ms | 23.87 ms | -38.8% |
| Short output: P99 TPOT | 39.91 ms | 46.62 ms | +16.8% |
| 64-token output: output throughput | 35.73 token/s | 40.35 token/s | +13.0% |
| 64-token output: median TTFT | 162.08 ms | 49.33 ms | -69.6% |
| 64-token output: median TPOT | 25.81 ms | 24.21 ms | -6.2% |

Arm columns are medians of the two run metrics. Changes are medians of paired
ratios. The client requests four input tokens and records three after random
prompt tokenization. Outputs have exactly 8 or 64 tokens. All initial requests
are included; no first-request samples were removed. Compilation and startup are
outside these HTTP measurements.

All eight timing processes complete 128 requests and 4,608 output tokens without
client errors. All four generated-text digests agree within each profile.
The server exits normally and its final report shows no active requests,
reserved/writing/live/retiring/quarantined KV pages, references, reader pins or
pending lifecycle work. Every fixed-state slot is free with no owners or pending
transitions. Full graph builds fall from 32 to 2 per process; the report records
one versus two retained graphs just before decoder teardown.

Median inter-token latency stays around 23.6 ms. This is evidence of reduced
phase-switch overhead. It does not demonstrate faster kernels. The short-output
P99 TPOT regression is retained: one capacity-two process has a 287 ms first
decode interval, while the other has 137 ms. Two process samples cannot establish
its long-run distribution. Startup prewarming and broader tail qualification
remain necessary before changing the default capacity of one.

A separate final-source HTTP probe reproduces the first three independent
reference token IDs `[5, 0, 31]` as `&!@`. It then keeps a 512-token SSE request
open, waits for its first nonempty token and sends SIGTERM. The engine cancels
that request and drains KV/fixed state. Exit takes about 10.5 s, including the
frontend grace interval. `completed_requests` counts all terminal requests, so
that report has two completed requests, of which one is cancelled.

`ModelEngine::shutdown()` now closes admission, requests cancellation and joins
the worker with a checked `EngineShutdownReport`. Worker failures and incomplete
retirement propagate to the executable's exit status. The benchmark harness
records exit status, rejects forced SIGKILL, and observes the current Luminal
checkout path. All new test source is under the owning `tests/` directories.

The final executable SHA-256 is
`432c46434f7ccf0d00b3f9d98f8ee8d1d6fd1dfe93693581a9293fef17465598`.
Its 435 frozen build inputs match the workspace. The serving artifact contains
418 generated CUDA images; the final traced probe loads them with zero NVRTC
calls. Its state-pool geometry differs from the earlier decoder diagnostic
artifact, so artifacts were not relabelled or edited to bypass identity checks.

The record covers short-input B1/C1 execution on one H20 with an explicit shared
FP8 tuning profile. It makes no multi-request batching, long-prefill, sustained
load, other-device, statistical-significance or reference-engine comparison
claim. Full measurements and tail metrics are in `summary.json`; source,
artifact and environment identities are in `environment.json`. Raw binaries,
source snapshots, logs and audit scripts remain under
`.qualification/bucket-serving-20260913/`. Seven prior result packages retain
their checksums. See [Luminal design](../../docs/luminal-design.md),
[graph residency](../../docs/graph-residency.md) and the
[roadmap](../../docs/roadmap.md).
