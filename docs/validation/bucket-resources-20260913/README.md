# Bounded bucket residency — H20, 2026-09-13

Retaining the selected prefill and decode graphs avoids repeated full graph
construction. Each FlashInfer plan now owns its integer metadata; the old
shared workspace let preparing another shape overwrite a retained graph's plan.
The engine/CLI exposes a positive bucket capacity, defaulting to one. Use
`--graph-cache-capacity 2` to retain both buckets when memory allows it.

Two timing processes per capacity run in ABBA order with the same frozen binary,
27B FP8 checkpoint, eight-row oracle and selected two-bucket artifact. Each
process performs four complete request lifecycles. Sequences 1 and 2 (zero-based)
measure repeated requests; sequence 0 initializes residency and sequence 3
forces capacity one and eviction. Timings exclude compilation and disable stage
tracing and per-node CUDA profiling.

| Diagnostic metric | Capacity 1 | Capacity 2 | Change |
| --- | ---: | ---: | ---: |
| Repeated-request prefill | 127.91 ms | 29.41 ms | -77.0% |
| First decode after prefill | 128.68 ms | 25.93 ms | -79.8% |
| Eight-step request, oracle checks and drain | 420.42 ms | 217.05 ms | -48.4% |
| Warm diagnostic decode | 24.49 ms | 24.39 ms | -0.4% |

Each table value is the median of two process medians. Full graph builds
increase by two per repeated request at capacity one. At capacity two they stay
at two across both repeated requests; after forced eviction, one graph remains
and the cumulative build counter increases to three. Explicit-CSR attention
still replans its islands. The result identifies phase-switch savings; the small
warm-decode difference does not establish faster kernels or serving throughput.

All five model processes (four timings plus a separate traced probe) pass 160
full-vocabulary reference rows with identical per-phase errors, maximum 0.5
under the unchanged 1.0 gate, and drain all 20 requests' KV/fixed-state ownership.
The probe loads all 428 saved CUDA module images and performs no NVRTC calls.
The GPU reports 0 MiB used after processes exit.

A separate provider regression captures decode, ragged decode and causal
sliding-window prefill, compares them with a CPU oracle, alternates replay,
retires one graph and replays the survivors. Its metadata-preservation assertion
fails against the old implementation. Eight provider tests and seven graph
resource/ordering tests pass after the fix. Planning includes each private
8 MiB metadata allocation, shared 128 MiB float scratch, and a replacement-plan
generation while new plans coexist with the old captured islands.

The final executable SHA-256 is
`bd3b0ce0e2a2757c000e108ce30e702eb2f737b1d72f95a0c48c02701b045fa5`.
All 433 frozen build-input files match the workspace. The exact artifact,
reference and source identities are in `environment.json`; per-process timings,
graph counters and limitations are in `summary.json`. Raw source snapshots,
binaries, full logs, the failed pre-fix regression and audit scripts remain in
`.qualification/bucket-resources-20260913/`. Six prior result packages retain
their checksums.

Qualification is short-context B1 on one H20 and one owning execution stream.
Driver-internal allocations are not fully captured by provider payload budgets.
Automatic residency budgeting, broader model transitions and a complete serving
workload remain open. See [graph residency](../../graph-residency.md) for the
runtime ownership contract and [roadmap](../../roadmap.md) for follow-up work.
