# Current evidence

This directory contains only compact evidence that directly qualifies the
current `core + Luminal executor + Rust server + engine` architecture. A record keeps
the concrete model, hardware/software environment, measured result, source
identity, and claim boundary. It must not embed source trees, binaries, package
caches, model weights, or superseded integration snapshots.

Removed records remain recoverable from Git history. They are not current
product evidence. New experiments should first write to `.qualification/` and
move a compact reviewed result here only after its checks pass.

## Retained records

| Record | Evidence | Boundary |
| --- | --- | --- |
| `bucketed-decoder-correctness-20260904` | Released dense checkpoint executes prefill and repeated decode through one searched two-bucket Luminal runtime and one persistent OrbitKV K/V arena | Correctness only; tiny batch-one workload |
| `on-device-greedy-correctness-20260904` | Device argmax matches host argmax and the default path returns token IDs rather than vocabulary logits | Greedy only; no sampling-performance claim |
| `decode-cuda-graph-correctness-20260905` | Stable-input outer-graph replay is correct; flattened capture is 24.9% slower than eager | Retained negative result; no benefit claim |
| `decode-child-graph-benefit-20260905` | Selected Luminal executables composed as child graphs reduce matched fixed-step batch-one median wall time by 5.8-8.3% in two runs | Narrow dispatch result; not serving throughput or lifecycle benefit |
| `released-hybrid-lifecycle-20260906` | Released 18-layer Full+Sliding checkpoint crosses its native window on H20, matches an independent greedy-token reference, reuses retired storage, cancels a second request, and fully drains | Correctness and lifecycle only; diagnostic timing is not a benefit result |
| `released-hybrid-residence-benefit-20260906` | Ten paired release-mode runs compare compiled and request-lifetime residence through one searched graph; output parity, resident bytes, fixed-budget reach, and paired timing interval pass | Narrow batch-one same-executor L5 result; not serving throughput or SGLang comparison |
| `continuous-batching-engine-20260907` | Released hybrid model executes B=2 prefill/decode with reference-token parity and admits late prefill during active decode | Continuous-batching correctness only; no HTTP, fairness, capacity, or performance claim |
| `single-process-http-engine-20260907` | Real released checkpoint serves OpenAI non-streaming and SSE completions through `orbitkv-serve`; concurrent requests, dropped-stream cancellation, shutdown, and final drain pass | HTTP correctness only; no fairness, capacity, or throughput claim |
| `serving-load-qualification-20260907` | One warm single-process engine completes fixed 16-request traces at C1/C2/C4/C8 with full outputs; B=8 row isolation and bounded B=1/B=8 logit parity pass | Narrow internal scaling frontier; no fairness, soak, capacity-limit, SGLang comparison, or performance-advantage claim |
| `sglang-product-comparison-20260907` | Four artifact-fixed alternating H20 epochs compare the released hybrid checkpoint against clean stock SGLang v0.5.17; request gates and per-arm repeatability pass | Negative serving-performance result: 0.534x throughput, 1.76x TPOT, 6.50x TTFT; configured KV payload is 40.4% smaller; cross-engine output digest differs |
| `compiler-constrained-schedule-benefit-20260907` | Required persistent-state aliases plus deeper search improve the same OrbitKV engine by 14.5% throughput, 32.2% TTFT, 10.2% TPOT, and 12.8% E2E; B2 reference probe passes | Current SGLang comparison remains negative at 0.598x throughput, 1.59x TPOT, and 4.79x TTFT; random-trace digests differ |

The normative current support boundary is the
[Capability Matrix](../docs/capability-matrix.md). The mechanism ablation proves
a same-semantics physical-residence reduction and host fixed-capacity admission
difference. The released hybrid benefit record now establishes a narrow
same-executor compiler benefit. The HTTP record closes model-backed serving
correctness. The serving-load record adds a narrow C1-C8 throughput/latency
frontier. The first SGLang record closes R4 as a negative performance result.
The compiler-constrained follow-up proves a real same-engine schedule-selection
improvement while keeping R4.1 open for graph-internal kernel and fusion work.

## Result-package policy

A promoted result directory contains only:

- `environment.json`: hardware, driver/runtime, model and weight identity,
  dependency versions, and source revisions;
- raw benchmark JSON or a compact reviewed projection of its measured fields;
- `summary.json`: paired metrics, correctness gate, and qualified claims;
- optional checksums for those files.

The matched serving harness and promotion rules are documented in
[benchmarking.md](../docs/benchmarking.md).
