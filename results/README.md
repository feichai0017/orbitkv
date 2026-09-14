# Current evidence

The [environment and search qualification](environment-search-20260914/README.md)
records strict provider/toolchain admission, 152 passing model reference
comparisons and a 73.7% reduction in an isolated decoder-query benchmark.
Complete cold compilation is 350 s in this run; steady serving performance
remains unqualified by this experiment.

The [CUDA backend qualification](cuda-backend-refactor-20260914/README.md) records
provider source/build identities, device-aware compiler facts, 148 backend test
executions and 152 passing B1/B8 reference comparisons. Cold compilation remains
costly; this reorganization does not establish a serving-performance improvement.

The [Rust 1.98 follow-up](workspace-integration-lint-20260914/README.md) records
typed byte conversion, all-target CUDA/executor Clippy and 144 B1/B8 reference
comparisons against the rebuilt binary.

The [integrated workspace qualification](workspace-integration-20260914/README.md)
records the compiler/crate migration, fresh compilation, and final-binary B1/B8
replay. Earlier records retain their original source identities.


This directory contains only compact evidence that directly qualifies the
current state-manager, compiler, CUDA executor and Rust engine/HTTP architecture. A record keeps
the concrete model, hardware/software environment, measured result, source
identity, and claim boundary. It must not embed source trees, binaries, package
caches, model weights, or superseded integration snapshots.

Removed records remain recoverable from Git history. They are not current
product evidence. New experiments should first write to `.qualification/` and
move a compact reviewed result here only after its checks pass.

The [hotspot-search qualification](hotspot-search-20260914/README.md) passes
296 reference comparisons and records 49 measured local neighbors with no
state/resource rejection. Two prefill neighbors replace generated vocabulary
GEMMs with cuBLASLt. B8 diagnostic decode is 37.29 ms, while B1 is unchanged;
the B8 seed already uses cuBLASLt, so historical differences are not a search
ablation or serving claim.

## Retained records

| Record | Evidence | Boundary |
| --- | --- | --- |
| [environment-search-20260914](environment-search-20260914/README.md) | Selected CUDA environment validation, four expected artifact rejections, staged MXFP4 matching and 152 passing B1/B8 reference comparisons | Isolated decoder-query median 103→27 s; small routed graphs regress slightly; complete cold compile 350 s is a historical comparison, not a serving-speedup claim |
| [cuda-backend-refactor-20260914](cuda-backend-refactor-20260914/README.md) | Unified provider lock/build cache, separated Rust/CUDA/egglog sources, device facts joined with state constraints, and final-binary B1/B8 parity and replay | Unchanged provider pins and package versions; cold compile 943 s and replay preparation about 17.5 s; no serving-speedup or joint KV-layout search claim |
| [state-preflight-20260914](state-preflight-20260914/README.md) | Required state aliases checked before CUDA preparation; 296 reference comparisons and final drains pass | Rejected-candidate evaluation is 8.00 s versus the preceding 81.06 s observation; independent snapshots and reused caches prevent causal speedup claims; warmed decode remains close |
| [search-coverage-20260914](search-coverage-20260914/README.md) | Fixed-snapshot sampling, bounded initial exploration, 56 measured graphs and 296 passing logit comparisons on H20 | Mixed performance: B8 prefill improves against the prior observation while decode regresses; all 263 rejected graphs violate state aliases; no serving speedup claim |
| `bucketed-decoder-correctness-20260904` | Qwen2.5-0.5B-Instruct executes prefill and repeated decode through one searched two-bucket Luminal runtime and one persistent OrbitKV K/V arena | Correctness only; tiny batch-one workload; removed relocation observations were dropped from the compact record |
| `on-device-greedy-correctness-20260904` | Device argmax matches host argmax and the default path returns token IDs rather than vocabulary logits | Greedy only; no sampling-performance claim |
| `decode-cuda-graph-correctness-20260905` | Stable-input outer-graph replay is correct; flattened capture is 24.9% slower than eager | Retained negative result; no benefit claim |
| `decode-child-graph-benefit-20260905` | Selected Luminal executables composed as child graphs reduce matched fixed-step batch-one median wall time by 5.8-8.3% in two runs | Narrow dispatch result; not serving throughput or lifecycle benefit |
| `released-hybrid-lifecycle-20260906` | Gemma 3 270M text checkpoint crosses its native Full+Sliding window on H20, matches an independent greedy-token reference, reuses retired storage, cancels a second request, and fully drains | Correctness and lifecycle only; diagnostic timing is not a benefit result |
| `released-hybrid-residence-benefit-20260906` | Ten paired release-mode runs compare compiled and request-lifetime residence through one searched graph; output parity, resident bytes, fixed-budget reach, and paired timing interval pass | Narrow batch-one same-executor L5 result; not serving throughput or SGLang comparison |
| `continuous-batching-engine-20260907` | Released hybrid model executes B=2 prefill/decode with reference-token parity and admits late prefill during active decode | Continuous-batching correctness only; no HTTP, fairness, capacity, or performance claim |
| `single-process-http-engine-20260907` | Real released checkpoint serves OpenAI non-streaming and SSE completions through `orbitkv-serve`; concurrent requests, dropped-stream cancellation, shutdown, and final drain pass | HTTP correctness only; no fairness, capacity, or throughput claim |
| `serving-load-qualification-20260907` | One warm single-process engine completes fixed 16-request traces at C1/C2/C4/C8 with full outputs; B=8 row isolation and bounded B=1/B=8 logit parity pass | Narrow internal scaling frontier; no fairness, soak, capacity-limit, SGLang comparison, or performance-advantage claim |
| `sglang-product-comparison-20260907` | Four artifact-fixed alternating H20 epochs compare the released hybrid checkpoint against clean stock SGLang v0.5.17; request gates and per-arm repeatability pass | Negative serving-performance result: 0.534x throughput, 1.76x TPOT, 6.50x TTFT; configured KV payload is 40.4% smaller; cross-engine output digest differs |
| `compiler-constrained-schedule-benefit-20260907` | Required persistent-state aliases plus deeper search improve the same OrbitKV engine by 14.5% throughput, 32.2% TTFT, 10.2% TPOT, and 12.8% E2E; B2 reference probe passes | Current SGLang comparison remains negative at 0.598x throughput, 1.59x TPOT, and 4.79x TTFT; random-trace digests differ |
| `deepgemm-luminal-bringup-20260909` | Qwen3.5 27B block-FP8 executes through compiler-selected DeepGEMM candidates; required in-place fixed-state writes remove the large arena copy path, and eight-step independent logit parity passes | Negative C1 serving result at 0.445x SGLang and 0.390x vLLM throughput; a near tie reverses the fixed trace's generated token-four argmax |
| [fp8-region-tuning-20260912](fp8-region-tuning-20260912/README.md) | Artifact-bound workload tuning, searchable shared FP8 preparation, and captured scratch lifetime; Qwen3.8 27B passes B4/B8 aligned and ragged reference/replay/drain on H20 | Shared preparation remains opt-in. Two-epoch same-server diagnostic: throughput +1.82%, TPOT -5.82%, TTFT +8.26%; two of eight texts differ and other kernel choices change, so no performance benefit is qualified |
| [fp8-logit-diagnosis-20260912](fp8-logit-diagnosis-20260912/README.md) | Frozen-v3 strict off/on replay localizes both serving divergences using full-vocabulary independent model-reference traces: OFF maximum ties and the existing highest-index argmax rule explain actual token choices | Sequential B1 teacher-forced diagnosis; no sampling-policy change, internal-provider attribution, or performance qualification |
| [compiler-boundaries-20260912](compiler-boundaries-20260912/README.md) | Executor/provider responsibility split, semantic program identity, and complete candidate trace; final-source Qwen3.8 27B passes fresh search, strict replay, eight reference steps, and drain on H20 | B1 two-bucket correctness and trace-identity validation; no serving-performance or search-quality claim |
| [engine-stage-attribution-20260913](engine-stage-attribution-20260913/README.md) | Merged engine/frontend crate, buffered compiler/runtime stage timing, and five H20 search/replay/control processes with eight reference steps and drain | B1/two-bucket attribution: egglog dominates cold startup, weight loading dominates replay preparation, and first decode incurs graph materialization; no serving-speedup claim |
| [module-image-artifact-20260913](module-image-artifact-20260913/README.md) | Decoder artifact embeds selected CUDA images; nine H20 processes pass eight reference steps and drain; two fixed-artifact timing pairs eliminate 428 NVRTC calls, reduce schedule load 56.5% and diagnostic process time 14.0% | Startup only; fresh capture adds 7.43 s, warm diagnostic decode remains about 24.5 ms; no serving-throughput or cold-search benefit claim |
| [weight-loading-20260913](weight-loading-20260913/README.md) | Fallible dtype-driven loader eliminates intermediate host byte copies; two fixed-artifact H20 timing pairs reduce weight loading 71.4% and complete diagnostic process time 48.9%; all eight processes preserve reference errors and drain | Startup only; existing caches and buffered stage tracing, warm diagnostic decode stays about 24.6 ms; old decoder formats now require regeneration |
| [bucket-resources-20260913](bucket-resources-20260913/README.md) | Private FlashInfer plan metadata, replacement-peak budgeting and finite bucket residency; H20 capacity 1/2 comparison reduces repeated prefill 127.91→29.41 ms and first decode 128.68→25.93 ms; 160 reference rows and 20 drains pass | Short-context B1 phase transitions; default capacity remains one, warm diagnostic decode stays about 24.4 ms; broader serving and automatic residency budgeting remain open |
| [bucket-serving-20260913](bucket-serving-20260913/README.md) | Same-server capacity 1/2 HTTP comparison completes 128 requests and 4,608 tokens with identical output and checked final state drain; observed throughput +86.8% for short output and +13.0% for 64-token output | C1 and short input, two timing processes per arm/profile; steady ITL stays 23.6 ms, short-output P99 TPOT +16.8%; default capacity remains one |
| [startup-preparation-20260913](startup-preparation-20260913/README.md) | Same-binary startup off/on comparisons at separately fixed capacities 2 and 1 complete 352 requests and 13,824 tokens in 24 processes; capacity-two short-output first stream interval 138.9→29.9 ms and P99 TPOT 37.7→24.5 ms; 128 reference row comparisons and final drain pass | H20/C1 paired observations; preparation cost moves before ready, steady ITL remains about 23.6 ms; default-capacity eviction regression corrected, but its observed P99 TPOT and throughput remain negative; rejected long-prefill workload retained; no automatic residency selection |

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
