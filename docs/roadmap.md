# Roadmap

The acceptance model is **Qwen3.8-27B-FP8 on one H20**. The engine, compiler and
state manager live in one Rust workspace. Model-specific facts belong in import
contracts, workload manifests and tests; implementation selection belongs in
compiler rules and measured backend search.

[Model support](capability-matrix.md) defines the current scope.
[Results](../results/README.md) record measured inference performance.

The [three-engine baseline](../results/qwen3.8-27b-fp8-h20-20260915/README.md)
now uses one official `vllm bench serve` client and fixed token-ID traces.
All 576 requests complete, but OrbitKV has C8 text variation and repeated latency
outliers. This establishes a diagnostic baseline, with input/concurrent
correctness required before an output-equivalent comparison.

Prepared execution now records complete DeepGEMM tiles and proved row bounds.
FlashAttention-3 records context capacity and reads actual lengths from GPU
metadata, so context growth can reuse its plan and capture. A controlled H20
four-workload trace finds no native JIT, compilation or library load after the
first request starts. Strict replay, independent numerical checks and captured
resource-lifetime tests cover these contracts. Model reports retain the
intermediate C1 regression and the final untraced measurements; removing native
first-use stalls does not establish a general throughput improvement.

## Next milestones

| Priority | Deliverable | Acceptance |
| --- | --- | --- |
| 1. Input and concurrent correctness | Resolve HTTP tokenizer differences; reproduce C8 text variation with identical teacher-forced histories and recorded batch geometry | Tokenize/detokenize parity against the checkpoint reference; identify the violated numerical/selection/state contract or prove measured near-tie behavior; preserve tolerances and test changing batches and state reuse |
| 2. Profile-driven regions | Improve expensive recurrent, normalization/gating or projection regions through equivalent egglog candidates | Independent operator references, full-model logits, ragged prefill/decode and drain pass; complete-workload measurements improve over the existing composition |
| 3. Wider workload coverage | Longer prefill/context, concurrency and memory pressure; shared-prefix and cancellation traces | Find actual capacity and tail-latency limits, preserve correctness under admission pressure, record all failures and memory budgets |
| 4. Joint state and compute plans | Search multiple legal state realizations under one device-memory budget | Include KV, fixed state, workspace and resident graphs; validate layout transitions and count movement/preparation costs in the measured objective |

## Compiler direction

1. **Provider alternatives.** Keep mature library implementations as candidates
   with explicit numerical, layout, target and workspace contracts. Preserve
   source pins, wrapper identity and build/cache ownership in the CUDA crate.
2. **Algorithm regions.** Add compositional alternatives only where profiles show
   useful work to eliminate or share. A Triton/TileLang adapter is justified by a
   measured candidate, and compiles outside request execution.
3. **Joint realization.** The executor coordinates state and compute search.
   The state manager keeps exclusive authority over pages, publication and reuse.
   External KV restore versus recomputation must include transport and overlap.
4. **Persistent execution.** Explore larger generated regions after dependency,
   progress, resource and effect contracts can be proved. A whole-model
   megakernel is an option, not a prerequisite or a promised speedup.

See [joint compilation](joint-compilation.md) for the design and
[compiler boundaries](compiler-boundaries.md) for current extension points.

## Performance and release gates

Use the [benchmark method](benchmarking.md) for every published model comparison.
Keep compile/startup cost separate from HTTP serving and compare representative
workloads rather than one favorable row. Small fixed traces are diagnostic;
a performance-win claim needs adequate repeated samples and confidence bounds
against both baselines, with no P95 TTFT/TPOT regression.

Further models enter the public support list only after checkpoint identity,
independent numerical checks, lifecycle closure and an H20 serving report.
Quantization and CPU offload are permitted by the deployment budget, but each
requires explicit storage, transfer and execution contracts before admission.

Engineering gates remain formatting, Clippy, host tests, relevant CUDA/model
checks and website validation. Keep tests under their owning `tests/` directory,
retain upstream licenses, and remove superseded notes instead of maintaining a
second implementation history in documentation.
