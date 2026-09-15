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

Compiler preparation now separates reusable definitions/query plans, model
facts and independent bucket interval analysis. Hotspot search can enumerate
bounded connected dependency choices and records every changed binding; full
candidate GPU measurements and deployment finalist checks remain authoritative.
The dependency workload profile opts into three changes per attempt. This adds
search coverage, not a promise that existing kernels become faster.

Numerical probes now accept explicit submission and lifetime plans for different
request histories, chunked prompts and reordered rows. Gather/cast regions retain
the original views and conversion boundaries while avoiding an intermediate.
Retained FlashInfer graphs now validate the query and page segmentation consumed
by their prepared plans: unchanged shapes and addresses do not prove unchanged
CSR contents. The regression covers changed query/KV segmentation, bucket reuse
and reuse without replanning when metadata is identical. Full-model C8 replay
with multiple resident buckets matches single-bucket residency on the measured
teacher-forced histories. These checks do not close the C8 gate: shape-dependent
output variation and longer-history reference errors still require diagnosis
before promotion. Wider region candidates follow that numerical diagnosis.
The [serving rerun](../results/qwen3.8-27b-fp8-h20-20260915-regions/README.md)
records C8 throughput changes alongside output variation and C1 tail latency;
it does not pass the performance-promotion gate.

The first upstream-inspired work elimination is available as an artifact-bound
decoder output policy: select request-final hidden rows before normalization and
LM-head projection. Isolated checkpoint projection preserves complete logits,
but independently searched full programs still exceed the existing equivalence
gate. Serving therefore retains all-token projection pending qualification.

FP8 preparation now uses one warp per activation group. Layer-boundary probes
identified a separate numerical defect: rounded division changed scale bits and
FP8 midpoint decisions relative to the independent Torch/DeepGEMM reference.
The shared and combined providers now use explicit F32 reciprocal/multiply
rounding; frozen external bits reject the old implementation and accept the
new one. First-layer QKV and Z projections now match exactly on the measured
input. Full-model error remains above the unchanged gate, with convolution,
gated normalization and residual/MLP boundaries requiring further diagnosis.
These checks establish the repaired quantization contract, not C8 acceptance.

## Next milestones

| Priority | Deliverable | Acceptance |
| --- | --- | --- |
| 1. Input and concurrent correctness | Resolve remaining convolution, recurrence, normalization and MLP rounding/selection boundaries; finish HTTP tokenizer parity | Independent same-input operator probes and unchanged full-model gates pass for serial, changing batches, state reuse and tokenize/detokenize parity |
| 2. Profile-driven regions | Qualify request-final projection and shared FP8 preparation; reduce gather/cast and recurrent-state movement; qualify normalization/partial-RoPE regions; compare coordinate and connected search at the same budget | Independent operator references, full-model logits, ragged prefill/decode and drain pass; complete-workload measurements improve over the existing composition |
| 3. Joint state and compute plans | Search multiple legal state realizations under one device-memory budget | Include KV, fixed state, workspace and resident graphs; validate layout transitions and count movement/preparation costs in the measured objective |
| 4. Wider workload coverage | Longer prefill/context, concurrency and memory pressure; shared-prefix and cancellation traces | Find actual capacity and tail-latency limits, preserve correctness under admission pressure, record all failures and memory budgets |

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
Further cold-compilation work should measure repeated generated-module
compilation and specialization plans after definition reuse, preserving the
graph/trial budget and all correctness checks. A reusable module-image cache
must bind source, target and compiler identity, bound host memory, and keep
strict artifact completeness checks.

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
