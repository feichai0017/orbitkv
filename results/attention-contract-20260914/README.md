# Attention contract and provider admission qualification

Logical attention now separates mathematical semantics from an explicit KV view.
Provider capability records generate egglog eligibility rules before lowering,
extraction and native preparation. Native CUDA remains an independently tested,
bounded experimental implementation and requires explicit opt-in to enter search.
See the [design and upstream comparison](../../docs/attention-providers.md).

The graph API validates ownership, dtype, geometry, contiguous storage and CSR
shapes, including statically known page capacity. Regression tests verify that
unsupported dtype, dimensions, layout, visibility and phase combinations do not
become executable FlashInfer candidates. Logical scales use round-trip float
serialization, including a scale of `1e-20`. No model mathematics or reference
tolerance was changed.

## Evidence

| Gate | Result |
| --- | --- |
| Root workspace, default features | 247 passed; 2 external tests ignored |
| NN, including backend-free attention contracts | 27 passed |
| Executor CUDA-feature unit suite | 71 passed; 2 external checkpoint tests ignored |
| Directed CUDA/provider/runtime suites | 37 passed on H20, including native independent-reference parity, FlashInfer capture, search exclusions, resource ownership, graph residency and module artifacts |
| 27B fresh search and two strict replay configurations | 96 full-vocabulary row comparisons; maximum absolute error 0.90625 under the unchanged 1.0 gate; all lifecycle drains passed |
| Old schema rejection | A real schema 7 artifact was rejected before weight loading |
| HTTP fresh search and strict replay | Independent three-token output `&!@`, active-stream cancellation and final state drain passed in both processes |
| Serving graph resources | Two buckets prepared; no additional graph builds during serving; all 64 KV pages and all recurrent/conv slots recovered |
| Strict HTTP module-image replay | 425 image hits, zero NVRTC compilations; decoder artifact unchanged |
| Source and historical evidence | All 346 frozen build inputs and 109 pre-existing result files verified unchanged |

Both workspace formatting checks, the active-source layout checker and Clippy
with `-D warnings` passed. An initial Clippy run reported redundant field names
in migrated tests; that log is retained and the corrected final run passed.
The CUDA runtime checks include host-side resource invariants as well as GPU
execution; the count is not a claim of 37 distinct numerical kernel tests.

## Compilation observations

The fresh model `compile_or_load` interval was 240.30 seconds, including 192.63
seconds building/saturating the search space and 27.85 seconds in CUDA search.
Prepared replay took 16.81 seconds and capacity-one replay took 16.75 seconds.
HTTP readiness took 237.80 seconds for fresh search and 19.03 seconds for replay.
These are CPU-side wall intervals; nested durations must not be added together.
Fresh compilation and replay perform different work, so these observations are
not a matched serving-performance comparison or a refactor speedup claim.

The next priority is to attribute and reduce search-space construction and
improve exploration of measured expensive regions. This result does not
establish attention as the primary runtime bottleneck.

## Identity and limits

[build.json](build.json) identifies the frozen server and test binaries, toolchain,
GPU and independent reference. [checks.json](checks.json) contains gate receipts;
[summary.json](summary.json) includes stage attribution and final resource state;
[files.json](files.json) hashes the raw evidence under
`.qualification/attention-contract-20260914/`. The final server SHA-256 is
`0e1859f1c6ff7f3e156572bdbd9ff8a21c60db9e786d60f633cb20b71fd0c188`.

Qualification is limited to H20 SM90 and the local `qwen3.8-27b-fp8` text
checkpoint, whose config declares Qwen3.5 architecture. The reference is
Transformers 5.12.1 with vocabulary 248320. Each model run checks four repeated
request lifecycles, each with four prompt tokens and seven teacher-forced decode
steps. The requested search budget was `search_graphs=1`, seed 7; runtime
population initialization also profiles candidates. These runs do not establish
optimal search, long-context accuracy, serving-scale concurrency or soak readiness.

Decoder schema 8 requires fresh artifacts. The experimental policy participates
in artifact identity. HND, unequal QK/V dimensions and unmasked attention have
semantic vocabulary but no admitted CUDA implementation here. Contiguous/ragged,
latent and sparse KV execution contracts remain unimplemented. No FlashMLA,
FlashAttention, Triton or TileLang provider, or attention megakernel, was added.
