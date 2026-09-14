# Mature attention provider qualification — 2026-09-14

Status: passed on the local H20 (SM90), with the Qwen3.8-27B-FP8 text checkpoint.
This change removes the handwritten native attention provider and its experimental
flag, makes FlashInfer CUDA-core/tensor-core algorithms explicit, and adds a
standalone C ABI adapter to pinned upstream FlashAttention-3 kernels.

## Selected programs

The model and HTTP artifacts independently selected:

| Bucket representative | Selected attention algorithm |
| --- | --- |
| B=1, Q=1, compact page references=1 | FlashInfer tensor-core decode |
| B=1, Q=4, compact page references=1 | FlashAttention-3 paged prefill |

[Saved-choice traversal](selection.json) follows the serialized program roots.
Its counts refer to templates before loop expansion, not the number of runtime
attention calls. FA3 is also tested as the sole registered attention provider,
including independent numerical checks and saved-schedule replay.

## Validation

- Workspace: 247 passing tests; NN: 27; CUDA-enabled executor: 71. These suites
  have overlapping host cases, so their sum is not a count of unique tests.
- 79 directed CUDA/provider/runtime checks pass, including 29 structural
  FlashInfer regressions, both decode algorithms, mixed-library recapture,
  source identity, resource accounting and module-artifact replay.
- FA3 directly matches independent CPU softmax references in seven configurations:
  F16/BF16, head dimensions 64/128/256, paged decode/prefill, ragged queries,
  reversed page mappings, non-power-of-two GQA and sliding windows. K/V payloads
  remain unchanged. Page sizes 16/32 and contexts through 513 tokens are exercised.
- Captured FA3 graphs read changed CSR contents and retain scratch across
  replanning, other graph retirement and operation-cache release. The initial
  slice-event lifetime failure was fixed by acquiring the stable scratch pointer
  before capture; the failing development logs remain in the evidence manifest.
- Fresh 27B search, prepared strict replay and capacity-one replay pass 96
  full-vocabulary logit comparisons. Maximum absolute error is **0.71875** under
  the unchanged **1.0** gate. All four sequences in each process drain their state.
- Fresh and replayed HTTP serving both produce the independent three-token
  reference `&!@`. SIGTERM with an active SSE client cancels the remaining request;
  all **64 KV pages** and both fixed-state pools return to free state. Preparation
  retains two graphs and these requests cause no extra graph builds.
- HTTP strict replay has **423 generated-module image hits and zero NVRTC
  compilations**. Schema 8 is rejected; current decoder artifacts use schema 9.
- Formatting, both workspace Clippy checks (`-D warnings`) and active source
  layout validation pass. All **353 frozen build inputs** and **115 historical
  result files** remain unchanged. GPU memory returns to zero after qualification.

## Performance and scope

Warm full-logit diagnostic decode medians are 24.14 ms (fresh-search process),
24.10 ms (prepared replay) and 24.20 ms (capacity-one replay). These are diagnostic
measurements, not a matched throughput comparison against the previous backend,
vLLM or SGLang. No whole-model speedup is established here.

The fresh HTTP process becomes ready in 234.80 s; strict replay in 19.03 s.
They perform different work: fresh search records 1,617 NVRTC compilations,
including candidates, while replay loads selected images. External provider
libraries were already cached. These numbers do not measure a fully cold
provider/toolchain installation or establish a cross-version startup speedup.

The initial FA3 adapter uses SM90, F16/BF16, equal head dimensions 64/128/256,
NHD paged K/V, packed GQA, non-TMA paged loads and one unsplit KV partition.
It converts CSR metadata to a rectangular table on the GPU; conversion,
scheduler work and scratch are included in execution/resource accounting.
Current metadata storage is O(requests × compact page references).

FA2/FA4, split-KV/TMA candidates, FP8 KV, MLA/sparse/contiguous attention,
other devices, serving-scale batches, long contexts and megakernels are not
qualified by this result. Search uses seed 7 and `search_graphs=1`, with additional
runtime population initialization; it does not establish a global optimum.

## Evidence

- [Summary, model and HTTP traces](summary.json)
- [Commands, test counts, lifecycle reports and resolved failures](checks.json)
- [Frozen build and reference identities](build.json)
- [Pinned upstream FlashAttention/CUTLASS revisions](provider.json)
- [Local evidence file hashes](files.json)
- [Provider architecture and setup](../../docs/attention-providers.md)

Logs, frozen executables and generated artifacts remain under
`.qualification/provider-kernels-20260914/`. The source is also installed at
`/root/.cache/luminal/providers/flashattention/8d3a3b80d475`; the explicit
`fetch-provider flashattention` command resolves it without a network fetch.
Historical qualification directories and their indexes were not modified.
