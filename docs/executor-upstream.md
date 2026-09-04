# Executor fork and upstream policy

OrbitKV embeds a complete Luminal fork as the graph compiler and device
executor. The fork is not a second inference engine: `core/` remains the only
authority for logical visibility, page selection, generations, retirement, and
reuse.

## Fork delta

The current OrbitKV patch stack adds the following general executor contracts:

- externally managed paged attention: page size and CSR request/page metadata
  are graph inputs authored by OrbitKV instead of an executor-private page
  table;
- grouped-query specialization: the query-to-KV head ratio participates in JIT
  identity and both planning and execution compile for that exact ratio;
- guarded native calls: FlashInfer failures cross the C boundary as Rust errors
  instead of unwinding a C++ exception through Rust;
- stream-ordered tensor-range copies and completion events, used to execute
  manager-authored token relocation before the new page view is published;
- device regressions for external block pages and non-power-of-two grouped-query
  attention.

These changes live in generic runtime, paged-attention, and device-copy files.
No production path or type is named for one GPU or model family.

## Supported graph vocabulary

The OrbitKV decoder path currently composes token embedding, BF16 linear
projections with optional Q/K/V bias, optional per-head Q/K normalization,
RoPE, RMS normalization, paged MHA/GQA, SwiGLU, residuals, KV scatter writes,
and an untied or tied language-model head. Head dimensions 64, 128, and 256 are
admitted by the graph contract.

This is an architectural family, not a model-name allowlist. A checkpoint is
executable only when its tensor names, dense decoder topology, dimensions,
dtype, and attention-state plan match this vocabulary. The current real-device
closure is one released dense decoder checkpoint with full token KV.

The following remain outside the executable model closure: MoE routing, MLA or
latent KV, recurrent and convolution state, quantized weights, multimodal
encoders, speculative decoding, tensor/pipeline parallelism, and a complete
multi-class hybrid model graph. Sliding and chunked policies compile and lower,
but do not yet have independent model-level device qualification.

## Updating Luminal

The submodule has separate `origin` (the OrbitKV fork) and `upstream` (Luminal).
An update is performed in the fork, never by copying upstream files into the
parent repository:

1. fetch the upstream branch and create an integration branch from the current
   fork head;
2. merge or rebase the desired upstream commit while preserving the small
   generic patch stack above;
3. run Luminal's host tests and the paged-attention/device-copy regressions;
4. run OrbitKV host gates, CUDA compile checks, and the released-model and
   relocation device closures;
5. push the fork commit, then update both the parent submodule pointer and the
   three Luminal dependency revisions in `executor/Cargo.toml`;
6. let `tools/verify_active_source.py` reject a mismatched pin or submodule.

An upstream update is therefore an explicit compiler-backend upgrade with
qualification, rather than an automatic floating dependency.
