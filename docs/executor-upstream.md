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
- runtime inspection that proves whether persistent-state outputs alias their
  registered inputs in every compiled dynamic-shape bucket;
- direct range copies within persistent graph inputs, so relocation always
  targets the stable K/V arena even when a bucket materializes its update;
- caller-owned capture of an already-warmed execution, plus a preparation-only
  path that refreshes stable input bindings before replay without replanning or
  executing the model;
- explicit parent-graph composition that retains the searched materialized
  executables as child nodes and orders persistent-output D2D copies after them;
- device regressions for external block pages and non-power-of-two grouped-query
  attention, and for captured execution reading updated stable inputs.

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

## Current whole-graph runtime

`CompiledDecoder` owns one graph, one runtime, and one K/V arena. It compiles
separate `s=1` decode and `s>=2` prefill buckets once; batch and context-page
dimensions have explicit capacity buckets. Tokens, positions, write slots, and
CSR tensors are allocated to their maximum configured capacity before search,
so later `set_data` calls update their contents and logical lengths without
changing device addresses.

The language-model head feeds a fused dynamic-row argmax in the same graph. The
default execution API transfers only one `i32` token ID per query row. An
explicit diagnostic API additionally transfers logits and was used to prove the
device token matches the previous host `max_by` result across prefill and decode.

After one same-shape warmup, `CompiledDecoder::capture_decode` records the
prepared decode work into one outer CUDA Graph. `replay_decode` first updates
the persistent input allocations, refreshes runtime bindings without executing
or replanning, and then launches that graph on the original stream. The current
contract fixes `s`, `b`, `c`, `query_indptr`, and `page_indptr`; page indices and
last-page lengths may vary within that signature. A mismatch fails closed and
requires recapture. Both the generic Luminal path and a released-checkpoint
OrbitKV lifecycle pass on H20.

The first implementation flattened every selected executable back into raw
launches and was 24.9% slower in a matched diagnostic. The current implementation
preserves each selected graph as one child node. On the same fixed batch-one
decode step, it reduced median wall time by 8.3% over 20 alternating iterations
and 5.8% over a 100-iteration confirmation. This narrow result does not qualify
continuous batching or end-to-end serving throughput.

The persistent K/V buffers are registered as paired input/output state before
profiling. Search may select an in-place scatter or a materialized update with
a graph-visible D2D epilogue back into the same arena. Both preserve the stable
address contract; `cache_updates_in_place()` reports which result was selected.
The current two-candidate real-device smoke selected the materialized-copy
form, so no zero-copy KV-write benefit is claimed. The final source run observed
one-time search/compile at roughly 183 seconds, a four-token prefill dispatch at
roughly 13 ms, first decode dispatch at roughly 36 ms, and a warm second decode
at roughly 5 ms. These are diagnostic timings from one correctness run, not an
L5 benchmark or speedup claim.

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
