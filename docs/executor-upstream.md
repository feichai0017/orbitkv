# Executor fork and upstream policy

OrbitKV embeds a complete Luminal fork as the graph compiler and device
executor. The fork is not a second inference engine: `orbitkv` remains the only
authority for logical visibility, page selection, generations, retirement, and
reuse.

## Fork delta

The current OrbitKV patch stack adds the following general executor contracts:

- externally managed paged attention: page size and CSR request/page metadata
  are graph inputs authored by OrbitKV instead of an executor-private page
  table;
- dynamic paged-attention geometry: page size is resolved from the current
  runtime dimensions and participates in capture invalidation, allowing one
  installed graph and K/V arena to switch between token-slot selection and
  packed physical pages;
- grouped-query specialization: the query-to-KV head ratio participates in JIT
  identity and both planning and execution compile for that exact ratio;
- guarded native calls: FlashInfer failures cross the C boundary as Rust errors
  instead of unwinding a C++ exception through Rust;
- stream-ordered tensor-range copies and completion events, used to execute
  manager-authored token relocation before the new page view is published;
- runtime inspection that proves whether persistent-state outputs alias their
  registered inputs in every compiled dynamic-shape bucket;
- caller and custom-op compiler facts injected into every e-graph bucket, with
  paged-attention nodes bound to an external persistent-state class ID;
- direct range copies within persistent graph inputs, so relocation always
  targets the stable K/V arena even when a bucket materializes its update;
- fresh profiles of an already installed executable at exact runtime geometry,
  including device time, sample count, bucket identity, and dynamic dimensions,
  plus timing-capable device-copy batches used by the executor's relocation
  cost contract;
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
global or local RoPE, direct or unit-offset RMS normalization, pre-norm or
sandwich-norm residuals, paged MHA/GQA, SwiGLU or GeGLU, KV scatter writes, and
an untied or tied language-model head. Head dimensions 64, 128, and 256 are
admitted by the graph contract.

This is an architectural family, not a model-name allowlist. A checkpoint is
executable only when its tensor names, dense decoder topology, dimensions,
dtype, and attention-state plan match this vocabulary. Configuration semantics
and the safetensors header are both checked before search. Current real-device
closures include released dense Full and Full+Sliding checkpoints.

The following remain outside the released-model execution closure: MoE routing,
MLA or latent KV, recurrent and convolution state, quantized weights, multimodal
encoders, speculative decoding, tensor/pipeline parallelism, and independently
qualified exact Chunked execution.

## Current whole-graph runtime

`CompiledDecoder` owns one graph, one runtime, and one stable K/V arena per
attention class. It compiles separate `s=1` decode and `s>=2` prefill buckets
once; batch and context-page
dimensions have explicit capacity buckets. Tokens, positions, write slots, and
CSR tensors are allocated to their maximum configured capacity before search,
so later `set_data` calls update their contents and logical lengths without
changing device addresses.

For a relocatable Full class, the physical K/V allocation retains its compiled
page width while attention page size is a dynamic symbol. A sparse
manager-authored view binds that symbol to `1` and expands physical token slots
into CSR indices; a packed view binds it to the storage page width. The switch
does not compile another decoder or allocate another cache. Because page size
affects FlashInfer planning, it is included in the capture-sensitive dimension
set and safely rematerializes the library plan when changed.

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

The persistent K/V buffers are registered as required paired input/output
state before profiling. Unlike an ordinary output registration, this contract
rejects a candidate or loaded artifact unless every selected bucket resolves
the output directly to the input arena. `cache_update_buckets()` reports the
per-bucket in-place tensor count and any copy-back bytes. The qualified
16-candidate artifact has 36/36 in-place K/V tensors and zero copy-back in both
buckets. This compiler constraint produced a measured same-engine C2 benefit;
it does not by itself close the remaining kernel gap to SGLang.

The embedded decoder keeps all searched decode/prefill buckets but bounds active
CUDA Graph materialization to one bucket. Explicit-CSR attention can rebuild
captured library resources when context geometry changes; retaining another
phase's materialization across such pool reclamation produced stale device
state on the tested driver. Switching phase therefore rematerializes the target
bucket without re-running graph search. A dedicated bucket-switch regression and
the repeated released-hybrid residence workload cover this contract.

Selected schedules, including graphs with explicit paged-attention custom ops,
can be serialized independently of weights and KV contents. Loading replays the
deterministic graph normalization, resolves the current custom-op table, and
verifies the unrolled LLIR fingerprint of every bucket. OrbitKV wraps this in a
decoder artifact identity covering the canonical manifest, decoder and weight
family geometry, arena shape, and compile buckets. Incompatible artifacts fail
closed. The artifact removes cross-process search variation; CUDA module and
FlashInfer prepared-resource materialization are not yet fully serialized.

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
5. push the fork commit, then update the parent submodule pointer;
6. keep `crates/orbitkv-executor/Cargo.toml` path dependencies pointed at that visible
   submodule and let `tools/verify_active_source.py` reject any second remote
   Luminal source.

An upstream update is therefore an explicit compiler-backend upgrade with
qualification, rather than an automatic floating dependency.

The current fork includes upstream through `d18376d1`; the parent pin is updated
with each qualified fork commit. The fork retains its own upstream workspace so it can be
built and tested independently even though the parent explicitly excludes it
from the four owned OrbitKV workspace members. The latest sync includes the
upstream CUDA correctness fixes, dynamic-bucket warmup behavior, and scatter
reuse rules while preserving OrbitKV's external-page, required-alias, artifact,
and child-graph contracts. Independent Luminal core tests and CUDA-lite compile
checks pass. Existing H20 measurements remain compatibility evidence for the
fork, not a new broad performance claim.
