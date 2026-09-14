# OrbitKV CUDA backend

CUDA code generation, measured candidate selection, provider integration and
execution for OrbitKV. See the [backend architecture](../../docs/cuda-backend.md)
for source boundaries, provider versions, cache policy and joint state/compute
compilation.

- `kernel/` contains generated kernels and fusion rules.
- `providers/` contains native adapters and shared source/build management.
- Egglog rules and CUDA templates are separate source assets beside their owner.
- `runtime/` and `search/` plan memory, profile candidates and execute schedules.
- Tests live under `tests/`; all native source pins live in `providers.lock.json`.

`KernelOp` describes a generated kernel. `HostOp` describes host-launched GPU
work, including library calls, workspace and CUDA Graph capture contracts.
Existing rules fuse supported generated regions. CUDA Graph capture retains
multiple launches; persistent megakernel generation remains future work.

### Semantic search contract

Backend rewrites add legal implementations with `union`; they do not remove a
legal implementation merely because another implementation is usually faster.
The profiling search, rather than cleanup, chooses between alternatives such as
generic kernels, specialized kernels, and host-library calls.

That includes GenericMatmul/cuBLASLt/GEMV, direct/decomposed Conv2D,
materialized/absorbed fusion and casts, copying/no-copy scatter, and
materialized/fused RoPE-scatter paths. These alternatives are matched in
egglog; selected LLIR is not rewritten into a different operator pattern after
extraction.

Block-scaled linear follows the same contract. A provider-neutral semantic
custom op has an independent CUDA reference, while four DeepGEMM
tile schedules are unioned into its e-class. Provider JIT preparation completes
before timed trials; normal device profiling chooses the implementation per
bucket, and the selected node records resolved source/wrapper identity and tile
variant in the serialized schedule. The independent reference is an operator
oracle, not a deployment-eligible fallback. Paged attention is likewise a
semantic custom op.
FlashInfer CUDA-core decode, FlashInfer tensor-core attention and optional
FlashAttention-3 enter through egglog provider rules. Algorithm identity remains
in the schedule and plan/capture keys. FA3 currently admits SM90 F16/BF16,
equal head dimensions 64/128/256 and NHD paged K/V with causal/sliding visibility.
Its adapter converts CSR metadata on the GPU and launches upstream paged non-TMA,
packed-GQA, unsplit kernels; K/V payloads keep their allocations. Captured graphs
retain private scratch and refresh metadata on replay. The handwritten native
attention implementation has been removed.

For diagnosis, `ORBITKV_CUDA_PROFILE_GRAPH_STEPS=1` prints aggregate CUDA
Graph step timings. Adding `ORBITKV_CUDA_PROFILE_GRAPH_STEP_DETAILS=1` groups
those timings by concrete operation identity, tensor geometry, and provider
variant. The detailed strings are intentionally diagnostic output, not a stable
artifact format.

Generated modules can be saved independently of any frontend through
`CudaRuntimeImpl::capture_module_artifact(&graph)` and loaded through
`load_selected_schedule_with_modules(&graph, &artifact)`. `CudaModuleArtifact`
serializes source-keyed, checksummed images and validates target architecture,
NVRTC version and compile options. Capture rebuilds only the selected schedule
before live state is bound; rejected search programs are not retained. Loading
keeps strict image lookup active through subsequent bucket materialization and
execution. Missing images never fall back to NVRTC. Process-wide provider helper
caches participate through `artifact::observe_cached_module`. The dynamic
backend shares this mechanism. Module schema 3 replaces the older schema-2
backend blobs; external provider libraries and prepared plans remain separate.

`CudaRuntimeImpl::load_safetensors(&graph, path)` returns
`anyhow::Result<runtime::WeightLoadReport>` with bound/converted tensor counts
and source/device byte totals. Matching storage borrows immutable mapped bytes;
floating-point conversion owns one typed buffer. Neither path creates an
intermediate byte vector or retains a weight host mirror. Encodings are checked
before rebinding a shard, and its uploads complete before the mapping is dropped.
File and CUDA errors propagate to the caller; a device error can leave earlier
bindings loaded, so use this during initialization. Extra checkpoint tensors
and graph inputs absent from an individual shard are ignored. Loading and pure
conversion have separate modules under `src/runtime/weights` and tests under
`tests/unit/runtime/weights`.

Prepare locked provider sources before model compilation:

```sh
cargo run -p orbitkv-cuda --bin providers -- list
cargo run -p orbitkv-cuda --bin providers -- fetch all
cargo run -p orbitkv-cuda --bin providers -- inspect flashinfer
```

Explicit checkouts use `ORBITKV_DEEPGEMM_DIR`, `ORBITKV_FLASHINFER_DIR` and
`ORBITKV_FLASHATTENTION_DIR`. `ORBITKV_CACHE_DIR` controls the common cache root.
An invalid explicit checkout fails without falling back to another installation.
Model compilation never fetches sources. Each provider retains its own pinned
CUTLASS dependency. cuBLASLt comes from the installed CUDA Toolkit.

All three providers hash the actual provider/dependency header contents and embedded
wrapper, including local edits present when the process first resolves sources.
Egglog-selected provider nodes record this identity and reject old or mismatched
identities during strict schedule replay. This covers OrbitKV's semantic
paged-attention and block-FP8 paths; directly inserting a
`FlashInferAttention` custom op bypasses the egglog extraction gate.

Compiled-library keys additionally include `nvcc` executable/version, target,
arguments, and selected compiler environment inputs. These keys prevent reuse
across changed declared compilation inputs, including on the direct-provider
path. They are not a hermetic toolchain fingerprint: host compiler binaries,
supporting CUDA compiler tools, and system headers are not hashed, and these
extra library inputs are not all bound into selected-schedule identity. Sources
and toolchains must remain unchanged for the process lifetime; restart after
editing them because their digests are memoized.

Large recurrent decoder graphs use at most two automatic rolling regions by
default. This bounds e-graph memory while preserving the outer layer-period
and one nested repeated body; `ORBITKV_MAX_ROLLED_REGIONS` can override the
limit for compiler experiments.

Cleanup may remove only representations that are not executable plans: cycles,
malformed shape/stride metadata, unsupported type/layout combinations, and
proven alias or ownership violations. Candidate resource checks may reject a
plan that cannot fit or launch on the target device. The intermediate-memory
cap applies to the peak planned bucket arena. Buckets share a stable high-water
intermediate allocation with separate offset plans; switching buckets reuses
that allocation. Growth invalidates captures and retires the old arena before
allocating its replacement. The device-memory check includes that peak arena, persistent host-op state
retained by all compiled buckets, the peak transient host-op allocation, and
deduplicated shared workspaces. That check is a necessary planned-capacity bound,
not an available-memory guarantee: external allocations, CUDA context and
allocator overhead, and pool reservations are not observable in the plan. Arena
growth likewise drops the synchronized old arena before allocating its
replacement, so replacement itself does not introduce an old-plus-new peak. The
intermediate-memory and synchronous-NVRTC source budgets are reported as resource
rejections and can be adjusted independently of rewrite semantics. Otherwise, a
plan that is legal but merely expensive remains available for measured search.

Choice-set validation detects correlated e-class cycles before LLIR loading.
Random initial genomes repair only those reachable cycles; later mutations may
still produce them, in which case candidate filtering discards them without
profiling and continues searching the remaining legal alternatives.

Captured FlashInfer plans own private integer metadata; float scratch remains
shared by dependency-ordered execution. Device planning includes one replacement
plan generation during recapture. Pinned planner staging is locked and its
upload drained before reuse. Finite bucket residency is controlled independently
of selected schedule identity. See OrbitKV
[graph residency](../../docs/graph-residency.md) and
[OrbitKV compiler design](../../docs/compiler.md) for the checked-in integration.
