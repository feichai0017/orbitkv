# Generated CUDA module artifacts

The decoder artifact now carries both the selected bucket schedules and their
generated CUDA module images. This removes NVRTC from compatible replay of
those modules. Graph normalization, LLIR extraction/validation, weight loading,
provider preparation, memory binding and CUDA Graph materialization still run.

## Ownership and lifecycle

`luminal_cuda_lite::CudaModuleArtifact` owns image serialization and validation.
`CudaRuntimeImpl::capture_module_artifact(&graph)` reloads only the selected
schedule into a fresh capture. Rejected candidates are excluded without keeping
every candidate's binary in host memory. This currently adds one selected-program
compilation pass when creating an artifact; it is a replay optimization, not a
reduction in fresh search time.

`CudaRuntimeImpl::load_selected_schedule_with_modules(&graph, &modules)` attaches
the validated images to the runtime and reloads the schedule. It clears the
runtime's function cache so old CUDA handles cannot conceal missing images.
The strict session remains attached during bucket compilation, materialization,
execution and captured-execution preparation. A source absent from the artifact
returns an error; it never silently falls back to NVRTC.

The dynamic backend and OrbitKV share the same mechanism. OrbitKV captures
after search and before replacing its profiling scratch with live fixed-state
arenas. Callers must use this API during initialization because reloading
rebuilds executable resources. It is not an operation on a live serving session.

Providers that cache generated helper modules across runtime instances must
call the backend's `observe_cached_module` on cache reuse. During capture this
obtains the image if missing; during replay it enforces completeness even when
a process already holds the CUDA function. DeepGEMM quantization/reference
and MoE decode helpers participate in this contract.
External FlashInfer/FlashAttention/DeepGEMM shared libraries keep their own revision/compiler
cache keys; their libraries and prepared plans are not embedded here.

## Format and validation

| Format | Behavior |
| --- | --- |
| Decoder schema 10 | Explicit checkpoint import, NN semantics and attention algorithms with compiled request geometry; requires a CUDA module artifact and strict image replay |
| CUDA module schema 3 | Sorted source-digest map, base64 images, per-image SHA-256 |

The existing model/arena/tuning identity and per-bucket LLIR fingerprints remain
in force. Module lookup uses SHA-256 of the current generated source. Loading
requires the same compute target, loaded NVRTC version and compile options;
OrbitKV checks this before reading weights. Deserialization rejects invalid
source keys, malformed/empty images, checksum mismatches and unknown fields.
An empty map is valid for a program with no generated modules, but fails when
loading a schedule that requires one.

The image checksum detects corruption; it does not authenticate an artifact.
The signature does not fingerprint the entire CUDA installation/header tree.
Artifacts are build/toolchain-bound execution inputs, not a portability promise
across arbitrary SDK installations or an expansion of supported GPU families.
Only decoder schema 10 and CUDA module schema 3 are accepted. Older decoder
formats predate the updated [provider request-geometry ABI](attention-providers.md)
and must be regenerated. The decoder owns a required module artifact and exposes
`module_image_count() -> usize`; there is no schedule-only decoder mode or
compatibility conversion API.

## Qualification

The CUDA regressions execute a saved two-bucket program against independent
elementwise results, repeatedly switch buckets, and assert that replay emits
module hits with zero `cuda.nvrtc.compile` spans. They also remove images from
an artifact loaded into a warm runtime and exercise process-wide helper reuse.
All test code lives under the owning crate's `tests/` directory.

For a fixed-program model comparison, run the decoder qualification tool against
the same `--replay-artifact` using frozen baseline and changed binaries.
`--stage-trace` separates schedule-loading/NVRTC costs from weights and device
execution.
Keep provider caches and numeric/state gates unchanged and record the binary,
artifact and oracle identities. These diagnostics do not establish serving TPOT.

The [H20 model result](../results/module-image-artifact-20260913/README.md) records
two fixed-artifact pairs: 428 module hits and zero NVRTC calls on cached replay,
11.40 s to 4.96 s median schedule loading, and 38.69 s to 33.28 s complete
diagnostic process time. All nine processes pass eight reference steps and drain.
Fresh creation pays 7.43 s for capture; warm diagnostic decode stays near 24.5 ms.
That historical experiment used an image-omission harness retained with its raw
source snapshot. Its compatibility path and ablation option are removed from
the current API and qualification runner.
