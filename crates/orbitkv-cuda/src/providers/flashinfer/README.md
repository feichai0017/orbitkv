# FlashInfer integration

`FlashInferAttention` is an attention provider selected through egglog. The
structural rewrite offers it as an equivalent implementation of supported
paged GQA attention; OrbitKV supplies logical attention and explicit KV-view
facts. Declarative capabilities filter supported combinations before the semantic
provider rewrite. Runtime extraction and GPU profiling choose among
the legal providers.

## Modules

| File | Responsibility |
| --- | --- |
| `flashinfer_attention.egg` | Structural attention equivalence rules |
| `paged_attention.egg` | Logical-attention/KV-view provider lowering after capability admission |
| `algorithm.rs` | Stable CUDA-core/tensor-core identities and declarative capabilities |
| `../flashinfer.rs` | shape/pointer validation, resource planning, preparation and execution |
| `workspace.rs` | Prepared-plan metadata ownership, shared execution scratch and pinned host staging |
| `jit.rs` | Pinned provider sources, CUDA wrapper compilation/cache and native entry points |
| `find_indptrs.rs` | Recover compact indices from supported lowered gathers |
| `wrapper.cu`, `wrapper.h` | Native planning, execution and layout helpers |

The public paged constructor requires an explicit `FlashInferAlgorithm` and
accepts positive page sizes and grouped-query
ratios with supported head dimensions (64, 128, 256, 512; 512 requires 16-bit
inputs). Explicit query/KV CSR and last-page lengths support ragged batches and
block pages. Native BF16/F16 decode and prefill, including causal sliding-window
attention, are implemented. The F32 path has narrower native support; admitted
shapes and dtypes must satisfy both the Rust semantic guard and native wrapper.
See those guards for exact applicability rather than assuming every constructor
geometry works on every GPU.

## Preparation and capture

CUDA-core decode and tensor-core attention are separate egglog candidates for
16-bit decode. Only tensor-core attention admits packed prefill; F32 remains
CUDA-core decode only. The serialized algorithm participates in plan/capture
identity and is never silently changed based on query count at launch.

The op resolves shape and CSR metadata, prepares the selected algorithm's plan,
enqueues attention, and returns the layout required by the graph. Wrapper JIT
specializes head dimension, grouped-query ratio, sliding-window mode and CUDA
architecture. Its cache includes source identity. Selected wrapper kernels are
external provider calls; OrbitKV compiler-generated surrounding regions remain separate
candidates and captures.

Every `PreparedFlashInferAttention` owns an 8 MiB integer workspace. FlashInfer's
planner writes request/tile indices, output indptrs and split-KV metadata there;
retaining offsets without retaining private contents is unsafe across shapes.
Captured islands retain the prepared allocation until retirement.

A 128 MiB float workspace remains shared scratch. Graph dependency edges
serialize its users, and the current decoder executes on its owning stream.
Pinned host staging is shared under a lock; preparation synchronizes the upload
before the next planner can overwrite it, including native error returns.
These capacities are provider budgets, independent of checkpoint geometry.

Pointer-free resource planning charges private metadata and auxiliary buffers to
each prepared plan, and shared float scratch once. CUDA Graph planning also
charges a replacement generation because recapture prepares all new plans
before retiring old islands. Derived decode may share plans only when semantic
specs and metadata producer match and data dependencies prove ordered use.
Explicit CSR contents are not inferred from pointer equality and are replanned.

Tests live in `tests/unit/providers/flashinfer/`. The explicit CUDA regression compares
retained decode, ragged decode and causal sliding-window prefill graphs against
a CPU reference, then alternates them and retires one graph. Graph resource and
ordering tests live in `tests/unit/kernel/to_host/`. Current qualification is
single-device, serialized execution; global scratch is not a cross-stream or
multi-device ownership contract.
