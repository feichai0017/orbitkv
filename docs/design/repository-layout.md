# Repository layout

Loom Infer keeps three crates. A module or crate needs a distinct functional,
ownership, build, or safety boundary.

## Top level

```text
loom-infer/
|-- crates/
|   |-- loom-infer/             contracts and CPU references
|   |-- loom-infer-cuda/        CUDA plans, providers, and Rust kernels
|   `-- loom-infer-validation/  non-published H20 programs
|-- docs/
|   |-- design/                 architecture and layout decisions
|   |-- development/            build and validation procedures
|   `-- results/                immutable machine-readable evidence
|-- tools/flashinfer/           matched-provider harnesses and summaries
|-- website/                    Astro documentation site
|-- Makefile                    local, CI, CUDA, and evidence entry points
`-- .github/                    CI policy and Pages workflows
```

Generated PTX, cubins, profiler captures, model weights, `target/`, and website
build output are not product source.

## Three-crate rule

```text
consumer engine
  -> loom-infer-cuda
       -> loom-infer
       -> cuda-oxide runtime and Rust device artifacts
       -> explicit vendor providers

loom-infer-validation
  -> loom-infer-cuda
  -> loom-infer
```

`loom-infer` builds without CUDA. Product crates do not depend on validation,
documentation, the website, or result records. The validation crate is a
workspace member but not a default member.

GEMM, attention, KV-cache operations, and Graph execution remain modules in
these crates. Loom does not add `loom-gemm`, `loom-runtime`, or
`loom-cuda-kernels` for namespace convenience. A fourth crate requires an
independent build artifact, dependency direction, safety boundary, or release
cycle.

## Operator family namespaces

The final public namespace follows operator semantics. The project creates a
family only when its first contract or provider exists.

| Family | Responsibility | Current physical source |
| --- | --- | --- |
| `attention` | Decode, prefill, masking, and attention-state merge | Implemented under `attention` |
| `kv_cache` | Paged append, gather, scatter, compaction, and remapping | Paged append is currently under `attention/paged_append` and the CUDA RoPE module |
| `gemm` | Dense, grouped, and quantized matrix operations | One contiguous BF16 dense contract and cuBLASLt provider exist in `gemm/` |
| `normalization` | RMSNorm and later normalization contracts | RMSNorm is currently exposed as `rms_norm` |
| `position` | RoPE and other position transforms | RoPE is currently exposed as `rope` |
| `activation` | Activation and gated-activation operators | Planned |
| `sampling` | Logits transforms, sampling, and RNG contracts | Planned |
| `speculation` | Draft verification and token compaction | Planned |
| `quantization` | Scale, pack, unpack, dequantize, and layout conversion | Planned |
| `moe` | Routing, permutation, expert compute inputs, and combine | Planned |
| `communication` | Qualified collectives for measured distributed workloads | Planned |

The source migration moves an implemented domain directly to its final family.
It does not add aliases, compatibility modules, or empty target directories.
The public facade re-exports only admitted contracts.

## Contract crate

`crates/loom-infer` defines behavior that every provider must share.

| Current path | Responsibility | Target family |
| --- | --- | --- |
| `src/lib.rs` | Re-exports for admitted public contracts | Crate facade |
| `src/dtype.rs` | Backend-independent storage types | Shared type |
| `src/error.rs` | Recoverable contract and host-buffer errors | Shared error |
| `src/rms_norm/mod.rs` | RMSNorm specifications and CPU references | `normalization/rms_norm` |
| `src/gemm/mod.rs` | Contiguous BF16 GEMM specification and CPU reference | `gemm/dense` |
| `src/rope/mod.rs` | Standard RoPE specification and CPU reference | `position/rope` |
| `src/attention/single_decode/mod.rs` | Contiguous decode and split-K state | `attention/single_decode` |
| `src/attention/paged_decode/mod.rs` | Read-only paged decode and page-table view | `attention/paged_decode` |
| `src/attention/ragged_prefill/mod.rs` | Ragged causal prefill and `indptr` view | `attention/ragged_prefill` |
| `src/attention/paged_prefill/mod.rs` | Read-only paged causal prefill | `attention/paged_prefill` |
| `src/attention/paged_append/mod.rs` | Fused RoPE append and exclusive-page ownership contract | `kv_cache/paged_append` |
| `src/**/tests.rs` | Tests owned by the matching operator domain | Matching family |

This crate contains no CUDA type, FFI, provider policy, launch configuration,
or engine scheduler policy. A GPU provider starts with a contract and CPU
reference unless the project records an independent oracle.

Family `mod.rs` files form the public facade. Private operator directories own
their contracts, references, metadata views, and tests.

## CUDA crate

`crates/loom-infer-cuda` owns CUDA execution and asynchronous resource safety.
The current runtime source remains authoritative until the namespace migration
lands.

| Current path | Responsibility | Target area |
| --- | --- | --- |
| `src/memory.rs` | Typed owned and external device regions, checked spans, and retained leases | `runtime/memory` |
| `src/command/` | Binding resolution, admission, retention, status, and completion | `runtime/command` |
| `src/device_status.rs` | Device status packet codes and typed host decoding | `runtime/status` |
| `src/driver.rs` | Raw-driver cleanup helpers | `runtime/driver` |
| `src/graph/mod.rs` | Fixed-address Graph capture and replay | `runtime/graph` |
| `src/interop.rs` | External streams and engine handoff | `runtime/interop` |
| `src/rms_norm/mod.rs` | RMSNorm plan and Rust device kernels | `normalization/rms_norm` |
| `src/gemm/` | Dense BF16 planning, one plan facade, and private cuBLASLt and Loom providers | `gemm/dense` when a second GEMM contract needs the split |
| `src/attention/decode.rs` | Single and paged decode providers | Matching attention domains |
| `src/attention/prefill.rs` | Ragged and paged prefill providers | Matching attention domains |
| `src/rope/mod.rs` | Standard RoPE and fused append providers | `position/rope` and `kv_cache/paged_append` |

The current GEMM area has one public dense operation and two private providers:

```text
gemm/
|-- mod.rs
|-- planner.rs
|-- plan.rs
`-- provider/
    |-- mod.rs
    |-- cublaslt.rs
    `-- loom/
        |-- mod.rs
        `-- sm90/
            `-- mod.rs
```

`GemmPlanner` exposes the provider-neutral plan path. `provider/cublaslt.rs`
owns the general vendor implementation. `provider/loom/sm90` owns the
experimental H20 `sm_90a` M=1 kernel. Both use the same plan, operands,
command, completion, and Graph path.

When a second GEMM contract lands, `gemm/mod.rs` remains the family facade and
the existing files can move under `gemm/dense`. Grouped and quantized contracts
then get separate directories. Loom does not pre-create those directories or
add forwarding APIs.

The cuda-oxide `#[cuda_module]` macro discovers one inline kernel bundle. Split
a bundle only at a complete provider domain with a separate safety proof. File
length alone does not justify another device bundle.

Keep region construction in the runtime memory area, operand resolution in
the command area, resource retention in submission, and settlement in
completion. Every external pointer carries its span, context, access mode, and
lifetime lease.

Loom will not add a generic multi-backend runtime before a second device
backend exists.

## Validation, tools, and evidence

`crates/loom-infer-validation` contains permanent Rust hardware programs.

| Path | Responsibility |
| --- | --- |
| `src/gates/*.rs` | Operator-specific H20 correctness and lifecycle gates |
| `src/benchmarks/*.rs` | Loom-side matched and Graph measurements |
| `src/support/fixture.rs` | Deterministic shared fixtures |
| `src/support/comparison.rs` | Finite comparisons and stable digests |
| `src/support/reporting.rs` | Stable gate output prefixes |
| `src/bin/*.rs` | Thin process entry points |

Validation modules follow the same operator families after a source domain
moves. `tools/flashinfer` contains pinned comparison scripts. These scripts are
validation infrastructure, not a product Python API.

`docs/results` contains immutable records. Each file binds its claim to a
source tree, toolchain, artifact, device, contract, and command matrix. The
2026-08-06 fused-append files remain historical because the current contract
adds exclusive-page reference counts.

Correctness records do not imply performance. Performance records do not
imply Graph, engine, or serving behavior.

## Adding a vertical slice

1. Select the final operator family and pin the contract.
2. Add the backend-independent `Spec`, error cases, and reference.
3. Add one `Provider`, one named `Algorithm`, and one immutable `Plan`.
4. Define typed `Operands` and extend resolution only for their access pattern.
5. Enqueue through `CommandScope` and return `Completion`.
6. Add CPU tests and one hardware validation program.
7. Record host, device, lifecycle, sanitizer, performance, Graph, and engine
   evidence as separate gates.
8. Update the catalog and parity matrix only after source and evidence agree.

For a KV mutation, define read sharing, write ownership, copy-on-write owner,
and metadata lifetime before device implementation.

## FlashInfer mapping

| FlashInfer concept | Loom family |
| --- | --- |
| Single decode and state merge | `attention/single_decode` |
| Paged decode semantics | `attention/paged_decode` |
| Ragged prefill semantics | `attention/ragged_prefill` |
| Paged prefill semantics | `attention/paged_prefill` |
| Fused RoPE plus paged append | `kv_cache/paged_append` with `position/rope` semantics |
| Dense and grouped GEMM | `gemm/dense` and `gemm/grouped` |
| Workspace, stream, and Graph ownership | CUDA runtime area |
| Hardware tests and benchmarks | `loom-infer-validation` and `tools/flashinfer` |

The mapping follows semantic domains. It does not copy FlashInfer's Python
wrapper hierarchy into the Rust API.
