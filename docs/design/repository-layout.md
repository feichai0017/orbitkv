# Repository layout

Loom Infer adds a module or crate only when the code needs a separate
functional, ownership, or safety boundary.

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

## Dependency direction

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

## Contract crate

`crates/loom-infer` defines behavior that every backend must share.

| Path | Responsibility |
| --- | --- |
| `src/lib.rs` | Re-exports for admitted public contracts |
| `src/dtype.rs` | Backend-independent storage types |
| `src/error.rs` | Recoverable contract and host-buffer errors |
| `src/rms_norm/mod.rs` | RMSNorm specifications and CPU references |
| `src/gemm/mod.rs` | Contiguous BF16 GEMM specification and CPU reference |
| `src/rope/mod.rs` | Standard RoPE specification and CPU reference |
| `src/attention/single_decode/mod.rs` | Contiguous decode and split-K state |
| `src/attention/paged_decode/mod.rs` | Read-only paged decode and page-table view |
| `src/attention/ragged_prefill/mod.rs` | Ragged causal prefill and `indptr` view |
| `src/attention/paged_prefill/mod.rs` | Read-only paged causal prefill |
| `src/attention/paged_append/mod.rs` | Fused RoPE append and exclusive-page ownership contract |
| `src/**/tests.rs` | Tests owned by the matching operator domain |

This crate contains no CUDA type, FFI, provider policy, launch configuration,
or engine scheduler policy. A GPU provider starts with a contract and CPU
reference unless the project records an independent oracle.

Family `mod.rs` files form the public facade. Private domain directories own
implementation and tests. MLA or new KV operations should extend the
attention family before they justify another crate.

## CUDA crate

`crates/loom-infer-cuda` owns CUDA execution and asynchronous resource safety.

| Path | Responsibility |
| --- | --- |
| `src/memory.rs` | Typed owned and external device regions, checked spans, and retained leases |
| `src/command/mod.rs` | Command API facade and shared errors |
| `src/command/binding.rs` | Typed buffer ownership, leases, and handles |
| `src/command/resolve.rs` | Disjoint operand resolution and alias rejection |
| `src/command/submission.rs` | Queue admission, retention, and capture transfer |
| `src/command/completion.rs` | Completion settlement and quiescence fallback |
| `src/command/status.rs` | Retained status readbacks and decoder registrations |
| `src/device_status.rs` | Device status packet codes and typed host decoding |
| `src/driver.rs` | Raw-driver cleanup helpers |
| `src/rms_norm/mod.rs` | RMSNorm plans and Rust device kernels |
| `src/gemm/mod.rs` | Fixed-algorithm cuBLASLt provider |
| `src/attention/decode.rs` | Single and paged decode providers |
| `src/attention/prefill.rs` | Ragged and paged prefill providers |
| `src/rope/mod.rs` | Standard RoPE and fused append providers |
| `src/graph/mod.rs` | Fixed-address Graph capture and replay |

The cuda-oxide `#[cuda_module]` macro discovers one inline kernel bundle.
Split a large bundle only at a complete provider domain with a separate safety
proof. Do not split files only to reduce line count.

Keep region construction in `memory.rs`, operand patterns in
`command/resolve.rs`, resource retention in `command/submission.rs`, and
settlement in `command/completion.rs`. External pointers must enter through a
region that binds pointer, span, context, access mode, and lifetime lease.

Do not create a generic multi-backend runtime before a second device backend
exists.

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

`tools/flashinfer` contains the pinned external-provider scripts and evidence
summaries. These scripts are validation infrastructure, not a product Python
API.

`docs/results` contains immutable records. Each file binds its claim to a
source tree, toolchain, artifact, device, contract, and command matrix. The
2026-08-06 fused-append files remain historical because the current contract
adds exclusive-page reference counts.

Correctness records do not imply performance. Performance records do not
imply Graph, engine, or serving behavior.

## Adding a vertical slice

1. Pin the matching contract and state admitted differences.
2. Add the backend-independent specification, error cases, and reference.
3. Add one immutable provider plan and checked enqueue path.
4. Extend binding resolution only for the required access pattern.
5. Add CPU tests and a hardware validation program.
6. Record host, device, lifecycle, sanitizer, performance, Graph, and engine
   evidence as separate gates.
7. Update the catalog and parity matrix only after source and evidence agree.

For a KV mutation, define read sharing, write ownership, copy-on-write owner,
and metadata lifetime before device implementation.

## FlashInfer mapping

| FlashInfer concept | Loom location |
| --- | --- |
| Single decode and state merge | `loom-infer/attention/single_decode/mod.rs` |
| Paged decode semantics | `loom-infer/attention/paged_decode/mod.rs` |
| Ragged prefill semantics | `loom-infer/attention/ragged_prefill/mod.rs` |
| Paged prefill semantics | `loom-infer/attention/paged_prefill/mod.rs` |
| Fused RoPE plus paged append | `loom-infer/attention/paged_append/mod.rs`, `loom-infer-cuda/rope/mod.rs` |
| CUDA decode providers | `loom-infer-cuda/attention/decode.rs` |
| CUDA prefill providers | `loom-infer-cuda/attention/prefill.rs` |
| Wrapper planning lifecycle | Immutable Rust plan types |
| Workspace and stream ownership | `loom-infer-cuda/command`, `loom-infer-cuda/graph` |
| Hardware tests and benchmarks | `loom-infer-validation`, `tools/flashinfer` |

The mapping follows semantic domains. It does not copy FlashInfer's Python
wrapper hierarchy into the Rust API.
