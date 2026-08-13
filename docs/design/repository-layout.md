# Repository layout

The current source keeps three crates. The accepted standalone-engine target
adds separate model-execution and server crates plus an offline TileLang
kernel tree. A module or crate still needs a functional, dependency, build,
ownership, safety, or release boundary.

## Current top level

```text
oxide-infer/
|-- crates/
|   |-- oxide-infer/        contracts and CPU references
|   |-- oxide-infer-cuda/   CUDA plans, providers, runtime, and Rust kernels
|   `-- oxide-infer-lab/    non-published hardware gates and benchmarks
|-- docs/
|   |-- design/             architecture and layout decisions
|   |-- development/        build and validation procedures
|   |-- integrations/       external engine source pairs and boundaries
|   `-- results/            immutable machine-readable evidence
|-- tools/
|   |-- flashinfer/         pinned comparison provider and summaries
|   `-- gemm/               workload census tools
|-- website/                Astro documentation site
|-- Makefile                local, CI, CUDA, and evidence entry points
`-- .github/                CI and Pages workflows
```

Generated PTX, cubins, profiler captures, model weights, `target/`, and website
build output are not product source.

## Accepted target layout

```text
oxide-infer/
|-- apps/
|   `-- oxide-infer/          CLI and server binary
|-- crates/
|   |-- oxide-infer/          contracts and CPU references
|   |-- oxide-infer-cuda/     CUDA resources and checked artifact launch
|   |-- oxide-infer-engine/   model IR, plans, KV pager, executor, sampling
|   |-- oxide-infer-server/   API, tokenizer, streaming, process lifecycle
|   `-- oxide-infer-lab/      correctness and benchmark programs
|-- kernels/
|   `-- tilelang/             sources, build profiles, and manifests
|-- benchmarks/
|   |-- kernels/              matched FlashInfer comparisons
|   `-- engines/              neutral reference-engine comparisons
|-- artifacts/
|   `-- manifests/            release metadata; binaries remain generated
|-- docs/
|-- website/
`-- .github/
```

The binary composes `oxide-infer-server` and `oxide-infer-engine`. The server
depends on engine request and event types, not GPU tensors. TileLang and Python
run only in the artifact build and qualification environment, never in the
serving process.

The new `oxide-infer-engine` crate has a real boundary: it owns model and KV
state and depends on execution facilities that the backend-independent
contract crate must not acquire. `oxide-infer-server` has a separate dependency
boundary for HTTP, tokenizer, and process concerns. The target structure and
external-source rules are defined in the
[standalone engine architecture](standalone-oxide-engine.md).

## Transitional three-crate rule

```text
consumer engine or adapter
  -> oxide-infer-cuda
       -> oxide-infer
       -> native Oxide provider
       -> explicit vendor providers

oxide-infer-lab
  -> oxide-infer-cuda
  -> oxide-infer
```

`oxide-infer` builds without CUDA. Product crates do not depend on the lab,
documentation, website, tools, or result records. The lab crate is a workspace
member but not a default member.

This rule describes current source only. The engine migration may add
`oxide-infer-engine` and `oxide-infer-server` with their first executable
vertical slices; it does not add empty placeholder crates.

GEMM, attention, KV-cache operations, and Graph execution remain modules in
these crates. Do not add `oxide-gemm`, `oxide-runtime`, or
`oxide-cuda-kernels` for namespace convenience.

## Operator family namespaces

The public namespace follows operator semantics. Fusion remains an algorithm
property.

| Family | Responsibility | Source state |
| --- | --- | --- |
| `attention` | Decode, prefill, masking, and attention-state merge | Current |
| `gemm` | Dense, grouped, and quantized matrix operations | BF16 dense current; native M=1 path experimental |
| `kv_cache` | Paged append, gather, scatter, compaction, and remapping | Paged append current under legacy physical paths; family migration pending |
| `normalization` | RMSNorm and later normalization contracts | RMSNorm current under `rms_norm`; family migration pending |
| `position` | RoPE and later position transforms | RoPE current under `rope`; family migration pending |
| `activation` | Activation and gated-activation operations | Planned |
| `sampling` | Logits transforms, sampling, and RNG | Planned |
| `speculation` | Draft verification and token compaction | Planned |
| `quantization` | Scale, packing, conversion, and dequantization | Planned |
| `moe` | Routing, permutation, expert inputs, and combine | Planned |
| `communication` | Qualified tensor-parallel and expert-parallel collectives | Planned |

Move an implemented domain directly to its final family. Do not add forwarding
modules, compatibility aliases, or empty target directories.

## Contract crate

`crates/oxide-infer` defines behavior that every provider must share.

| Current path | Responsibility | Target family |
| --- | --- | --- |
| `src/lib.rs` | Public facade for admitted contracts | Crate facade |
| `src/dtype.rs` | Backend-independent storage types | Shared type |
| `src/error.rs` | Recoverable contract and host-reference errors | Shared error |
| `src/attention/single_decode/` | Contiguous decode and split-K state | `attention/single_decode` |
| `src/attention/paged_decode/` | Read-only paged decode and page-table view | `attention/paged_decode` |
| `src/attention/ragged_prefill/` | Ragged causal prefill and index views | `attention/ragged_prefill` |
| `src/attention/paged_prefill/` | Read-only paged causal prefill | `attention/paged_prefill` |
| `src/attention/paged_append/` | Fused RoPE append and exclusive-page contract | `kv_cache/paged_append` |
| `src/gemm/` | Contiguous BF16 dense specification and reference | `gemm`, then `gemm/dense` when a second contract exists |
| `src/rms_norm/` | RMSNorm specifications and references | `normalization/rms_norm` |
| `src/rope/` | Standard RoPE specification and reference | `position/rope` |

The contract crate contains no CUDA type, FFI, launch configuration, provider
policy, or engine scheduler policy. A GPU provider starts with a contract and
CPU reference unless the project records an independent oracle.

## CUDA crate

`crates/oxide-infer-cuda` owns CUDA execution and asynchronous resource safety.

| Current path | Responsibility | Target area |
| --- | --- | --- |
| `src/memory.rs` | Owned and external device regions, spans, and leases | `runtime/memory` |
| `src/command/` | Binding resolution, admission, retention, status, and completion | `runtime/command` |
| `src/device_status.rs` | Device status codes and typed host decoding | `runtime/status` |
| `src/driver.rs` | Raw-driver cleanup helpers | `runtime/driver` |
| `src/graph/` | Fixed-address Graph capture and replay | `runtime/graph` |
| `src/interop.rs` | External streams and engine handoff | `runtime/interop` |
| `src/attention/` | Decode and prefill plans and native kernels | Matching attention families |
| `src/gemm/` | Dense planning and native or vendor providers | `gemm`, then contract subdirectories as needed |
| `src/rms_norm/` | RMSNorm plans and native kernels | `normalization/rms_norm` |
| `src/rope/` | RoPE and fused append plans and native kernels | `position/rope` and `kv_cache/paged_append` |

The runtime namespace stays inside this crate. A generic multi-backend runtime
requires a second implemented device backend.

### GEMM provider layout

The dense GEMM family has one public execution path and two private providers:

```text
gemm/
|-- mod.rs
|-- planner.rs
|-- plan.rs
`-- provider/
    |-- mod.rs
    |-- cublaslt.rs
    `-- oxide/
        |-- mod.rs
        `-- sm90/
            `-- mod.rs
```

`GemmPlanner` creates a provider-neutral plan. `provider/cublaslt.rs` owns the
vendor implementation. `provider/oxide/sm90` owns the experimental SM90a
`sm_90a` M=1 kernel.

When a second GEMM contract lands, keep `gemm/mod.rs` as the family facade and
move dense files under `gemm/dense`. Add grouped or quantized directories only
with their first admitted contract.

### Device bundles

The cuda-oxide `#[cuda_module]` macro discovers an inline device bundle. Split
a bundle only at a complete provider domain with a separate safety proof.
File length alone does not justify another artifact.

Keep these ownership rules:

- `runtime/memory` constructs typed regions.
- `runtime/command` resolves operands and retains resources.
- operator plans declare workspace and launch requirements.
- operator operands carry page tables and invocation metadata.
- provider architecture modules own target-specific kernels.
- completion settlement releases retained capabilities.

## Lab crate and evidence

`crates/oxide-infer-lab` contains permanent Rust hardware programs.

| Path | Responsibility |
| --- | --- |
| `src/gates/*.rs` | Operator correctness, negative, and lifecycle gates |
| `src/benchmarks/*.rs` | Native, vendor, and Graph measurements |
| `src/support/fixture.rs` | Deterministic fixtures |
| `src/support/comparison.rs` | Finite comparisons and stable digests |
| `src/support/reporting.rs` | Stable output records |
| `src/bin/*.rs` | Thin process entry points |

Lab modules follow final operator families after a source domain moves.
`tools/flashinfer` remains comparison infrastructure, not a product Python API.

`docs/results` contains immutable records. Each record binds a claim to an
exact source tree, toolchain, artifact, device, contract, and command matrix.
Historical records keep their original project and provider identifiers.

## Add one vertical slice

1. Select a final operator family from a real engine call site.
2. Pin the `Spec`, error cases, and independent reference.
3. Add one provider, one named algorithm, and one immutable plan.
4. Define typed operands and exact access modes.
5. Enqueue through `CommandScope` and return `Completion`.
6. Add host tests and one permanent hardware gate.
7. Record each applicable evidence level separately.
8. Update the catalog only after source and evidence agree.

For KV mutation, define read sharing, write ownership, copy-on-write ownership,
and metadata lifetime before device implementation.

## FlashInfer concept mapping

| FlashInfer concept | Oxide Infer family |
| --- | --- |
| Single decode and state merge | `attention/single_decode` |
| Paged decode | `attention/paged_decode` |
| Ragged prefill | `attention/ragged_prefill` |
| Paged prefill | `attention/paged_prefill` |
| Fused RoPE plus paged append | `kv_cache/paged_append` with `position/rope` semantics |
| Dense and grouped GEMM | `gemm/dense` and future `gemm/grouped` |
| Workspace, stream, and Graph ownership | CUDA runtime area |
| Hardware tests and benchmarks | `oxide-infer-lab` and `tools/flashinfer` |

The mapping follows semantic domains. It does not copy another project's
wrapper hierarchy into the Rust API.
