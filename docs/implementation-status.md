# Implementation status

Support is reported separately for compiler representation, manager lifecycle,
executor lowering, device execution, complete-model execution, and measured
benefit. A model is supported end to end only when every required state and
operator passes all applicable layers.

## Current implementation

| State or attention family | Compiler and manager | Executor | Complete model | Benefit |
| --- | --- | --- | --- | --- |
| Full MHA/GQA token KV | Implemented and host-tested | Paged attention, Prefix/COW, writes, relocation | Narrow released-checkpoint H20 closure | No lifetime-management L5 result |
| Sliding Window token KV | Periodic placement, retirement, ACK, and generation reuse host-tested | CSR/window lowering implemented | Not independently device-qualified | Unproven |
| Full + Sliding interleaving | Independent class lifetimes and joint transactions host-tested | Manifest-driven per-layer graph construction, independent arenas, write slots, CSR metadata, and capture signatures pass host tests; a synthetic-policy H20 plumbing smoke passes | No released hybrid-architecture checkpoint run | Unproven |
| Exact Chunked attention | Resettable epoch arena host-tested; one whole-domain class only | Metadata lowering implemented | Not independently device-qualified | Unproven |
| MLA/latent KV | Component-aware latent/RoPE lifecycle compiles | Matching Luminal attention kernel contract missing | Unsupported | Unproven |
| Mamba/GDN/KDA/linear attention | Recurrent checkpoint geometry compiles and checkpoint pool is host-tested | Recurrent operators are not integrated into one transaction | Unsupported | Unproven |
| Convolution state | Generation-checked checkpoint lifecycle host-tested | Convolution operator/state transaction missing | Unsupported | Unproven |
| Sparse, tree, speculative, cross-attention | No complete general contract | Missing | Unsupported | Unproven |

The native dense decoder now accepts multiple token-KV classes with exact,
non-overlapping layer coverage. Each class owns independent dynamic page
metadata and arena geometry. A real released dense checkpoint also completes a
short H20 prefill and captured decode under a synthetic interleaved Full/Sliding
policy whose window exceeds the test context. This proves device plumbing, not
that a released hybrid-architecture checkpoint executes correctly.
Unknown or incomplete contracts fail closed instead of silently becoming Full
attention.

## External tiers

Backend-neutral export, restore, replica admission, deletion acknowledgement,
completion ordering, and ambiguity quarantine are implemented. A host-memory
reference adapter moves real bytes and verifies a partial-tail round trip across
sessions. Mooncake/NIXL, remote lease renewal, eviction races, shared Prefix
restore, and distributed recovery remain open.

## Serving

The async local `Engine` contract and optional vLLM Rust frontend adapter are
implemented and host-tested. A complete scheduler that connects HTTP requests,
continuous batches, `RuntimeSession`, Luminal execution, sampling, cancellation,
and final drain is still required before the repository is a production server.

## Evidence interpretation

The child CUDA Graph result demonstrates a narrow dispatch optimization. It does
not validate the central compiler claim. That claim requires a same-executor
ablation between conservative and compiled retention, followed by a matched
end-to-end comparison with a tuned reference engine.
