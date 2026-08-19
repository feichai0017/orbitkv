# Capability Matrix

This file is the normative OrbitKV capability boundary. README, website, and
result summaries link here rather than defining independent support claims.

## Levels

| Level | Meaning |
| --- | --- |
| L1 Compiler | OrbitKV parses the semantics and emits a checked plan. |
| L2 Reference Runtime | Simulator or reference Manager executes the plan with correctness tests. |
| L3 GPU Primitive | An isolated CUDA or physical-backend primitive is qualified. |
| L4 Engine E2E | A pinned external engine and real model/checkpoint execute the path end to end. |
| L5 Production | Complex-feature, pressure, version-matrix, and failure qualification is complete. |

`Implemented` never means L5 unless the row explicitly says L5.

## Current Matrix

| Capability | Level | Qualified Boundary | Evidence |
| --- | --- | --- | --- |
| Full Attention retention | L2 | `AppendOnly`; unbounded lifetime; reference runtime | `tests/retention_cli.rs`, `src/runtime.rs` |
| Fixed Sliding retention | L4 | Uniform SWA and Full+SWA on pinned SGLang; page size 16 paths qualified | `results/applicability-h20-20260817/`, `results/h20-gpt-oss-20b-real-20260817/` |
| Dilated Local retention | L2 | Exact finite last-read inference; periodic reference layout | `examples/dilated_local_retention.json`, `tests/retention_cli.rs` |
| Sink + Sliding | L2 | `Pinned + PeriodicFrom`; exhaustive host/reference proof | `results/sink-sliding-20260817/` |
| Same-Chunk retention | L2 | `ResettableArena`; epoch retirement in reference runtime | `results/chunked-local-20260817/` |
| Per-head fixed windows | L2 | Lifetime-normalized stripes and exact Retention Amplification | `results/lifetime-normalization-20260817/` |
| HF Full/SWA frontend | L4 | Explicit layer types and allowlisted uniform SWA; unknown semantics fall back to Full | `results/applicability-h20-20260817/` |
| Unified runtime StatePlan artifact | L4 | One fingerprinted JSON drives semantic source, layout, SGLang policy, owner mode, and Capsule limits; deployment paths remain external | `tests/retention_cli.rs`, `integrations/sglang/tests/test_shadow_plugin.py` |
| SGLang physical-plan optimizer | L4 | Non-overlap Full+SWA ChunkCache; one Full domain and one Sliding domain | `results/h20-gpt-oss-20b-real-20260817/` |
| Proof-carrying reclamation | L4 | Non-overlap SGLang ChunkCache; FFI owner; radix/speculation/disaggregation disabled | `results/owner-ffi-20260817/`, `results/h20-gpt-oss-20b-real-20260817/` |
| Paged-periodic pure SWA | L4 | Mistral, page size 16, eager prefill; decode CUDA Graph qualified | `results/applicability-h20-20260817/page16-graph-manifest.json` |
| Pure-SWA live-tail Capsule | L4 | SGLang hydration; 1K live tail; single request and one decode token | `results/h20-live-tail-capsule-20260818/` |
| Full+SWA Hybrid Capsule | L4 | GPT-OSS 20B; Full history plus 128-token SWA tail; measured host-file crossover | `results/h20-hybrid-capsule-20260818/` |
| Holt persistent Capsule catalog | L4 | Sole catalog backend; content-addressed payload files; longest-prefix restore | `results/h20-capsule-export-20260818/`, `results/h20-hybrid-capsule-20260818/` |
| Component-aware Prefix ownership | L4 | GPT-OSS 20B; exact 1K Prefix; SGLang `UnifiedRadixCache`; authenticated Full+SWA components; atomic shared lease; single request, non-overlap | `results/h20-radix-prefix-20260819/` |
| CUDA-event execution frontier | L4 | GPT-OSS 20B; SGLang overlap scheduler; Full+SWA ChunkCache; one forward-stream domain; one- and two-request event-backed reclamation | `results/h20-cuda-event-overlap-20260819/` |
| Dense ownership runtime | L2 | Fixed Class IDs and request stripes; bounded generation arenas; 1,000-event Reference differential test; release control-plane benchmark | `results/dense-runtime-20260819/` |
| Dense SGLang physical binding | L4 | Pure SWA; 4K logical Prefix/1K live tail; paged-periodic allocator; single request; 128 continuation tokens across seven rounds; typed-struct FFI v1; CUDA-event view pin; non-overlap | `results/h20-dense-sglang-20260819/` |
| Generation-aware CUDA VMM slot | L3 | H20 reserve/map/remap/unmap primitive; not SGLang tensor storage | `results/h20-generation-vmm-20260817/` |
| Transactional physical reclamation | L3 | Certificate, backend receipt, commit; reference/CUDA lifecycle | `results/h20-generation-vmm-20260817/` |
| Transactional allocation/binding | L4 | Rust prepare/commit/abort coordinator drives SGLang Capsule hydration; binding uses the owner sidecar while reclamation may use FFI | `src/binding.rs`, `src/manager.rs`, `integrations/sglang/tests/test_shadow_plugin.py` |

## Not Qualified

The following remain below L4:

- One runtime resource manifest for deployment paths, model identity, and external service endpoints.
- A versioned FFI ABI for component-aware Prefix binding; that L4 path still
  uses the Rust owner sidecar. Dense page binding has a separate typed FFI v1.
- Drop-in replacement of SGLang's complete allocator/ReqToToken/Prefix contract;
  the qualified Dense path owns page binding and lifetime for one pure-SWA
  execution slice while SGLang still supplies allocation primitives and kernel
  views.
- SGLang Radix/Prefix pressure eviction, concurrent sharing, and broader
  shared-page qualification beyond the exact-prefix Full+SWA path.
- Multiple simultaneous completion domains, Radix+overlap, and CUDA Graph plus
  overlap execution-frontier qualification.
- Dense SGLang overlap, Radix, speculation, disaggregation, CUDA Graph, and
  Hybrid Full+SWA physical binding.
- Speculative decoding, fork, rollback, beam/tree state, and COW.
- Mamba/SSM or other recurrent state.
- Cross-attention and dynamic/content-dependent sparse attention.
- A general Dense-runtime SGLang adapter beyond the qualified pure-SWA path.
- A vLLM adapter.
- Native TWO-span or mirrored-VMM attention data planes.
- VMM-backed SGLang KV tensors.
- Multi-GPU, remote-memory, or production version-matrix qualification.

Unsupported or unproved semantics fail closed or fall back to unbounded Full
state. Performance results apply only to their recorded hardware, model,
engine revision, and workload. In particular, Hybrid Capsule restore is slower
than cold prefill at 1K and 4K in the recorded host-file backend, and faster at
16K; the current runtime does not yet compile that crossover into an automatic
decision artifact.
