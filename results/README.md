# Validation Records

Capability levels and unsupported paths are defined only in
`docs/capability-matrix.md`. This index lists evidence snapshots.

| Record | Scope |
| --- | --- |
| `h20-runtime-state-plan-20260819` | One fingerprinted runtime artifact drives owner and Capsule contracts |
| `h20-hybrid-capsule-20260818` | GPT-OSS Full+SWA component restore crossover |
| `h20-live-tail-capsule-20260818` | Pure-SWA minimal-state hydration and E2E A/B |
| `h20-capsule-export-20260818` | Real-checkpoint KV export and host restore |
| `applicability-h20-20260817` | Qwen Full fallback, Mistral bounded state, GPT-OSS Hybrid |
| `h20-gpt-oss-20b-real-20260817` | Primary real-checkpoint SGLang result |
| `lifetime-normalization-20260817` | Per-head windows and Retention Amplification |
| `chunked-local-20260817` | Same-chunk → ResettableArena |
| `sink-sliding-20260817` | Sink + local lifetime partitioning |
| `retention-ir-20260817` | Declarative IR and legacy equivalence |
| `h20-generation-vmm-20260817` | Generation-aware CUDA VMM lifecycle |
| `owner-ffi-20260817` | In-process owner ABI |

Other directories are earlier calibration runs retained for auditability.

Boundaries:

- GPT-OSS is a real checkpoint systems result, not a model-quality claim.
- Mistral page16 execution qualifies decode CUDA Graph replay with eager prefill.
- Per-head, Sink+Sliding, and Same-Chunk records are host compiler/Manager
  proofs, not GPU performance results.
- `h20-capsule-export-20260818` covers export and host restore only; later
  live-tail and Hybrid records qualify selected hydration paths.
- Live-tail Capsule hydration is qualified only for compiler-proven pure SWA.
- Hybrid Capsule restore is beneficial only beyond the measured host-restore
  crossover; short prefixes should use cold prefill.
- VMM does not yet back SGLang KV tensors.
- Every primary record includes a hash manifest.
