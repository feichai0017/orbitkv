# Validation Records

| Record | Scope |
| --- | --- |
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
- VMM does not yet back SGLang KV tensors.
- Every primary record includes a hash manifest.
