# Validation Records

OrbitKV result records separate host proofs, allocator traces, physical-memory
measurements, and end-to-end serving behavior.

## 2026-08-16/17 H20 records

| Directory | Scope |
| --- | --- |
| `h20-shadow-ab-20260816` | Stock SGLang versus non-owning OrbitKV observation |
| `h20-policy-ab-20260816` | Initial interval-16 policy screen |
| `h20-policy-sweep-20260816` | Interval-32 and interval-64 Pareto screen |
| `h20-policy32-notrace-ab-20260817` | Interval-32 normal-path overhead without allocation tracing |
| `h20-62l-memory-20260817` | Large-geometry physical KV-pool reduction |
| `h20-admission-ab-20260817` | Fixed-budget long-request admission and makespan |
| `h20-owner-ab-20260817` | Rust owning control plane versus the same generated policy |
| `h20-cuda-vmm-20260817` | Isolated H20 CUDA VMM stable-address remap qualification |
| `owner-ffi-20260817` | In-process C ABI transport microbenchmark and balanced H20 A/B |
| `h20-generation-vmm-20260817` | Layout-driven cell versions, CUDA events, VMM generation receipts |

The reviewed interpretations are:

- `docs/h20-sglang-validation-20260817.md`;
- `docs/h20-owning-vmm-validation-20260817.md`.

The owning record is end-to-end SGLang evidence for the strict SWA chunk-cache,
non-overlap, non-speculative path. The VMM record is isolated physical-backend
qualification; it is not evidence that SGLang KV tensors already use VMM.

Raw logs and JSONL allocator traces are intentionally excluded from Git. Matrix
records include exact commands, source paths, environment fields, output
digests, timing samples, and references to the local raw artifacts used during
the run.

Source and record hashes for the owning/VMM milestone are stored in
`h20-owning-vmm-manifest-20260817.json`.
