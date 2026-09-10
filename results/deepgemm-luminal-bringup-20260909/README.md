# Luminal block-scaled DeepGEMM bring-up — 2026-09-09

This is operator-level bring-up evidence on one NVIDIA H20 (SM90, 96 GB), not
a Qwen 27B serving result. The inspected checkpoint directory is named
`qwen3.8-27b-fp8`, while its metadata identifies
`Qwen3_5ForConditionalGeneration` / `qwen3_5_text`.

## Integration result

- DeepGEMM is pinned at `559d79fb6994a58b8a15b4b93bf13ccc16edf247`.
- A raw CUDA Driver API probe launched the generated SM90 1D2D CUBIN without
  Python or PyTorch in the launch path. The ABI is `sfb*, grouped_layout*, m, n,
  k, TMA(A), TMA(B), TMA(D), TMA(SFA)`.
- Luminal now exposes a provider-neutral 128x128 block-scaled linear custom op.
  Its independent CUDA reference and four legal DeepGEMM tile candidates are
  unioned into one e-class. Candidate JIT runs before timing; the ordinary CUDA
  search profiles candidates and the selected LLIR carries the provider revision
  plus tile variant.
- H20 tests cover direct native DeepGEMM execution, independent-reference parity,
  e-class alternatives, device search, and strict selected-schedule replay.
  DeepGEMM currently remains a standalone HostOp rather than being absorbed
  into a parent CUDA Graph.

## Upstream DeepGEMM probe

These warm-cache figures use DeepGEMM's official Python wrapper only as a
bring-up/reference harness. They do not measure OrbitKV or full-model execution.
Activation quantization is outside the timed GEMM loop.

| M | N | K | time (us) | TFLOP/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5,120 | 5,120 | 27.822 | 1.9 |
| 16 | 5,120 | 5,120 | 25.960 | 32.3 |
| 128 | 5,120 | 5,120 | 35.270 | 190.3 |
| 1 | 17,408 | 5,120 | 54.788 | 3.3 |
| 16 | 17,408 | 5,120 | 54.652 | 52.2 |
| 128 | 17,408 | 5,120 | 91.207 | 250.2 |
| 1 | 5,120 | 17,408 | 68.639 | 2.6 |
| 16 | 5,120 | 17,408 | 68.835 | 41.4 |
| 128 | 5,120 | 17,408 | 107.617 | 212.0 |

## Remaining qualification

The complete 64-layer checkpoint now passes a bounded H20 prefill, one decode
step, and joint token/fixed-state final drain. With two searched programs per
bucket and the final semantic LLIR fingerprint, the cold search/compile took
273.563 s; strict artifact reload took 38.047 s, the four-token prefill took
2.698867 s and returned token id 5, and the one-token decode took 2.843313 s
and returned token id 0. These are bring-up timings with diagnostic logits,
not optimized serving numbers. Each selected bucket contains 28 DeepGEMM
choices. Remaining qualification includes an independent logits/token oracle,
multi-token decode soak/capacity checks, and matched vLLM/SGLang runs.

The bring-up also exposed a compiler-scale boundary: before source-only values
were treated as loop inputs, the 17,185-node full graph reached about 219 GiB
anonymous RSS and was killed by the 229.5 GB memory cgroup. The corrected
rolling prepass reduced the search graph to roughly 7.6k HLIR nodes, after
which both bucket e-graphs and device search completed.
