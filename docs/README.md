# Loom Kernels documentation

Loom documents contracts, integration boundaries, and measured results. Start
with the page that answers your current question.

[Project overview](../README.md) · [Website](https://feichai0017.github.io/loom-kernels/) · [Operators](operator-catalog.md) · [Evidence](results/README.md)

## Choose a path

| Goal | Read |
| --- | --- |
| Understand the product boundary | [Operator-library design](design/operator-library.md) |
| Trace an operator through the repository | [Code layout](design/code-layout.md) |
| Check implemented and open work | [Implementation status](status.md) |
| Check versions and the native wheel | [Compatibility and distribution](compatibility.md) |
| Integrate PyTorch or vLLM | [Python guide](../python/README.md) and [vLLM guide](guides/vllm-ir-provider.md) |
| Inspect measurements | [H20 evidence index](results/README.md) |
| Choose the next operator | [Roadmap](roadmap.md) and [operator catalog](operator-catalog.md) |

## Design references

| Topic | Document |
| --- | --- |
| Operator admission and vendor boundaries | [Operator library](design/operator-library.md) |
| Paged decode and split-K/LSE | [Paged decode attention](design/paged-decode-attention.md) |
| Static FP8 KV write and rejected system gate | [FP8 KV cache](design/fp8-kv-cache.md) |
| Stable MoE permutation and combine | [MoE movement](design/moe-movement.md) |
| Explicit-state Philox sampling | [Counter-based sampling](design/counter-based-sampling.md) |
| Greedy draft verification | [Speculative verification](design/greedy-speculative-verify.md) |

## Evidence levels

Loom keeps six claims separate:

1. The contract rejects invalid inputs.
2. CUDA matches the CPU or high-precision oracle.
3. A warmed operator beats a named baseline.
4. Framework dispatch and CUDA Graph replay work.
5. A real engine reaches the operator.
6. A model or serving workload improves.

Passing one level does not imply the next. Only JSON files under
[`docs/results`](results/) support performance claims. The evidence index also
keeps parity, fallback, and rejected results.

## Status terms

| State | Meaning |
| --- | --- |
| `supported` | Contract, oracle, CUDA, framework adapter, and H20 evidence exist |
| `in progress` | Source exists, but a required engine or system gate remains open |
| `profile-gated` | Loom adds the path only after a real workload shows material cost |
| `vendor-backed` | Loom owns adjacent work while the engine keeps the base primitive |
