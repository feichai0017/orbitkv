# Loom Infer documentation

Loom Infer is a Rust-native CUDA operator layer for LLM inference engines.
This documentation separates implemented contracts, planned work, and recorded
evidence.

## Read in this order

1. [Architecture](design/loom-infer-architecture.md) defines ownership, plans,
   bindings, submission, and completion.
2. [Repository layout](design/repository-layout.md) separates current paths
   from the target family namespaces.
3. [Operator catalog](operator-catalog.md) lists every admitted operator
   combination and its current gate.
4. [FlashInfer parity](flashinfer-parity.md) tracks the pinned comparison
   surface without claiming complete parity.
5. [Roadmap](roadmap.md) orders the remaining correctness, integration, and
   performance work.
6. [Mistral.rs integration](integrations/mistralrs.md) defines the
   paired-repository boundary and POC qualification.
7. [Evidence](results/README.md) indexes machine-readable device and benchmark
   records.

## Development

- [Environment](development/environment.md) pins Rust, Node.js, CUDA, and
  cuda-oxide.
- [H20 validation](development/h20-validation.md) defines device correctness,
  sanitizer, Graph, and performance gates.
- [Dense GEMM shape census](development/gemm-shape-census.md) defines the
  untimed model-call profile used to select Loom GEMM candidates.
- [Experimental SM90 M=1 GEMV](development/sm90-simt-gemv-m1.md) freezes the
  first measured Loom GEMM contract and its promotion gates.

## Current boundary

The product contains three crates:

| Crate | Responsibility |
| --- | --- |
| `loom-infer` | Backend-independent contracts and CPU references |
| `loom-infer-cuda` | CUDA providers, command runtime, Graphs, and vendor calls |
| `loom-infer-validation` | H20 gates, benchmarks, and evidence generation |

The engine retains models, requests, scheduling, KV allocation policy, and
distributed control. Loom owns the checked operator boundary.

Engine adapters stay in their engine repositories.
Loom records their pinned source pairs and qualification status without vendoring engine source or raw evidence.

Fused KV append now requires exclusive write pages. Its historical records do
not qualify the revised ownership contract. New device evidence is required.

## State sources

Use one source for each kind of fact:

| Source | Question answered |
| --- | --- |
| Rust source and Cargo manifests | What exists in the current checkout? |
| [Operator catalog](operator-catalog.md) | What source surface is admitted for users? |
| [Evidence index](results/README.md) and immutable records | What passed on an exact source and environment? |
| [Integration documents](integrations/mistralrs.md) | What external source pair and adapter boundary passed? |
| [Roadmap](roadmap.md) | What work is planned, and what ends each milestone? |
| Design documents | Which boundaries and names guide future source? |

Source presence does not prove device correctness or performance. Old evidence
does not qualify changed source. Planned work does not count as an admitted
operator.

When these surfaces disagree, keep result records unchanged. Correct the
catalog or design projection against the source, then create new evidence for
the new commit.

## Documentation rules

- State implemented behavior in the present tense.
- Mark planned work as planned.
- Name each admitted algorithm and shape combination.
- Use the public lifecycle terms from `Spec` through `Completion` without
  synonyms.
- Keep correctness, performance, Graph, engine, and serving claims separate.
- Bind each performance claim to source, hardware, contract, method, and raw
  result.
- Preserve historical records. Do not project them onto changed source.

The root [README](../README.md) provides the short project overview.
