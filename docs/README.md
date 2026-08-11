# Loom Infer documentation

Loom Infer is a Rust-native CUDA operator layer for LLM inference engines.
This documentation separates implemented contracts, planned work, and recorded
evidence.

## Read in this order

1. [Architecture](design/loom-infer-architecture.md) defines ownership, plans,
   bindings, submission, and completion.
2. [Operator catalog](operator-catalog.md) lists every admitted operator
   combination and its current gate.
3. [FlashInfer parity](flashinfer-parity.md) tracks the pinned comparison
   surface without claiming complete parity.
4. [Roadmap](roadmap.md) orders the remaining correctness, integration, and
   performance work.
5. [Mistral.rs integration](integrations/mistralrs.md) defines the paired-repository boundary and POC qualification.
6. [Evidence](results/README.md) indexes machine-readable device and benchmark
   records.

## Development

- [Repository layout](design/repository-layout.md) maps each crate and module to
  one responsibility.
- [Environment](development/environment.md) pins Rust, Node.js, CUDA, and
  cuda-oxide.
- [H20 validation](development/h20-validation.md) defines device correctness,
  sanitizer, Graph, and performance gates.

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

## Documentation rules

- State implemented behavior in the present tense.
- Mark planned work as planned.
- Name each admitted algorithm and shape combination.
- Keep correctness, performance, Graph, engine, and serving claims separate.
- Bind each performance claim to source, hardware, contract, method, and raw
  result.
- Preserve historical records. Do not project them onto changed source.

The root [README](../README.md) provides the short project overview.
