# Oxide Infer documentation

The current source is a Rust-native CUDA operator layer. The accepted target
is a complete Rust inference engine that embeds a Mistral.rs control-plane
shell, owns model execution in Oxide, and uses offline TileLang artifacts as
its only product custom compute kernels. The target does not describe a
completed migration.

The documentation separates three facts:

- **Current** means the source implements the stated contract.
- **Experimental** means the source implements the contract, but promotion
  gates remain open.
- **Planned** means the roadmap admits the work. No public module or provider
  exists yet.

Source presence does not prove device correctness or performance. Each result
applies only to its recorded source, contract, artifact, hardware, and command.

## Core documents

1. [Target engine architecture](design/tilelang-engine-architecture.md)
   defines the accepted Mistral.rs, Oxide, and TileLang ownership model.
2. [Current operator architecture](design/oxide-infer-architecture.md)
   defines the source that is being migrated.
3. [Engine benchmark plan](development/engine-benchmark-plan.md) defines
   kernel, model, and serving comparisons.
4. [Repository layout](design/repository-layout.md) separates current and
   target source trees.
5. [Operator catalog](operator-catalog.md) lists current, experimental, and
   planned contracts.
6. [Roadmap](roadmap.md) orders work and defines admission and exit gates.
7. [FlashInfer parity](flashinfer-parity.md) tracks the pinned comparison
   surface without claiming full parity.
8. [Mistral.rs integration](integrations/mistralrs.md) records the first engine
   adapter boundary.
9. [Evidence index](results/README.md) lists immutable device and benchmark
   records.

The [rename provenance](design/rename-provenance.md) maps the former project
name to current identifiers. Historical records and links keep their original
names.

## Development documents

- [Environment](development/environment.md) pins the current transitional
  Rust, CUDA, cuda-oxide, and website toolchains.
- [Engine benchmark plan](development/engine-benchmark-plan.md) fixes the
  future FlashInfer, vLLM, and SGLang comparison protocol.
- [Current device validation](development/h20-validation.md) defines correctness,
  sanitizer, Graph, performance, and engine gates.
- [Dense GEMM shape census](development/gemm-shape-census.md) defines the
  untimed workload profile used to select native GEMM candidates.
- [Experimental SM90a M=1 GEMV](development/sm90-simt-gemv-m1.md) fixes the
  first native GEMM contract and its promotion gates.

## Crates

| Crate | Responsibility | Published |
| --- | --- | --- |
| `oxide-infer` | Backend-independent `Spec` types and CPU references | Yes |
| `oxide-infer-cuda` | CUDA plans, native and vendor providers, command runtime, and Graph execution | No |
| `oxide-infer-lab` | Hardware gates, benchmarks, fixtures, and evidence generation | No |

`oxide-infer` has no CUDA dependency. Product crates never depend on
`oxide-infer-lab`.

## State sources

| Source | Fact |
| --- | --- |
| Rust source and Cargo manifests | Current implementation |
| [Operator catalog](operator-catalog.md) | Admitted public and experimental surface |
| [Evidence index](results/README.md) | Qualified source and hardware pairs |
| [Integration documents](integrations/mistralrs.md) | External adapter source pairs |
| [Roadmap](roadmap.md) | Planned work and exit gates |
| Design documents | Target boundaries and names |

If these sources disagree, keep reviewed result records unchanged. Correct the
catalog or design projection, then create evidence for the new source.

## Documentation rules

- Use the lifecycle names `Spec`, `Provider`, `Algorithm`, `Plan`, `Operands`,
  `CommandScope`, and `Completion`.
- State whether a capability is current, experimental, or planned.
- Keep correctness, sanitizer, Graph, performance, engine, and serving claims
  separate.
- Name each admitted dtype, layout, shape class, algorithm, and hardware
  target.
- Preserve historical records. A rename does not transfer qualification to a
  new source commit.
- Do not create a target module until its first contract or provider exists.

The root [README](../README.md) gives the short project overview.
