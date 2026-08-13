# Oxide Infer documentation

The current source is a Rust-native CUDA operator layer. The accepted target
is a standalone Rust inference engine with an Oxide-owned server, scheduler,
model and KV data plane, and offline TileLang artifacts as its only product
custom compute kernels. External engines remain provenance sources and
performance baselines, not product components. The target does not describe a
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

1. [Standalone engine architecture](design/standalone-oxide-engine.md)
   defines the accepted server, engine, runtime, TileLang, and provenance
   boundaries.
2. [Control-plane source map](design/control-plane-source-map.md) defines what
   is imported, split, referenced, or excluded from the upstream baseline.
3. [Current operator architecture](design/oxide-infer-architecture.md)
   defines the source that is being migrated.
4. [Engine benchmark plan](development/engine-benchmark-plan.md) defines
   kernel, model, and serving comparisons.
5. [Repository layout](design/repository-layout.md) separates current and
   target source trees.
6. [Operator catalog](operator-catalog.md) lists current, experimental, and
   planned contracts.
7. [Roadmap](roadmap.md) orders work and defines admission and exit gates.
8. [FlashInfer parity](flashinfer-parity.md) tracks the pinned comparison
   surface without claiming full parity.
9. [Historical reference integration](integrations/mistralrs.md) records the
   former paired-repository proof and benchmark provenance.
10. [Evidence index](results/README.md) lists immutable device and benchmark
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
| [Integration documents](integrations/mistralrs.md) | Historical and external reference source pairs |
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
