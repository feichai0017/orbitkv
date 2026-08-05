# Loom Infer documentation

Loom Infer is a Rust-native GPU operator library for LLM inference.

- [Architecture](design/loom-infer-architecture.md) defines ownership and the
  operator lifecycle.
- [Repository layout](design/repository-layout.md) maps source directories to
  architectural responsibilities and defines when they should split.
- [Operator catalog](operator-catalog.md) lists current and planned work.
- [FlashInfer parity](flashinfer-parity.md) pins the comparison surface and
  records every upstream operator domain.
- [Roadmap](roadmap.md) gives the implementation order and exit gates.
- [Development environment](development/environment.md) pins host, website,
  and CUDA toolchains and defines the common validation commands.
- [H20 validation](development/h20-validation.md) defines the device test
  process.
- [Evidence](results/README.md) indexes results from permanent providers.

The root [README](../README.md) gives the short project overview. Current
correctness evidence covers Rust RMSNorm, single-decode attention, and paged
batch-decode kernels, plus one fixed BF16 cuBLASLt GEMM plan on H20.

## Documentation rules

Document implemented behavior in the present tense. Mark planned work as
planned. Keep correctness, kernel latency, graph behavior, engine integration,
and serving results separate.

Every performance claim must name the source revision, hardware, operator
contract, baseline, measurement method, and result file.
