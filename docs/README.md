# Loom Infer documentation

Loom Infer is a Rust-native GPU operator library for LLM inference.

- [Architecture](design/loom-infer-architecture.md) defines ownership and the
  operator lifecycle.
- [Operator catalog](operator-catalog.md) lists current and planned work.
- [Roadmap](roadmap.md) gives the implementation order and exit gates.
- [H20 validation](development/h20-validation.md) defines the device test
  process.
- [Evidence](results/README.md) indexes results from permanent providers.

The root [README](../README.md) gives the short project overview. Current
correctness evidence covers Rust RMSNorm kernels and one fixed BF16 cuBLASLt
GEMM plan on H20.

## Documentation rules

Document implemented behavior in the present tense. Mark planned work as
planned. Keep correctness, kernel latency, graph behavior, engine integration,
and serving results separate.

Every performance claim must name the source revision, hardware, operator
contract, baseline, measurement method, and result file.
