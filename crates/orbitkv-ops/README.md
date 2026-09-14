# OrbitKV operations

Backend-independent inference operation contracts and tensor-graph builders.
Attention separates mathematical semantics from its KV view. Block-scaled
linear describes quantization geometry without selecting a CUDA provider.

This crate depends on `orbitkv-compiler` and has no CUDA dependency. Backend
rules add applicable implementations to the compiler's search space.

See [attention contracts](../../docs/attention-providers.md) and
[source origin and licenses](../../docs/compiler-maintenance.md).
