# OrbitKV Core

`core/` is the native half of the single OrbitKV Engine product. It contains
the attention-state compiler, runtime target and binding contracts, and the
Rust `RuntimeSession` that is the sole KV-lifecycle authority for an admitted
SGLang profile. It is not a separate serving product.

`RuntimeSession` owns request and snapshot identities, physical-page selection
and generations, Prefix sharing where admitted, copy-on-write, semantic and
execution frontiers, retirement, acknowledgement, and safe reuse. SGLang still
owns scheduling, model execution, tensors, attention and copy kernels, and CUDA
stream/event execution.

```text
core/
├── src/       compiler, RuntimeSession, runtime target, checkpoint pool, CLI
├── ffi/       Rust library, typed C header, and C/C++ smoke tests
├── tests/     Rust CLI integration tests
├── examples/  attention-state and manager-plan examples
├── fixtures/  fixed-state compiler fixtures
└── Cargo.toml
```

The live typed boundary is `WIRE_VERSION = 14`. The C header and Rust library
expose exactly 48 typed symbols; the SGLang adapter freezes exactly 78 ctypes
layouts. The packaged SGLang `RuntimeTarget` remains contract version 4 and
requires wire 14. Version, symbol, layout, target, and binding mismatches fail
before an operational call. The public admission API and compiler CLI always
use this packaged target; callers cannot provide an alternate target contract.

Build and test the native workspace from the repository root:

```bash
cargo build --release --workspace
cargo test --workspace
python tools/verify_capability_matrix.py
```

See the [engine product](../compat/README.md) for assembly and the
`orbitkv-engine compile`/`serve` workflow, and the
[Capability Matrix](../docs/capability-matrix.md) for the exact implementation
and qualification boundary.
