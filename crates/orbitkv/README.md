# orbitkv crate

`crates/orbitkv/` contains the attention-state compiler and the sole KV lifecycle
authority. It has no C ABI, Python package, device runtime, or engine-specific
target contract.

```text
crates/orbitkv/
├── src/       compiler, manager, RuntimeSession, checkpoint pool, CLI
├── tests/     CLI integration tests
├── examples/  generic attention-state and manager-plan inputs
└── fixtures/  generic compiler fixtures
```

`RuntimeManifest` is the canonical compiler artifact. `RuntimeSession` owns
request and snapshot identities, physical-page selection and generations,
Prefix/COW, semantic and execution frontiers, retirement, acknowledgement, and
safe reuse. Device execution is supplied in-process by
`orbitkv-executor`.

```bash
cargo test --locked -p orbitkv --all-targets
```
