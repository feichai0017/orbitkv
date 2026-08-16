# Host Validation — 2026-08-16

> Historical host-only baseline. Current H20 system evidence is recorded in
> `docs/h20-sglang-validation-20260817.md` and
> `docs/h20-owning-vmm-validation-20260817.md`.

## Scope

This record validates the independent Rust OrbitKV compiler, reference
lifecycle runtime, trace analyzer, and the source-level SGLang plugin contract.
It contains no GPU or end-to-end serving performance claim.

## Environment boundary

- OrbitKV repository: `/workspace/orbitkv`
- SGLang validator checkout: `/workspace/sglang`
- SGLang revision: `095ec6c997bfdd25d3864cb0ce77a6562a934b96`
- NVIDIA driver: unavailable on this host
- Complete SGLang Python dependencies: unavailable on this host

## Commands

```bash
cargo test --offline --all-targets
cargo fmt --all -- --check
cargo clippy --offline --all-targets -- -D warnings
cargo run --offline -- check-sglang /workspace/sglang
cargo run --offline -- compile examples/full_swa.json --boundary 32768
python3 -m compileall -q integrations/sglang/src
git diff --check
```

## Results

- Seven Rust unit tests passed after the Python prototype was retired.
- Exhaustive small-domain checks matched the compiled sliding-window slot
  formula for page sizes 1, 4, and 16 and windows 1 through 65.
- The reference runtime rejected modulo-slot reuse while the previous logical
  block generation remained GPU-pinned.
- The SGLang source contract check found the required plugin framework and
  `SWATokenToKVPoolAllocator` methods.
- The thin Python plugin was applied by SGLang's pinned real `HookRegistry`
  implementation to a controlled allocator contract. All five hook points
  preserved allocator return values, emitted ordered JSONL events, and the Rust
  CLI consumed the resulting trace.
- The 10-Full/52-SWA, W=1024, P=16, T=32768 illustrative plan compiled to:
  - 2,048 Full blocks;
  - 65 physical SWA slots per request;
  - 1,040 physical SWA token slots;
  - 1,563,688,960 resident bytes under the illustrative 4,096-byte
    per-layer-per-token geometry;
  - 8,321,499,136 bytes for the all-Full diagnostic baseline.

The compiled footprint is 18.79095% of that diagnostic baseline, a theoretical
81.20905% reduction. This is not a result against SGLang's existing hybrid SWA
manager. The strong comparison requires the Stage A and Stage C GPU experiments
defined in `docs/sglang-e2e.md`.

## Qualified conclusions

The current evidence supports:

- correct block-level lifting for Full and fixed-window SWA;
- the bounded-slot formula in the tested domain;
- `W-1` old-token continuation semantics at a pre-query boundary;
- completion-safe reuse behavior in the reference simulator;
- source compatibility of the shadow plugin with the pinned SGLang checkout.
- cross-language plugin-to-Rust trace-analysis behavior through SGLang's real
  plugin hook machinery.

The current evidence does not support:

- lower GPU memory than stock SGLang;
- higher throughput or concurrency;
- production replacement of SGLang's allocator;
- prefix-cache, P/D connector, or HiCache benefits;
- arbitrary attention-mask compilation.
