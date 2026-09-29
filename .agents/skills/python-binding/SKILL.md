---
name: python-binding
description: Modify OrbitKV PyO3 APIs, Python type stubs, vLLM/SGLang adapters, or native wheel packaging. Use for binding and engine-integration changes in this repository.
---

# OrbitKV Python bindings and adapters

Read `AGENTS.md` from the Git root. Keep Python responsible for framework callbacks,
layout inspection, GPU allocations and ownership handoff; shared cache state
machines, batching, waiting and execution belong to the existing Rust owners.

- Public bindings: `python/src/lib.rs`, `python/src/client.rs` and
  `python/orbitkv/orbitkv.pyi`. Update stubs when an exposed API changes. Verify
  actual call sites rather than restoring removed aliases or client facades.
- Common client ownership: `crates/orbitkv-channel/src/cache_client.rs`.
- vLLM: `python/orbitkv/vllm/{connector,scheduler,worker}.py`; use the pinned
  `third-party/vllm` contract, currently v0.29.0.
- SGLang: `python/orbitkv/sglang/linker.py` and its registered plugin. The direct
  GPU linker exists; inspect its admission and recovery hooks against the pinned
  SGLang v0.5.20 before changing callbacks.
- Model/layout identity and hybrid recovery are shared contracts. Do not equate
  compatible API shapes with cross-engine byte compatibility.

Use the gate table in `AGENTS.md`: default Python tests must remain source-only;
native lifecycle changes need process integration, and consumed adapter changes
need the relevant engine correctness/restart gate. Run `benches/tests` when
changing benchmark helpers. Put runtime outputs outside the checkout.

Use `scripts/build-wheel.sh` for a complete installed artifact; `maturin develop`
is a development build. See `docs/releases.md` for CUDA variants, bundled native
dependencies and installed-package checks. Freeze native artifacts before GPU
tests: Cargo can restage shared libraries used by live Managers.
