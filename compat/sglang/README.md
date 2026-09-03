# SGLang engine integration

This directory turns the complete pinned SGLang source tree into OrbitKV
Engine. SGLang remains responsible for serving, scheduling, model execution,
attention kernels, tensors, communication, and CUDA execution. OrbitKV replaces
only the admitted KV-manager lifecycle.

```text
compat/sglang/
├── source/   complete pinned SGLang checkout; materialized locally, not vendored
├── overlay/  reviewed source changes at the KV lifecycle boundary
├── bridge/   `orbitkv_sglang` Python package
├── tools/    source preparation and qualification entry points
└── tests/    bridge, lifecycle, source-contract, and E2E tests
```

`source/` is intentionally ignored by the OrbitKV repository because it is an
independent upstream Git checkout. Its exact revision and every changed path are
validated before use. Run `tools/prepare_source.py check-base` before applying
the overlay, or `tools/prepare_source.py verify` afterwards; both default to this
local `source/` directory. The checked-out development tree normally contains
the reviewed overlay, so `verify` is the routine integrity check. The
distributable product is created by
`../../assemble.py` from this complete tree plus the reviewed overlay and
OrbitKV Core.
