# OrbitKV

[![Python package v0.1.0 — not yet published](https://img.shields.io/badge/Python_package-v0.1.0%20%28unreleased%29-203b30)](https://feichai0017.github.io/orbitkv/docs/releases/)

**KV cache for vLLM and SGLang, backed by Rust.** Reuse computed prefixes from
pinned DRAM and optional SSD after GPU eviction or an engine restart. Both
adapters use the same native Cache Manager API. Unencoded DRAM Restore runs in
the engine process using shared payload arenas; SSD/codec Restore and Publish
retain Manager workers and CUDA IPC tensor bindings.

[Documentation](https://feichai0017.github.io/orbitkv/docs/) ·
[Quickstart](https://feichai0017.github.io/orbitkv/docs/single-node/) ·
[Architecture](https://feichai0017.github.io/orbitkv/architecture/) ·
[Source](https://github.com/feichai0017/orbitkv)

## Features

- DRAM/SSD prefix reuse with engine-owned GPU allocation.
- Automatic Rust SSD backend selection for both engines, with native cuFile
  writes/restores where available and io_uring otherwise; see the
  [GPU storage configuration and qualification gates](../docs/gds.md).
- Compiled recovery ranges for supported attention, window and checkpoint layouts.
- Optional GPU ANS lossless compression and experimental FP8/3-bit/4-bit
  TurboQuant storage in DRAM, SSD and peer transfers; see [storage formats](../docs/storage-formats.md).
- Native query ownership, source grants, byte budgets and completion fences.
- Engine-local raw Restore with actual tensor ownership and an explicit engine
  readiness stream; both adapters use the whole-operation completion gate.
- Prometheus metrics and optional request timelines.
- Experimental peer cache sharing through Mooncake TENT.
- Experimental vLLM and SGLang P/D payload transfer through the same Rust TENT
  runtime; SGLang retains its native handoff control plane.

Pinned engine releases: **vLLM 0.29.0** and **SGLang 0.5.20**. The new
[engine-local Restore path](../docs/engine-local-restore.md) passes single-H20
Qwen3-8B DRAM correctness and engine-restart reuse in both pinned engines.
The [recorded vLLM end-to-end results](../docs/single-node-performance.md#matched-vllm-end-to-end-comparison)
compare native HBM, native CPU offload, OrbitKV and LMCache MP. Multiple-GPU,
huge-page and broader serving qualification remain separate. See
[the measured Restore improvements and remaining overhead](../docs/communication-performance.md),
[model qualification](https://feichai0017.github.io/orbitkv/docs/models/)
and [deployment support](https://feichai0017.github.io/orbitkv/docs/deployment/).
Interfaces may change before 1.0.

## Native registration and Restore contract

`register_context_batch(..., tensors=[...])` takes the real tensor/exporter
objects as well as their IPC metadata. Keep tensor order aligned with layer
registration. `start_restore(..., ready_stream=...)` takes the CUDA stream of
the engine's previous destination-page users; the native worker establishes
readiness before copying. Both bundled adapters provide these arguments.

The native worker owns accepted operations even if a Python handle is dropped
or `wait_restore` times out. Connectors must retain logical destination page IDs
until a terminal result. Local completion means DMA drained and does not wait
for the Manager's source-retirement ACK. Repeated registration of the same
binding is rejected; unregister and close drain accepted operations first.

Build the native client and Manager together: this cutover uses bootstrap 6,
channel ABI 9, and lifecycle 4, with no old-wire compatibility path. An encoded
raw Restore plan above 1 MiB after allocation-aware compaction is rejected
before consuming its leases. Idle destination streams need no additional GPU
event; busy streams are fenced with a reusable event. Automatic partitioning,
layer overlap, and graph replay dependencies remain future work.

## Installation

| CUDA runtime | Distribution | Python import |
| --- | --- | --- |
| CUDA 12 | `orbitkv-llm` | `orbitkv` |
| CUDA 13 | `orbitkv-llm-cu13` | `orbitkv` |

Install only one distribution and one engine extra (`vllm` or `sglang`) per
environment. The extras pin the validated engine releases. The wheel includes
the native extension, Cache Manager and Mooncake runtime; the Manager also
needs compatible PyTorch. The base package does not install PyTorch.

Version 0.1.0 is being prepared for release. Build a complete wheel from the
repository, then install the produced file:

```bash
git submodule update --init --recursive third-party/mooncake
./scripts/build-wheel.sh --release --no-default-features --features cuda-13,mooncake
python -m pip install /absolute/path/to/the-built-wheel.whl
```

Use `./scripts/build-wheel.sh --release` for CUDA 12. See the
[installation guide](https://feichai0017.github.io/orbitkv/docs/single-node/)
for complete engine environments and the
[release guide](https://feichai0017.github.io/orbitkv/docs/releases/) for artifact checks.

## Quickstart

Start an independent Manager in a compatible PyTorch/CUDA environment:

```bash
orbitkv-cache-manager --addr 127.0.0.1:50055 --pool-size 8gb
```

To add SSD capacity, append
`--ssd-cache-path /data/orbitkv/cache.bin --ssd-cache-capacity 100gb`.
The Manager automatically tries native cuFile and falls back to io_uring;
normal deployments do not need `--ssd-backend` or a separate cuFile service.
Engines on the same host can connect to this Manager and share its external
capacity; see [deployment requirements](../docs/deployment.md) for runtime
compatibility, shared resources and current qualification limits.

Then enable the adapter on the same host:

```bash
# vLLM
vllm serve /path/to/immutable-model --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"OrbitKVConnector","kv_role":"kv_both","kv_connector_module_path":"orbitkv.vllm"}'

# SGLang, in its own environment
ORBITKV_SGLANG_ENDPOINT=unix:///tmp/orbitkv-50055.sock \
  sglang serve --model-path /path/to/immutable-model --page-size 64 \
  --enable-unified-cache-external-linker --radix-cache-backend orbitkv
```

Keep the Manager alive, restart the engine and repeat a multi-block prompt to
verify an external restore. SSD contents are recreated when the Manager restarts.
The engine remains responsible for HBM and scheduling. Automatic request
preparation is experimental and disabled by default.

## Development

[Adapter reference](https://feichai0017.github.io/orbitkv/docs/adapters/) ·
[Test gates](https://github.com/feichai0017/orbitkv/blob/main/python/tests/README.md) ·
[Benchmarks](https://github.com/feichai0017/orbitkv/tree/main/benches) ·
[Roadmap](https://feichai0017.github.io/orbitkv/docs/roadmap/)

The runtime package contains only the native client and engine adapters. Tests
live in `python/tests/`; workloads and results live in `benches/`.

## License

[Apache-2.0](https://github.com/feichai0017/orbitkv/blob/main/LICENSE).
