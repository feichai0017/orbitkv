# OrbitKV

[![Python package v0.1.0 — not yet published](https://img.shields.io/badge/Python_package-v0.1.0%20%28unreleased%29-203b30)](https://feichai0017.github.io/orbitkv/docs/releases/)

**KV cache for vLLM and SGLang, backed by Rust.** Reuse computed prefixes from
pinned DRAM and optional SSD after GPU eviction or an engine restart. Both
adapters use CUDA IPC for GPU transfers and the same Cache Manager API.

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
- Native query ownership, byte budgets and completion fences.
- Prometheus metrics and optional request timelines.
- Experimental peer cache sharing through Mooncake Transfer Engine.

Validated engine releases: **vLLM 0.29.0** and **SGLang 0.5.20**. See
[model qualification](https://feichai0017.github.io/orbitkv/docs/models/)
and [deployment support](https://feichai0017.github.io/orbitkv/docs/deployment/).
Interfaces may change before 1.0.

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
