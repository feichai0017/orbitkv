<p align="center">
  <img src="website/public/readme-banner.svg" alt="OrbitKV — Compile the lifetime of state" width="100%" />
</p>

<p align="center">
  <a href="https://feichai0017.github.io/orbitkv/">Website</a> ·
  <a href="https://feichai0017.github.io/orbitkv/docs/">Documentation</a> ·
  <a href="docs/single-node.md">Quickstart</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/roadmap.md">Roadmap</a>
</p>

<p align="center">
  <a href="https://github.com/feichai0017/orbitkv/actions/workflows/ci.yml"><img src="https://github.com/feichai0017/orbitkv/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="docs/releases.md"><img src="https://img.shields.io/badge/Python_package-v0.1.0%20%28unreleased%29-203b30" alt="Python package v0.1.0 — not yet published" /></a>
  <a href="docs/releases.md#packages"><img src="https://img.shields.io/badge/Python-3.10%E2%80%933.14-203b30" alt="Wheel targets: Python 3.10–3.14" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-203b30" alt="Apache 2.0 license" /></a>
</p>

## About

**OrbitKV extends the KV cache of vLLM and SGLang beyond GPU memory.** Keep
reusable prefixes in DRAM and SSD, then restore them when a matching request
arrives. This helps workloads with repeated documents, shared system prompts,
and conversations whose prefixes no longer fit in the engine's GPU cache.

Run an independent Cache Manager per host and connect the engines on that host
to its shared cache. Engines own GPU memory and scheduling; OrbitKV manages
external replicas and transfers. See [deployment patterns](docs/deployment.md)
for shared-instance budgets and container qualification limits.
The single-node path is GPU-tested on **vLLM 0.29.0** and **SGLang 0.5.20**.
Multi-node cache sharing is experimental. Interfaces may change before 1.0.

## Key features

- **DRAM and SSD caching.** Reuse prefixes after GPU eviction or an engine
  restart while the Cache Manager remains alive.
- **Optional reuse policies.** Rust can protect reused pages within a byte cap
  and admit SSD writes selectively. See the [policy controls](docs/cache-policies.md)
  and their cold-reuse tradeoff before enabling them.
- **Direct GPU transfers.** Both engines register GPU buffers through CUDA IPC;
  adapters fence the producing CUDA stream, and Rust handles cache queries,
  reads and transfer completion.
- **Model-aware recovery.** Cache identity includes model artifacts, computation
  settings and storage layout. Compiled recovery rules select the required
  attention pages, sliding windows and recurrent/conv checkpoints, including
  supported layouts that combine all three.
- **Bounded resource use.** Byte budgets cover pending reads, ready pages and
  active GPU transfers. Cancellation retains submitted I/O until completion.
- **Observable behavior.** Inspect Prometheus metrics and optional request
  timelines, and reproduce the published latency and throughput measurements.
- **Experimental shared cache.** Embedded catalog shards locate peer replicas,
  Mooncake Transfer Engine moves bytes, and etcd tracks cluster membership.
  Source allocations remain budgeted through timeout; bounded completion records
  reconcile lost authorization replies and retry completion acknowledgements
  using reusable windows and generation-fenced tickets.

See [supported deployments](docs/deployment.md) and
[model qualification](docs/models.md) before selecting a checkpoint and topology.
Compiled recovery uses engine-declared state requirements; arbitrary model-graph
analysis and future-token prediction are outside the current implementation.

## Quickstart

Follow the [installation guide](docs/single-node.md) to build and install a
wheel for your Python and CUDA runtime. Use separate environments for vLLM and
SGLang. The wheel includes the Cache Manager and Mooncake libraries; the Manager
also requires compatible PyTorch. The first Python release is being prepared;
see [release preparation](docs/releases.md) for package names and artifact checks.

Start a Cache Manager:

```bash
orbitkv-cache-manager --addr 127.0.0.1:50055 --http-addr 127.0.0.1:9091 --pool-size 8gb
```

In another terminal on the same host, start **vLLM**:

```bash
vllm serve /path/to/immutable-model \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"OrbitKVConnector","kv_role":"kv_both","kv_connector_module_path":"orbitkv.vllm"}'
```

Or start **SGLang**:

```bash
ORBITKV_SGLANG_ENDPOINT=unix:///tmp/orbitkv-50055.sock \
  sglang serve --model-path /path/to/immutable-model --page-size 64 \
  --enable-unified-cache-external-linker --radix-cache-backend orbitkv
```

To enable SSD caching, add
`--ssd-cache-path /data/orbitkv/cache.bin --ssd-cache-capacity 100gb` to the
Manager command. The SSD cache is recreated when the Manager restarts.
No backend flag or engine-side storage setting is needed. The default
[automatic SSD backend](docs/gds.md) tries native cuFile on supported
mounts and uses io_uring when unavailable. cuFile is an optional library loaded
inside the Manager, not a separate service. It writes complete GPU state
groups and restores SSD demand hits through bounded GPU staging. Hardware
selection and native GDS performance qualification are separate. When cuFile
is selected, the Manager reserves the configured disk capacity before serving
and coalesces adjacent reads across cached blocks within each file. Rust submits
asynchronous I/O through two 4 MiB slots, keeping demand reads progressing
alongside bounded GPU writeback and event-tracked host copies.

[GPU storage encoding](docs/storage-formats.md) supports nvCOMP ANS lossless
compression, FP8 and 3/4-bit TurboQuant with bounded batches and reusable GPU
workspace. Encoded pages stay compact in DRAM, SSD and peer transfers; cuFile can
write complete encoded groups and restore encoded SSD hits through GPU validation
and decode. GPU restore reconstructs the engine layout; FP8 CPU fallback selects
AVX-512F, AVX2 or scalar code at runtime. Set
`--storage-codec ans|fp8|turboquant-4|turboquant-3` on the Manager. The default
is exact storage; lossy modes require model-quality qualification.

Keep the Manager alive, restart the engine, and repeat a multi-block prompt.
An increase in `orbitkv_load_bytes_total` confirms an external restore.
See the [complete quickstart](docs/single-node.md) for identity, metrics and
container setup. Standalone caching requires neither etcd nor a gRPC listener.

## Architecture

![OrbitKV architecture: engine-owned GPU memory, compiled page demand, and cache tiers](website/public/architecture.svg)

The engine adapter identifies missing state and supplies GPU destinations.
OrbitKV selects compatible cached ranges, reads them from the configured tiers,
and retains page ownership until the GPU copy finishes. Newly computed KV is
published for later reuse. The same adapter API serves DRAM, SSD and experimental
remote fetches; physical placement stays inside the Cache Manager.

Read the [architecture](docs/architecture.md),
[hybrid recovery contract](docs/hybrid-recovery.md), and
[distributed design](docs/distributed-cache.md). Cross-engine byte conversion,
production catalog HA and KV-aware request routing remain planned work.

## Performance

Performance depends on prefix reuse, cache capacity, storage and engine scheduling.
Reports include configurations, final results and reproduction commands:

| Report | Coverage |
| --- | --- |
| [Single-node comparisons](docs/single-node-performance.md) | Native HBM, engine CPU caches, OrbitKV, LMCache and FlexKV compatibility |
| [SSD recovery](docs/ssd-performance.md) | Restore readiness and sustained read/write pressure beyond DRAM capacity |
| [Ordinary recovery](docs/recovery-performance.md) | Qwen3-8B host reads, GPU transfers, notification delays and resource drain |
| [Request preparation](docs/request-preparation.md) | Repeated preparation controls, DRAM recovery and read stopping policies |
| [Shared-cache qualification](docs/shared-cache-qualification.md) | Independent replicas, remote GPU restoration, catalog replay and restart gates |

Request preparation remains **off by default**: the current Qwen3-8B controls
improve throughput in both engines, but SGLang P95 latency regresses. These
single-H20 measurements do not establish a universal advantage over other caches.
Benchmark code and final summaries live in [`benches/`](benches/README.md).

## Documentation and contributing

- **Get started:** [Installation](docs/single-node.md) · [Adapter configuration](docs/adapters.md) · [Manager options](docs/server.md)
- **Operate:** [Metrics](docs/metrics.md) · [Fault qualification](docs/fault-qualification.md) · [Deployment patterns](docs/deployment.md)
- **Develop:** [Contributor guide](AGENTS.md) · [Python package](python/README.md) · [Test gates](python/tests/README.md) · [Releases](docs/releases.md)
- **Plan:** [Roadmap](docs/roadmap.md) · [Work queue](TODO.md)

Technical pages in `docs/` are also published on the website. Contributions
should include the relevant checks and documentation changes.

## License

OrbitKV is licensed under [Apache-2.0](LICENSE).
