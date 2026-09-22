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
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-203b30" alt="Apache 2.0 license" /></a>
</p>

**Reuse KV across requests with vLLM and SGLang.** OrbitKV keeps reusable
prefixes in pinned DRAM and optional SSD, and restores them into engine-owned
GPU memory. Run one Cache Manager per host; enable an engine adapter to use it.
An experimental distributed path extends the same cache API to peer managers.

## Why OrbitKV?

- **Extend cache capacity.** Keep prefixes beyond the engine's HBM cache using
  NUMA-aware host memory and SSD backing.
- **Use either engine.** vLLM and SGLang register their GPU buffers through
  CUDA IPC and share the same UDS + iceoryx2 manager API.
- **Keep ownership explicit.** The engine controls HBM allocation and execution.
  OrbitKV retains external replicas and transfer leases through completion.
- **Bound preparation.** Byte budgets cover pending reads, ready leases and GPU
  consumers; identical backing reads can share preparation. Pressure reclaim
  works in byte-bounded batches and rechecks actual contiguous capacity.
- **Prepare queued demand (experimental).** Both engine adapters can warm missing prefixes
  within a separate budget share, then revalidate them at admission. Warming
  yields to foreground ownership and tracks restored, unused and pending pages.
  It remains opt-in: [pressure controls](docs/queued-warming.md#page-use-and-reclamation-controls)
  still show no established throughput gain.
  The [next policy steps](docs/queued-warming.md#reference-implementations-and-policy-order)
  draw on reviewed LMCache, HiCache, FlexKV and Dynamo implementations.
- **Identify compatible state.** Versioned keys bind immutable model artifacts,
  computation settings and registered storage geometry. SGLang compiles prefix,
  sliding-window and recurrent-checkpoint requirements, then validates complete
  recovery boundaries before loading. See [hybrid recovery](docs/hybrid-recovery.md).
- **Build toward shared caching.** Embedded catalog shards discover peer
  replicas, Mooncake Transfer Engine moves bytes, and etcd tracks membership
  and placement. Multi-node serving is still experimental.

The [delivery plan](docs/roadmap.md#current-delivery-priorities) prioritizes
single-node failure recovery, bounded preparation experiments, then real
two-host DP and P/D qualification. Catalog HA gates production distributed use.

## Get started

The validated release targets are **vLLM 0.29.0** and **SGLang 0.5.20**. Use a
separate environment for each engine. Follow the [installation guide](docs/single-node.md)
to build and install an OrbitKV wheel matching your Python ABI and CUDA runtime;
the Cache Manager currently also needs compatible PyTorch packages.

Start a manager in one terminal:

```bash
orbitkv-cache-manager \
  --addr 127.0.0.1:50055 \
  --http-addr 127.0.0.1:9091 \
  --pool-size 8gb
```

Then start your chosen engine in another terminal on the same host.

**vLLM**

```bash
vllm serve /path/to/immutable-model \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"OrbitKVConnector","kv_role":"kv_both","kv_connector_module_path":"orbitkv.vllm"}'
```

**SGLang**

```bash
ORBITKV_SGLANG_ENDPOINT=unix:///tmp/orbitkv-50055.sock \
  sglang serve --model-path /path/to/immutable-model --page-size 64 \
  --enable-unified-cache-external-linker --radix-cache-backend orbitkv
```

To add SSD capacity, pass
`--ssd-cache-path /data/orbitkv/cache.bin --ssd-cache-capacity 100gb` to the
manager. SSD contents are recreated on manager startup. Standalone deployment
requires neither etcd nor a gRPC listener.

Local model artifacts are fingerprinted at startup. Hub models require an
immutable revision or a verified `ORBITKV_MODEL_FINGERPRINT`. Keep the manager
alive, restart the engine, and repeat a multi-block prompt to distinguish an
external restore from a native HBM hit. See the
[full setup and verification steps](docs/single-node.md).

## Architecture

![OrbitKV architecture: engine-owned HBM, per-host Cache Managers, DRAM and SSD, embedded catalogs, Mooncake transfers and etcd membership](website/public/architecture.svg)

1. The engine adapter identifies a reusable prefix and registers GPU buffers.
2. The manager prepares matching DRAM, SSD or remote blocks within byte budgets.
3. Leases retain sources and destinations until transfers finish.
4. The engine resumes computation and publishes completed KV for later reuse.

The engine always connects to its host's manager. Remote discovery and transfer
stay inside OrbitKV. Candidate indexes reduce repeated catalog lookups; the
source manager checks live residency before authorizing a transfer. etcd is
outside per-block lookup and transfer paths. Catalog shards currently have one
copy each; replication and online handoff are planned.

Read the [architecture and crate boundaries](docs/architecture.md),
[transport contracts](docs/transport.md), and
[distributed design](docs/distributed-cache.md).

## Deployment support

| Scenario | Current scope |
| --- | --- |
| Single-node DRAM and SSD cache | GPU recovery gates and Qwen3-8B measurements for both pinned engines |
| SGLang model layouts | Full-attention MHA/MLA, Full + SWA, and Full + recurrent/conv; explicit layout checks and TP=1 recovery gates |
| Independent replicas / DP cache sharing | Embedded catalogs, etcd membership and Mooncake fetch implemented; cross-host serving qualification is next |
| Prefill/decode separation | Experimental vLLM Mooncake `PdConnector`; current-request handoff is separate from reusable cache |
| Cross-host TP/PP, layout conversion | Not qualified; the current vLLM scheduler cannot query remote TP shards through its local endpoint |
| Catalog HA and KV-aware routing | Planned after distributed-cache recovery gates |

The [deployment guide](docs/deployment.md) covers each topology and explains
the role of upstream NIXL. The [LMCache/Mooncake comparison](docs/distributed-comparison.md)
separates shared caching, engine parallelism and P/D handoff. A common cache API
does not make vLLM and SGLang KV bytes interchangeable.

## Measurements

Results include environment, workload, transfer evidence and limitations:

| Report | What it measures |
| --- | --- |
| [Single-node comparisons](docs/single-node-performance.md) | Native HBM, built-in CPU caches, OrbitKV and LMCache; FlexKV compatibility attempts |
| [SSD recovery](docs/ssd-performance.md) | Forced DRAM eviction, SSD restoration and engine readiness |
| [Concurrent bursts](docs/concurrent-performance.md) | Shared and mixed prefixes at concurrency 1/4/8 with byte budgets |
| [Sustained serving](docs/sustained-performance.md) | Bounded mixed reuse/cold traffic, throughput, tail latency and post-run drain |

Workloads, harness tests and recorded results live in [`benches/`](benches/README.md).
These are scoped measurements, not a claim that every workload is faster.

## Documentation and development

- [Single-node setup](docs/single-node.md) · [Manager options](docs/server.md) · [Metrics](docs/metrics.md)
- [Multi-node setup](docs/p2p.md) · [P/D integration](docs/pd.md)
- [Model and state identity](docs/state-identity.md) · [Transfer planning](docs/state-planning.md)
- [Python packages and tests](python/README.md) · [Rust checks](docs/rust-quality.md)
- [Roadmap](docs/roadmap.md) · [Work queue](TODO.md) · [Development guide](AGENTS.md)

The implementation order is single-node stability and performance, independent
replica sharing, then P/D with cache reuse. Replicated catalogs, broader
parallelism and routing have separate gates. Before 1.0, interfaces may change.
Technical documentation lives in `docs/` and is rendered directly on the website;
behavior and deployment changes must update the affected documentation.

OrbitKV began from PegaFlow 0.24.5. The remote transfer implementation now uses
pinned upstream Mooncake Transfer Engine. Upstream measurements are not
presented as OrbitKV results. The workspace is licensed under [Apache-2.0](LICENSE).
