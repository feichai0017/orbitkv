<p align="center">
  <img src="website/public/readme-banner.svg" alt="OrbitKV — Compile the lifetime of state" width="100%" />
</p>

<p align="center">
  <a href="https://feichai0017.github.io/orbitkv/">Website</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/roadmap.md">Roadmap</a> ·
  <a href="TODO.md">TODO</a>
</p>

**A framework-neutral state cache and physical planner for vLLM and SGLang.**

OrbitKV combines a production-oriented Rust storage and transfer engine with a
new control plane that will compile attention semantics and workload evidence
into cache placement, retention, prefetch, movement, and routing decisions.
vLLM is the currently validated adapter; SGLang is the next first-class adapter.
The framework release targets as of 2026-09-20 are
[vLLM `0.29.0`](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)
and [SGLang `0.5.20`](https://github.com/sgl-project/sglang/releases/tag/v0.5.20);
the SGLang source submodule is pinned to that release. See
[`python/README.md`](python/README.md) for separate GPU environments.

## What exists now

- content-addressed KV blocks in pinned host memory, with optional SSD backing;
- NUMA-aware allocation and batched GPU/host transfer;
- cross-node discovery and Mooncake transfer over RDMA or TCP;
- prefix lookup, leases, eviction, metrics, and P/D transfer paths;
- a Rust server, Python bindings, and the imported vLLM connector.

The initial storage and control data plane was imported from PegaFlow `0.24.5`
and renamed throughout. The copied remote transfer stacks have since been
replaced by a pinned upstream Mooncake Transfer Engine. PegaFlow's published
measurements are not presented as OrbitKV results.

## Where OrbitKV goes further

The target is not another manually tuned external cache. OrbitKV will compile a
declarative state-liveness contract into physical cache plans:

- semantic death and execution completion are separate reclamation frontiers;
- HBM, DRAM, SSD, and remote replicas are placement choices under one cost model;
- ring buffers, prefix pages, checkpoints, and eviction classes are derived
  physical plans rather than hard-coded product features;
- SGLang remains the serving scheduler while OrbitKV becomes the authority for
  cache identity, placement, and safe reuse.

The first integration milestone is an SGLang HiCache backend over the imported
data plane. A deeper allocator boundary follows after the host-page path is
correct and measured.

## Workspace

| Path | Responsibility |
| --- | --- |
| [`orbitkv-contract`](crates/orbitkv-contract) | Framework-neutral state identity, format, page and recovery contracts |
| [`orbitkv-local`](crates/orbitkv-local) | Versioned iceoryx2 and UDS process IPC implementation |
| [`orbitkv-core`](crates/orbitkv-core) | Content-addressed blocks, leases, eviction, SSD and remote tiers |
| [`orbitkv-transfer`](crates/orbitkv-transfer) | Pinned upstream Mooncake Transfer Engine wrapper |
| [`orbitkv-server`](crates/orbitkv-server) | Cache Manager crate: shared cache operations, process endpoint, peer control, health and metrics |
| [`orbitkv-metaserver`](crates/orbitkv-metaserver) | Cross-node replica discovery |
| [`python/orbitkv/vllm`](python/orbitkv/vllm) | vLLM adapter |
| [`python/orbitkv/sglang`](python/orbitkv/sglang) | SGLang adapter contracts and upcoming HiCache backend |
| [`python/orbitkv/client`](python/orbitkv/client) | Framework-neutral cache API and transport selection |
| [`third-party/sglang`](third-party/sglang) | Pinned SGLang source used to develop and validate integration |
| [`website`](website) | OrbitKV project website and brand assets |

The process IPC, network control, Mooncake integration boundary, and measured
IPC baselines are documented in
[`docs/transport.md`](docs/transport.md).

## Build

The default build targets CUDA 12.8. Host-only inspection can disable default
features; GPU and RDMA qualification requires matching local hardware and drivers. The
workspace MSRV is Rust 1.89, required by iceoryx2 0.10.

```sh
cargo check --workspace

cd python
maturin develop --release
```

Initialize Mooncake before the first native build:

```sh
git submodule update --init --recursive third-party/mooncake
```

To run the vLLM adapter while SGLang support is under construction:

```sh
orbitkv-cache-manager
vllm serve Qwen/Qwen3-0.6B \
  --kv-transfer-config '{"kv_connector":"OrbitKVConnector","kv_role":"kv_both","kv_connector_module_path":"orbitkv.vllm"}'
```

For a same-host Cache Manager, the connector uses UDS bootstrap +
iceoryx2 for Query/Publish/Restore/Release. If the derived same-host socket is
missing, startup fails with a clear error instead of switching to gRPC.
Registration, health, sessions, and cleanup use the same authenticated
Unix socket. Standalone mode does not start a gRPC listener. Distributed mode
(`--metaserver-addr`) enables a peer-only gRPC control endpoint; remote KV
bytes still use Mooncake. Every inference process connects to a Cache Manager
on its own host.

OrbitKV's current workspace is Apache-2.0 licensed. Earlier experiments remain
available in repository history but are not part of the current build.
