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

## What exists now

- content-addressed KV blocks in pinned host memory, with optional SSD backing;
- NUMA-aware allocation and batched GPU/host transfer;
- cross-node discovery and RDMA fetch;
- prefix lookup, leases, eviction, metrics, and P/D transfer paths;
- a Rust server, Python bindings, and the imported vLLM connector.

The storage and transfer data plane was imported from PegaFlow `0.24.5` and
renamed throughout. PegaFlow's published measurements are not presented as
OrbitKV results.

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
| [`orbitkv-core`](crates/orbitkv-core) | Content-addressed blocks, leases, eviction, SSD and RDMA tiers |
| [`orbitkv-transfer`](crates/orbitkv-transfer) | CUDA-aware and RDMA transfer engines |
| [`orbitkv-server`](crates/orbitkv-server) | Local sidecar, health/metrics endpoints, and P/D router |
| [`orbitkv-metaserver`](crates/orbitkv-metaserver) | Cross-node replica discovery |
| [`python/orbitkv/vllm`](python/orbitkv/vllm) | vLLM adapter |
| [`python/orbitkv/sglang`](python/orbitkv/sglang) | SGLang adapter contracts and upcoming HiCache backend |
| [`python/orbitkv/client`](python/orbitkv/client) | Framework-neutral local client exports |
| [`third-party/sglang`](third-party/sglang) | Pinned SGLang source used to develop and validate integration |
| [`website`](website) | OrbitKV project website and brand assets |

## Build

The default build targets CUDA 12.8. Host-only inspection can disable default
features; GPU and RDMA tests require matching local hardware and drivers.

```sh
cargo check --workspace

cd python
maturin develop --release
```

To run the vLLM adapter while SGLang support is under construction:

```sh
orbitkv-server
vllm serve Qwen/Qwen3-0.6B \
  --kv-transfer-config '{"kv_connector":"OrbitKVConnector","kv_role":"kv_both","kv_connector_module_path":"orbitkv.vllm"}'
```

OrbitKV's current workspace is Apache-2.0 licensed. Earlier experiments remain
available in repository history but are not part of the current build.
