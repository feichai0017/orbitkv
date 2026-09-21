# P2P KV Cache Sharing

Experimentally share KV cache across OrbitKV nodes through Mooncake Transfer Engine. When node
B needs blocks that node A already has, OrbitKV authorizes the block ranges and
Mooncake reads them over RDMA (or TCP fallback).

This path is not yet a high-availability deployment. The current MetaServer is
an independent in-memory directory: it lacks complete resident-inventory
replay after restart and is a remote-hit-rate and availability risk. The
Cache Manager verifies and pins source bytes before transfer. The target
recoverable catalog is described in [architecture](architecture.md).

**When to use**: experimental multiple-node reuse for matching model,
tokenizer, format, rank topology, and namespace.

## How It Works

### Step 1: Save & Register

Node A saves KV blocks to pinned memory. Block hashes are registered with the MetaServer in the background.

```mermaid
sequenceDiagram
    participant vLLM as vLLM (Node A)
    participant A as OrbitKV (Node A)
    participant M as MetaServer

    vLLM->>A: save KV blocks
    A->>A: GPU → pinned memory
    A-->>M: register block hashes (async)
```

### Step 2: Discover & Fetch

Node B needs the same blocks. It queries the MetaServer, discovers Node A has
them, and reads them through Mooncake's selected transport.

```mermaid
sequenceDiagram
    participant B as OrbitKV (Node B)
    participant M as MetaServer
    participant A as OrbitKV (Node A)

    B->>M: who has these blocks?
    M-->>B: Node A
    B->>A: gRPC authorize + pin blocks
    A-->>B: Mooncake endpoint + ranges + lease
    B->>A: Mooncake READ
    B->>A: gRPC release lease
    B->>B: pinned memory → GPU
```

## Quick Start

### 1. Start MetaServer

One per test cluster in the current implementation. It is in-memory and not HA.

```bash
orbitkv-metaserver --addr 0.0.0.0:50056
```

### 2. Start OrbitKV nodes

`--metaserver-addr` enables P2P. `--nics` is optional:

- **`--nics <NAME>...`** — optional Mooncake RDMA allow-list (for example
  `mlx5_0`, `mlx5_0 mlx5_1`). OrbitKV passes it to `MC_TE_FILTERS`. When it is
  omitted, Mooncake selects the available transport and can fall back to TCP.

- **`--metaserver-addr <URL>`** — the MetaServer address. Once set, this node
  starts Mooncake, registers its block hashes, and fetches remote blocks when needed.

When P2P is enabled, `--addr` must use a routable IP. The gRPC service and the
Mooncake P2P endpoint use that host with separate ports.

**Node A** (e.g. `10.0.0.1`):

```bash
orbitkv-cache-manager \
  --addr 10.0.0.1:50055 \
  --pool-size 30gb \
  --nics mlx5_0 \
  --metaserver-addr http://10.0.0.100:50056
```

**Node B** (e.g. `10.0.0.2`):

```bash
orbitkv-cache-manager \
  --addr 10.0.0.2:50055 \
  --pool-size 30gb \
  --nics mlx5_0 \
  --metaserver-addr http://10.0.0.100:50056
```

### 3. Launch inference engine

Same adapter setup as single-node; the node-local Cache Manager handles remote
fetch after a local miss. A vLLM/SGLang process does not connect to a remote
manager or MetaServer directly.

```bash
vllm serve /path/to/immutable-model \
  --kv-transfer-config '{"kv_connector": "OrbitKVConnector", "kv_role": "kv_both", "kv_connector_module_path": "orbitkv.vllm"}'
```

### 4. Verify

Use `--log-level debug` to confirm P2P is working. Look for MetaServer
registration, the advertised Mooncake endpoint, and fetch summaries.

## Fallback Behavior

P2P is opportunistic. On discovery or transfer failure, the request must fall
back to local cache or recomputation; timeout and retry costs still affect
latency and require qualification.

| Scenario | What happens |
|---|---|
| MetaServer unreachable | Remote discovery is unavailable; local cache operations continue. |
| Remote node unreachable | Authorization or Mooncake segment open fails; the request proceeds without remote blocks. |
| Transfer timeout | Mooncake segment cache entry is invalidated and the transfer lease is released. |

## Tuning

### Hugepages

For pools >64 GB, hugepages significantly reduce RDMA memory registration overhead and transfer latency. Configure before starting OrbitKV:

```bash
# Allocate hugepages (example: 64 GB of 2MB pages)
echo 32768 > /proc/sys/vm/nr_hugepages

orbitkv-cache-manager --pool-size 64gb --use-hugepages --nics mlx5_0 ...
```

### NUMA affinity

OrbitKV automatically detects GPU–NIC NUMA affinity at startup. For best performance, ensure GPUs and RDMA NICs share the same NUMA node. Check the topology log at startup.

### MetaServer sizing

The service is single-process and in-memory. Its footprint grows with owner
registrations; measure it under your actual block count and churn. There is no
tested cluster-size or memory-capacity recommendation yet.

### Metrics

P2P-related Prometheus metrics (on `:9091/metrics` by default):

| Metric | Type | Description |
|---|---|---|
| `orbitkv_remote_fetch_total` | Counter | Total per-segment Mooncake fetch operations |
| `orbitkv_remote_fetch_duration` | Histogram | Mooncake fetch latency distribution |
| `orbitkv_remote_fetch_bytes` | Counter | Total bytes fetched via Mooncake |
| `orbitkv_remote_fetch_plan_segments` | Histogram | Planned segment count per executed Mooncake fetch plan |
| `orbitkv_remote_fetch_plan_completed_segments` | Histogram | Completed segment count before a plan stops |
| `orbitkv_transfer_lock_active` | UpDownCounter | Currently held transfer locks |
| `orbitkv_transfer_lock_timeouts_total` | Counter | Transfer lock timeout events |

## Troubleshooting

**Blocks not discovered on remote nodes**

- Both nodes must point to the same MetaServer and serve the same model. Namespace is derived from model name and TP config — mismatched models or TP sizes will result in different namespaces.
- Check MetaServer logs for `InsertBlockHashes` — if absent, the source node isn't registering.

**High Mooncake fetch latency**

- Check NUMA affinity in the startup topology log — cross-NUMA transfers add latency.
- Enable hugepages for large pools (`--use-hugepages`).

For all P2P issues, `--log-level debug` shows the authorization and Mooncake
fetch flow.
