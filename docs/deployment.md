# Deployment

For LMCache/Mooncake integration boundaries and OrbitKV's single-node, DP and
P/D execution order, see [distributed deployment comparison](distributed-comparison.md).

| Mode | Processes | Status |
| --- | --- | --- |
| Single-node vLLM or SGLang cache | Engine + one local Cache Manager | Single-rank DRAM and forced-SSD recovery validated on both; concurrent and multi-rank workloads need separate qualification |
| Shared cache across nodes | One Cache Manager per host with embedded catalog + etcd | Experimental; fixed directory shards have one metadata copy |
| vLLM P/D through OrbitKV `PdConnector` | Prefill, decode, P/D proxy; Mooncake transfers KV | Experimental; does not need Cache Manager or Catalog for the handoff |
| vLLM P/D through upstream NIXL | Prefill, decode, NIXL-aware router | Upstream vLLM connector; separate from OrbitKV cache |

For exact wheel installation, vLLM and SGLang commands, socket/container
requirements, capacity settings, and external-hit verification, follow the
[single-node guide](single-node.md). The engine owns HBM capacity and
allocation; configure OrbitKV's pinned host-memory and optional SSD capacity
independently. See the [SSD measurements](ssd-performance.md) for
tier-specific evidence and measurement limits. SSD cache files are truncated on
manager startup, so they are not durable across a Cache Manager restart. For cross-node cache
sharing, run the
[embedded catalog and a local Cache Manager on each host](p2p.md); inference
processes still connect only to their *own* host's UDS endpoint. Multi-host TP
query fan-out is not supported yet.

## P/D: Mooncake or NIXL

P/D moves KV for the same request from prefill to decode. Remote caching finds
reusable KV from an earlier request. These are independent paths; see
[P/D and NIXL](pd.md) for the ownership and control-flow distinction.

OrbitKV's vLLM `PdConnector` pushes KV through Mooncake directly between GPU
workers. Try the [local P/D example](../scripts/run_pd_local.sh) for that path.
vLLM `0.29.0` also includes its own NIXL connector; the
[NIXL comparison example](../scripts/run_nixl_local.sh) uses vLLM's code.
OrbitKV does not ship a NIXL connector, and its SGLang adapter currently
implements external caching only.

### Experimental vLLM P/D with NIXL plus OrbitKV cache

This configuration is illustrative and has not passed the multi-host
qualification gate. It enables MTP and `MultiConnector` on both sides. vLLM's
NIXL connector handles the P-to-D handoff; OrbitKV uses `read_write` on P and
`save_only` on D to retain KV for later requests without competing with NIXL
loads. `MultiConnector` chooses the first connector advertising a load and
saves to all configured connectors in order.

Run one Cache Manager beside each vLLM instance, with the same etcd cluster and catalog host set,
and use a P/D-aware request router. Configure each manager's routable `--addr`,
`--nics`, `--etcd-endpoints`, `--node-id`, and `--catalog-nodes` as described in the [P2P guide](./p2p.md).
P and D must use compatible model, tokenizer, block size, KV dtype, KV layout,
and `PYTHONHASHSEED`.

Replace `<p_node_ip>` and `<d_node_ip>` with the addresses assigned to the P
and D nodes. The example assumes P and D run on separate nodes; to colocate
them on one host, keep the distinct NIXL side-channel ports and point both
connectors at one local Cache Manager — each vLLM instance registers with its
own instance ID.

### Prefill

```bash
PYTHONHASHSEED=42 \
VLLM_NIXL_SIDE_CHANNEL_HOST=<p_node_ip> \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve GLM-5.2-FP8 \
  --served-model-name glm-5.2 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --enable-prefix-caching \
  --trust-remote-code \
  --kv-transfer-config '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "connectors": [
        {
          "kv_connector": "NixlConnector",
          "kv_role": "kv_producer"
        },
        {
          "kv_connector": "OrbitKVConnector",
          "kv_role": "kv_both",
          "kv_connector_module_path": "orbitkv.vllm",
          "kv_connector_extra_config": {
            "orbitkv.host": "http://<p_node_ip>",
            "orbitkv.port": 50055,
            "orbitkv.mode": "read_write"
          }
        }
      ]
    }
  }'
```

### Decode

```bash
PYTHONHASHSEED=42 \
VLLM_NIXL_SIDE_CHANNEL_HOST=<d_node_ip> \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve GLM-5.2-FP8 \
  --served-model-name glm-5.2 \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --enable-prefix-caching \
  --trust-remote-code \
  --kv-transfer-config '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "connectors": [
        {
          "kv_connector": "NixlConnector",
          "kv_role": "kv_consumer",
          "kv_connector_extra_config": {
            "kv_recompute_threshold": 0
          }
        },
        {
          "kv_connector": "OrbitKVConnector",
          "kv_role": "kv_both",
          "kv_connector_module_path": "orbitkv.vllm",
          "kv_connector_extra_config": {
            "orbitkv.host": "http://<d_node_ip>",
            "orbitkv.port": 50055,
            "orbitkv.mode": "save_only"
          }
        }
      ]
    }
  }'
```

### Validate D-to-P reuse

Generate a multi-block response, then send the full history in turn two. A hit
for the first response exercises D-to-P reuse. Verify:

- `orbitkv_remote_fetch_bytes` increases on the prefill-side Cache Manager if
  a remote cache hit occurs, and peer-transfer authorization is visible in the
  decode-side Cache Manager logs.
- Every requested block is transferred with no block-count mismatch.
- `vllm:orbitkv_load_failure_total`, `vllm:orbitkv_save_failure_total`, and
  peer-transfer failures stay at zero.
