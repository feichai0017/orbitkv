# Shared-cache qualification

This gate covers independent replicas of the same model, engine and TP=1 storage
layout. Every engine uses its host's Cache Manager. Rust owns candidate lookup,
source validation, memory budgets and Mooncake transfers; Python only drives
serving requests and checks the exported results.

The driver accepts already running replicas, so the same requests can run on one
machine or on two hosts. A deployment label records the operator's setup; it is
not automatic proof of distinct physical hosts or an RDMA transport. Keep these
three results separate: same-host TCP, two-host TCP and two-host RDMA.

## Recorded result

The [2026-09-23 Qwen3-8B gate](../benches/results/20260923-shared-cache/README.md)
passed on one H20 with vLLM 0.29.0 and SGLang 0.5.20, tested separately. Each
engine completed three remote GPU restores, including catalog restart recovery,
and one correct recomputation after source payload loss. Outputs matched and
the checked resource counters drained in every case. Each engine transferred
288 MiB remotely and restored the same amount to HBM. In the final rerun,
513/1025-token requests used 3/6 discovery RPCs on each engine, down from
8/14 on vLLM and 6/16 on SGLang in the preceding gate. All 12 source-release
acknowledgements per engine were observed, and requester completion slots drained.

This is a same-host TCP correctness result. Short-prompt restoration was not
consistently faster than recomputation. Physical two-host/RDMA deployment and
performance comparisons still require separate measurements.

## Start matching replicas

Follow [distributed startup](p2p.md) for etcd and two Managers, then the
[single-node instructions](single-node.md) for the selected engine on each host.
Use the same immutable model, engine version, KV dtype, block/page size, TP=1
and `PYTHONHASHSEED=0`. Use separate Manager pools, instance IDs, sockets and
ports. Do not serve unrelated traffic during the gate.

For deterministic Qwen3 controls, set `VLLM_BATCH_INVARIANT=1` for vLLM or
`--enable-deterministic-inference` for SGLang. Keep preparation disabled. Start
with DRAM-only Managers; remote SSD staging is not implemented. A two-host TCP
run sets `MC_FORCE_TCP=1` on both Managers. For RDMA, expose the devices and
select the appropriate `--nics`; record the actual Mooncake transport and NIC
counters with the result.

The Manager's HTTP endpoint must be reachable by the qualification driver on a
trusted test network. It includes administrative operations and is not a public
inference endpoint. The engine still uses UDS/iceoryx2, regardless of the driver
location.

## Run requests

Create `prompts.json` containing fresh token-ID arrays of at least 128 tokens,
tokenized with the deployed model's tokenizer. Prefer distinct first pages and
several prompt lengths. Each prompt is first computed by the source, then sent
to a consumer that has never seen it.

```bash
python -m benches.shared_cache \
  --engine vllm --model /path/to/immutable-model \
  --source-url http://10.0.0.1:8000 --target-url http://10.0.0.2:8000 \
  --source-manager http://10.0.0.1:9091 --target-manager http://10.0.0.2:9091 \
  --prompts /path/to/prompts.json --deployment two-host-tcp \
  --output benches/results/runs/shared-cache-vllm.json
```

Use `--engine sglang` for its native serving endpoint. The driver requires only
the benchmark HTTP dependencies, not an installed engine or CUDA runtime.

The source must publish new bytes. `POST /cache/sync` waits for already submitted
saves and acknowledged catalog residency, with a bounded error when synchronization
cannot finish. Each consumer request must increase both Mooncake READ and GPU
restore bytes, match the cold source output, and drain query, source-transfer
and I/O reservations, including requester completion records awaiting a source
acknowledgement. A response without these counters does not pass as a
remote hit. This gate proves recovery, not throughput superiority.

## Restart and ownership gates

The repository's model-serving test starts etcd, two Managers and two replicas
on one GPU. It checks ordinary sharing, replay after restarting the sole catalog
host, and a clean recomputation after the source restarts without its payload.
It runs separately in the pinned vLLM and SGLang environments:

```bash
cd python
ETCD_BIN=/path/to/etcd \
ORBITKV_CACHE_MANAGER_BINARY=/path/to/orbitkv-cache-manager \
  ../.venv/vllm-release/bin/python -m pytest -m e2e \
  tests/e2e/test_shared_cache.py -k vllm --model /workspace/models/qwen3-8b
```

Repeat with `.venv/sglang-release/bin/python` and `-k sglang`. Build the Manager
before starting these processes. Native builds restage Mooncake libraries.
Raw logs belong in ignored `benches/results/runs/` or CI artifacts; retain only
the final summary and reproduction commands in a PR.

Rust tests separately verify stale owner/residency rejection, source budget
exhaustion, retained source allocations after timeout, cancellation during a
blocking transfer and bounded retry of lost release replies. They cannot prove
transport revocation after a permanently lost requester. Such source pins remain
charged until safe release or coordinated Manager teardown. Real partitions,
two-host serving, multi-rank replicas and catalog HA remain separate gates.
