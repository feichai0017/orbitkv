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

### Two-host TCP, 2026-09-28

Native source `d2ef3a60` runs on a local H20 and a remote A100 over IPv6 TCP,
with identical Manager/extension artifacts and Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`. Both Managers use separate 2 GiB
DRAM pools, TP=1 and 64-token pages; preparation is disabled. vLLM 0.29.0 uses
explicit FlashAttention 2 on both hosts and `VLLM_BATCH_INVARIANT=1`. SGLang
0.5.20 uses its deterministic flag. Both use eager execution.

| Gate | vLLM | SGLang |
| --- | --- | --- |
| Fresh 513/1025-token remote Restore | Passed; 72 + 144 MiB READ/H2D | Passed; 72 + 144 MiB READ/H2D |
| Sole catalog host restart, inventory replay and Restore | Passed; 72 MiB READ/H2D | Passed; 72 MiB READ/H2D |
| Source restart without payload | Passed; correct recomputation, zero remote/H2D bytes | Passed; correct recomputation, zero remote/H2D bytes |
| Output controls and source-release/resource drain | Passed in every recorded case | Passed in every recorded case |

The natural-text prompts require retrieving a specific cabinet key from the
cached first page among distractors. Complete eight-token outputs match the
source controls and contain the correct key. Each engine transfers and restores
288 MiB in total. Raw launches, request outputs and counter deltas are under
`benches/results/runs/two-host-natural-20260928/`.

A separate [byte-exact gate](#byte-exact-gpu-recovery) passes 8 MiB in each
direction, including re-serving the received replica after original-source
eviction. Every K/V byte, destination-page permutation and sentinel gap matches,
with source-release acknowledgements and drained counters. It found and now
regresses a real placement error: H20's NUMA 0 had been used for allocations
on the A100 host, whose GPU-local pool is NUMA 1. The fix derives destination
placement from the receiving instance's registered slots; host NUMA identifiers
never define cross-host storage identity. The final normal release with bounded
Restore partitioning repeats this bidirectional byte gate successfully; its
outputs and matching native hashes are retained under
`benches/results/runs/partitioned-restore-20260928/`.

**Numerical scope:** the initial random-token model suite did not pass strict
cross-GPU output equality. Native monolithic H20/A100 controls also differ for
some of those prompts; forcing FA2 on both hosts alone does not close the
remote-output failure. A100 native HBM prefix hits match their own cold controls. The same random
513/1025-token requests also pass through two vLLM/FA2 replicas on one A100,
with 72/144 MiB remote/GPU restores; those diagnostic logs are under
`benches/results/runs/same-host-a100-20260928/`.
Keep the rejected cross-GPU runs and native controls under
`benches/results/runs/two-host-20260928/`; do not describe the natural-text or
byte gate as proving arbitrary cross-GPU token equality.

These are source-build correctness results. Neither container exposes RDMA
hardware, and the hosts have different GPUs. They do not qualify RDMA,
GPUDirect, distributed throughput, installed release artifacts, P/D or
mid-transfer crash/partition revocation.

### Earlier same-host gates

The [2026-09-25 ownership-layout recheck](implementation-plan.md#ownership-layout-final-evidence)
passed on native source `046b16f5` with a matching query-body-v6 client and frozen
Manager. Both engines completed three remote GPU restores, catalog restart
replay and correct recomputation after source payload loss. Each transferred
and restored 288 MiB; outputs matched and checked resource counters drained.
This remains single-H20, same-host TCP evidence.

The [2026-09-23 Qwen3-8B gate](../benches/results/20260923-shared-cache/README.md)
passed on one H20 with vLLM 0.29.0 and SGLang 0.5.20, tested separately. Each
engine completed three remote GPU restores, including catalog restart recovery,
and one correct recomputation after source payload loss. Outputs matched and
the checked resource counters drained in every case. Each engine transferred
288 MiB remotely and restored the same amount to HBM. The final rerun includes
reusable transfer windows and slot generations for lost authorization replies;
513/1025-token requests used 3/6 discovery RPCs on each engine, retaining the
catalog-host batching reduction from 8/14 on vLLM and 6/16 on SGLang in the
preceding gate. All 12 source-release
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
with DRAM-only Managers for the existing recorded baseline. Peer SSD routing is
implemented but not yet qualified; force source DRAM eviction and record SSD
read/staging evidence in a separate run. A two-host TCP
run sets `MC_FORCE_TCP=1` on both Managers. For RDMA, expose the devices and
select the appropriate `--nics`; record the actual Mooncake transport and NIC
counters with the result.

The Manager's HTTP endpoint must be reachable by the qualification driver on a
trusted test network. It includes administrative operations and is not a public
inference endpoint. The engine still uses UDS/iceoryx2, regardless of the driver
location.

## Byte-exact GPU recovery

Use `benches.shared_cache_bytes` before model comparisons on different GPU
architectures. It checks four split-K/V layers, source pages spaced by one gap,
permuted destination pages and untouched sentinel bytes. After an 8 MiB remote
Restore, the driver evicts the original source's DRAM and requires the received
replica to serve the same bytes back. Both directions require 8 MiB of Mooncake
READ and GPU Restore, source-release acknowledgements, matching SHA-256 digests
and drained resource counters. This checks the full GPU-to-Manager-to-peer-to-GPU
path; it does not claim direct GPU-to-GPU RDMA or model-output equivalence.

Start fresh DRAM-only test Managers, with no unrelated traffic, using the
cluster setup above. Run one fixture worker beside each Manager from a checkout
with a built or installed OrbitKV extension and the engine's Torch environment.
The workers expose only fixed test operations; use the trusted test network:

```bash
# Host A; on host B use 10.0.0.2 and --instance byte-target.
python -m benches.shared_cache_bytes worker \
  --host 10.0.0.1 --port 8002 --manager-socket /tmp/orbitkv-50055.sock \
  --instance byte-source --namespace byte-qualification

# Controller; leave both workers running during this command.
python -m benches.shared_cache_bytes run \
  --source-url http://10.0.0.1:8002 --target-url http://10.0.0.2:8002 \
  --source-manager http://10.0.0.1:9091 --target-manager http://10.0.0.2:9091 \
  --output benches/results/runs/gpu-byte-roundtrip.json
```

The source DRAM cleanup is intentional. Stop both workers after the gate; use
fresh test Managers and a matching namespace on both workers for each rerun.

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

## Forced source-SSD gate

The 2026-09-28 two-host H20→A100 TCP run passes on vLLM 0.29.0 and SGLang
0.5.20 with Qwen3-8B and the same natural-text controls described above. After
source DRAM eviction, each engine recovers 513/1025-token requests with
72/144 MiB of source io_uring reads, remote TENT READs and destination GPU
restores. Complete outputs match the source controls, source staging succeeds,
and both sides' checked reservations drain. Raw launches, output controls and
counter snapshots are under
`benches/results/runs/partitioned-restore-20260928/{vllm,sglang}-shared-ssd/`.
This qualifies ordinary source-SSD recovery over TCP; it does not cover
cancellation during SSD staging/READ, RDMA or a throughput advantage.

Run this as a separate result from the DRAM baseline. Configure SSD on the
source Manager with write policy `all`; the target may use its ordinary local
configuration. Add `--source-medium ssd` to the command above; the driver
performs the synchronization, DRAM-only cleanup and source/target evidence
checks below. For each prompt:

1. Execute it on the source, call `POST /cache/sync`, and verify the SSD write
   completed before changing residency.
2. Call `POST /cache/memory/cleanup` on the source. Require positive evicted
   blocks, zero `still_referenced_blocks`, and surviving Catalog evidence with
   medium `ssd`; do not clear the SSD ring or restart the source.
3. Send the prompt to a target replica with a cold local namespace. Require an
   increase in source `orbitkv_ssd_prefetch_bytes_total`, target
   `orbitkv_remote_fetch_bytes_total`, and target GPU restore bytes. The output
   must match the deterministic cold control.
4. After completion, require source `orbitkv_ssd_read_pinned_bytes`,
   `orbitkv_transfer_reserved_bytes` and `orbitkv_transfer_lock_active` plus
   target query and `orbitkv_transfer_completion_outstanding` reservations to
   return to baseline. Repeat cancellation
   while source I/O is queued and while Mooncake READ is active.

Run first with `MC_FORCE_TCP=1`, then on two physical hosts with the intended
RDMA rails and `--nics`. Record the actual Mooncake transport, NIC counters,
source SSD device/mount, byte amplification, authorization/staging/READ
latencies and peak sender/receiver memory. A successful same-host or TCP run
does not qualify RDMA, and a remote byte increase without a source SSD-read
increase does not qualify the peer-SSD route.

For cost evidence, first set only `ORBITKV_COST_OBSERVATIONS=1`. Keep
`ORBITKV_COST_SELECTION` unset so the fixed route remains the control. Record
`local_ssd_host_ready`, `peer_dram_host_ready`, `peer_ssd_host_ready` and shadow
decisions for identical stored-byte/block shapes. Multi-owner or different-
coverage rows are not valid cross-medium comparisons.

Only after that control passes, run a separately labelled experiment with all
three variables set to `1`: `ORBITKV_COST_OBSERVATIONS`,
`ORBITKV_COST_SELECTION`, and `ORBITKV_CROSS_MEDIUM_SELECTION`. Require positive
`orbitkv_cost_route_decisions_total{decision="selected",scope="cross_medium"}`
evidence, the same
output/byte/drain checks, and a matched fixed-route control. Do not carry the
third flag into ordinary serving until both engines pass TCP and RDMA cells.

For the later GPU-memory gate, register each granted region with the exact
Mooncake `cuda:N` location and keep the engine page-generation owner beside the
registration token until completion. Exercise explicit unregister, cancellation,
engine-handle release and instance teardown. Do not treat successful host-memory
RDMA or a configured `--nics` value as GPUDirect evidence; record GPU/NIC counters
from the external H20 hosts.

The artifact under test must contain `libtent_shared.so` and must not contain
`libtransfer_engine.so`. Capture TENT startup logs showing the installed
transports. A CUDA-enabled build alone is not evidence that a batch selected
RDMA, NVLink or GPUDirect.

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
blocking transfer, bounded retry of lost release replies, lost authorization
replies after pinning, close-before-authorize races, stale slot generations and
idle-window eviction. Connection recovery verifies that an old requester
completion cannot release a new runtime’s hold. These tests cannot prove
transport revocation after a permanently lost requester. Such source pins remain
charged until safe release or coordinated Manager teardown. Real partitions,
two-host serving, multi-rank replicas and catalog HA remain separate gates.
