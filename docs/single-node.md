# Single-node cache: vLLM and SGLang

OrbitKV runs one Cache Manager per host, next to the inference engine. The
engine owns HBM allocation and decides when to look up, save, and restore KV.
The manager owns external pinned DRAM and optional SSD replicas. Both adapters
register engine-owned GPU buffers through CUDA IPC, send cache commands through
iceoryx2, and use an authenticated Unix socket for bootstrap and lifecycle.
No Catalog or peer gRPC listener is needed for this deployment.

| Adapter | Validated release | Single-node path | Current limit |
| --- | --- | --- | --- |
| vLLM `OrbitKVConnector` | `0.29.0` | KV connector callbacks, CUDA IPC, UDS/iceoryx2 | Attention + aligned recurrent layouts share compiled page demand and validation; SWA remains unsupported, cross-host TP and further hybrid layouts need separate qualification |
| SGLang `OrbitKVLinker` | `0.5.20` | RadixCache external linker, CUDA IPC, UDS/iceoryx2 | Full-attention MHA/MLA, Full + SWA, or Full + recurrent/conv; TP=1 DRAM/SSD recovery gates; combined SWA + recurrent unsupported, multi-rank serving unqualified |

Start with a single-rank model and one Manager. Check
[hybrid model compatibility](hybrid-recovery.md) before enabling other layouts.
The first release is being prepared; the commands below build from source.

## Install and start the common manager

Build from the repository root after selecting the wheel variant that matches
the host Python ABI and CUDA runtime. This example uses Python 3.11 and CUDA 13:

```bash
git submodule update --init --recursive third-party/mooncake
./scripts/build-wheel.sh --release --no-default-features --features cuda-13,mooncake
WHEEL="$(find target/wheels -maxdepth 1 -name 'orbitkv_llm_cu13-*.whl' -print -quit)"
test -n "$WHEEL"
```

Run the installation commands below from the repository root in this same
shell, or set `WHEEL` to the built wheel's absolute path in each shell.

Use `./scripts/build-wheel.sh --release` for the CUDA 12 distribution. The
complete wheel contains the Cache Manager binary, Python extension, and
Mooncake runtime; `maturin build` alone does not stage all of them. The Cache
Manager still imports PyTorch to reconstruct framework CUDA IPC registrations,
so start it in a Python environment with compatible PyTorch/CUDA packages.
The engine and manager may use separate Python environments; install the same
OrbitKV wheel in both. The examples below use one environment per engine and
start the manager from that environment.

Start **one** manager on the host before its engines. For the commands below,
the bootstrap socket is `/tmp/orbitkv-50055.sock`:

```bash
orbitkv-cache-manager \
  --addr 127.0.0.1:50055 \
  --http-addr 127.0.0.1:9091 \
  --pool-size 8gb
```

This reserves an external pinned-host-memory budget; it does not take over the
engine's HBM allocator. Set `--pool-size` according to host RAM and the desired
cache budget. To enable an SSD backing cache, add for example
`--ssd-cache-path /data/orbitkv/cache.bin --ssd-cache-capacity 100gb` to the
manager command. The current SSD cache file is truncated on manager startup;
it is not durable across a manager restart. Both engines have single-rank recovery gates after forced DRAM eviction.

Request preparation and queued warming are optional experiments and are disabled
by default. Leave them off for the first deployment. See
[request preparation](request-preparation.md) for measured policy tradeoffs and
[Manager configuration](server.md) for capacity and queue controls.

The manager, each engine process, and their GPU buffers must be on the same
host. The Unix socket verifies peer credentials; use the same UID and make the
socket and iceoryx2 shared-memory resources visible to both processes. If they
run in separate containers, their Unix socket, IPC/shared-memory namespace,
and GPU access must be shared. A remote HTTP address is not a replacement for
this node-local connection.

## vLLM

Install the validated vLLM release and the wheel in one environment:

```bash
uv venv .venv/vllm-release --python 3.11
uv pip install --python .venv/vllm-release/bin/python 'vllm==0.29.0' --torch-backend=cu130
uv pip install --python .venv/vllm-release/bin/python --reinstall "$WHEEL"
```

In terminal 1, start the manager with
`.venv/vllm-release/bin/orbitkv-cache-manager` in place of
`orbitkv-cache-manager` in the common command. In terminal 2, start vLLM:

```bash
.venv/vllm-release/bin/vllm serve /path/to/immutable-model \
  --enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector": "OrbitKVConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "orbitkv.vllm"
  }'
```

`OrbitKVConnector` defaults to `read_write`. vLLM first uses its own HBM
prefix cache; the connector queries OrbitKV for missing reusable blocks,
restores hits into vLLM-owned GPU slots, and saves newly computed full blocks
to the manager. To prove an **external** hit, keep the manager alive, restart
vLLM, repeat a multi-block prompt, and check manager load/hit counters. A
same-process repeated prompt may be served entirely from vLLM's HBM and does
not by itself demonstrate an OrbitKV restore.

The default connector endpoint derives the same socket from
`http://127.0.0.1:50055`. If you choose a different manager port, set
`orbitkv.port` in `kv_connector_extra_config` or `ORBITKV_PORT` in the vLLM
environment. Use `orbitkv.bootstrap_socket` for a custom socket path.
For multiple **same-host** TP shards, see
[the ordered shard endpoint configuration](adapters.md#tp-shards-and-host-boundary).
Do not configure a scheduler to query TP shards on another host through this
local connector.

## SGLang

Install the validated SGLang release and the same wheel in a separate
environment:

```bash
uv venv .venv/sglang-release --python 3.11
uv pip install --python .venv/sglang-release/bin/python 'sglang==0.5.20' --torch-backend=cu130
uv pip install --python .venv/sglang-release/bin/python --reinstall "$WHEEL"
```

Start the manager from this environment if it is not already running. Then:

```bash
ORBITKV_SGLANG_ENDPOINT=unix:///tmp/orbitkv-50055.sock \
  .venv/sglang-release/bin/sglang serve \
  --model-path /path/to/immutable-model \
  --page-size 64 \
  --enable-unified-cache-external-linker \
  --radix-cache-backend orbitkv
```

The wheel registers the SGLang plugin. Both flags are required to schedule GPU
restores and handle their completion. SGLang owns the radix tree and HBM pages;
OrbitKV saves page-aligned KV externally and restores it into engine-owned slots.
Pending reads keep their request queued until ready or until its waiting budget
expires, when the engine can recompute. Supported hybrid layouts use compiled
ranges and validate every selected window/checkpoint before recovery. See
[hybrid recovery](hybrid-recovery.md) for model limits and GPU test coverage.

Both engines fingerprint local weights, tokenizer and processor artifacts at
startup, and bind the computation and registered storage layout to the cache
identity. Hub models require a full commit in `--revision`. Large deployments
can set `ORBITKV_MODEL_FINGERPRINT` to their verified 64-digit SHA-256 artifact
digest to avoid startup file reads. Change that digest when covered artifacts
change. `ORBITKV_CACHE_SCOPE` provides optional tenant/experiment isolation; it
does not override the model identity. Dynamic LoRA is rejected; live weight
updates require an engine restart. See [state identity](state-identity.md).

For a recovery check, keep the manager alive, flush the radix cache or restart
SGLang, repeat the prompt, compare against a cold run, and verify that the
manager's load counter rose.

## Check behavior and capacity

```bash
curl --fail http://127.0.0.1:9091/health
curl --fail http://127.0.0.1:9091/metrics | \
  grep -E 'orbitkv_(save_bytes_total|load_bytes_total|cache_block_hits_total)'
```

Save completion confirms the GPU-to-DRAM copy. SSD writes proceed
asynchronously and may be dropped under pressure; a completed save does not
guarantee an SSD replica. Both adapters record a CUDA event on the producing
stream before handing pages to their save worker. vLLM records it after the
forward launch, outside graph capture; saving waits for those events rather
than synchronizing the whole device. Request and checkpoint pages remain held
until the native D2H operation completes.

The SSD backend separates read and write submission queues across its existing
io_uring workers. Reads rotate across read workers, including when there is only
one cache file; each file's writes retain a stable queue. This avoids a read
waiting in the submission queue of an unrelated write. It does not increase the
configured in-flight read/write limits or remove device-level I/O contention.

Later requests can restore ready matching blocks,
subject to each adapter's readiness handling. On a cache miss the engine
computes the state normally. HBM pressure and active-page eviction remain engine
decisions; OrbitKV's `--pool-size` and SSD options control only external cache
capacity. Pinning too much host memory or saving every low-reuse block can
increase latency, so size and admission policy should be measured against the
workload. To validate the current adapters, use the
[vLLM correctness test](../python/tests/e2e/test_vllm_e2e_correctness.py) and
[SGLang direct GPU test](../python/tests/e2e/test_sglang_direct_e2e.py) on the
matching GPU host.

For experimental multi-node cache sharing, use [P2P deployment](p2p.md).
vLLM P/D handoff via OrbitKV's `PdConnector` or upstream NIXL is a separate
request-transfer path; see [P/D and NIXL](pd.md). SGLang has no OrbitKV P/D
adapter today.

## Deployment variants

| Topology | vLLM | SGLang |
| --- | --- | --- |
| One engine and one manager on a host | Validated DRAM recovery and measured SSD restoration on the pinned release | Validated single-rank full-attention DRAM and SSD recovery through plugin admission |
| Multiple engine instances sharing one host manager | Instances can use the same local socket; use immutable model identities and qualify concurrency for the workload | Instances can use the same local socket; rank/layout-scoped namespaces isolate incompatible pages, and concurrent multi-rank recovery still needs a GPU gate |
| Replicas on separate hosts | One manager per host with embedded catalog, etcd membership and Mooncake fetch; experimental | The same node-local adapter connection with one manager per host; remote fetch and multi-rank behavior still need qualification |
| One TP replica split across hosts | Unsupported by the current scheduler-to-manager query fan-out | Not qualified by the current single-rank GPU gate |
| P/D handoff | Experimental OrbitKV Mooncake `PdConnector`, or upstream vLLM NIXL; separate from external cache | No OrbitKV P/D adapter |

The current embedded directory has one metadata copy per shard. Do not infer
production multi-node resilience from the validated single-node paths.
