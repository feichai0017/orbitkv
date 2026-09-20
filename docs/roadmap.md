# OrbitKV roadmap

Every milestone ends with an executable gate. Future design is not reported as
current capability. The detailed work queue lives in [TODO.md](../TODO.md).

## M0: framework-neutral foundation

Deliver:

- move Rust packages under `crates/` and keep the repository root a virtual
  workspace;
- introduce `orbitkv-contract`;
- place the vLLM cache connector and Mooncake P/D adapter under `orbitkv.vllm`;
- establish `orbitkv.sglang` and `orbitkv.client` package boundaries;
- establish the `orbitkv-local` iceoryx2 control ABI;
- connect lifecycle probes, epoch fencing, and shutdown to the real Cache Manager and
  Python client;
- bootstrap a generation-checked descriptor arena over UDS and execute
  `QueryBundle` through the cache service;
- add a framework-neutral Python cache client for local query, publish,
  restore, and lease release;
- integrate a pinned upstream Mooncake Transfer Engine as the single remote
  movement implementation;
- preserve current vLLM behavior.

Gate:

- Cargo metadata, format, workspace check, and host-safe tests pass;
- default Python unit tests pass;
- the vLLM plugin registers its connectors without loading the engine at
  package import time;
- the SGLang plugin registers the direct GPU-page linker.

## M1: SGLang direct GPU-page linker

Deliver:

- register SGLang GPU KV buffers with the Cache Manager through CUDA IPC;
- switch framework adapters to the available local `QueryBundle`, publish,
  restore, completion, and lease-release APIs;
- support full-attention MHA and MLA layouts;
- reject hybrid, draft, DSA, and auxiliary GPU state until complete recovery
  contracts are available;
- expose cold miss, partial prefix, warm hit, cancellation, and restart metrics.

Gate:

- exact generated-text parity against a cold-control namespace;
- a real GPU load after flushing radix cache and after a worker restart;
- SGLang worker restart preserves Cache Manager-resident cache;
- unsupported state fails during initialization rather than reporting a hit.

## M2: common StateBundle query and native local transport

Deliver:

- move vLLM hybrid reconciliation from the adapter into common bundle logic;
- use local restore operations and eventfd wakeups for both adapters;
- define framework-neutral region registration and transfer-plan operations;
- add generation validation to every local page reference.

Gate:

- vLLM and SGLang generate equivalent recovery contracts for a shared test
  model;
- adapter code contains no tier-selection or bundle-completeness policy;
- load/save throughput is not regressed against the M0 baseline.

## M3: KV-aware routing and replica catalog

Deliver:

- consume vLLM and SGLang KV placement events;
- track replicas by worker and tier;
- reproduce a Dynamo-style weighted-overlap worker selector as a baseline;
- add queue, transfer, recompute, and eviction costs;
- return target worker plus source/restore plan.
- qualify Mooncake topology-aware slicing, endpoint pooling, and alternate-rail
  retry against OrbitKV transfer plans.

Gate:

- baseline selector agrees with Dynamo on captured traces;
- joint planning beats load-only and overlap-only baselines on a held-out trace;
- stale events and worker restarts cannot route to a dead replica.

## M4: OrbitKV page authority

Deliver:

- generation-qualified page handles;
- explicit semantic and execution frontiers;
- CUDA/RDMA/SSD completions advance one execution-fence abstraction;
- SGLang Radix nodes and vLLM adapter consume manager-authored handles.

Gate:

- generation reuse cannot race an outstanding operation under stress and fault
  injection;
- the frameworks no longer mint external page identities;
- cache cleanup is safe across cancellation, preemption, and process death.

## M5: semantic state compiler

Deliver:

- a `may_read(query, state)` lifetime IR;
- full, sliding, sink-local, recurrent, and hybrid recovery plans;
- Minimum Persistent State Realization;
- compiled retention, checkpoint, placement, and replication policies.

Gate:

- report Retention Amplification alongside TTFT, TPOT, throughput, and network
  traffic;
- compiled plans reduce physical state without changing exact-model outputs;
- ring/checkpoint/tier choices are derived from the state contract and measured
  cost, not selected by model-name branches.

## M6: backend and ecosystem expansion

Deliver only after M1-M5 gates:

- agentic multi-turn value model;
- multi-DC replica planning;
- signed plan/evidence bundles if deployment requires them.
