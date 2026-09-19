# OrbitKV roadmap

Every milestone ends with an executable gate. Future design is not reported as
current capability. The detailed work queue lives in [TODO.md](../TODO.md).

## M0: framework-neutral foundation

Deliver:

- move Rust packages under `crates/` and keep the repository root a virtual
  workspace;
- introduce `orbitkv-contract`;
- move the implementation to `orbitkv.vllm` with `orbitkv.connector` as a
  compatibility alias;
- establish `orbitkv.sglang` and `orbitkv.client` package boundaries;
- establish the `orbitkv-local` iceoryx2 control ABI;
- connect lifecycle probes, epoch fencing, and shutdown to the real sidecar and
  Python client;
- bootstrap a generation-checked descriptor arena over UDS and execute
  `QueryBundle` through the same core path as compatibility gRPC;
- introduce `RemoteMover` with the native RDMA implementation as its first
  backend;
- preserve current vLLM behavior.

Gate:

- Cargo metadata, format, workspace check, and host-safe tests pass;
- default Python unit tests pass;
- the vLLM compatibility module and new canonical module export the same
  connector classes;
- SGLang contract helpers import without SGLang installed.

## M1: SGLang HiCache backend

Deliver:

- implement the dynamic `HiCacheStorage` backend;
- register SGLang shared host regions with the sidecar over UDS;
- execute restore, publish, and completion over iceoryx2 (`QueryBundle` and
  lease release are already available to local clients);
- support KV, MLA, Mamba/recurrent, SWA, and explicit opaque pools;
- map SGLang hit policies into `RecoveryContract`;
- expose cold miss, partial prefix, warm hit, cancellation, and restart metrics.

Gate:

- numerical parity with SGLang's file backend;
- no second host-page copy in the steady state;
- SGLang worker restart preserves sidecar-resident cache;
- multi-pool queries never report a boundary with missing required state.

## M2: common StateBundle query and native local transport

Deliver:

- move vLLM hybrid reconciliation from the adapter into common bundle logic;
- replace per-load shared-memory status files with iceoryx2 completions;
- use UDS file-descriptor passing for shared regions;
- add framework-neutral query, lease, register-region, and transfer-plan RPCs;
- preserve the legacy vLLM protocol until its adapter migrates.

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
- implement an optional production `MooncakeMover` while retaining native
  RDMA as the lightweight and validation backend;
- qualify topology-aware slicing, endpoint pooling, and alternate-rail retry
  against identical transfer plans.

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
