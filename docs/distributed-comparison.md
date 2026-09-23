# Distributed cache deployment comparison

Reviewed against official documentation on 2026-09-23. These are documented
integration paths, not OrbitKV benchmark results or a claim that every model,
engine release and parallel layout works. OrbitKV's tested engine versions
remain the pinned releases in [single-node setup](single-node.md).

## LMCache

The current MP documentation distinguishes three deployments:

- [P2P sharing](https://docs.lmcache.ai/mp/p2p.html): each node has an LMCache
  server. A coordinator supplies membership; a peer locks and locates the
  requested KV before transfer into the requesting server's L1 memory.
  NIXL is the default transport; Mooncake TE is another documented option.
  Independent inference replicas can therefore reuse matching prefixes.
- [Multi-server TP](https://docs.lmcache.ai/mp/configuration.html#vllm-client-configuration):
  one vLLM deployment assigns contiguous rank groups to multiple servers.
  This mode explicitly rejects PP > 1 and DP > 1. That restriction concerns
  one deployment's multi-server connector topology; it does not prohibit
  P2P sharing between independent inference replicas.
- [P/D plus cache reuse](https://docs.lmcache.ai/mp/disaggregated_prefill.html):
  vLLM `MultiConnector` composes upstream `NixlConnector` for the current
  request's handoff with `LMCacheMPConnector` for offload and later reuse.
  The guide uses separate LMCache servers for P and D and optionally adds
  P2P sharing. It also names a required vLLM MultiConnector fix; a command
  example alone is not a release-compatibility guarantee.

The old in-process PD examples describe a different integration. Do not mix
their constraints with the MP deployment when comparing capabilities.

## FlexKV

- [Engine integration](https://github.com/taco-project/FlexKV) covers vLLM and
  SGLang, CPU/SSD offload, TP/PP work and Mooncake-based remote reuse. These
  features do not establish every model/topology combination.
- [Distributed reuse](https://github.com/taco-project/FlexKV/blob/main/docs/dist_reuse/README_en.md)
  uses a local snapshot of the global index, Redis metadata and Mooncake
  transfer. Avoiding a centralized lookup on each request is therefore not
  unique to OrbitKV. Its example still pins an older vLLM; verify compatibility
  before using it as a current-release benchmark.
- [SGLang integration](https://github.com/taco-project/FlexKV/blob/main/flexkv/integration/sglang/README.md)
  distinguishes standard released-engine support from model-specific adaptations
  tied to an upstream PR. Record the exact engine release and FlexKV commit.

OrbitKV prioritizes independent-replica sharing and P/D reuse, followed by catalog
HA. Compare bytes read, index/synchronization costs, tail latency and useful
throughput at equal capacity. Native code, separate processes and hybrid-model
support alone do not establish a performance advantage.

## Mooncake

- [vLLM shared storage](https://kvcache-ai.github.io/Mooncake/deployment/integrations/vllm/kv-cache-storage.html):
  `MooncakeStoreConnector` uses the Store's distributed CPU/SSD pool for
  offload and prefix reuse across instances. Its P/D example composes
  `MooncakeConnector` with `MooncakeStoreConnector` through `MultiConnector`.
- [SGLang HiCache](https://kvcache-ai.github.io/Mooncake/design/hicache-design.html):
  engine HBM and host caches are private tiers; Mooncake Store supplies the
  shared L3. Multi-rank coordination takes the minimum successful prefix
  across ranks before claiming readiness.
- [SGLang P/D](https://kvcache-ai.github.io/Mooncake/deployment/integrations/sglang/pd-disaggregation.html):
  the framework's handoff uses Mooncake TE for cross-node movement. The
  transfer path is distinct from HiCache's shared storage backend.

Transfer Engine moves registered memory; it does not define a framework's
legal recovery boundary. TP/PP, heterogeneous rank layouts and hybrid models
must be qualified against the selected engine connector. Shared storage does
not by itself prove that a TP=4 representation can be restored by TP=8.

## OrbitKV execution order

1. Qualify sustained single-node DRAM/SSD serving for both engines, with
   bounded requests and resources, cancellation/restart correctness, and
   native-engine controls. Profile actual copy destinations before attempting
   to merge H2D restores. See [benchmarks](../benches/README.md).
2. Qualify two real hosts running independent matching TP=1 replicas, separately
   for vLLM and SGLang. Count real remote restore bytes, compare outputs and
   bound failure-to-recompute behavior. Then qualify same-host TP per replica.
3. Qualify P/D and cache reuse together: a P-side cache hit must still reach D;
   completed D-side blocks should be available to a later request on another
   P. Start from the existing vLLM Mooncake adapter and separately integrate
   SGLang's handoff lifecycle. A basic P/D proxy is sufficient for this gate.
4. Add replicated catalogs and operational HA before a production distributed
   deployment. Cross-host TP/PP, general resharding, remote SSD and KV-aware
   routing have separate later gates.

Transport completion and page-lifetime safety are requirements at every stage.
Directory hints never authorize memory reuse. Keep engine-to-Manager UDS/iceoryx2,
peer control gRPC, and Mooncake TE payload transfer. Full recovery semantics
remain tracked in [state identity](state-identity.md).
