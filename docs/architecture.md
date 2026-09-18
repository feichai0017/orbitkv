# OrbitKV architecture

## Thesis

OrbitKV is a KV-cache system for SGLang, not a second inference server. SGLang
owns request scheduling and model execution. OrbitKV owns persistent cache
identity and will progressively own placement and lifetime decisions.

The imported storage engine is a strong physical substrate: content-addressed
blocks, pinned DRAM, SSD, RDMA, prefix lookup, leases, topology awareness, and
observability. Those mechanisms are the starting point, not the research
contribution.

The contribution is a compiler boundary:

```text
attention semantics + request phase + hardware topology + measured costs
                              |
                              v
                    OrbitKV plan compiler
             liveness / value / placement / movement
                              |
                              v
      HBM pages <-> pinned DRAM <-> SSD <-> remote replicas
                              ^
                              |
                  SGLang execution evidence
```

## Safety invariant

A physical generation may be reused only when both conditions hold:

```text
SemanticDead(block, semantic_frontier)
and
ExecutionComplete(block, execution_frontier)
```

Semantic death proves that no future token permitted by the attention/state
contract can read the block. Execution completion proves that no submitted GPU
or transport work still references its storage. Neither fact implies the other.

## Current data plane

| Component | Responsibility |
| --- | --- |
| `orbitkv-core` | block identity, pinned allocation, leases, cache admission, SSD/RDMA coordination |
| `orbitkv-transfer` | topology-aware CUDA and RDMA movement |
| `orbitkv-server` | process boundary, gRPC, sessions, health, metrics, P/D router |
| `orbitkv-metaserver` | remote replica discovery and liveness |
| `python/orbitkv` | Python binding and framework connectors |
| `orbitkv-proto` | versioned wire contract |

This data plane was imported from PegaFlow 0.24.5 and renamed. Its vLLM path is
upstream-derived and must be requalified as OrbitKV before performance claims
are made.

## SGLang integration

Integration proceeds in two explicit stages.

1. **HiCache storage backend.** SGLang keeps its allocator and moves pages to
   its host pool; an OrbitKV dynamic backend implements batch existence, get,
   and put across KV and recurrent-state pools. This establishes correctness,
   namespace isolation, failure behavior, and a measurable baseline.
2. **Authority handoff.** OrbitKV-authored page handles and generation leases
   replace duplicated physical identity. SGLang consumes those handles, while
   OrbitKV chooses HBM/DRAM/SSD/remote placement and retires pages only after
   both frontiers advance.

Stage one is intentionally not described as sole authority: SGLang still owns
the L1/L2 allocation. That claim becomes true only after stage two is wired and
tested.

## Innovation direction

The strongest direction is **Minimum Persistent State Realization**: compile the
smallest state representation sufficient for future legal execution, then pick
its physical realization from measured costs. This subsumes several mechanisms:

- full attention becomes append-only paged KV with prefix sharing;
- sliding attention becomes bounded cyclic storage;
- sink-plus-local attention becomes a two-region plan;
- recurrent models become checkpoint/value-state plans rather than fake token KV;
- tiering and remote routing become next-touch placement decisions.

The key metric is Retention Amplification: physical resident bytes divided by
semantically live bytes. Bandwidth and TTFT remain required constraints, but a
faster copy engine alone is not the contribution.
