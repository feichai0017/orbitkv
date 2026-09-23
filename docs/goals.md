# Overview

OrbitKV is an external KV cache for vLLM and SGLang. It retains reusable state
in pinned DRAM and optional SSD so requests can recover prefixes that no longer
fit in GPU memory. Run one Cache Manager per inference host and enable the
engine adapter; begin with the [single-node quickstart](single-node.md).

## What you can use today

| Capability | Scope |
| --- | --- |
| Prefix reuse | DRAM/SSD recovery after GPU eviction or engine restart, with the Manager kept alive |
| Engine integration | vLLM 0.29.0 and SGLang 0.5.20, direct GPU transfers through CUDA IPC |
| Hybrid recovery | Compiled prefix/window/checkpoint requirements for [supported layouts](hybrid-recovery.md) |
| Resource control | Byte budgets for reads, leased results and GPU transfers; cancellation and completion fences |
| Observability | Prometheus metrics, optional request timelines and reproducible Qwen3-8B workloads |
| Request preparation | Opt-in dense-layout lookahead; ready pages remain budgeted until consumed, cancelled or expired |

Use OrbitKV for repeated documents, shared system prompts and conversation
prefixes that extend beyond HBM capacity. A cold workload with little reuse can
pay extra copy/storage costs; use the [measurements](single-node-performance.md)
and your own traffic to choose capacities and policies.

## Ownership and compatibility

The inference engine owns HBM allocation and scheduling. OrbitKV owns external
replicas and retains source/destination holds through transfer completion.
Cache keys bind model artifacts, computation and byte layout. Compiled recovery
validates the requirements declared by the adapter; it does not analyze an
arbitrary model graph or predict future tokens. A shared API does not imply
that different engines' KV bytes are interchangeable.

## Experimental and planned work

The embedded catalog, etcd membership and Mooncake transfer path support
experimental remote-cache development. Real two-host serving qualification,
catalog replication, broader parallelism and P/D with cache reuse have separate
gates. A future KV-aware router can use cache location and transfer costs without
moving inference scheduling into the Manager.

General lifetime analysis, page-generation enforcement and joint
retention/placement planning remain open. See [deployment support](deployment.md),
[architecture](architecture.md), and [the roadmap](roadmap.md).
