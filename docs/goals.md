# Goals and boundaries

OrbitKV provides reusable KV replicas outside framework-owned HBM. The
single-node path prioritizes exact recovery, bounded leases, predictable GPU
to pinned-DRAM/SSD movement, and observability. The multi-node path should
offer the same cache API while Mooncake moves remote bytes and a recoverable
catalog identifies candidate owners. A later KV-aware router may use that
catalog together with queueing and transfer costs.

The long-term state planner must validate model, format, component coverage,
and page generation before claiming that a prefix can resume execution. The
SGLang path now uses `orbitkv-state` to validate registered identity, absolute
coverage and prefix/window/checkpoint requirements before restoring. Shared
vLLM validation, page-generation enforcement and model-independent planning
remain open. See [compiled hybrid recovery](hybrid-recovery.md).

OrbitKV does not own model execution, GPU allocation, or inference scheduling.
It is not a general RPC framework, network stack, or replacement for Mooncake
Transfer Engine. A future router decides where requests run; it does not make
the Cache Manager an inference server. See [architecture](architecture.md) and
[roadmap](roadmap.md) for current and planned responsibilities.
