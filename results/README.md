# Current evidence

This directory contains only compact evidence that directly qualifies the
current `core + Luminal executor + Rust server` architecture. A record keeps
the concrete model, hardware/software environment, measured result, source
identity, and claim boundary. It must not embed source trees, binaries, package
caches, model weights, or superseded integration snapshots.

Removed records remain recoverable from Git history. They are not current
product evidence. New experiments should first write to `.qualification/` and
move a compact reviewed result here only after its checks pass.

## Retained records

| Record | Evidence | Boundary |
| --- | --- | --- |
| `bucketed-decoder-correctness-20260904` | Released dense checkpoint executes prefill and repeated decode through one searched two-bucket Luminal runtime and one persistent OrbitKV K/V arena | Correctness only; tiny batch-one workload |
| `on-device-greedy-correctness-20260904` | Device argmax matches host argmax and the default path returns token IDs rather than vocabulary logits | Greedy only; no sampling-performance claim |
| `decode-cuda-graph-correctness-20260905` | Stable-input outer-graph replay is correct; flattened capture is 24.9% slower than eager | Retained negative result; no benefit claim |
| `decode-child-graph-benefit-20260905` | Selected Luminal executables composed as child graphs reduce matched fixed-step batch-one median wall time by 5.8-8.3% in two runs | Narrow dispatch result; not serving throughput or lifecycle benefit |

The normative current support boundary is the
[Capability Matrix](../docs/capability-matrix.md). No retained record proves
that compiled lifetime management improves admission capacity, tail latency,
memory use, or end-to-end throughput. That is the next L5 gate.

## Result-package policy

A promoted result directory contains only:

- `environment.json`: hardware, driver/runtime, model and weight identity,
  dependency versions, and source revisions;
- raw benchmark JSON emitted by the common client;
- `summary.json`: paired metrics, correctness gate, and qualified claims;
- optional checksums for those files.

The matched serving harness and promotion rules are documented in
[benchmarking.md](../docs/benchmarking.md).
