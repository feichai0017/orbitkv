# OrbitKV

OrbitKV is a Rust attention-state compiler and the sole KV lifecycle authority
inside a modular inference engine. The primary product is being reorganized
around a forked Luminal compiler/executor and a thin Rust server. The previous
SGLang integration remains under `compat/` as a regression and migration path,
not as the architecture owner.

The product boundary is intentionally asymmetric:

| Owner | Responsibilities |
| --- | --- |
| OrbitKV Rust `RuntimeSession` | Compiled retention plan, request and snapshot identity, physical-page selection and generations, Prefix/COW decisions, semantic and execution frontiers, retirement, acknowledgement, and safe reclamation |
| OrbitKV Luminal executor | Tensor-graph compilation, attention/kernel selection, CUDA Graph execution, and consumption of OrbitKV-authored physical metadata |
| OrbitKV server | API, tokenization, continuous-batch intent, sampling, cancellation, and backpressure; it cannot allocate or recycle KV pages |
| SGLang compatibility path | Existing correctness/performance baseline and temporary serving integration while the Rust-native path is completed |

The server emits request intent, OrbitKV emits the physical state plan, and
Luminal executes that plan. None may keep a second authoritative KV table. The
compatibility bridge follows the same rule and fails closed for unsupported
profiles.

## Product layout

```text
orbitkv/
├── core/                    Rust compiler, RuntimeSession, and typed C wire
├── executor/
│   ├── src/                 OrbitKV-to-executor plan lowering
│   └── luminal/             complete OrbitKV Luminal fork (Git submodule)
├── server/                  engine-neutral Rust control-plane contract
├── compat/
│   ├── profile.json         legacy SGLang product/source contract
│   ├── assemble.py          immutable compatibility assembler
│   └── sglang/              pinned source, bridge, tools, tests, and overlay
├── docs/                    active product documentation
├── tests/                   repository-level verifier tests
├── tools/                   read-only product/evidence verifiers
└── results/                 append-only evidence, never active source
```

`compat/sglang/assemble.py` combines one verified pinned checkout, the reviewed
overlay, the SGLang bridge, and the Rust manager into a self-contained source
snapshot. It preserves every upstream tracked path; OrbitKV does not construct
a reduced engine by selecting a subset of SGLang modules. The bridge declares
no package dependency on `sglang`, so assembly cannot silently introduce a
second SGLang distribution.

```bash
python compat/sglang/assemble.py --output /path/to/orbitkv-engine-source
```

The default input is `compat/sglang/source`. A different `--sglang-root` is
accepted only when it satisfies the same pinned source contract. See
[Engine source overlay](compat/README.md).

## From semantics to safe reuse

OrbitKV compiles attention-state semantics into physical address and retirement
programs. Ring layouts, append-only regions, Prefix sharing, and resettable
arenas are derived plans rather than separate products.

Every reusable page must cross two independent frontiers:

1. The **Semantic Frontier** proves that the admitted attention semantics can no
   longer read the old state.
2. The **Execution Frontier** proves that the executor's earlier CUDA work has
   completed.

Detach alone is not reuse. The executor must apply exact device metadata
cleanup, and Rust must validate ordered completion, retirement evidence, and
ACK before recycling a page generation. The Luminal path is intended to record
completion in the same Rust-owned CUDA stream as the model graph. The
compatibility path still uses its recorded SGLang event protocol.

## Runtime artifacts and admission

`RuntimeManifest` is the canonical compiler artifact. It carries a declarative
attention-state or retention source, its derived physical plan, capability
requirements, and a stable fingerprint. `RuntimeTarget` and `RuntimeBinding`
currently preserve the qualified SGLang compatibility contract. The
engine-neutral `ExecutorPlan` is the first Luminal consumer of the same
canonical manifest; serialized Luminal target admission remains migration work.

The current typed boundary is `WIRE_VERSION = 14`: 48 exported typed symbols
and 78 frozen ctypes layouts. The header, Rust library, Python loader, packaged
target, binding, and capability verifier must agree before the bridge calls an
operational symbol.

Static admission checks topology, class order, address/retirement pairing,
storage geometry, layer coverage, cache-sharing policy, and required wire
compatibility. The SGLang bridge still performs dynamic checks for the loaded
model, tensors, backend, device, and execution mode before allocation or
mutation.

Example compiler flow:

```bash
cargo run --locked --manifest-path core/Cargo.toml --bin orbitkv -- \
  compile-runtime-manifest \
  core/examples/hybrid-attention-state-plan.json \
  > runtime-manifest.json

cargo run --locked --manifest-path core/Cargo.toml --bin orbitkv -- \
  bind-runtime-manifest runtime-manifest.json \
  > runtime-binding.json
```

The binding is an admission proof, not execution evidence and not a performance
result.

## RuntimeSession profiles

The qualified compatibility route is:

```text
RuntimeManifest
  -> RuntimeBinding and dynamic SGLang admission
  -> Rust RuntimeSession (sole KV lifecycle/page-selection authority)
  -> typed session wire
  -> SGLang bridge and reviewed overlay seams
  -> checked SGLang tensor and mirror effects
```

The new primary route is source- and host-contract tested but not yet
accelerator-qualified:

```text
RuntimeManifest -> ExecutorPlan -> RuntimeSession physical plan
  -> forked Luminal paged-attention inputs -> CUDA execution
  -> completion -> OrbitKV publication and safe reuse
```

The source contract currently describes five narrow, eager, non-overlap,
single-device profiles. Their evidence levels are deliberately separate:

| Profile | Cache policy | Current evidence boundary |
| --- | --- | --- |
| Whole-domain Full token KV | Shared page-aligned Prefix | Host lifecycle coverage. The latest exact-source accelerator record predates the live wire and establishes output correctness plus clean drain only for its recorded closure. Timing shows overhead, not a benefit. Live-wire device qualification is pending. |
| Ordered whole-domain Full+Sliding token KV | Shared page-aligned Prefix and checked Full-to-Sliding LUT | Host-tested. A current-wire real-device diagnostic on a released checkpoint passed paired token correctness, native current-stream event ordering, Sliding retirement/reuse, and clean drain. Its three-epoch throughput gate failed and a later exact-floor correction prevents sealing it as current-HEAD qualification. Capacity remains pending. |
| Whole-domain pure Sliding token KV | Request-private; Prefix APIs rejected | Implemented and host-tested. Current real-device qualification, including observed Sliding retirement/reuse evidence, is still pending. |
| Exact whole-domain Chunked token KV | Request-private; Prefix APIs rejected | Narrow resettable-arena lifecycle is host-tested. Host admission checks do not observe actual kernel or scheduler execution; real-device qualification is pending. |
| Whole-domain Full latent KV | Request-private; Prefix APIs rejected | Manifest, geometry, lifecycle, cleanup, and release are host-tested. Device and engine-E2E qualification are pending. |

Shipping the complete SGLang source does not widen this table. Other upstream
features remain present, but unsupported OrbitKV manager topologies fail closed.

## Prefix, copy-on-write, and release

Full and ordered Full+Sliding sessions can publish and attach page-aligned
Prefix snapshots. A Prefix node contains token/digest/LRU indexing plus an
opaque native lease; it never owns a page allocator. Extending a shared or
pinned partial tail produces checked copy-on-write effects, and Rust publishes
the new root only after the bridge proves that required copies completed before
new writes.

Ordinary release is ordered: wait for submitted forward work, prepare the native
release, validate detached pages and mirror effects, clear only the proven-dead
SGLang mappings, synchronize that cleanup, confirm exact retirement receipts,
ACK and recycle in Rust, then release the SGLang request row.

## Token relocation migration

Token-relocation core logic and an SGLang CUDA copy path exist, but relocation
integration is being migrated onto the now-available opaque `RuntimeSession`
wire operations. The current product claim is host-verified migration state
only. An independent CUDA component-conformance harness does not establish
native-session engine E2E, capacity, memory-saving, performance, or production
qualification. Relocation must not be presented as a qualified product path
until the live SGLang route and evidence gates close together.

## Evidence boundary

The latest published Full/shared-Prefix accelerator record establishes output
correctness, forward-stream event observation, and clean final drain for its
exact earlier-wire source and workload closure. Its manager timings are slower
than stock, so OrbitKV makes no speedup claim. Full retention also offers no
bounded-retention memory advantage for that workload. It is provenance, not a
live-wire qualification; exact details remain in the
[Results Index](results/README.md).

That scope does not transfer to Full+Sliding, pure Sliding, Chunked, latent KV,
relocation, fixed state, overlap, CUDA Graphs, distributed execution, capacity,
memory savings, or production readiness. Historical records retain their exact
device, model, source, and protocol identities only as provenance in the
[Results Index](results/README.md).

A separate current-wire Full+Sliding accelerator diagnostic is also retained in the
Results Index. It passed paired correctness and stream/event/reuse checks, but
the frozen throughput gate reported a 3.92% paired median regression and a
5.28% bootstrap upper regression. A subsequent exact-floor formula correction
means the record is diagnostic rather than a sealed current-HEAD qualification;
it establishes no speedup or capacity benefit.

## Removed product paths

The former `orbitkv-runtime` wheel, `orbitkv-reference` adapter, optional
`structured-data-plane` route, generic adapter SPI, and second-engine story are
removed history. They are not installable components or current capabilities.
Append-only results may still name them because a record must preserve the
source closure it measured.

## Documentation

- [Architecture](docs/architecture.md) — primary core/executor/server ownership
  and migration boundary.
- [Capability Matrix](docs/capability-matrix.md) — normative implemented and
  qualified boundary.
- [SGLang Integration Contract](docs/sglang-compatibility.md) — reviewed overlay,
  bridge, mirror, and completion responsibilities.
- [RuntimeSession Architecture](docs/runtime-session.md) —
  native ownership and transaction model.
- [State Lifetime and Reclamation](docs/state-lifecycle.md)
  — frontiers, retirement, and heterogeneous-state constraints.
- [Implementation and Qualification Roadmap](docs/roadmap.md)
  — remaining gates without promoting planned work into current capability.

## Build and verify

```bash
cargo test --workspace
python tools/verify_active_source.py
python tools/verify_engine_profile.py
python tools/verify_capability_matrix.py
pytest -q compat/sglang/tests tests
```

Hardware-dependent qualification is separate from these host and source gates.
The [Capability Matrix](docs/capability-matrix.md) is authoritative when code,
historical records, and roadmap text have different scopes.
