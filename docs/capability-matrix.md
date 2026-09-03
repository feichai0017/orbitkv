# Capability Matrix

This matrix is the normative boundary for the live source tree. OrbitKV is one
product: a complete pinned SGLang source tree, a reviewed lifecycle overlay, the
packaged SGLang bridge, and a Rust `RuntimeSession`. SGLang owns scheduling,
model execution, tensors, kernels, and CUDA execution. OrbitKV owns the admitted
KV lifecycle, physical-page selection, frontiers, retirement, acknowledgement,
and safe reuse.

Host implementation, device correctness, measured benefit, and production
readiness are separate claims. An archived result qualifies only its exact
source and workload closure; it does not promote the current runtime. Historical
identities and immutable records are routed through the
[Results Index](../results/README.md). `qualification_runner.py` produces
observations; an independent verifier evaluates them against explicit gates.

The pinned source contract and product profile define the only live engine
assembly. The removed `orbitkv-runtime`/`orbitkv-reference` packages, structured data
plane, generic adapter SPI, and second-engine route are not current capabilities.

## Levels

| Level | Meaning |
| --- | --- |
| L1 Compiler | Declarative semantics parse and lower into checked plans. |
| L2 Host/ABI | Native, wire, bridge, and failure contracts pass host gates. |
| L3 GPU Primitive | An isolated device primitive passes exact-source conformance. |
| L4 Engine E2E | A pinned engine and released checkpoint pass scoped end-to-end gates. |
| L5 Production | Pressure, cancellation, feature combinations, benefit, and release matrices are qualified. |

## Single product boundary

| Capability | Current level | Exact boundary | Evidence |
| --- | --- | --- | --- |
| Complete pinned SGLang source product | L2 source/host | `compat/sglang/assemble.py` preserves every upstream tracked path and combines one verified source origin with the reviewed overlay, `orbitkv-sglang` bridge, and Rust manager. The bridge does not resolve a second SGLang distribution. Complete-source preservation does not imply that every SGLang topology is managed by OrbitKV | `compat/sglang/profile.json`, passing source-contract/assembly verifier, and engine-owner host tests |
| Ownership split | L2 host | SGLang owns scheduling, model execution, tensor allocation, kernels, serving, and CUDA streams/events. One Rust `RuntimeSession` exclusively owns its canonical manager, session identities, physical-page selection, snapshots, Prefix/COW decisions, semantic and execution frontiers, retirement, ACK, and reuse. On the admitted SGLang route, Python keeps only SGLang keys/rows, session tickets, events, completion high-water marks, and checked mirrors. The raw manager C ABI is removed | native session tests, session FFI tests, SGLang lifecycle/ownership tests |
| Closed product admission | L2 host | Unsupported topology, geometry, class order, cache policy, backend, execution mode, or wire contract fails before mutation. There is no generic adapter or raw-manager fallback for an admitted product session | source profile, runtime admission tests, failure-path tests |
| Removed alternate data-plane products | Removed history | The former neutral runtime/reference wheels, optional structured engine bridge, and generic adapter SPI are absent from the live product. Names may remain only in append-only evidence provenance or an explicit removed-history note | active-source path/package scans and packaging tests |

## Compiler and admission

| Capability | Current level | Exact boundary | Evidence |
| --- | --- | --- | --- |
| Retention and attention-state compiler | L1 GO | Compiles Full, Sliding, exact Chunked, latent/component, and fixed-state semantics into checked address and retirement programs while preserving class order, domains, and component geometry | Rust compiler and property tests |
| `RuntimeManifest` | L1 compiler + L2 load | Canonical executable compiler artifact with one tagged declarative source, derived physical plan, capability requirements, and stable fingerprint. It is an admission input, not an execution or benefit result | Rust/Python cross-language and CLI tests |
| `RuntimeTarget` | L2 static contract | Packages SGLang executor identity, supported manifest forms, page geometry, capabilities, required wire, and closed topology profiles | Rust/Python target tests and packaged target resource |
| `RuntimeBinding` | L2 static admission | Binds one exact manifest fingerprint to one exact target fingerprint and complete structural execution signature. Device, loaded model/tensor, backend, and execution checks remain dynamic | binding and pre-allocation tests |
| Retention-IR source | L1 compiler | Preserves explicit head ranges, block domains, `Pinned`, `PeriodicFrom`, and `ResettableArena` semantics. Only the exact admitted Chunked shape currently crosses the product runtime; broader shapes fail closed | compiler, manifest, and admission tests |
| Fixed-state compiler/checkpoint core | L1 compiler + L2 host component | Recurrent/convolution geometry and generation-checked checkpoint transactions exist at host level. They are not one atomic lifecycle with token state and are not an admitted current native-session product profile | compiler and checkpoint-pool host tests |

## Native RuntimeSession profiles

| Profile | Lifecycle/cache policy | Current qualification |
| --- | --- | --- |
| Whole-domain Full token KV | Native session; shared page-aligned Prefix, request fork, COW, atomic publish-and-release | Host lifecycle coverage. The latest accelerator record predates the live wire and establishes exact-source output correctness, forward-stream event observation, and final drain only for its recorded closure. It is unsealed and its descriptive timing shows overhead, not a speed, capacity, memory, production, or general-replacement benefit. Live-wire device qualification is pending |
| Ordered whole-domain Full+Sliding token KV | Native two-class session; shared Prefix and joint COW; checked `ReqToToken` and Full-to-Sliding LUT effects | Host-tested. A current-wire real-device diagnostic on a released checkpoint completed three paired epochs: token correctness and native stream/event/SWA-reuse gates passed; throughput failed with 3.92% paired median regression and 5.28% bootstrap upper regression. A later exact-floor correction invalidates its capacity metadata for current HEAD, so it is unsealed diagnostic evidence, not a capacity, speedup, L4-seal, or production claim |
| Whole-domain pure Sliding token KV | Native request-private periodic session; every Prefix operation rejected; no Full-to-Sliding LUT | Implemented and host-tested. Current real-device correctness and page-retirement/reuse qualification are pending |
| Exact whole-domain Chunked executor | Native request-private resettable session using canonical Retention IR; every unsupported shape is fail-closed | These host checks do not observe real kernel or scheduler execution. No real-accelerator, released-model, performance, or complete-engine qualification |
| Whole-domain Full latent KV | Native request-private component-aware session; every Prefix operation rejected | Manifest, geometry, lifecycle, mirror cleanup, release, and drain are host-tested. Device, engine-E2E, broader precision/profile, relocation, performance, and production qualification are pending |

All five profiles use the single `RuntimeSession` route and eager, non-overlap,
single-device execution. The current source product may carry other SGLang
features, but an unsupported OrbitKV manager profile fails closed.

## Lifecycle capabilities

| Capability | Current level | Exact boundary | Evidence |
| --- | --- | --- | --- |
| Identity and arena ownership | L2 GO | Generation-checked request, snapshot, page, step, submission, Prefix, control, release, reclamation, and session operation identities; independent class arenas | Rust lifecycle, stale-ID, and property tests |
| Persistent snapshots | L2 GO | Immutable class roots, expected-head checks, incremental path copy, reader pins, and stale-root rejection | Rust host/property tests |
| Append transaction | L2 GO | Acquire/fork, prepare, checked copy/write effects, submit, completion, publication, abort-before-observation, and quarantine/fail-stop after uncertainty | Rust, wire, Python, and SGLang host tests |
| Prefix lifecycle | L2 GO for Full and ordered Full+Sliding | Page-aligned lookup, publish, attach, eviction, fork, joint COW, and atomic publish-and-release. Cache nodes carry opaque native leases, not allocator state. Request-private profiles reject every Prefix operation | native session and SGLang cache/fault tests |
| Checked engine mirrors | L2 host | One checked `SGLang key <-> engine request ID <-> ReqToToken row` relation per live request; ordered Full+Sliding additionally maintains SGLang's Full-to-Sliding LUT. Mirrors never select pages independently | lowering, row, LUT, cleanup, and hostile-output tests |
| Dual-frontier reclamation | L2 GO | Semantic death and GPU execution completion are independent. Exact mirror cleanup and ordered retirement ACK are required before page-generation or request-row reuse | native session reclamation/release and bridge completion tests |
| SGLang event provenance | L2 host; current-wire Full+Sliding device diagnostic | SGLang records an event on its current forward stream for the submitted ticket; Python observes it and advances a monotonic completion assertion. Rust validates ownership and ordering, not the external CUDA fence itself | host event/fault tests plus the unsealed current-wire Full+Sliding accelerator diagnostic; other profiles and a sealed L4 closure remain pending |
| Request-private pressure telemetry | L2 host | Separates consumed capacity, resident, request-reachable, semantic-live, free, and high-water values where supported. It is not a capacity or memory-saving result | host pressure tests |

## Token relocation migration

| Capability | Current level | Exact boundary | Evidence |
| --- | --- | --- | --- |
| Relocation compiler/core transaction | L2 host | Canonical token views, explicit semantic/policy dispositions, collective prepare/copy/submit/complete/publication/ACK, exact receipts, quarantine, and generation-safe reuse exist behind opaque `RuntimeSession` wire operations. Product integration is still migrating to that sole route | Rust and Python host transaction/fault tests |
| SGLang CUDA relocation copy | Implemented component; product qualification pending | Device copy code and scheduling integration exist, but code existence is not a current native-session engine-E2E claim | bridge tests and independent CUDA component harness |
| Current product relocation | Host-verified migration state | The opaque session wire exists; the live SGLang relocation integration is still migrating to it. No current engine seal, capacity, memory saving, performance, or production qualification | current host gates; archived evidence does not transfer |
| Independent CUDA relocation harness | L3 component conformance only | Exercises payload, stream ordering, evacuation, ACK-gated generation reuse, and drain outside a current product E2E qualification | component harness; not L4, capacity, performance, or production evidence |

Relocation is collective but not end-to-end rollback-capable. After disposition
state commits, a later observed or uncertain failure is quarantined or
fail-stopped; it does not restore an invented earlier request head.

## Current typed wire

The live C header, Rust library, Python loader, packaged target, and this matrix
must agree exactly before operational symbols are used. During a wire migration,
the code and verifier are authoritative; no document may advertise a partially
landed surface.

| Capability | Current level | Exact boundary |
| --- | --- | --- |
| Current typed C wire | L2 GO | Exactly 48 typed symbols at `WIRE_VERSION = 14`; the library, header, Python loader, `RuntimeTarget.required_wire_version`, and `RuntimeBinding.required_wire_version` must agree. The packaged target stores `id = "sglang"`, `contract_version = 4`, and `required_wire_version = 14` |
| Current Python FFI/runtime | L2 GO | The SGLang bridge validates the wire before binding symbols and freezes exactly 78 ctypes layouts. Session APIs expose engine/session identities and effects while keeping manager capabilities private |

### Exact current C wire surface

```text
orbitkv_session_abort_control
orbitkv_session_abort_prepared
orbitkv_session_abort_prepared_relocation
orbitkv_session_acquire_requests
orbitkv_session_arena_identities
orbitkv_session_arena_stats
orbitkv_session_cancel_pending_attach
orbitkv_session_commit_control
orbitkv_session_complete_execution
orbitkv_session_complete_relocation
orbitkv_session_confirm_control
orbitkv_session_confirm_publication
orbitkv_session_confirm_release
orbitkv_session_confirm_relocation_publication
orbitkv_session_create
orbitkv_session_destroy
orbitkv_session_finalize_pending_attach_cancel
orbitkv_session_mark_token_dispositions_batch
orbitkv_session_prefix_lookup_batch
orbitkv_session_prefix_publish_batch
orbitkv_session_prefix_publish_release_batch
orbitkv_session_prepare_append
orbitkv_session_prepare_prefix_attach
orbitkv_session_prepare_prefix_evict
orbitkv_session_prepare_release
orbitkv_session_prepare_relocation_batch
orbitkv_session_prepare_request_fork
orbitkv_session_quarantine_control
orbitkv_session_quarantine_prepared
orbitkv_session_quarantine_relocation
orbitkv_session_quarantine_submitted
orbitkv_session_read_control_plan
orbitkv_session_stats
orbitkv_session_submit_execution
orbitkv_session_submit_relocation
orbitkv_session_token_views_batch
orbitkv_state_pool_abort_batch
orbitkv_state_pool_acknowledge_batch
orbitkv_state_pool_complete_batch
orbitkv_state_pool_create
orbitkv_state_pool_current_batch
orbitkv_state_pool_destroy
orbitkv_state_pool_identity
orbitkv_state_pool_prepare_batch
orbitkv_state_pool_retire_owners_batch
orbitkv_state_pool_stats
orbitkv_state_pool_submit_batch
orbitkv_wire_version
```

## Evidence boundary

The latest published Full engine records establish output correctness, real
forward-stream event observation, and zero final drain for their exact earlier-
wire Full/shared-Prefix closures. They are unsealed. Manager timing shows
overhead and `speedup_qualified=false`; no performance, capacity, memory-saving,
production, or general-replacement claim is made. Full attention itself
provides no bounded-retention memory advantage on those workloads. Exact device,
model, source, and protocol provenance lives in the
[Results Index](../results/README.md). The records do not qualify the live wire.

The earlier-wire Full record does not qualify ordered Full+Sliding. The new
current-wire Full+Sliding diagnostic independently establishes paired
correctness and current-stream/SWA lifecycle observation for its exact closure,
but fails throughput and cannot seal after the exact-floor correction. Neither
record qualifies pure Sliding, exact Chunked, Full latent KV, relocation, fixed
state, overlap, graph replay, capacity, or distributed execution.

## Not qualified

- sealed L4 or L5 qualification for any current profile;
- sealed current-HEAD Full+Sliding qualification or any pure Sliding
  real-device correctness and safe retirement/reuse evidence;
- current device/engine qualification for exact Chunked or Full latent KV;
- current native-session relocation E2E, capacity, memory-saving, performance,
  or production behavior;
- atomic token-plus-fixed-state lifecycle;
- packed Prefix; broader head/domain/Chunked/latent/fixed-state profiles;
- asynchronous overlap, multiple completion domains, CUDA Graph replay,
  speculation, beam search, distributed or remote state;
- a same-capacity memory reduction or general latency/throughput improvement; or
- a general takeover of every SGLang KV topology.

Unsupported profiles and unproved combinations must fail closed before
mutation. Implementation and qualification order is tracked in the
[Implementation and Qualification Roadmap](roadmap.md).
