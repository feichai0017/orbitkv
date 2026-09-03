# SGLang Integration Contract

This document defines the only live engine integration: the complete pinned
SGLang source product, its reviewed OrbitKV overlay, and the packaged
`orbitkv-sglang` bridge. The filename is retained as a stable documentation
link; it no longer describes a generic multi-engine adapter SPI.

## Product scope

The deployable product is not a standalone KV service and not an alternative
inference engine. Assembly preserves the complete tracked SGLang source tree
and applies a closed, reviewed set of lifecycle seams. The bridge has no
package dependency on `sglang`; deployment installs SGLang from the verified
source product rather than resolving a second copy.

The former `orbitkv-runtime`, `orbitkv-reference`, optional
`structured-data-plane`, and generic second-engine adapter contracts are
removed history. They are not live packages, optional product modes, or
qualification targets. Historical result archives may retain their names as
provenance.

## Authority boundary

| Component | Owns | Must not own |
| --- | --- | --- |
| SGLang | Serving, scheduling, request protocol, model execution, tensors, attention/copy kernels, CUDA streams and events | OrbitKV page generations, native request/snapshot identity, or reclamation decisions |
| Rust `RuntimeSession` | Accepted plan, lifecycle state, physical-page selection, snapshots, Prefix/COW state, frontiers, retirement, ACK, and reuse | Scheduler policy, tensor allocation, kernel launch, or CUDA event creation |
| SGLang bridge | Admission, identity correlation, checked tensor/table effects, event provenance, and error containment | Independent allocation, shadow page ownership, or fallback lifecycle |

One native `RuntimeSession` owns one `CanonicalKvManager`. On the admitted
SGLang lifecycle route, manager request, snapshot, page, Prefix, step,
reclamation, and relocation capabilities remain private to that session; the
route uses engine identities, physical effects, session-scoped operation IDs,
and evidence needed to confirm those effects. The raw manager C ABI has been
removed; it is not a second live SGLang lifecycle route.

Python `SessionRuntime` is a coordinator around the opaque native handle. It
may keep the stable SGLang request key, the corresponding engine request ID and
positive `ReqToToken` row, submitted tickets, CUDA events, and monotonic
completion high-water marks. It must never allocate a page independently or
infer reuse from a local mirror.

## Product flow

```text
complete pinned SGLang source + reviewed overlay
  -> RuntimeManifest
  -> RuntimeTarget admission and RuntimeBinding
  -> dynamic engine/tensor/backend checks
  -> Rust RuntimeSession
  -> typed session effects
  -> reviewed SGLang lifecycle seams
  -> checked tensor and mirror mutations
```

`RuntimeManifest` describes the compiler-derived state contract.
`RuntimeTarget` describes the packaged SGLang execution boundary.
`RuntimeBinding` binds their exact fingerprints and structural execution
signature. The product CLI and Rust admission API always use that packaged
target; there is no external target-file or alternate-engine selection. None
of these artifacts proves that a request executed correctly; runtime checks
and evidence remain separate.

The live typed boundary is `WIRE_VERSION = 14`, with 48 exported typed symbols
and 78 frozen ctypes layouts. The C header, Rust library, Python loader, target,
binding, and capability verifier must agree exactly before the bridge resolves
operational symbols.

The product profile is closed. A topology not named by the source contract, or
one whose runtime geometry differs from the binding, fails before mutation.
There is no generic adapter fallback.

## Reviewed source seams

The overlay classifies its changes by authority:

- manager takeover points delegate KV configuration, allocator construction,
  extend/decode placement, Sliding eviction, and release decisions;
- lifecycle notifications connect waiting-request cleanup and scheduler batch
  boundaries to the native session;
- observability exposes OrbitKV state without transferring scheduler ownership;
  and
- the local-attention page-ID change is a compatibility fix, not lifecycle
  authority.

Around-wrappers may call preserved SGLang implementations. That preserves
SGLang scheduler and execution semantics; it does not make the wrapper their
owner. The exact reviewed path inventory lives in `compat/sglang/profile.json`.

## Admission and initialization

Initialization must complete in this order:

1. verify the pinned SGLang base or exact reviewed patched checkout;
2. validate `RuntimeManifest`, `RuntimeTarget`, and `RuntimeBinding`;
3. check loaded wire compatibility before resolving operational symbols;
4. validate the exact engine topology, class order, storage, geometry, backend,
   execution mode, and cache policy;
5. construct SGLang tensors and request tables;
6. register those arenas with one native `RuntimeSession`; and
7. transfer the unique native handle to `SessionRuntime`.

Any partial initialization must close its unique handle. Unsupported or
ambiguous state must not silently select a legacy manager.

## Checked mirrors

Every live request has a checked correspondence:

```text
SGLang request key <-> session-scoped engine request ID <-> ReqToToken row
```

The native session selects physical bindings. The bridge validates them and
updates SGLang's `ReqToToken` table. For the ordered Full+Sliding profile it
also maintains SGLang's `full_to_swa_index_mapping`, referred to here as the
Full-to-Sliding LUT. These tables are device execution inputs, but they are not
allocators or sources of lifecycle truth.

Before the first write, the bridge validates the complete batch of effects:
request identity, class order, pool and page range, generation, logical-token
coverage, destination uniqueness, table row, and applicable LUT mapping. A
partial or uncertain device mutation poisons the route unless the protocol has
an explicit safe abort state.

## Prefix and request-private cache policies

Whole-domain Full and ordered Full+Sliding token KV use shared, page-aligned
Prefix operations. The SGLang cache is an index over tokens, digests, LRU state,
and opaque native Prefix leases. Lookup, publication, attach, eviction, request
fork, and atomic publish-and-release remain native controls.

Pure Sliding, exact Chunked, and Full latent KV are request-private. Their
session creation declares that policy, shared Radix caching is disabled, and
every Prefix wire operation fails closed. Cache policy is independent of the
fact that all profiles use the same native lifecycle route.

## Completion provenance

Semantic death and GPU completion are independent facts. When an eager forward
returns, the integration records a CUDA event on SGLang's current forward
stream and associates it with the exact submitted session ticket. The bridge
queries or waits for that event before asserting completion to Rust.

Rust checks that the assertion belongs to the correct session and batch and
that each completion domain advances monotonically. It does not authenticate
the CUDA event. Event creation, stream provenance, and observation are SGLang
integration responsibilities and must be covered by engine evidence.

## Release and reuse

Request release is a protocol, not a boolean callback:

1. wait for all submitted work for the request;
2. ask the native session to prepare release;
3. validate the exact detached pages and required mirror cleanup;
4. clear only those `ReqToToken` and Full-to-Sliding mappings;
5. synchronize or otherwise prove cleanup completion;
6. confirm ordered retirement receipts to Rust;
7. ACK and recycle the native state; and
8. only then release the SGLang request row.

Atomic Prefix publish-and-release transfers the request reference to the Prefix
reference before recycling the request. Request-private profiles reject that
operation. No page generation or request row may be reused before the exact
cleanup and native ACK complete.

## Relocation migration boundary

Relocation has Rust transaction logic, opaque session wire operations, SGLang
bridge code for CUDA copies, and an independent device component-conformance
harness. Product integration is still in migration. Therefore the live product
claim is host-verified migration state only: there is no native-session engine-
E2E, capacity, memory-saving, performance, or production qualification.

The removed raw-manager relocation ABI and archived engine results do not
authorize a second lifecycle path in the current product. The migration is complete only
when the live SGLang relocation integration uses the available opaque session
operations and current-source evidence verifies copy, mirror publication,
completion, retirement ACK, generation reuse, and final drain together.

## Current qualification boundary

- The latest Full/shared-Prefix accelerator record predates the live wire. It
  establishes output correctness and clean drain for its exact recorded closure,
  and its timing shows overhead. Live-wire device qualification, performance,
  and production readiness remain pending.
- Ordered Full+Sliding and pure Sliding are implemented and host-tested, but
  their current real-device qualification is pending.
- Exact Chunked and Full latent KV also remain host-only at the product level.
- The relocation CUDA harness is component evidence, not product E2E.

See the [Capability Matrix](capability-matrix.md) for the normative boundary and
the [Results Index](../results/README.md) for immutable provenance.

## Verification

From the repository root, the source and host integration gates include:

```bash
python tools/verify_active_source.py
python tools/verify_engine_profile.py
python tools/verify_capability_matrix.py
pytest -q compat/sglang/tests tests
```

Passing these gates does not substitute for a profile-specific real-device
qualification run.
