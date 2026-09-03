# Implementation and Qualification Roadmap

This roadmap is for one product: the complete pinned SGLang source tree with the
OrbitKV overlay and Rust `RuntimeSession`. It separates code availability, host
verification, device correctness, measured benefit, and production readiness.
A future milestone is not a current capability.

The [Capability Matrix](capability-matrix.md) is normative. Historical evidence
in the [Results Index](../results/README.md) qualifies only the archived source
closure that produced it.

## Current checkpoint

The product architecture is established:

- SGLang owns serving, scheduling, model execution, tensors, kernels, and CUDA
  stream/event execution;
- Rust `RuntimeSession` is the sole lifecycle and physical-page-selection
  authority for an admitted profile;
- the SGLang bridge performs admission, checked tensor/table effects, event
  provenance, and error containment; and
- the complete pinned upstream source is assembled with a closed reviewed
  overlay rather than replaced by a reduced or second engine.

The former neutral runtime/reference packages, structured data plane, and
generic adapter SPI have been removed from the product. Roadmap work must not
reintroduce them as a compatibility route.

Five native-session shapes have host coverage. Their current qualification is
not uniform:

| Profile | Implementation state | Qualification state |
| --- | --- | --- |
| Full token KV with shared Prefix | Native session and SGLang integration | Latest earlier-wire accelerator record has exact-source output correctness, event observation, and final drain; unsealed timing shows overhead. Live-wire device, benefit, and production qualification are pending |
| Ordered Full+Sliding token KV | Native two-class session, Prefix/COW, mirrors, cleanup | Host-tested; current real-device correctness, Sliding retirement/reuse, capacity, and performance runs pending |
| Pure Sliding token KV | Native request-private periodic session | Host-tested; current real-device correctness and retirement/reuse runs pending |
| Exact Chunked token KV | Native request-private resettable session | Host-tested; actual kernel/scheduler execution and device qualification pending |
| Full latent KV | Native request-private component-aware session | Host-tested; device and engine-E2E qualification pending |

Token relocation has Rust transaction logic, opaque `RuntimeSession` wire
operations, and an SGLang CUDA copy path, but product integration is still
migrating to that single session route. Its current claim is host-verified
migration state. A separate CUDA component harness and archived records do not
establish current native-session E2E.

## Definition of done

OrbitKV becomes a qualified compiled attention-state manager only when all of
the following layers close for the claimed profile.

| Layer | Required outcome |
| --- | --- |
| Compiler | Declarative visibility/lifetime semantics lower deterministically to checked physical address and retirement programs |
| Admission | One exact manifest and SGLang target bind before allocation; unsupported topology, geometry, backend, or execution mode fails closed |
| Lifecycle | One Rust session owns identities, placement, frontiers, retirement, ACK, and generation reuse across every operation in scope |
| Engine effects | The bridge validates complete batches, applies exact SGLang tensor/table effects, and reports real event provenance |
| Correctness | Stock/manager outputs match for the same semantic policy, and lifecycle evidence proves exact cleanup and final drain |
| Benefit | Matched runs establish the claimed capacity, memory, latency, or throughput improvement without changing the semantic workload |
| Operations | Cancellation, pressure, long-running reuse, failures, supported feature combinations, and release/version matrices pass |

Host tests can close the compiler, admission, and many lifecycle invariants.
They cannot substitute for engine execution, device event provenance, or a
benefit measurement.

## R1: Preserve the single authority boundary

Status: **implemented; continuously enforced**.

The architecture gate is that every admitted KV lifecycle mutation enters one
native `RuntimeSession`. Python and SGLang retain only engine identities,
mirrors, events, and reconciliation state. Future features must extend the
session protocol instead of restoring raw manager handles or a parallel
allocator.

Exit gates:

- one unique native handle per session and deterministic teardown;
- no fallback lifecycle when session creation or admission fails;
- the admitted SGLang lifecycle route never receives manager capabilities;
- exact session/request/operation identity and stale-ID rejection; and
- assembly/source verifiers prove there is one complete pinned SGLang origin.

## R2: Stabilize the typed session wire

Status: **implemented for lifecycle and relocation controls; product relocation
integration remains in progress**.

The wire must remain an engine-facing effect protocol, not a public copy of the
internal manager API. Each extension needs:

- capability-free, fixed-layout records;
- exact symbol and layout inventory;
- bounded workspaces and hostile-output validation;
- explicit short-buffer, retry, fail-stop, and quarantine semantics;
- Rust header/library/Python parity; and
- version agreement among the library, target, binding, bridge, and verifier.

The current boundary is `WIRE_VERSION = 14`, with 48 exported typed symbols and
78 frozen ctypes layouts. The removed raw manager C ABI must not be
reintroduced as a second product route.

## R3: Qualify ordered Full+Sliding

Status: **implementation and host lifecycle complete; current device
qualification pending**.

The run must exercise the exact ordered class pair, shared Prefix behavior,
joint COW, class-specific placement, `ReqToToken`, the Full-to-Sliding LUT,
Sliding semantic retirement, execution completion, exact ACK, generation reuse,
and final drain.

Required evidence:

- pinned source, manifest, binding, checkpoint, backend, and workload identity;
- stock/manager output equivalence;
- real current-stream completion events;
- Sliding activity reported as applicable rather than silently zero/not
  applicable;
- page-generation trace connecting retirement, event completion, ACK, and later
  reuse;
- shared-Prefix and joint-COW lifecycle counters;
- capacity and memory census separated from configured tensor reservation; and
- matched latency/throughput distributions with no positive claim unless the
  predefined gate passes.

Host geometry tests or an archived older-source Hybrid record do not close this
milestone.

## R4: Qualify pure Sliding

Status: **implementation and host lifecycle complete; current device
qualification pending**.

Pure Sliding must remain request-private, with every Prefix operation rejected
and no Full-to-Sliding LUT. Its verifier must independently admit the exact
single Sliding class and bind its window, page geometry, periodic slot count,
retirement rule, cache policy, and disabled Radix state.

The evidence must distinguish:

- `not_applicable`: no Sliding class exists;
- `applicable` with zero activity: the workload has not crossed a retirement
  boundary; and
- `applicable` with observed activity: the workload crosses the boundary and
  records retirement, reclamation, wrap, ACK, and later generation reuse.

Aggregate free-page counts alone cannot prove safe reuse. The evidence needs a
per-page or equivalently strong trace tied to the Execution Frontier.

## R5: Complete opaque-session relocation

Status: **host-verified migration state; product E2E pending**.

The target route is one collective session transaction:

```text
canonical token views
  -> explicit disposition mark
  -> session prepare
  -> checked SGLang CUDA copies
  -> session submit and completion
  -> atomic engine mirror publication
  -> exact source retirement and ACK
  -> later generation reuse
```

Completion requires:

- opaque session relocation IDs and no leaked manager leases;
- exact Rust/C/Python layout and symbol parity;
- fixed session workspaces for queries, moves, evidence, and publications;
- complete-batch validation before the first tensor/table write;
- clear safe-abort versus quarantine/fail-stop boundaries;
- no raw-manager fallback used by the live SGLang path;
- real engine tests for batch ordering, mirror publication, event provenance,
  retirement, ACK, repeated generation reuse, and final drain; and
- matched capacity/performance experiments before any benefit statement.

The existing CUDA copy path proves that device movement code exists. The
component harness proves only its scoped component contract. Neither is a
current product E2E claim.

## R6: Qualify exact Chunked execution

Status: **narrow host lifecycle implemented; engine qualification pending**.

The current host validator checks the configured chunk-local backend and safe
scheduler indicators before pool allocation. A qualification must additionally
observe the actual loaded layers, kernel path, scheduler behavior, epoch
boundaries, old-epoch mirror cleanup, completion events, exact ACK, physical
generation reuse, and final drain under the released execution closure.

Broader sink+window, per-head, multi-domain, dynamic, or mixed chunking remains
outside this exact milestone and must fail closed.

## R7: Qualify latent and fixed state

Status: **host-only slices; unified lifecycle open**.

Full latent KV needs device validation of its exact component geometry,
attention backend, row addressing, cleanup, and output correctness. Relocation
requires a component-aware copy contract and cannot reuse ordinary K/V claims.

Recurrent/convolution checkpoints currently use an independent pool. A broader
hybrid product profile needs a single compiled commit group or an explicitly
proved containment protocol across token and fixed state, including
replacement triggers, failure injection, completion, mirror publication, and
final drain. Until then, independent host tests and archived pair records do
not make fixed state a current native-session product capability.

## R8: Concurrency and graph execution

Status: **pending**.

Current admitted profiles are eager and non-overlap. Multiple completion
domains, asynchronous copy/compute overlap, and CUDA Graph replay require
stable addresses or generation-aware indirection, explicit stream dependencies,
and graph-safe lifecycle operations. Reclamation decisions must remain outside
captured execution unless the graph contract proves them replay-safe.

No eager-path evidence transfers automatically to these modes.

## R9: Branching, cancellation, and distributed state

Status: **pending beyond narrow host controls**.

Speculation, beam search, cancellation pressure, disaggregation, and remote or
multi-GPU state add ownership edges and completion domains. Each requires
explicit snapshot/fork semantics, placement identities, transport completion,
failure ownership, and reclamation proofs. Same-device relocation evidence does
not qualify fabric transfer.

## Benefit gates

Any capacity, memory, or performance claim must use matched source, model,
inputs, sampling, backend, tensor geometry, device allocation budget, and
execution mode. Required measurements include:

- output-token or numerical-state correctness;
- semantic-live, request-reachable, resident, reserved, padding, and temporary
  relocation bytes;
- admission capacity under the same budget;
- TTFT, inter-token latency, throughput, tail latency, and CPU/GPU profiles;
- cold/warm Prefix behavior where applicable;
- long-running arrival/departure and pressure behavior; and
- exact artifact/source hashes with an independent verifier.

Reclaimed pages do not alone prove reduced tensor reservation. A smaller
semantic state does not alone prove higher throughput. A current correctness
record with slower manager timing is evidence of correctness and overhead, not
a hidden benefit.

## Release order

The near-term order is:

1. finish SGLang adoption of the completed opaque session-relocation wire and
   retire any remaining raw-manager relocation integration;
2. keep all source, wire, layout, and documentation verifiers synchronized;
3. expose trustworthy Sliding retirement/reuse evidence;
4. run independent current-source Full+Sliding qualification;
5. run independent current-source pure Sliding qualification;
6. qualify exact Chunked and latent profiles separately; and
7. measure benefit only after correctness and lifecycle gates pass.

This order keeps physical mechanisms subordinate to the compiler and lifecycle
contract: a ring, relocation copy, or new backend becomes a product capability
only when it is derived, admitted, executed, and qualified through the single
`RuntimeSession` route.
