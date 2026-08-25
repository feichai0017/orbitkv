# Engine-neutral KV data-plane adapter SPI

## Scope

`orbitkv-runtime` defines a small typed Python boundary between OrbitKV's
ownership runtime and engine-owned KV tensor arenas. `orbitkv-reference` is a
package-tested reusable tensor-arena implementation and executable contract
oracle.
It is not a second serving engine: it has no scheduler, attention kernel, model
runner, page allocator, free list, or policy for deciding which state is live.

The packages are deliberately separate from `orbitkv-sglang`. Neither imports
SGLang. The neutral package has no third-party dependency. The reference
package always supports writable CPU buffers and lazily recognizes contiguous
CPU or CUDA Torch tensors when Torch is installed. Importing either package
does not import Torch.

## Authority boundary

OrbitKV remains authoritative for page selection, logical-to-physical binding,
page generation, snapshots, semantic death, retirement certificates, and the
manager ACK that permits physical reuse. An adapter realizes exact reads,
writes, copies, completion fences, and checked engine mirrors over externally
allocated storage. A mirror or a resolved tensor address is never allocation
authority.

```text
manager-issued PageLease + backend coordinate
              |
              v
       KvDataPlaneAdapter
       /       |        \
 exact bytes  fences   checked mirrors
       \       |        /
        adapter reuse evidence
              |
              v
       exact manager ACK
              |
              v
  same physical page, higher generation only
```

This SPI is a Python-side data-plane contract layered on the current manager
wire. It does not change or replace the native ABI.

## Typed contract

The installed module is `orbitkv_runtime`. Its central values are:

- `PageLease`: manager authority containing engine epoch, pool epoch, physical
  page identity, and generation.
- `BackendPageAddress` and `BackendTokenAddress`: an exact lease plus class,
  backend domain/index, and (for a token) offset. The neutral contract does not
  embed any engine's dummy-page convention.
- `OperationContext`: the exact request lease and manager transaction that
  authorized a data effect. Append uses a `StepLease`; relocation uses a
  `RelocationLease`. A transaction is one-shot, while different requests may
  legitimately use the same logical token id in one batch.
- `CompletionFence`: an adapter-issued, positive, monotonic point in one
  completion domain. The caller cannot choose its value. Each adapter instance
  adds a private identity nonce even when given the same human-readable label.
- `CompletionEvidence`: confirmation returned only by querying or waiting on a
  fence issued by the same adapter; it also names every exact page generation
  covered by that fence.
- `ExternalTokenWrite(context, token_id, destination, byte_count)`: a
  payload-free description of one append that an engine-owned kernel will
  perform. It binds an APPEND `OperationContext`, logical token id, exact
  `BackendTokenAddress`, and exact positive byte count.
- `ExternalAppendTicket(adapter_id, ticket_id, writes,
  resolved_destinations, completion_domain, launch_context)`: adapter-issued
  write authorization for one non-empty batch. It retains the original writes
  and their one-to-one `ResolvedTokenAddress` values; each resolved address and
  byte length must
  exactly match its write. It also carries the positive completion domain and
  opaque adapter-selected launch context bound before kernel enqueue. A ticket
  is not completion evidence.
- `AdapterCapabilities`: explicit feature disclosure. CUDA support describes
  the supplied arenas, while `external_kernel_writes` advertises the optional
  external-write extension. Neither capability is a graph, overlap, or
  performance claim.
- `ReclamationCertificate`: an exact, generation-bearing retirement statement
  supplied by the manager rather than minted by the adapter.
- `ReuseEvidence`: the exact certificate batch, completed last-use fences, and
  completed mirror-cleanup fence that may be converted into manager receipts.
- `KvDataPlaneAdapter`: the runtime-checkable batch-first protocol.

`append` performs exact new-token writes and optional partial-tail COW copies.
All sources are gathered before any destination write, so overlapping copies
remain byte exact. `relocate` uses the same gather/scatter rule for arbitrary
token moves. Copies cannot cross a class, backend domain, or pool. Complete
batch validation precedes the first storage mutation; a
bad later member therefore cannot partially write an earlier member. An error
after external mutation begins poisons the adapter because rollback cannot be
claimed.

Page and token mirrors use compare-and-swap-like updates with an expected old
address and an optional replacement. `replacement=None` means CLEAR. A non-null
replacement means REPLACE. REPLACE is valid independently of retirement: a COW
source may remain shared and therefore produce no reclamation certificate. All
updates are preflighted before the dictionaries are published.

## Optional external append protocol

`ExternalWriteCompletionAdapter` is an optional extension for engines whose own
kernel writes new-token bytes directly into an arena. It complements the
adapter-executed `append`; it does not replace `KvDataPlaneAdapter`, grant the
engine allocation authority, or permit retrospective validation after storage
has already been mutated. The required order is:

```text
prepare_external_append(writes, completion_domain=...)
    -> predecessor waits are enqueued; ticket pages/domain are reserved
enqueue the engine-owned write kernel using the ticket's resolved destinations
record_external_data_ready(ticket)
    -> data-ready fence; the pending ticket is consumed
enqueue every eventual consumer of those page generations
record_last_use(pages)
    -> explicit last-use fence for reclamation
```

The engine must call `prepare_external_append` with the complete write batch
before the first kernel launch or any other destination mutation. Preparation
validates the full batch, resolves every exact destination, binds a completion
domain, enqueues its predecessor waits on the current stream when needed, and
registers the ticket as pending. The kernel must use those resolved
destinations in `ticket.launch_context` (the selected CUDA stream for the
reference adapter; `None` for synchronous CPU storage). A foreign,
stale, already-consumed, or write-mismatched ticket fails closed. A page
generation named by a pending ticket cannot be otherwise resolved, accessed,
mutated, declared last-used, or pass `prepare_reuse`; its completion domain is
also reserved until data-ready recording. An abandoned or uncertain ticket
must not be bypassed to recover either resource.
Once an issued ticket is pending, changing any of its public fields is an
unrecoverable protocol violation: the kernel may already have used a changed
descriptor, so the reference adapter poisons instead of allowing repair and
retry. Completion receipts and reservation cleanup are derived only from the
adapter's private immutable snapshot.

Only after all external writes have been enqueued in the ticket's completion
domain may the engine call `record_external_data_ready`. This records a fence
after the enqueued writes, consumes the ticket, and returns the corresponding
`DataPlaneEvidence`. Calling it before enqueue would make readiness unsound.
The resulting evidence says when the produced bytes may be consumed. It does
not say that their final consumer has completed and must never be supplied as
last-use evidence for reuse.

After the final consumer has been enqueued, the engine calls the semantically
explicit `record_last_use` for the exact page generations involved. The
existing `record_completion` spelling remains the legacy compatibility alias
for this last-use operation; new integrations should use `record_last_use`.
Querying or waiting on that fence produces the last-use `CompletionEvidence`
required by the reuse path.

## Completion and ACK-gated reuse

Data readiness and last use are distinct. `append` and `relocate` return a
data-completion fence. The engine records a separate last-use fence after the
final consumer has been enqueued, explicitly naming the exact page generations
it covers. Mirror cleanup is ordered after that explicit last-use evidence and
emits its own fence. Any later read, write, copy, or mirror operation invalidates
the earlier last-use claim.

A completion domain is bound to one CPU or CUDA device timeline on first use.
CUDA events in a domain wait on their predecessor, and cross-domain dependent
operations wait on the named predecessor event. One operation cannot mix CPU
and CUDA storage or span CUDA devices.

Physical reuse requires every gate below:

1. The manager proves semantic unreachability and issues an exact
   `ReclamationCertificate`.
2. The adapter confirms exact last-use completion points matching every
   certificate.
3. No pending external-append ticket names a retiring generation. Data-ready
   evidence is not accepted in place of last-use evidence.
4. No adapter page or token mirror still names a retiring generation.
5. Mirror cleanup has completed after last use.
6. The coordinator derives exact reclamation receipts and sends the manager ACK
   once.
7. Only after that ACK succeeds does it call `note_reuse_acknowledged`.

`prepare_reuse` produces evidence; it neither frees storage nor acknowledges
the manager. Before step 7, old and new generations of the retiring page are
both rejected. Afterwards the same physical page is accepted only with a
strictly higher generation. A duplicate ACK notification is rejected. If the
manager may have consumed an ACK but its return is lost, the caller must
fail-stop; retrying and calling `note_reuse_acknowledged` would overstate what
is known.

The initial reuse batch contract accepts exactly one shared last-use fence.
Completion values are independently monotonic in multiple domains, but a
single mirror fence cannot establish ordering after multiple unordered domains.
A future join-fence primitive is required before that case can be admitted.

The reference adapter is synchronous for CPU buffers. For CUDA tensor arenas,
it gathers and scatters byte rows with device-native Torch operations on the
current stream and records a page-scoped CUDA event. It never stages copy
payloads through the host. This mechanism does not qualify optimized overlap,
CUDA Graphs, a model, an engine, or performance.

## External arena geometry

Register each storage object with `ArenaRegistration`. The object must contain
exactly:

```text
page_count * page_tokens * token_bytes
```

bytes. A CPU arena may be `bytearray`, writable `memoryview`, or another
contiguous writable buffer. A Torch arena may have any shape and element dtype
whose contiguous byte size matches; the adapter views it as
`[page_count, page_tokens, token_bytes]` bytes. Token records are opaque. The
contract therefore supports ordinary K/V rows, latent K/V, or test records
without baking a model layout into the SPI. Torch devices other than CPU and
CUDA fail closed.

## Running the contract suite

The tests add both `src` directories themselves, so package installation is not
required. From the repository root run:

```bash
python -m pytest -q \
  python/orbitkv-runtime/tests \
  integrations/reference/tests
```

The required CPU suite runs without Torch. Torch-CPU parity is exercised when
Torch is available and otherwise reported as skipped. Static tests reject any
SGLang import in either package and any Torch import in the neutral package.

## Current boundary

This initial SPI intentionally does not adapt the existing SGLang plugin or
claim a second complete serving-engine integration. It supplies the reusable
effect boundary and a real external-arena implementation needed for a future
engine adapter. The optional external append protocol likewise leaves native
ABI8 unchanged, and SGLang has not migrated to it. It establishes ordering and
reclamation safety only; it makes no throughput, latency, kernel-overlap, or
CUDA Graph claim. Distributed completion, multi-device operations in one
fence, graph-stable descriptor storage, structured tensor layouts, and sealed
hardware qualification remain outside this contract.
