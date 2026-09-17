# OrbitKV Next engine design

Status: implementation architecture after importing `kern` 0.2.3 at
`05df6d9cf8233b2438a7a584ce4ed7a0666abf53`.

## Purpose

OrbitKV Next is a source-integrated inference engine for recent hybrid language
models. Its distinguishing capability is compilation: model topology, numerical
contracts, persistent-state effects, available kernels, and target hardware are
compiled into a verified, bucket-specific GPU program.

The engine is not successful merely because it serves a checkpoint. It must
demonstrate both:

1. model-specialized performance through generated or selected execution
   islands; and
2. family-level reuse, so GLM-5.3 and DeepSeek-V4.1 add semantic operations and
   kernels rather than private runtimes.

## Source architecture

```text
OrbitKV-owned model/compiler layer
  src/model              pinned model contracts and weight semantics
  src/ir                 values, tasks, dependencies, state effects
  src/compiler           validation, bufferization, islands, schedules
  src/lower              physical plan -> kern manifest v5
  kernels                generated and adapted kernel packages

Imported and locally owned execution layer
  crates/kern-manifest   executable format, verifier, serving protocol
  crates/kern-pool       pages, sequence slots, checkpoints, host tier
  crates/kern-runtime    CUDA load/link, buffers, VMM, graph replay
  crates/kern-test       differential program and kernel qualification
  crates/kern-run        CLI, artifact loading, direct run and benchmark
  crates/kern-serve      batching, admission, prefix reuse, HTTP engine

External frontend/providers
  PegaInfer/vLLM Rust frontend
  DeepGEMM, FlashInfer, FlashAttention/MLA, FlashKDA, NCCL/DeepEP
```

The imported execution layer is editable source, not a black-box dependency.
Its ownership boundary nevertheless remains strict: model names, model math,
fusion policy, and schedule search stay above the manifest. Runtime and pool
changes must implement a cross-model mechanism or remove a measured bottleneck.

## Offline compilation

The compilation path is not part of request serving:

```text
checkpoint config + tensor index
  -> exact model contract
  -> semantic task graph
  -> effect and numerical validation
  -> value lifetime and state-layout planning
  -> legal execution-island candidates
  -> provider/generated-kernel capability matching
  -> device measurement for declared buckets
  -> selected physical plan
  -> manifest v5 + cubins + hashes + provenance
  -> kern verifier and differential qualification
```

The initial buckets are target decode batches 1, 2, 4, and 8 on H20. Prefill
uses a correct provider composition until a measured prefill bottleneck justifies
its own island work.

### Two Qwen oracles

The imported `examples/qwen3.8-27b.json` is a BF16 executable oracle. It is
authoritative for model topology, physical recurrent/KV state, call ordering,
kernel ABI, manifest protocol, and serving behavior. It is not authoritative
for the official FP8 checkpoint's weight dtypes, block scales, or FP8 GEMM
numerics.

The official `Qwen3.8-27B-FP8` checkpoint is the weight oracle. Its config and
safetensors headers define dynamic E4M3, 128 by 128 scale blocks, and the exact
matrix/scale tensor pairs. Compiler qualification combines the two oracles; it
must never claim FP8 support merely because it reproduced the BF16 manifest.

### Compiler/runtime contract

The compiler decides:

- which tensors and persistent states exist;
- state shapes, layouts, effects, versions, and legal aliases;
- which calls form one execution island;
- which provider or generated kernel implements the island;
- buffer lifetimes, materialization, and communication tasks;
- program variants for prefill, decode, and speculative verification.

The execution layer decides:

- how verified buffers and opaque state bytes are allocated;
- how checkpoint tensors are loaded into declared weight buffers;
- how cubins are loaded and launch ABIs are wired;
- how pages and per-sequence slots are leased;
- how CUDA Graphs are captured and replayed;
- how requests are admitted, batched, streamed, retired, or checkpointed.

## Artifact

A deployable model is a closed artifact:

```text
artifact/
  manifest.json          verified model program and serving protocol
  kernels/               content-addressed cubins
  qualification.json     model/kernel numerical evidence
  provenance.json        model, compiler, provider and source revisions
  schedule.json          selected target and bucket decisions
```

Weights remain the official checkpoint unless a qualified load-time transform
or packed deployment format is explicitly selected. A serving process does not
compile or search.

## Startup flow

```text
read manifest
  -> verify schema, references, ABI and dataflow
  -> derive the serving Protocol from fills and batch shapes
  -> load cubins and resolve entries
  -> allocate fixed buffers and reserve state virtual arenas
  -> create the shared page/sequence-slot VMM pool
  -> bind checkpoint tensors to weight buffers
  -> lower named calls to flat launches with stable addresses
  -> connect peer mappings for a multi-GPU artifact
  -> run once programs
  -> create Tray and Scheduler
  -> begin serving
```

After loading, the hot path contains flat launch lists or CUDA Graph replay. It
does not inspect model topology or search for implementations.

## Request flow

```text
OpenAI-compatible request
  -> chat rendering and tokenization
  -> waiting queue
  -> prefix/session checkpoint lookup
  -> all-or-nothing lease for prompt + output bound + speculative headroom
  -> chunked prefill
  -> running set
  -> batch-bucket selection and padding
  -> stage token, position, validity, slot, length and table inputs
  -> issue the manifest forward on every rank
  -> first shape: capture CUDA Graph; later shapes: replay
  -> read token/count outputs and stream accepted tokens
  -> repeat decode/verify
  -> release state or retain a checkpoint at completion
```

The first serving policy remains intentionally conservative: prefill-first,
continuous decode batching, greedy output, and worst-case admission reservation.
It is a correctness and performance baseline, not a permanent scheduler design.

## KV and persistent-state management

The state plane starts from `kern-pool`, not the archived OrbitKV manager.

### State classes

Manifest state declarations lower into three physical classes:

- `bytes_per_token`: paged KV, sparse index state, or other token-addressed
  state;
- `bytes_per_seq`: GDN/KDA recurrent matrices and convolution history;
- `bytes`: immutable or globally fixed model state.

A live request owns a `Lease` containing all of its logical token pages and,
when required, one sequence slot. Page-table rows and line-table indices are
derived exclusively from that lease; the scheduler never invents physical page
or slot identifiers.

### Shared physical budget

Token pages and sequence slots use separate reserved virtual arenas but draw
from one set of physical VMM chunks. Free chunks can therefore be remapped from
pages to slots for many short sequences, or from slots to pages for fewer long
contexts. Held objects do not move.

The initial scheduler reserves the request's worst-case page count at admission.
This prevents mid-decode allocation failure and gives stable addresses at the
cost of conservative capacity. We keep this policy through Qwen M1/M2 and change
it only if serving traces show material lost throughput.

### Prefix and session reuse

For pure paged state, whole pages are immutable after checkpointing and can be
shared through a prefix chain. A partial final page is copied on fork.

For a model with large per-sequence recurrent state, arbitrary page-boundary
checkpoints would also copy the entire recurrent slot. The baseline therefore
retains a finished request's complete pages and slot as a session checkpoint,
which efficiently supports multi-turn continuation but not arbitrary partial
prefix reuse.

The compiler eventually supplies checkpointability facts for each state family:

```text
page-shareable       whole token pages may be shared
slot-snapshot        a complete sequence slot is required
recomputable         state may be reconstructed from a declared boundary
versioned            speculative writes require accepted-frontier commit
immutable-lookup     resident table, not request-owned state
```

Those facts should become a manifest/runtime mechanism only after Qwen proves a
specific reuse or speculative-state requirement that v5 cannot express. Until
then, the manifest remains the compatibility boundary. Schema v6 is currently
the emitted format and the runtime continues to accept v5 artifacts.

### Host tier

An optional pinned-host tier parks cold checkpoints by copying their pages and
sequence slots off the compute stream. A hit wakes the entire retained state
into a fresh lease. It is session-state spill, not a distributed KV service.
Remote KV and multi-node disaggregation remain out of scope until local
execution wins.

## Speculative execution

The compiler IR represents base, tentative, and committed state versions. An
optimized target program may lower this to overwriteable rows beyond the
accepted frontier instead of copying full state. The manifest exposes a fixed
multi-row `round` program and token-count output; the scheduler advances each
sequence only by the accepted count. Rejected token slots are overwritten by a
later round.

The runtime must not infer model rollback semantics. The compiler proves and
tests the physical lowering against forced-accept and forced-reject traces.

## Multi-GPU execution

The compiler emits topology groups and explicit collective/peer-transfer tasks.
The runtime creates one rank-local instance per GPU, imports peer mappings,
stages all ranks, issues every rank before waiting, and then collects results.
NCCL, DeepEP, or custom collective kernels remain implementations selected by
the physical plan.

GLM-5.3 is the first milestone allowed to make substantial changes here. Qwen
M1 must not grow speculative distributed abstractions.

## Modification rules for the imported substrate

Modify `kern-manifest` when a required, cross-model executable contract cannot
be represented in the current schema. The first such extension is schema v6's
scratch-backed TMA descriptor for provider-private packed activations.

Modify `kern-pool` for state ownership, page/slot/checkpoint, VMM, or host-tier
mechanisms shared by model families.

Modify `kern-runtime` for verified program loading, CUDA execution, graph replay,
or measured hot-path overhead.

Modify `kern-serve` for request scheduling and batching policy. It must consume
protocol/artifact declarations and must not contain Qwen-, GLM-, or
DeepSeek-specific branches.

Every modified imported file must retain Apache-2.0 attribution and record that
it was changed locally. Upstreamable fixes should remain separable from
OrbitKV-specific compiler work.

## Immediate implementation order

1. Import the official Qwen3.8 config and tensor index into checked model and
   weight contracts.
2. Parse the imported hand-authored Qwen3.8 manifest as an executable oracle.
3. Produce the same declarations and call graph from the compiler IR, without
   per-layer JSON templates.
4. Diff and verify compiler output using local `kern-manifest` and `kern-test`.
5. Run the generated artifact through local `kern-runtime` and `kern-serve`.
6. Replace the GDN non-GEMM region with a qualified stateful kernel, then measure
   the complete decoder.
7. Add low-M FP8 schedule selection and only then attempt a persistent three-GDN
   execution island.

No scheduler or KV redesign is on the Qwen critical path unless the imported
implementation blocks correctness or a measured end-to-end gate.
