# Architecture

OrbitKV is organized as one inference product with three ownership layers and
one temporary compatibility path.

```text
OpenAI HTTP / gRPC
        |
        v
server/                 request admission, tokenization, scheduling, sampling
        | BatchIntent (no physical addresses)
        v
core/                   attention-state compilation and KV lifecycle authority
        | ExecutionPlan (pages, writes, copies, visibility, retirement)
        v
executor/               forked Luminal graph compiler and device executor
        | one ordered CUDA stream / graph
        v
CUDA + FlashInfer       kernels and completion evidence
        |
        v
core/                   publication, retirement acknowledgement, safe reuse
```

## Ownership rules

- `core/` is the only owner of page allocation, page generations, request and
  Prefix snapshots, token disposition, retirement, and reuse.
- `executor/` consumes immutable physical plans. It may cache device metadata
  and compiled graphs, but it cannot invent, retain, or recycle a page.
- `server/` emits scheduling intent and receives tokens. Its public contract
  intentionally contains no page identifier.
- `compat/sglang/` is a regression and migration route. Its tables are checked
  execution mirrors, not an alternative manager.

Safe reuse requires both semantic death and completion of every device command
that may still observe the old state. Device mirror cleanup and the exact
retirement acknowledgement occur before a page generation returns to the free
pool.

## Source layout

```text
core/
  src/                    compiler and RuntimeSession
  ffi/                    compatibility C boundary
executor/
  src/                    engine-neutral plan lowering
  luminal/                complete fork, tracked as a Git submodule
server/
  src/                    engine-facing control-plane contract
compat/
  sglang/                 previous SGLang integration and qualification tools
docs/                     active architecture and capability contracts
results/                  append-only measured evidence
```

The Luminal fork preserves upstream history and tracks
`luminal-ai/luminal`. OrbitKV-specific runtime changes are committed in
`feichai0017/orbitkv-luminal`; the parent repository pins the exact commit.

## Migration state

The repository currently has a host-tested `BatchIntent`, an engine-neutral
OrbitKV-to-executor plan, and a forked Luminal paged-attention operation that
accepts externally owned page tables including last-page lengths. The complete
same-stream model execution and server integration remain in progress.

No performance result from the SGLang compatibility route transfers to the
Luminal executor. A speedup claim requires paired real-model runs against the
same stock SGLang revision, model, dtype, workload, device, and capacity.
