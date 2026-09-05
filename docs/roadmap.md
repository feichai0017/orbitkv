# Roadmap

The target product is one Rust inference process: OrbitKV compiles and owns
attention-state lifetimes, the Luminal fork compiles and executes model graphs,
and the Rust server schedules requests and exposes client protocols. Planned
work is not a current capability.

## Current checkpoint

- `core`, `executor`, and `server` are the only active product layers.
- `RuntimeManifest` is the shared source of truth for manager and executor plans.
- OrbitKV owns page allocation, generations, Prefix/COW, token disposition,
  retirement, publication, and reuse.
- Full, Sliding, Full+Sliding, and exact Chunked lifetimes compile and pass host
  lifecycle tests.
- Luminal consumes manager-authored page metadata, keeps stable K/V arenas,
  compiles decode/prefill buckets once, samples greedy tokens on device, and can
  replay fixed-signature decode through child CUDA graphs.
- One released Full checkpoint has a narrow H20 model closure. The current
  decoder still rejects multi-class model graphs.
- External export/restore/delete transactions and an async transport contract
  move real bytes through a host reference adapter. Production transports and
  remote leases remain open.
- The server has a local async `Engine` contract and optional vLLM Rust frontend,
  but no complete continuous-batching model-backed executable.
- No experiment yet proves that compiled lifetime management improves admission
  capacity, memory, tail latency, or end-to-end throughput.

## R1: Execute a multi-class hybrid graph

Remove the single-class restriction from `DecoderGraph`. Build every layer from
its manifest-assigned class, bind independent persistent arenas and CSR metadata,
and select Full or Sliding attention parameters per layer. Complete prefill,
repeated decode, cancellation, publication, retirement, generation reuse, and
final drain on a released Full+Sliding checkpoint.

This is the first priority because hybrid state is where compiler-derived
lifetimes differ materially from conservative Full retention.

## R2: Prove the compiler contribution

Run a same-executor ablation with identical Luminal graphs, kernels, weights,
dtype, scheduler, request trace, and device budget:

```text
conservative retention  versus  manifest-compiled retention
```

Measure semantic-live bytes, physical-resident bytes, Retention Amplification,
admission failures, maximum admitted requests, allocator work, relocation bytes,
TTFT, TPOT, throughput, and p95/p99 latency. Require output equivalence and final
drain before interpreting performance.

This is the experiment that can validate the central compiler claim. CUDA Graph
dispatch speedups are useful but do not substitute for this ablation.

## R3: Complete the single-process serving engine

Implement the concrete coordinator behind the server `Engine` trait: request
queues, continuous batching, tokenization, sampling state, cancellation,
backpressure, `RuntimeSession` transactions, Luminal execution, and final drain.
Exercise the optional vLLM Rust OpenAI/tokenizer/chat/SSE frontend against the
real model engine rather than its current test engine.

PegaInfer is a useful reference for a small Rust server/model boundary and for a
full-plus-linear-attention execution loop. It is not a Dynamo integration: its
current source has no Dynamo, KVBM, or NIXL dependency. Do not copy its
model-named dispatch or model-owned contiguous KV cache into this architecture.

## R4: Run the product comparison

Use `tools/run_matched_serving.py` and one vLLM benchmark client for both
OrbitKV/Luminal and a tuned stock SGLang. Alternate launch order across an even
number of epochs and hold model, weights, dtype, request trace, device budget,
sampling, and concurrency constant. Report correctness separately from TTFT,
TPOT, ITL, throughput, tail latency, memory, and capacity.

The same-executor ablation from R2 establishes attribution to the compiler. The
SGLang comparison establishes product competitiveness. Neither replaces the
other.

## R5: Add distributed KV tiers

Implement Mooncake first behind `ExternalKvTransport`, reusing the host adapter's
conformance suite. Then add remote lease epochs, renewal, eviction intent, active
restore pins, exact deletion acknowledgement, timeout reconciliation, and node
failure recovery. Add shared Prefix restore and independently qualify Sliding,
Full+Sliding, and Chunked transfers.

Use Dynamo as an architectural and optional outer-control-plane source: routing,
events, topology, discovery, and telemetry may feed OrbitKV policy. Do not import
`kvbm-logical`, Dynamo `KvBlockManager`, lifecycle pins, or another allocator.
Add NIXL as a transport-only adapter after Mooncake semantics are stable, then
compare both data planes with identical restore/offload workloads.

## R6: Execute heterogeneous state

Integrate one component-aware heterogeneous model path. The preferred first
target is Full attention interleaved with recurrent/linear attention and
convolution state because it exercises token pages and fixed-size checkpoints
in one transaction. MLA is the alternative path and requires latent/RoPE-aware
attention kernels. Model configuration—not model-name branches—must drive both
OrbitKV state compilation and Luminal graph construction.

## R7: Formalize MPSR and harden production

State the Minimum Persistent State Realization objective and constraints
formally. Prove optimality for bounded-window and exact-chunked subclasses, and
compare generated plans with a small exact oracle for randomized instances.

Then add graph recapture policy, overlap, bounded queues, authenticated completion
envelopes, metrics, tracing, crash recovery, soak tests, multi-device placement,
release artifacts, and a supported-combination matrix. Production claims require
all applicable correctness, pressure, cancellation, and long-running gates.
