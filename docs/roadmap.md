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
- A released 18-layer Full+Sliding checkpoint has an H20 correctness and
  lifecycle closure: native 3 Full / 15 Sliding layer assignment, 512-token
  prefill, 33 decode steps, independent reference-token parity, Sliding
  retirement and generation reuse, cancellation, and complete arena drain.
- External export/restore/delete transactions and an async transport contract
  move real bytes through a host reference adapter. Production transports and
  remote leases remain open.
- The server has a local async `Engine` contract and optional vLLM Rust frontend,
  but no complete continuous-batching model-backed executable.
- A released-hybrid same-executor residence experiment passes ten paired
  release-mode epochs with full token parity. Compiled residence reduces live
  payload by 27.8%, raises the fixed-budget boundary from 528 to 560, and has a
  positive paired total-time confidence interval. Serving concurrency, tail
  latency, and throughput remain unproven.

## R1: Execute a multi-class hybrid graph — completed

The single-class restriction is removed from `DecoderGraph`: every layer is
built from its manifest-assigned class, with independent persistent arenas,
write slots, CSR metadata, context dimensions, and capture signatures. The
released-checkpoint qualification crosses the native Sliding window with 512
prefill tokens and 33 decode steps. It matches an independent Transformers
greedy-token sequence, observes retirement and post-ACK generation reuse, runs
a second 16-token request from reused storage, releases that request at a token
boundary, and verifies both final drains.

This closes R1 correctness and lifecycle qualification. It does not by itself
close a benefit or production-serving claim; the narrow R2 result below is a
separate matched experiment.

## R2: Prove the compiler contribution — narrow closure completed

The backend-neutral `PhysicalResidencePolicy` provides the two arms. A single
compiled Luminal graph executed both on H20 with identical weights, dtype,
request trace, kernel selection, and arena geometry:

```text
conservative retention  versus  manifest-compiled retention
```

Ten paired alternating release-mode epochs each ran a 512-token prefill and 255
decode steps. All 256 output tokens matched between arms, and the independently
known first 34 tokens matched Transformers. Compiled residence reduced physical
resident payload from 14,155,776 to 10,223,616 bytes (27.8%) and Sliding pages
from 48 to 32 (33.3%). Retention Amplification fell from 1.387 to 1.002. With
Full=35 and Sliding=33 pages, compiled residence
advanced to boundary 560 while request-lifetime residence stopped at 528. Median
total test-path time fell from 1.219458 s to 1.210898 s (0.70%); paired mean
improvement was 10.597 ms with a 95% confidence interval of 6.310-14.884 ms.
Model compute stayed nearly equal; the main difference was manager time.

This is a narrow same-executor L5 result: it proves that compiler-derived
lifetime management reduces real resident payload and improves admission without
adding net cost in this one batch-one workload. It does not establish
continuous-batching throughput, TTFT/TPOT tails, multi-user capacity, or a win
over SGLang. Those product-level measurements remain R3/R4. CUDA Graph dispatch
speedups remain separate evidence.

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
