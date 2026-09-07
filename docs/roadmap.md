# Roadmap

The target product is one Rust inference process: OrbitKV compiles and owns
attention-state lifetimes, the Luminal fork compiles and executes model graphs,
and the Rust server schedules requests and exposes client protocols. Planned
work is not a current capability.

## Current checkpoint

- `core`, `executor`, `server`, and the outer `engine` composition root are the
  active product crates.
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
- The server has a local async `Engine` contract and optional vLLM Rust frontend.
  `orbitkv-engine::ModelEngine` now implements that contract with one dedicated
  model thread, bounded admission/output queues, a decode-first token-budgeted
  active set, a real `RuntimeSession`, one compiled Luminal decoder, streamed
  greedy tokens, stop/cancel handling, and exact final drain. It is not yet
  performance-qualified.
- `orbitkv-serve` composes `ModelEngine` with the pinned vLLM Rust OpenAI,
  tokenizer/chat, SSE, request-ID, and stream-drop auto-abort components in the
  same process. Real H20 tests cover non-streaming and SSE completions,
  concurrency, disconnect cancellation, graceful shutdown, and final drain. A
  fixed 16-request load trace also completes at C1/C2/C4/C8 with every requested
  output token and no per-request errors.
- A released-hybrid same-executor residence experiment passes ten paired
  release-mode epochs with full token parity. Compiled residence reduces live
  payload by 27.8%, raises the fixed-budget boundary from 528 to 560, and has a
  positive paired total-time confidence interval. HTTP concurrency and latency
  metrics are now measured through C8, but comparative serving benefit remains
  unproven.

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
stable first 13 tokens matched Transformers. Compiled residence reduced physical
resident payload from 14,155,776 to 10,223,616 bytes (27.8%) and Sliding pages
from 48 to 32 (33.3%). Retention Amplification fell from 1.387 to 1.002. With
Full=35 and Sliding=33 pages, compiled residence
advanced to boundary 560 while request-lifetime residence stopped at 528. Median
total test-path time fell from 1.083926 s to 1.075590 s (0.77%); paired mean
improvement was 11.633 ms with a 95% confidence interval of 5.697-17.570 ms.
Model compute stayed nearly equal; the main difference was manager time.

This is a narrow same-executor L5 result: it proves that compiler-derived
lifetime management reduces real resident payload and improves admission without
adding net cost in this one batch-one workload. It does not establish
continuous-batching throughput, TTFT/TPOT tails, multi-user capacity, or a win
over SGLang. Those product-level measurements remain R3/R4. CUDA Graph dispatch
speedups remain separate evidence.

## R3: Complete the single-process serving engine — narrow load closure completed

Status: single-process HTTP correctness and a narrow C1-C8 load qualification
are completed for one released Full+Sliding checkpoint.

The concrete coordinator now implements the server `Engine` trait. A dedicated
thread owns `RuntimeSession`, stable class arenas, and `CompiledDecoder`; bounded
admission feeds an active set whose decode-first batches obey a configurable
token budget. Logical requests stream through bounded output queues and converge
through release/ACK on length, stop, cancellation, or disconnect. Invalid
continuation, overlength, duplicate, unsupported batch, queue pressure, and
insufficient per-class capacity inputs fail closed. Device ambiguity quarantines
state and fail-stops the worker. Released hybrid H20 tests pass B=2 prefill and
decode with reference-token parity, late prefill during decode, stop/cancel, and
final drain.

The generic `orbitkv-serve` binary now starts the real model engine and vLLM
Rust frontend together. It uses typed CLI configuration, supports separately
stored tokenizer assets, reports logical rather than class-summed KV capacity,
and handles Ctrl-C/SIGTERM through graceful frontend shutdown. A released model
passes non-streaming and SSE OpenAI completions, two concurrent HTTP requests,
client-disconnect auto-abort with an explicit cancellation counter, and final
manager drain on H20.

The pinned Rust `vllm-bench` client then ran one fixed trace through the same
warm server at C1/C2/C4/C8. Every arm completed 16 requests with 256 output
tokens each and zero detailed errors. Output throughput increased from 184.24
to 518.88 token/s, while median TTFT increased from 163.06 to 1127.81 ms and
median TPOT from 4.80 to 11.06 ms. A direct B=1/B=8 teacher-forced diagnostic
also proves bit-identical rows within B=8, maximum absolute cross-batch logit
difference 0.4296875, and zero argmax mismatches over 16 positions.

This closes a narrow R3 load gate and exposes a real throughput/latency tradeoff.
Fairness, soak, the capacity failure point, Chunked prefill, and richer sampling
remain open. The R4 comparison below now quantifies the remaining executor gap;
no SGLang-relative benefit follows from this internal scaling result.

PegaInfer is a useful reference for a small Rust server/model boundary and for a
full-plus-linear-attention execution loop. It is not a Dynamo integration: its
current source has no Dynamo, KVBM, or NIXL dependency. Do not copy its
model-named dispatch or model-owned contiguous KV cache into this architecture.

## R4: Run the product comparison — completed, competitiveness gate failed

Use `tools/run_matched_serving.py` and one vLLM benchmark client for both
OrbitKV/Luminal and a tuned stock SGLang. Alternate launch order across an even
number of epochs and hold model, weights, dtype, request trace, device budget,
sampling, and concurrency constant. Report correctness separately from TTFT,
TPOT, ITL, throughput, tail latency, memory, and capacity.

The same-executor ablation from R2 establishes attribution to the compiler. The
SGLang comparison tests product competitiveness. Neither replaces the other.

The first run exposed cross-process Luminal search variation, so it was not
promoted. Native schedule artifacts were then added: paged-attention custom-op
schedules are reloaded without search and are strictly bound to the manifest,
decoder/weight geometry, arenas, and compile buckets. Four independent H20
restarts produced one stable candidate digest and a 0.74% throughput coefficient
of variation.

The artifact-fixed four-epoch C2 comparison completed every 127-to-256-token
request without errors. OrbitKV median output throughput was 592.32 token/s
versus 1112.60 for stock SGLang v0.5.17 (0.534x). OrbitKV had 2.997 ms TPOT
versus 1.699 ms (1.76x) and 98.08 ms TTFT versus 15.14 ms (6.50x). Cross-engine
text digests differed, so this is a transparent product diagnostic rather than
a matched-output benefit claim.

OrbitKV did preserve a state-capacity advantage: its Full/Sliding class arenas
encode 85.875 MiB of persistent K/V payload for the tested capacity, versus
144.0 MiB for SGLang after SGLang disabled hybrid SWA memory on Gemma3, a 40.4%
reduction. Those are resolved tensor-payload bytes, not allocator peak. The R4
competitiveness gate therefore fails on serving speed while passing the narrower
persistent-state objective.

## R4.1: Close the executor gap

Do not expand the product surface until the fixed trace is competitive. Attack
the measured hot path in this order:

1. Fixed-signature capture now accepts pure decode batches, but automatically
   recapturing exact context-page signatures reduced C2 throughput by 13.7%; do
   not connect that policy to serving until padded/stable signatures remove the
   recapture cliffs.
2. Completed: persistent K/V aliases are now compiler hard constraints. Search
   candidates and loaded artifacts fail closed unless every selected bucket
   updates all K/V tensors in place. A 16-candidate constrained search produced
   36/36 in-place tensors and zero copy-back bytes in both retained buckets.
3. Completed narrow gate: four alternating epochs against the previous
   artifact improved throughput by 14.4%, TTFT by 32.0%, TPOT by 10.1%, and
   E2E by 12.6%. The independent B2 reference-token probe passes; random-trace
   text differs across schedules, so strict output equivalence remains false.
4. Continue with graph-internal kernel/fusion profiling against SGLang FA3.
   The improved artifact reaches 0.598x SGLang throughput, 1.59x TPOT, and
   4.69x TTFT, so the product competitiveness gate still fails.

Only after this gate passes should R5/R6 become the primary product work.

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
