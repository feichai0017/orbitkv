# Oxide Infer roadmap

Oxide Infer is a Rust operator layer for LLM inference engines. The roadmap
stabilizes the framework and current evidence before it expands the operator
surface.

The complete target stays within `oxide-infer`, `oxide-infer-cuda`, and
`oxide-infer-lab`. A fourth crate needs an independent dependency, artifact,
safety, or release boundary.

## Global admission rules

Every new vertical slice needs a real consumer or measured workload. A design
idea alone does not justify a public API or empty module.

### Contract gate

Before CUDA work starts, record:

- engine call site and tensor roles
- shape, dtype, layout, masking, and post-operations
- numerical and determinism limits
- alias, workspace, page-table, and stream ownership
- independent CPU reference or external oracle
- unsupported cases and typed errors

### Provider gate

Every provider exposes one named algorithm through this lifecycle:

```text
Spec -> Provider -> Algorithm -> Plan -> Operands -> CommandScope -> Completion
```

Planning fixes provider, algorithm, artifact, workspace, launch, and Graph
policy. Enqueue cannot tune, switch providers, or fall back.

### Device gate

An admitted device algorithm needs:

- positive and negative host cases
- device correctness with sentinels
- asynchronous lifetime and completion tests
- Compute Sanitizer for all applicable tools
- artifact hash and target identity
- SASS and spill review for a performance candidate
- fixed-address Graph proof or an explicit exclusion

### Performance gate

Performance records need identical fixtures, contracts, and timed boundaries.
Run each provider order separately. Reject a ranking when order drift exceeds
the recorded stability limit.

Operator speed alone does not promote an engine route. Measure TTFT, TPOT,
throughput, and memory in the real consumer path.

### Release gate

A stable public surface needs:

- current-source evidence for each claimed target
- no compatibility alias or duplicate execution path
- complete API and safety documentation
- one external engine integration at a pinned version
- package contents, CI, and installation checks

## R0: Rename and normalize the framework

**State:** complete.

The rename changes current identifiers to Oxide Infer. Historical records,
tags, and commit links keep their original names.

Work:

- Rename the repository and three crates to `oxide-infer`,
  `oxide-infer-cuda`, and `oxide-infer-lab`.
- Rename the native provider to `Oxide`.
- Rename current CLI programs and environment variables to `oxide_*` and
  `OXIDE_*`.
- Keep old result JSON files byte-for-byte unchanged.
- Use the final operator families: attention, GEMM, KV cache, normalization,
  position, activation, sampling, speculation, quantization, MoE, and
  communication.
- Move implemented domains directly to their final namespaces.
- Rename remaining `*Args` types to `*Operands` without aliases.
- Keep one public execution path per operator.
- Keep runtime memory, command, status, Graph, driver, and interop code inside
  `oxide-infer-cuda`.

Exit gate:

- All current source, Cargo metadata, CI, tools, website, and live docs use the
  new identifiers.
- Historical provenance remains readable and unmodified.
- Host tests, strict Clippy, formatting, documentation links, and package
  checks pass.
- No empty family, architecture, or provider module exists.

## R1: Requalify current source

**State:** complete.

The phase 2 device record binds clean source `477b47a`, all nine permanent
`sm_90a` runners, 89 machine-readable passing case lines, 12 named Graph case
lines, and 36 of 36 Compute Sanitizer cells. The record names its H20 test host;
that hardware identity is evidence scope, not the project identity. Performance
and real-model engine claims remain outside R1.

The rename and namespace migration change source identity. Historical device
records do not qualify the new tree.

Work:

- Requalify RMSNorm, RoPE, dense cuBLASLt GEMM, decode, prefill, and fused
  append on the declared device target.
- Cover host errors, device correctness, sentinels, lifecycle, and resource
  recovery.
- Run memcheck, racecheck, synccheck, and initcheck over permanent runners.
- Rebuild fixed-address Graph records with poisoned outputs and lease checks.
- Record the exact `sm_90a` artifact and source hashes.
- Keep engine evidence separate from the simulated-engine gate.

Exit gate:

- Each current operator has one reviewed device correctness record for the renamed
  source or an explicit source-level exclusion.
- Each performance-relevant plan has sanitizer and Graph evidence or a written
  exclusion.
- Dynamic metadata rejection preserves outputs and leaves the queue reusable.
- Paged append passes the exclusive-target-page contract.

## R2: Performance baseline and native M=1 GEMV decision

**State:** active; the frozen native M=1 GEMV decision is complete with a stop.

The first current-source matched baseline covers 14 BF16 attention shapes
against FlashInfer 0.6.17. All 14 pass the provider-order stability gate;
Oxide has lower combined median eager latency in eight and FlashInfer in six.
The current paged long-context GQA4 path measures 46.60 microseconds against
FlashInfer at 23.21 microseconds, a 2.01x gap on that exact eager contract. In
the separate source cohort it is 18.45% lower latency than source `49290b5`.
The current ragged optimization reduces its
source-bound parent from 39.42 to 37.04 microseconds. In the separate matched
provider cohort, Oxide measures 36.94 microseconds and FlashInfer 21.93
microseconds, a 1.68x gap.

The current native candidate is `OxideSm90SimtGemvM1N16K64`. It admits BF16
`M=1`, `N % 16 = 0`, `K % 64 = 0`, no post-operation, and zero workspace on
the currently recorded `sm_90a` target. Its matched decision run passed the
10% margin against both Mistral.rs custom GEMV and cuBLASLt on only one of five
declared shapes. Four shapes triggered the stop gate, so it remains
experimental and `CublasLtHeuristic` remains selected.

Work:

- Continue reducing the long ragged and paged GQA4 prefill gaps without
  regressing the current short paths.
- Keep eager-provider, Graph, engine, and serving measurements separate.
- Retain the five-shape census and the stopped algorithm identity as immutable
  evidence; do not add shape-aware production routing for it.
- Give any new M=1 design or larger-M experiment a new algorithm identity and
  workload census.

Promotion gate:

- Every declared shape has at least 10% lower combined median latency than
  each baseline.
- Provider-order median drift is at most 5%.
- No spill, local-memory regression, overread, overwrite, Graph failure, or
  lease failure occurs.
- TPOT and throughput improve beyond measured engine noise.

Stop gate:

- A safety failure removes the artifact from selectable plans until full
  requalification.
- A performance failure keeps the algorithm experimental. `CublasLt` remains
  the selected production plan.
- A larger-M WGMMA experiment gets a new algorithm identity and workload
  census. It cannot modify this frozen algorithm.

Exit gate:

- The frozen M=1 algorithm exited through its recorded stop result; retain the
  vendor route.

## R3: Stabilize engine adapters

**State:** experimental paired-repository work exists.

The consumer engine owns model execution, scheduling, batching, KV policy,
sampling, and serving. The adapter owns type conversion and stream handoff.

Work:

- Requalify the Mistral.rs decode path against renamed source.
- Bind engine allocations through typed external regions.
- Use the engine's non-default CUDA stream without adopting it.
- Prove that Q, KV, output, workspace, and metadata cross no adapter device
  copy.
- Record provider hits and algorithm identities.
- Compare model outputs or selected tokens with the engine baseline.
- Define fail-closed behavior for panic and abandoned forward execution.
- Replace local path dependencies with immutable source identities.
- Define a checked C ABI only when a non-Rust engine needs it.

Mistral.rs exit gate:

- At least two model configurations pass output, no-copy, stream-order,
  completion, and typed-recovery checks.
- The adapter has no process-global runtime state.
- The complete engine interval has TTFT, TPOT, throughput, and memory records.

vLLM admission gate:

- The Rust API and any C ABI have a versioned ownership contract.
- Pin one vLLM release and one attention or linear call site.
- The adapter remains outside the three core crates.
- PyTorch and vLLM types do not enter `oxide-infer` or
  `oxide-infer-cuda`.

vLLM exit gate:

- A real vLLM request records provider hits, unchanged tensor addresses,
  current-stream ordering, output comparison, and clean teardown.
- The adapter defines behavior for unsupported shapes without native-provider
  fallback during enqueue.
- Engine-level performance exceeds measured noise before a speed claim.

## R4: Complete attention and KV-cache contracts

**State:** current narrow contracts, broader work planned.

Work:

- Requalify direct, split-K, token-parallel, and tiled attention algorithms.
- Add paged-decode and split-K fixed-address Graph coverage.
- Replace average-only ragged selection with measured shape classes when data
  supports the change.
- Add sliding-window attention from a pinned engine call site.
- Add broader head dimensions and page sizes from measured model demand.
- Scope mixed-batch attention from the engine scheduler contract.
- Scope MLA from one exact model and cache layout.
- Add KV gather, scatter, compaction, and remapping after pager ownership is
  fixed.

Attention admission gate:

- The engine trace fixes mask, layout, head mapping, sequence distribution,
  and workspace requirements.
- The independent reference covers the exact contract.
- The plan selects one named algorithm before enqueue.

KV mutation admission gate:

- Specify read sharing, write ownership, copy-on-write owner, metadata epoch,
  and completion lifetime first.
- A failure preserves all cache pages and returns the writable capability.

Exit gate:

- Each new contract passes current-source correctness, lifecycle, sanitizer,
  Graph, matched performance, and one engine invocation where applicable.
- The catalog contains no generic claim beyond recorded shape classes.

## R5: Add activation, sampling, and speculation

**State:** planned.

Activation work:

- Start with a measured SwiGLU or gated-activation call.
- Define a standalone contract and reference before fusion.
- Fuse RMSNorm, GEMM, bias, or activation only when the complete path wins.

Sampling work:

- Define logits dtype, transforms, penalties, Top-K, Top-P, Min-P, logprobs,
  and output ordering.
- Bind deterministic RNG state to request and token position.
- Test finite behavior, ties, degenerate distributions, and reproducibility.

Speculation work:

- Define draft and target token spans.
- Bind accepted-token count, pending token, RNG state, and grammar state in one
  commit contract.
- Cover greedy, stochastic, and tree verification separately.

Exit gate:

- Each family starts with one real engine call and no empty sibling modules.
- Sampling matches its declared deterministic or statistical limits.
- Speculative decoding preserves baseline token output or the declared
  stochastic distribution.
- Engine TPOT improves beyond measured noise.

## R6: Add quantization and MoE

**State:** planned.

Quantization work:

- Pin scale granularity, packing format, zero-point policy, accumulator type,
  and quality limit.
- Add conversion and dequantization before fused compute.
- Keep FP8 and FP4 native kernels out until cuda-oxide supports the required
  types and instructions or the project contributes that support upstream.

MoE work:

- Pin routing, capacity, permutation, expert input, and weighted-combine
  semantics.
- Add grouped GEMM only after the routing trace fixes expert size
  distributions.
- Keep dense, grouped, and quantized GEMM as separate contracts.

Exit gate:

- Quantized outputs pass numerical and model-quality limits.
- Grouped GEMM passes per-expert correctness and end-to-end MoE output checks.
- Native and vendor providers use one plan and command lifecycle per contract.
- Real-engine throughput and memory improve beyond measured noise.

## R7: Add hardware targets

**State:** `sm_90a` is current. Future work targets `sm_100a` and `sm_120`.

Work:

- Keep H20 `sm_90a` as the first qualified architecture row.
- Add `sm_100a` as a separate provider artifact and evidence row.
- Add `sm_120` only from a measured consumer workload.
- Give TMA, WGMMA, and tcgen05 matrix algorithms stable names.
- Publish hashes for every admitted PTX or cubin.
- Reject incompatible devices before module load.

Target admission gate:

- Hardware is available for correctness, sanitizer, SASS, Graph, and
  performance work.
- cuda-oxide supports every required type and instruction with a pinned
  revision.
- One engine workload justifies the target-specific algorithm.

Exit gate:

- Each architecture has independent host, device, sanitizer, Graph,
  performance, and engine rows.
- No target inherits qualification from PTX forward compatibility.
- No empty `sm100` or `sm120` directory exists.

## R8: Add communication and release a stable API

**State:** planned.

Work:

- Start collectives from one measured tensor-parallel or expert-parallel
  workload.
- Define communicator ownership, stream order, topology, timeout, failure, and
  recovery boundaries.
- Use an explicit vendor provider unless a measured native alternative exists.
- Stabilize the Rust API after the first production-shaped engine adapter.
- Audit package contents, CI targets, documentation, and registry access.

Communication exit gate:

- Multi-GPU correctness covers ordering, partial failure, and teardown.
- Performance uses the same topology and payload distribution as the engine.
- A collective failure cannot release live device resources early.

Stable release exit gate:

- All documented current rows have current-source evidence.
- Public names follow one lifecycle and final family namespaces.
- The release contains no experimental default selection or silent fallback.
- A clean downstream install and one pinned engine integration pass.

## Evidence rule

Every result states its contract, source, provider, algorithm, artifact,
hardware, timed region, accepted claims, and excluded claims. Reviewed records
remain immutable. A faster kernel never proves a faster model or server by
itself.
