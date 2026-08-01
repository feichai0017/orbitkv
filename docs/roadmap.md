# Roadmap

## Product boundary

Loom does not implement matrix multiplication. cuBLASLt, CUTLASS, FlashInfer,
or the engine owns dense, quantized, sparse, and grouped GEMM.

A new Loom operator must satisfy three conditions:

1. Memory traffic, launch overhead, layout conversion, or scheduling metadata
   causes the cost.
2. A named engine exposes a zero-copy boundary that Loom can enter.
3. A real model or serving workload can prove latency, memory, throughput, or
   goodput value.

A microbenchmark alone does not admit an operator.

## Current order

| Order | Work | Exit |
| --- | --- | --- |
| 1 | Run the production MoE movement gate | A pinned pretrained workload preserves output and improves latency with the same vendor grouped GEMM |
| 2 | Add measured quantization plumbing | A named vendor-kernel consumer removes an HBM pass, launch, or large temporary |
| 3 | Build a Rust decode proof | One deterministic step uses borrowed tensors and a borrowed stream without private engine state |
| 4 | Revisit profile-gated work | A named workload first shows material KV movement, speculative metadata, or communication cost |

## Milestones

| Milestone | State | Result or remaining exit |
| --- | --- | --- |
| K0: backend foundation | Complete | Contracts, CPU oracles, safe CUDA resources, and result format |
| K0.5: Rust distribution | Complete | Self-contained source crates at `1.0.0-alpha.1` |
| K0.6: runtime interop | Complete | One checked path over borrowed tensors and current streams |
| K0.7: native wheel | Complete, unpublished | ABI12 passes the PyTorch 2.10/2.11 and vLLM 0.24/0.25 H20 matrix |
| K1: normalization | In progress | FP8 is qualified. INT8 still needs quality and stable engine benefit |
| K2: MLP fusion | In progress | Base and FP8 paths are qualified. INT8 remains explicit and profile-gated |
| K2.5: quantization plumbing | In progress | Add scale, pack, and layout paths only for measured consumers |
| K3: KV cache | In progress | Static FP8 write works, but the first system candidate failed quality |
| K4: decode tail | In progress | Current sampling and logits paths are qualified. Serving-scale proof remains open |
| K4.5: speculative decode | Profile-gated | Greedy verification works, but it is below `0.2%` of the measured batch latency |
| K5: MoE movement | In progress | Direct and synthetic-engine gates pass. Production-model evidence is next |
| K6: attention | In progress | Short paged decode is admitted. FA3 remains the long-context backend |
| K7: communication fusion | Planned | Requires reproducible multi-GPU engine baselines |
| K8: Rust decode proof | Planned | Starts after the next engine-level result |

## Active tracks

### Production MoE gate

The benchmark must pin the checkpoint, prompts, engine version, provider order,
and grouped-GEMM implementation. It must record TTFT, TPOT, throughput, memory,
token equality, Loom path hits, and the unchanged vendor GEMM call.

Routing enters Loom only if the same profile shows that routing is material.
The current synthetic Qwen2-MoE result proves integration, not production value.

### Quantization plumbing

Candidate work includes scale reduction, pack and unpack, dequantization,
requantization, and scale-layout conversion. Each path must sit next to a named
vendor GEMM and remove measured traffic or temporary storage.

Loom will not add a generic quantization checklist or a second matrix core.

### Rust decode proof

The example will borrow engine-produced CUDA memory and a non-owning stream. It
will chain a small cache, logits, sampling, and token-output slice while every
GEMM and model-owned attention call remains external.

The example will own no scheduler, model weights, tokenizer, or KV-cache
lifetime.

## Profile-gated backlog

- KV copy, gather, scatter, compaction, or remap for a named physical-movement
  path. Default vLLM prefix reuse and preemption did not copy data.
- Tree masks, stochastic speculative rejection, or speculative KV updates when
  a named draft and target pair exposes material overhead.
- FP8 or INT8 KV variants after a pinned model passes the quality precondition.
- Tensor-parallel or expert-parallel fusion after an equivalent NCCL or
  transport baseline exists.
- Broader paged attention only where it beats the engine's selected backend.

## Completion rule

Each operator advances through contract, oracle, CUDA correctness, named
baseline, framework dispatch, engine invocation, and system-value gates.
Passing one gate never closes the next. The [operator catalog](operator-catalog.md)
tracks the full candidate surface.
