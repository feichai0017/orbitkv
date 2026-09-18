# Roadmap and gates

Each milestone ends with executable evidence. Later features do not compensate
for a failed earlier gate.

## M0: trustworthy control plane

Deliver the portable contracts, validation, registry, deterministic selection,
provider ABI, fallback executor, source lock, CLI demo, and SGLang entry-point
skeleton. All tests run without a GPU.
The source workspace additionally pins SGLang, TensorRT-LLM AutoDeploy,
FlashInfer, DeepGEMM, and their nested native dependencies as Git submodules.

Gate: a malformed, mismatched, out-of-domain, over-budget, or unqualified plan
is rejected; a provider failure reaches an eligible fallback and is reported.

## M1: one correct GPU operator path

Add a source-pinned FlashInfer or vendor provider, device fingerprinting, AOT/JIT
artifact caching outside the request path, raw timing samples, and PyTorch
differential qualification for one decode operation.
Candidate bundles are admitted only through the artifact/trace/evidence staging
gate; the first schema uses exact shape buckets.

Gate: two implementations are qualified over explicit shape domains, selection
changes with hardware or shape, and every artifact is reproducible from source.

## M2: single-request dense decoder

Import one small dense decoder through `torch.export`, implement BF16 weights,
RoPE, GQA, KV state, greedy sampling, and layer/logit differential oracles.
Eager and CUDA Graph plans share one semantic model identity.

Gate: multi-step token parity and bounded numerical error across boundary shapes;
no allocation or compilation occurs in the measured decode loop.

## M3: production single-GPU serving

Add paged KV, continuous batching, chunked prefill, streaming, cancellation,
timeouts, admission control, OOM recovery, metrics, trace capture, and replay.
Promote the proven trace hook into the narrowest required SGLang model-runner
integration, upstreaming the interface where practical.

Gate: fault injection and soak tests show no leaked request/KV state; p50/p99
TTFT, TPOT, goodput, and memory are reproducible against pinned SGLang and vLLM
baselines.

## M4: closed-loop optimization

Generate bounded candidates across provider tactic, fusion, buffer layout, CUDA
Graph bucket, prefill chunk, and batch policy. Store raw evidence, fit a cost
model, qualify winners, canary them, and automatically roll back regressions.

Gate: on a held-out trace, the system produces and safely deploys distinct plans
for two GPUs or two workload distributions without source changes to the model.

## M5: differentiated state and scale

Add prefix/state value modeling, agentic multi-turn traces, recurrent or sparse
state, then multi-GPU placement and communication. Use Dynamo only when the
single-node executor contract is stable.

Gate: a cross-layer optimization beats strong production baselines end to end
and its correctness, state lifetime, and rollback evidence remain inspectable.

## Explicit non-goals before M3

- broad model and accelerator support;
- a new general tensor language;
- arbitrary online code generation;
- distributed control-plane work;
- custom GEMM or attention written without profiler evidence.
