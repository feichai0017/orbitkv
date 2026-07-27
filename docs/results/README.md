# H20 evidence index

Machine-readable correctness and performance artifacts for Loom Kernels. All
results in this directory were captured on NVIDIA H20 unless the artifact says
otherwise.

[Documentation](../README.md) · [Implementation status](../status.md) · [Benchmark page](https://feichai0017.github.io/loom-kernels/benchmarks/)

> [!IMPORTANT]
> Operator latency, dispatcher latency, CUDA Graph replay, engine latency, and
> serving performance are different claims. The decision column below states
> the narrow conclusion supported by each result set.

## Evidence ladder

| Level | What it proves |
| --- | --- |
| Correctness | The accelerator agrees with the declared oracle and contract |
| Operator | A warmed kernel or fused boundary beats a named equivalent baseline |
| Engine | A real framework/engine invokes Loom and preserves outputs |
| Serving | TTFT, TPOT, throughput, memory, or goodput improves under load |

## Compatibility

| Boundary | Result set | Current conclusion |
| --- | --- | --- |
| Native Python ABI8 cross-matrix wheel | [current clean-install H20 gate](h20-native-wheel-clean-install-abi8-20260727.json) | Revision `e2c2982` packages all seventeen operators and exactly two Loom `.so` files, then passes 293 tests with each vLLM minor plus 199 applicable tests on PyTorch 2.10 from fresh repository-free environments. It is qualified but not published. |
| Historical ABI7 vLLM 0.24 refresh | [refresh clean-install H20 gate](h20-native-wheel-clean-install-abi7-refresh-20260727.json) | Revision `f98a931` packages the final FP8 KV adapter and exactly two Loom `.so` files, then passes 286/286 full and 22/22 focused tests from a fresh repository-free environment. |
| Historical ABI7 cross-matrix wheel | [first complete clean-install H20 gate](h20-native-wheel-clean-install-abi7-20260727.json) | One exact `py3-none-linux_x86_64` wheel contains the two Loom `.so` files and passes 286 tests with each vLLM minor plus 193 applicable tests on PyTorch 2.10. |
| Historical ABI6 matrix wheel | [predecessor clean-install H20 gate](h20-native-wheel-clean-install-abi6-20260727.json) | Preserved as the earlier 277/186-test artifact before fused mixed-sampling logits preprocessing entered the packaged ABI. |
| Historical ABI5 matrix wheel | [predecessor clean-install H20 gate](h20-native-wheel-clean-install-abi5-20260727.json) | Preserved as the earlier 268/178-test artifact before fused top-p filtering and renormalization entered the packaged ABI. |
| Historical ABI4 matrix wheel | [predecessor clean-install H20 gate](h20-native-wheel-clean-install-abi4-20260725.json) | Preserved as the earlier 253/164-test artifact before exact top-k filtering entered the packaged ABI. |
| Historical ABI2 matrix wheel | [predecessor clean-install H20 gate](h20-native-wheel-clean-install-abi2-20260724.json) | Preserved as the earlier 225/138-test artifact before sparse penalties and top-k logprobs entered the packaged ABI. |
| Historical ABI1 matrix wheel | [first clean-install H20 gate](h20-native-wheel-clean-install-20260723.json) | Preserved as the earlier 192/123-test artifact. |
| LibTorch Stable ABI across PyTorch minors | [two-minor H20 binary gate](h20-libtorch-stable-abi-20260723.json) | The source-built predecessor established the PyTorch 2.10 Stable ABI target and same-binary 2.10/2.11 boundary; the packaged clean-install result is the row above. |
| Pre-Stable-ABI single Rust bridge | [breaking-change H20 gate](h20-single-rust-bridge-compatibility-20260723.json) | Historical revision `cb5feaf` first proved all ten framework families on the Rust-owned path and passed 191 tests on each vLLM minor. The current dispatcher result is the row above. |
| Historical partial-bridge baseline | [pre-unification 0.24/0.25 gate](h20-vllm-compatibility-rust-bridge-20260723.json) | Preserved as historical evidence for revision `3ae4210`; its raw-ABI routing description does not apply to the current architecture. |

## Normalization and activation

| Boundary | Result set | Current conclusion |
| --- | --- | --- |
| RMSNorm | [F32 bring-up](h20-rms-norm-f32-smoke-20260721.json) · [FP16/BF16 paths](h20-rms-norm-low-precision-20260721.json) | Handwritten CUDA correctness and low-precision vector paths are qualified |
| Add+RMSNorm | [Operator gate](h20-add-rms-norm-20260721.json) · [vLLM IR gate](h20-vllm-ir-add-rms-norm-20260721.json) | Double in-place fusion and current-stream engine dispatch are supported |
| RMSNorm→FP8 | [Operator gate](h20-rms-norm-dynamic-fp8-20260721.json) · [Qwen2.5 engine gate](h20-vllm-qwen25-05b-fp8-engine-20260722.json) | Exact path invocation is proven; real-model latency remains at parity |
| SiLU-and-Mul | [Operator and engine gate](h20-silu-and-mul-20260721.json) | Compatible and engine-valid; CUDA Graph latency is at parity |
| SiLU-and-Mul→block FP8 | [Fused operator gate](h20-silu-and-mul-dynamic-fp8-20260721.json) · [Qwen2.5 engine gate](h20-vllm-qwen25-05b-fp8-engine-20260722.json) | Operator-level advantage; exact real-model invocation; end-to-end parity |

## RoPE and paged-KV write

| Result set | Current conclusion |
| --- | --- |
| [Decode-sized operator sweep](h20-rope-paged-kv-20260722.json) · [large-token sweep](h20-rope-paged-kv-large-20260722.json) | Fusion wins most strongly at decode-sized token counts and narrows with larger batches |
| [Baseline-first engine gate](h20-vllm-qwen25-rope-paged-kv-engine-20260722.json) · [Loom-first engine gate](h20-vllm-qwen25-rope-paged-kv-engine-loom-first-20260722.json) | Exact tokens and Loom path hits are proven; order reversal crosses parity, so no model-level speedup is claimed |
| Static FP8 E4M3 cache: [cache-write gate](h20-fp8-kv-cache-write-20260724.json) · [rejected Qwen2.5 system candidate](h20-fp8-kv-system-rejected-20260727.json) | Exact bytes, framework/clean-wheel coverage, `1.317-1.378x` operator ratios, `1.99879x` cache-token capacity, and native-vLLM/Loom FP8 provider equivalence are qualified. An 8-sequence, 1,016-scored-token early-stop slice rejects the pinned Qwen2.5 candidate at about `3.07x` FP8/BF16 perplexity; the formal TTFT/TPOT matrix was not run. |
| [Rejected default prefix/preemption movement candidate](h20-vllm-kv-movement-admission-rejected-20260727.json) | A real vLLM V1 run records a 1,024-token prefix hit and three preemptions, but zero physical swap/copy calls or bytes. Prefix reuse is logical and preemption recomputes, so Loom adds no operator for this default path. |

## Sampling and log probabilities

| Boundary | Result set | Current conclusion |
| --- | --- | --- |
| Deterministic categorical sampling | [Direct ABI8 H20 gate](h20-categorical-sample-20260727.json) · [baseline-first engine](h20-vllm-engine-categorical-sample-20260727.json) · [Loom-first engine](h20-vllm-engine-categorical-sample-loom-first-20260727.json) · [source-pinned admission](h20-vllm-seeded-sampling-admission-20260727.json) | Rust/CUDA/PyTorch exact replay, state, distribution, stream, compile, and graph gates pass. Persistent vLLM 0.24/0.25 request state survives remove/condense/resume/swap. Direct 4/7/8/32-row ratios are `1.15–5.40x`; Qwen batch 32 is an order-stable `1.057–1.081x`, while batch 1–4 has a measured `1.5–2.4%` engine cost. |
| Greedy + sampled logprob | [Operator gate](h20-greedy-sample-logprobs-20260722.json) · [baseline first](h20-vllm-greedy-logprobs-baseline-first-20260722.json) · [Loom first](h20-vllm-greedy-logprobs-loom-first-20260722.json) | Exact tokens/ranks and an order-stable real-engine win for pure greedy `logprobs=0` |
| Selected-token logprob + rank | [Operator gate](h20-selected-token-logprobs-20260722.json) · [baseline first](h20-vllm-selected-logprobs-baseline-first-20260722.json) · [Loom first](h20-vllm-selected-logprobs-loom-first-20260722.json) | vLLM-owned top-k/top-p sampling preserves exact tokens/ranks and shows an order-stable engine win |
| Sampled-token + top-k logprobs | [Direct operator](h20-topk-sampled-logprobs-20260725.json) · [baseline first](h20-vllm-qwen25-topk-logprobs-baseline-first-20260725.json) · [Loom first](h20-vllm-qwen25-topk-logprobs-loom-first-20260725.json) | Direct deterministic reduction is `3.25x/2.60x/1.19x` at 1/8/32 rows and near parity at 128, with sharply smaller temporaries. The vLLM adapter preserves exact engine top-k order/ranks and records `1440/0` launches, but latency crosses parity after order reversal, so no stable engine speedup is claimed. |
| Exact in-place top-k filter | [1–7-row operator gate](h20-top-k-filter-20260727.json) | At 151,936 F32 logits and `top_k=50`, threshold masks and retained logits are exact; Loom is `1.42–2.15x` faster than vLLM's small-row full sort and uses `0.62–4.36 MB` rather than `4.90–47.01 MB` of peak temporary storage. |
| Fused top-p + renormalization | [151,936-vocabulary gate](h20-top-p-renorm-20260727.json) · [32,768-vocabulary boundary](h20-top-p-renorm-vocab32768-20260727.json) | Every admitted 2/4/7-row F32 top-p-only case beats vLLM's full sort plus softmax: `1.72–1.77x` at 151,936 and `1.15–1.34x` at the minimum admitted vocabulary. Retained logits are exact; differing parallel F32 scan order may move one cutoff token with per-row probability L1 at most `1e-4`. |
| Fused mixed-sampling logits preprocessing | [Direct operator](h20-logits-preprocess-20260727.json) · [baseline first](h20-vllm-logits-preprocess-baseline-first-20260727.json) · [Loom first](h20-vllm-logits-preprocess-loom-first-20260727.json) | One exact F32 mask/bias/suppression/temperature pass is `3.26–7.30x` faster at 1–32 rows with zero measured temporaries. Qwen2.5-0.5B preserves every token and records `720/0` launches per order; TPOT is order-stable at `1.010–1.084x`, while batch latency crosses parity at batch 32. |
| Min-P | [151,936-vocabulary sweep](h20-min-p-filter-20260722.json) · [65,536-vocabulary boundary](h20-min-p-filter-vocab65536-20260722.json) | The crossover is shape-dependent; the adapter routes only qualified larger rows/vocabularies |
| Repetition/frequency/presence penalties | [Sparse-history H20 gate](h20-token-penalties-20260725.json) · [baseline first](h20-vllm-qwen25-token-penalties-baseline-first-20260725.json) · [Loom first](h20-vllm-qwen25-token-penalties-loom-first-20260725.json) | Exact vLLM semantics; `5.82–34.30x` operator ratios with zero operator temporaries. Qwen2.5-0.5B preserves every token with `1440/0` Loom launches per provider order and order-stable `1.056–1.123x` batch-latency and `1.068–1.126x` TPOT ratios. |

## Speculative decoding

| Boundary | Result set | Current conclusion |
| --- | --- | --- |
| Greedy verify + accepted/bonus compaction | [15-case H20 gate](h20-greedy-speculative-verify-20260723.json) | Bit-exact with vLLM 0.24 across batches 1-256 and draft lengths 1/4/8; `1.101-1.128x` operator-level ratio. The source suite also passes 202 tests on vLLM 0.24 and 0.25.1. No end-to-end model claim. |
| Real draft/target engine gate | [native first](h20-vllm-qwen25-speculative-native-first-20260723.json) · [Loom first](h20-vllm-qwen25-speculative-loom-first-20260723.json) | Qwen2.5-1.5B target plus 0.5B draft on vLLM 0.24 preserves exact native/Loom tokens and draft statistics with `714/714` measured Loom calls in each order. Loom's verifier boundary is `1.026-1.133x` faster but only `0.048-0.200%` of batch latency; end-to-end native/Loom ratios cross parity and speculative decode is `3.18-4.97x` slower than target-only in these cases. |

The target-only and speculative providers use different target-model execution
shapes. At batch 32, both speculative providers follow the same deterministic
trajectory while two of 32 target-only requests diverge after generated token
51 or 53; batch 1 and 8 match fully. The reports retain those mismatches and
make target-only equality informational. Exact native-vLLM versus Loom
speculative output is the correctness gate.

## Paged-decode attention

| Evidence set | Result set | Current conclusion |
| --- | --- | --- |
| Bring-up | [Separate-cache report](h20-paged-decode-attention-20260722.json) | Base Rust/CUDA/PyTorch contract and correctness path |
| Native-layout breadth | [156-case interleaved sweep](h20-paged-decode-interleaved-shape-sweep-20260722.json) · [Qwen 32/8 batch sweep](h20-paged-decode-qwen-batch-sweep-20260722.json) | Performance is geometry-dependent; only the measured short envelope is admitted |
| vLLM short route | [Backend gate](h20-vllm-paged-decode-backend-20260722.json) · [baseline-first engine gate](h20-vllm-paged-decode-engine-baseline-first-20260722.json) · [Loom-first engine gate](h20-vllm-paged-decode-engine-loom-first-20260722.json) | All 24 admitted backend cases win; synthetic-engine path and stable tokens are proven, not pretrained-model acceleration |
| Odd GQA experiment | [72-case sweep](h20-paged-decode-odd-gqa-20260722.json) · [rejected Qwen2.5 route](h20-vllm-qwen25-paged-decode-rejected-20260722.json) · [32/8 non-regression gate](h20-vllm-paged-decode-tail-gqa-backend-20260722.json) | The broader Qwen2.5 route failed token and latency gates and was removed; the existing route remains qualified |
| Local split-K/LSE | [BF16 block-16](h20-paged-decode-split-k-20260722.json) · [FP16](h20-paged-decode-split-k-f16-20260722.json) · [BF16 block-32](h20-paged-decode-split-k-block32-20260722.json) | Faster than legacy Loom across the tested long-context matrix; FA3 remains the engine fallback |

## Reproducing a claim

1. Use the commit, GPU, software versions, shapes, warm-up, iterations, and
   sample counts recorded in the JSON artifact.
2. Run correctness before timing.
3. Keep the named baseline and dispatch boundary unchanged.
4. Reverse provider order for engine comparisons.
5. Report regressions and rejected routes alongside wins.
