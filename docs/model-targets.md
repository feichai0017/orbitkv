# Model targets on one H20

Reviewed **2026-09-14**. The deployment budget is one H20; quantized weights
and CPU offload are allowed. Qwen, GLM, Kimi and DeepSeek are the primary
families. The current H20 reports 97,871 MiB of device memory. Qualification
must reserve space for state, activations, provider workspaces, retained CUDA
graphs and transfer staging as well as weights.

This is a dated development matrix. Only the bounded Qwen3.8-27B-FP8 text path
below has an OrbitKV model-level record. New targets need a pinned checkpoint
revision, tokenizer, tensor index and independent reference before admission.
An upstream engine's model support does not qualify our importer or kernels.

## Primary releases

| Family / official checkpoint | Structural requirements | Single-H20 route and current status |
| --- | --- | --- |
| [Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | Dense FFN, Gated DeltaNet + gated full attention, convolution/recurrent state, block-FP8 linear | **Current correctness and performance anchor.** Bounded text prefill/decode passes on H20. Long-context serving, vision and MTP remain unqualified. |
| [Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | GDN + Qwen Sparse Attention, MoE, gated residual, n-gram embeddings. The card lists 125B parameters with 6B active, plus 51B n-gram embeddings and 4B MTP | **First latest-architecture expansion target after shared MoE/attention foundations.** Requires import, sparse-index state and quantization/offload. Its architecture is `qwen4_exp`; the existing Qwen3.5-family importer does not cover it. |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | About 320B total / 18B active, MoE, hybrid sparse/linear attention, manifold-constrained Hyper-Connections | Requires semantic import, equivalent kernels and host weight residency. Even ideal uniform four-bit payload is about 160 GB before metadata and execution memory. It cannot be fully resident on one H20 at that encoding. |
| [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | 2.8T total / 104B active, KDA + gated MLA, latent MoE, Attention Residuals, MXFP4 weights / MXFP8 activations | A full-model offload research target after KDA/MLA, MoE and quantized execution pass independently. Ideal four-bit payload alone is about 1.4 TB. Host capacity and transfer/CPU-compute cost must pass explicit gates; single-H20 throughput is unqualified. |
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 552B backbone, causal encoder/decoder, CSA2 sparse attention and cross-layer reuse, compressed KV, MoE, Engram conditional memory and mHC | Requires new semantic/state contracts and quantization/offload. Backbone payload alone is at least 276 GB at ideal uniform four-bit storage. The currently documented FlashMLA V4.1 sparse-decode path is SM100-only, so this target also needs an equivalent H20 implementation. |

The Qwen organization also publishes a 2.4T-A95B member of Qwen3.8. Flash-Next
is the more bounded expansion target for this single-device project; 27B remains
the regression anchor. See the [official Qwen collection](https://huggingface.co/Qwen).

These payload calculations are **planning lower bounds**, using decimal GB/TB
and `parameter_count × bits / 8`. They are not checkpoint file sizes, available
quantizations, RAM reservations or measured deployments. Mixed precision
tensors, scales, padding, vision/MTP tensors and duplicated runtime
representations can increase storage. Active parameter counts describe compute
sparsity; they do not eliminate the other experts' weights.

The current machine reports approximately 1.88 TiB of host RAM across two NUMA
nodes. This machine-wide observation is not a guaranteed process allocation.
Record the process/cgroup budget, available RAM and pinned-memory ceiling before
any large-checkpoint download or offload experiment.

## Smaller architecture witnesses

Use released smaller checkpoints to qualify common mechanisms before paying
the full target's memory and compilation costs. These are explicitly **earlier
architecture witnesses**, not substitutes advertised as the latest models.

| Checkpoint | Mechanism to establish | Memory plan |
| --- | --- | --- |
| [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) | 30B-A3B MoE import, routing, expert dispatch/combine and the checkpoint's attention semantics | About 60 GB of ideal BF16 payload; a practical single-H20 bring-up candidate after tensor-index and workspace accounting. |
| [DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat) | 16B / 2.4B active, MoE + MLA; independent latent-state lifecycle and numerical reference | About 32 GB of ideal BF16 payload. The official card describes single-40-GB deployment, but OrbitKV still needs its own backend qualification. |
| [Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) | KDA + global MLA and MoE; the finer-grained KDA gate must not be treated as the existing GDN operation | About 96 GB of ideal BF16 payload, leaving little H20 headroom. Plan a validated quantized format or bounded offload; the official BF16 release is not a single-H20 qualification. |

Passing these witnesses proves only their admitted contracts. It does not prove
QSA, CSA2, cross-layer KV reuse, gated MLA, latent MoE, residual mixing, or any
other changed semantics in a newer release. A distilled dense model is not
evidence for its teacher's MoE or attention architecture.

## Delivery order and gates

1. **Keep Qwen 27B correct and reduce wasted compilation.** Reject broken
   persistent-state aliases before source generation; retain final-load checks.
   Initial local region exploration, staged `glumoe` matching and reusable
   bucket setup are implemented. Next, investigate `kernel_specialize`, coherent
   region choices and generated-output consistency across batch compositions.
   Retain workload/binary/artifact identities. Cold compilation,
   artifact loading, uninstrumented runtime and serving latency remain separate
   measurements. Broader candidate counts alone are not a performance result.
2. **Establish shared MoE and latent-attention contracts.** Describe router
   scoring, expert selection, normalization, shared experts, dispatch/combine,
   weight encoding and latent KV explicitly. Add equivalent provider candidates
   through egglog. Qualify GLM and DeepSeek witnesses with independent logits,
   routing and next-state references, dynamic shapes and final drain.
3. **Extend hybrid and sparse attention.** Add KDA with its own scan/state
   semantics; then explicit indexer, selection, compressed-state and residual
   contracts for the latest targets. Provider names never imply model coverage.
   [FlashMLA's support matrix](https://github.com/deepseek-ai/FlashMLA#available-kernels)
   currently restricts V4.1 sparse decoding to SM100 and its fused norm/RoPE/
   attention/cast path to SM100. A generic SM90 library label cannot admit them
   on H20. Pin and test the exact kernel, geometry, dtype and build toolchain.
4. **Add bounded host weight residency.** Start with immutable lookup tables
   and explicit expert staging, then evaluate CPU expert execution as a
   separate implementation. Quantization needs numerical qualification and a
   compatible SM90 kernel; MXFP4 is not interchangeable with the existing
   block-FP8 path. Admit the first offloaded latest checkpoint only after memory,
   transfer, cancellation and output gates pass.
5. **Qualify complete requests.** Pin a model revision and precision, prompt
   format, context lengths, concurrency, quantization and offload policy. Check
   logits/next state, EOS and output length, prefix reuse, cancellation and
   final drain. Report TTFT, inter-token latency, throughput, tail latency,
   peak GPU/host memory, transferred bytes, cache misses and cold/warm costs.
   Compare against a mature engine under the same one-H20 and host budget.

## Joint planning with offload

The existing memory-mapped checkpoint loader transfers admitted weights to the
GPU at load time. It is not a demand-paged expert cache or CPU execution backend.
External KV transport also does not implement weight offload.

The planned deployment object binds a compatible **compute schedule, mutable
state realization and immutable weight-residency plan**. OrbitKV keeps authority
over KV/recurrent/convolution ownership and publication. The executor coordinates
immutable host/device weight storage and transfers; OrbitKV compiler owns compute
candidates and their kernel/workspace requirements. Their physical allocations
share one device budget, with explicit host and staging budgets.

Weight cache eviction must wait for the last consuming CUDA event. Request-state
reuse must still satisfy OrbitKV's generation and completion protocol. Capture
must bind stable staging slots or use a validated recapture policy; an evicted
expert pointer cannot silently remain in a CUDA graph. Candidate profiling uses
scratch request state and records misses and transfers under the deployment
policy, not only an all-resident kernel time.

CPU expert execution has an explicit host execution and transfer boundary. It
cannot be treated as a GPU kernel inside an existing captured CUDA graph or
as an alias for copying expert weights to the device.

Start with a finite set of explicit placement/prefetch plans. Measure their
complete-request cost under memory limits, retaining a correct executable
alternative. GPU cache size, host budget, queue depth and staging capacity are
configuration and measured device facts, never checkpoint-name dispatch or
hidden constants. General automatic offload planning and persistent kernels
follow this evidence; they are not required for the first model bring-up.
