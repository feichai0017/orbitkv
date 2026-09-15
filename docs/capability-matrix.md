# Model support

The current execution target is **Qwen3.8-27B-FP8**, for experimental text inference
on one NVIDIA H20. The tested workload below does not cover every Qwen3.8 size,
precision or modality. Extended numerical qualification remains open.

## Verified checkpoint

| Property | Value |
| --- | --- |
| Official repository | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) |
| Verified revision | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Architecture class | `Qwen3_5ForConditionalGeneration` |
| Text geometry | 64 layers, hidden size 5120, vocabulary 248320 |
| Attention | 16 Full attention layers + 48 Gated DeltaNet layers with convolution state |
| Precision | E4M3 FP8 projection weights, 128×128 FP32 block scales, BF16 activations/KV |
| Execution | Single GPU, greedy text generation, OpenAI-compatible HTTP streaming |

Qwen3.8 retains the Qwen3.5 architecture class in its
[official configuration](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/config.json).
That class identifies the importer, not the model release. Local verification
compares all 66 indexed weight shards and five configuration/tokenizer files
with the official revision's LFS SHA-256 or Git blob digest. The
[model results](../results/README.md) retain checkpoint identity and measurements.

## Tested scope

The serving configuration admits up to eight active requests, 512 tokens per
sequence and 32 input tokens per request. The aggregate batch budget is 256
query tokens. Current performance profiles cover C1/C8, fixed token-ID input
lengths 4/32, and output lengths 64/128. Historical random-text measurements
retain their observed input lengths.
These are measured bounds, not a long-context or maximum-capacity claim.

Numerical preflight uses an independent full-vocabulary, teacher-forced reference
and checks token-KV plus recurrent/convolution state drain. Its exact artifact,
prompts, steps and tolerances are recorded with the results. A passing probe
does not establish arbitrary-prompt equivalence. A wider 16-history, 64-step probe
exceeds the existing reference-error gate, and changing batch shapes can change
selected tokens. The retained-graph CSR reuse regression is fixed; these wider
numerical failures remain open. Generated text differences across batches or
engines are reported separately and require logit diagnosis.
The HTTP tokenizer has a measured token-count difference from the Python
baselines on synthetic text. Current cross-engine measurements use token-ID
requests; text-tokenizer parity remains a qualification item.

Vision, MTP, speculative decoding, CPU weight offload, distributed execution and
production soak are outside the current support claim.

## Implementation contracts

| Component | Implemented boundary |
| --- | --- |
| State manager | Full, Sliding and Chunked token lifetimes; Prefix/COW; generation-safe reuse; recurrent/convolution ownership |
| Model import | Explicit architecture, tensor-shape and quantization validation; see [import contracts](checkpoint-import.md) |
| Compiler | Egglog alternatives, bounded GPU profiling, state-alias/resource checks and deployment-graph finalist measurement |
| CUDA providers | Generated kernels, cuBLASLt, DeepGEMM, FlashInfer and optional FlashAttention-3; see [provider contracts](attention-providers.md) |
| Execution artifacts | Selected bucket programs and generated module images, bound to the model, arenas, tuning and execution environment |
| Serving | Bounded continuous batching, greedy sampling, streaming, cancellation and final state drain |
| External state | Host-tested export/restore transactions and reference byte transport; production remote KV placement remains open |

Additional importers and state fixtures provide compiler regression coverage.
They are not additional supported models. Joint state-layout search, general
algorithm-region search and persistent megakernels remain [roadmap work](roadmap.md).
