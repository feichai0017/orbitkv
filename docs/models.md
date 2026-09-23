# Model qualification

OrbitKV qualifies a checkpoint, engine release, cache format and execution
path together. A model family name or an engine's ability to load weights is
not evidence that OrbitKV can restore every state it needs.

The current engine baselines are **vLLM 0.29.0** and **SGLang 0.5.20**. Engines
own GPU allocation; OrbitKV restores registered attention pages and sealed
checkpoints through the [compiled recovery contract](hybrid-recovery.md).
See [deployment patterns](deployment.md) for topology limits.

## Pretrained checkpoints

The larger-model qualification uses one H20 with approximately 96 GiB of
device memory, TP=1 and text-only requests. FP8 below describes model weights;
attention caches use BF16 and recurrent tensors retain their engine-declared
dtypes, including FP32 state. Results establish
recovery correctness, not model quality, throughput or multimodal support.

| Checkpoint | State layout | vLLM | SGLang |
| --- | --- | --- | --- |
| [Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a) | 16 full-attention layers + 48 GDN layers with conv/recurrent state | DRAM and forced SSD: 7 checks each passed | DRAM and forced SSD passed, including concurrent restart and output controls |
| [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash/tree/7dd20894a642a0aa287e9827cb1a1f7f91386b67) | MLA; BF16 weights | SSD-enabled: 6 checks passed, 1 recurrent-only check skipped | SSD-enabled recovery, concurrent restart and output controls passed |
| [DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/tree/85864749cd611b4353ce1decdb286193298f64c7) | MLA; BF16 weights | SSD-enabled: 6 checks passed, 1 recurrent-only check skipped | SSD-enabled recovery, concurrent restart and output controls passed |
| [Kimi-Linear-48B FP8](https://huggingface.co/nm-testing/Kimi-Linear-48B-A3B-Instruct-FP8-DYNAMIC/tree/c3a71758d772dc8ff6c207a034a722cb8decff77) | 7 MLA layers + 20 KDA layers with conv/recurrent state | SSD-enabled: 7 checks passed | SSD-enabled recovery, concurrent restart and output controls passed |

The Kimi checkpoint is a third-party quantization of
[Moonshot's Kimi Linear](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/tree/e1df551a447157d4658b573f9a695d57658590e9),
not an official Moonshot FP8 release. Its weights occupy about 46.5 GiB; the
official BF16 weights alone occupy about 91.5 GiB, leaving insufficient room
for this single-card gate. DeepSeek-V2-Lite is an older MLA regression target;
it does not establish DeepSeek-V4 compatibility. SSD-enabled runs keep DRAM
available during serving and require SSD recovery after eviction and restart;
they are not separate DRAM-only qualifications.

The Kimi quantization repository's tokenizer imports a removed Transformers
interface. The qualification artifact keeps those pinned weights and uses
`tokenization_kimi.py` from the pinned official Moonshot revision above. The
official file changes only the `bytes_to_unicode` import to
`transformers.convert_slow_tokenizer`; no OrbitKV compatibility shim is used.
The launches use `--trust-remote-code` for this tokenizer. Reproduce this
artifact after downloading the quantized checkpoint:

```bash
hf download moonshotai/Kimi-Linear-48B-A3B-Instruct tokenization_kimi.py \
  --revision e1df551a447157d4658b573f9a695d57658590e9 \
  --local-dir /path/to/kimi-linear-48b-fp8
```

The smaller Qwen3-8B dense and Qwen3.5-0.8B recurrent baselines retain their
existing gates. Mellum and Inkling use generated weights to exercise native
engine paths; they are listed separately in the
[state-layout qualification](hybrid-recovery.md#reproducible-gates).
Full + SWA + temporal recurrent serving still needs native-model evidence.

### Qwen3.8 on H20

In this environment, native vLLM with `DeepGemmFp8BlockScaledMMKernel`
returned HTTP 400 (`Out of range float values are not JSON compliant`) before
OrbitKV was enabled. With `VLLM_USE_DEEP_GEMM=0`, vLLM selected
`CutlassFp8BlockScaledMMKernel`; native short/long prompts returned finite
log probabilities and both cache-tier correctness gates passed. Keep this
setting on both the native and OrbitKV sides when reproducing these results. This is a
recorded kernel configuration, not an OrbitKV default or a fix to vLLM.

Run from `python/` with a freshly built extension and Manager. Keep native
builds separate from running GPU gates:

```bash
export ORBITKV_CACHE_MANAGER_BINARY=/absolute/path/to/orbitkv-cache-manager

VLLM_USE_DEEP_GEMM=0 ../.venv/vllm-release/bin/python -m pytest -m e2e \
  tests/e2e/test_vllm_e2e_correctness.py \
  --model /path/to/qwen3.8-27b-fp8 --max-model-len 4096 \
  --orbitkv-pool-size 4gb --vllm-cache-tier dram

# Repeat with --vllm-cache-tier ssd to force SSD-backed restart recovery.
SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 ../.venv/sglang-release/bin/python -m pytest -m e2e \
  tests/e2e/test_sglang_direct_e2e.py --model /path/to/qwen3.8-27b-fp8
```

The vLLM gate compares an ordered native-prefix execution plan with OrbitKV
and requires GPU restore bytes after engine restart. Its SSD mode waits for
writes to drain, evicts Manager DRAM after engine exit, and requires new SSD
reads as well as actual GPU restores. The SGLang gate runs both tiers and
checks native HBM reuse, HBM flush, concurrent restart recovery, changed-identity
misses and finite generated-token log probabilities. This container's SSD
cache file is on an overlay mount; it does not qualify physical NVMe speed.
Both SSD serving gates use an 8 GiB cache. Size it for attention pages and
the retained checkpoint set; a small cache can evict a required component and
correctly turn the next lookup into a miss.
For Qwen, SGLang's fast warmup option samples fewer initialization shapes;
CUDA graphs and the inference kernels remain enabled. Initialization is not
a performance measurement in these gates.

### MLA controls on H20

Run the SSD-enabled MLA gates with GLM-4.7-Flash, DeepSeek-V2-Lite or the
Kimi artifact prepared above:

```bash
VLLM_TEST_ATTN_BACKEND=FLASH_ATTN_MLA ../.venv/vllm-release/bin/python -m pytest -m e2e \
  tests/e2e/test_vllm_e2e_correctness.py --model /path/to/checkpoint \
  --max-model-len 4096 --orbitkv-pool-size 4gb --vllm-cache-tier ssd

../.venv/sglang-release/bin/python -m pytest -m e2e \
  'tests/e2e/test_sglang_direct_e2e.py::test_sglang_direct_gpu_cache_recovery[ssd]' \
  --model /path/to/checkpoint
```

The vLLM helper's batch-invariant control must retain native prefix caching:
vLLM disables that combination for `TRITON_MLA`, and the gate rejects it for
having no native hit.
The KDA workload uses a longer prefix and a 2,048-token scheduling budget to
cross its larger aligned checkpoint boundary; other recurrent workloads retain
their existing prompt lengths. Kimi's native vLLM hit boundary was 1,920 tokens.
For Kimi, SGLang automatically disables prefill CUDA graphs; decode graphs
remain enabled. No dummy-weight preflight is counted in the table above.

## Larger hybrid targets

Weight totals below are sums of the pinned repositories' top-level
`.safetensors` files, excluding runtime workspaces and cache/state memory.
MoE active parameters reduce per-token computation; ordinary GPU-resident
deployment still needs the complete expert weights.

| Target | Weight files | State work remaining |
| --- | ---: | --- |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a) | 305.79 GiB, FP8 | 34 KDA layers + 11 sparse MLA layers; include compressed indexer state and its token-to-page mapping in recovery |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1) | 148.66 GiB, FP8 | Sliding/compressed attention, indexer state and auxiliary request pools need complete registration and recovery |
| [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06) | 1,453.74 GiB | Qualify its MLA/KDA pools, layouts and multi-rank restore; Kimi Linear results cannot substitute for this checkpoint |

None fits this H20 as a fully GPU-resident TP=1 model. CPU/expert-weight
offloading or another quantization would be a separate deployment and requires
its own output and performance controls. OrbitKV offloads inference state;
it does not offload model weights.

GLM-5.3-Flash is a useful next state-contract target, not a qualified model.
SGLang's OrbitKV plugin currently rejects DSA and auxiliary request pools.
The [vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash) also calls for
a model-enabled Docker build while integration is pending in the public tree;
the pinned source release has no native `Glm5NextForConditionalGeneration`
registration. Qualify the engine build before adding a cache result. MTP,
draft state, cross-engine byte reuse and remote hybrid recovery each retain
separate gates.
