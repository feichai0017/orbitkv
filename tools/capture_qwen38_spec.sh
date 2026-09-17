#!/usr/bin/env bash
# Mine the DFlash2 speculative path of Qwen3.8-27B out of vLLM 0.28: the
# draft model's kernels (5 sliding-window non-causal layers, grouped conv,
# candidate selector), the target's verify forward (8-token block) and the
# GDN state handling under speculation (mtp / num_accepted_tokens kernels).
# Same backend pins as tools/capture_qwen38.sh.
#
#   CUDA_VISIBLE_DEVICES=0 tools/capture_qwen38_spec.sh   # -> dumped-kernels/pid<N>/
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hub=${HF_HUB:-$HOME/.cache/huggingface/hub}
model="${MODEL:-$hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
draft="${DRAFT:-$hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/dedf8df68adfb1afeaf7b7480c0a0243108177b4}"
out="${KERNEL_CAPTURE_DIR:-$repo/dumped-kernels}"

[ -f "$repo/tools/kernel-capture/libkernelcapture.so" ] || "$repo/tools/kernel-capture/build.sh"

mkdir -p "$out"
export PATH="$repo/.venv/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
export CUDA_INJECTION64_PATH="$repo/tools/kernel-capture/libkernelcapture.so"
export KERNEL_CAPTURE_DIR="$out"
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

"$repo/.venv/bin/python" - "$model" "$draft" <<'EOF'
import sys
from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig

llm = LLM(model=sys.argv[1], tokenizer=sys.argv[1], dtype="bfloat16",
          tensor_parallel_size=1, max_model_len=4096,
          gpu_memory_utilization=0.6, enforce_eager=True,
          limit_mm_per_prompt={"image": 0, "video": 0},
          attention_config=AttentionConfig(backend="TRITON_ATTN"),
          additional_config={"gdn_prefill_backend": "triton"},
          speculative_config={"method": "dflash", "model": sys.argv[2],
                              "num_speculative_tokens": 7})

prompts = [
    "The harbor master kept a ledger of every ship that wintered in the bay, "
    "noting cargo, crew, and the state of each hull in a cramped, looping hand.",
    "When the observatory finally reopened after the renovation, the docents "
    "discovered that the old refractor had been quietly recollimated by a "
    "retired machinist who lived nearby. He left no note, only a small brass "
    "shim on the pier and a chalked arrow pointing at Polaris.",
]
for p in prompts:
    o = llm.generate([p], SamplingParams(temperature=0.0, max_tokens=24))[0]
    print(f"prompt_tokens={len(o.prompt_token_ids)} out_tokens={len(o.outputs[0].token_ids)} "
          f"output={o.outputs[0].text!r}", flush=True)
EOF

echo "--- capture summary ---"
for d in "$out"/pid*/; do
  echo "$d: $(ls "$d" | grep -c cubin) cubins, $(wc -l < "$d/launches.jsonl") launches"
done
