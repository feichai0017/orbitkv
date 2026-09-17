#!/usr/bin/env bash
# Mine Qwen3.8-27B (Qwen3.5 hybrid GDN + full attention) kernels out of vLLM
# 0.28 with the CUPTI injection lib: every module's cubin + every launch's
# flat ABI into dumped-kernels/pid<N>/.
#
# Backend pins (the only flat-ABI combination, see docs/qwen38-bringup.md):
#   attention  TRITON_ATTN   (FlashInfer/trtllm-gen + FA4 are packed structs)
#   GDN prefill triton/FLA   (default FlashInfer GDN prefill is CuTe DSL)
#   GDN decode  VLLM_GDN_DECODE_KERNEL default; the CUDA mtp kernel needs
#               num_v_heads == 8*num_k_heads (48 != 8*16) so the non-spec
#               decode path is causal_conv1d_update + packed recurrent (Triton)
#
# Run on a free GPU:
#   CUDA_VISIBLE_DEVICES=0 tools/capture_qwen38.sh
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hub=${HF_HUB:-$HOME/.cache/huggingface/hub}
model="${MODEL:-$hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
out="${KERNEL_CAPTURE_DIR:-$repo/dumped-kernels}"

[ -f "$repo/tools/kernel-capture/libkernelcapture.so" ] || "$repo/tools/kernel-capture/build.sh"

mkdir -p "$out"
export PATH="$repo/.venv/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
export CUDA_INJECTION64_PATH="$repo/tools/kernel-capture/libkernelcapture.so"
export KERNEL_CAPTURE_DIR="$out"
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
# EngineCore in-process: the injection lib sees the pid that launches kernels.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

"$repo/.venv/bin/python" - "$model" <<'EOF'
import sys
from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig

llm = LLM(model=sys.argv[1], tokenizer=sys.argv[1], dtype="bfloat16",
          tensor_parallel_size=1, max_model_len=4096,
          gpu_memory_utilization=0.6, enforce_eager=True,
          limit_mm_per_prompt={"image": 0, "video": 0},
          attention_config=AttentionConfig(backend="TRITON_ATTN"),
          additional_config={"gdn_prefill_backend": "triton"})

# Several prompt lengths, one at a time (bs=1): each prefill is a sample of
# the same dispatch under a different `tokens`, so grid expressions
# (ceil_div(tokens,64) for the FLA chunk kernels, tokens*heads, ...) can be
# fitted instead of guessed. Diverse prose only.
prompts = [
    "hi",
    "The harbor master kept a ledger of every ship that wintered in the bay, "
    "noting cargo, crew, and the state of each hull in a cramped, looping hand.",
    "When the observatory finally reopened after the renovation, the docents "
    "discovered that the old refractor had been quietly recollimated by a "
    "retired machinist who lived nearby. He left no note, only a small brass "
    "shim on the pier and a chalked arrow pointing at Polaris. The director "
    "considered filing a complaint, then looked through the eyepiece at Saturn "
    "and decided some trespasses are better rewarded than reported.",
    "The floodplain census took three summers to complete. In the first "
    "summer the crews mapped oxbow lakes and counted heron rookeries from "
    "canoes, losing two clipboards and one outboard motor to the river. In "
    "the second they walked transects through willow thickets, recording "
    "beaver sign, sediment depth, and the stranded hulks of fence posts from "
    "farms abandoned in the forties. The third summer was all reconciliation: "
    "duplicate plots resolved, disputed species calls sent to the herbarium, "
    "and the great argument over whether the channel had truly migrated or "
    "merely braided settled by an afternoon with the 1962 aerial photographs. "
    "The final report ran to four hundred pages, and the appendix everyone "
    "actually read — the one with the flood marks painted on grain elevators "
    "— was written in a single evening by the youngest technician on the crew.",
    "The lighthouse keeper's daughter learned to read from shipping manifests "
    "and weather logs, so her first stories were inventories: forty barrels of "
    "salt cod, a crate of oranges gone soft, one bishop travelling incognito. "
    "By twelve she had catalogued every wreck on the reef by the sound it made "
    "in a north wind, and by fifteen she was writing letters to the harbor "
    "board in her father's name, proposing a second light on the outer rock. "
    "The board replied twice, both times to decline, and both times she filed "
    "the letters under 'weather', reasoning that a refusal was only a kind of "
    "fog. When the second light was finally built, thirty years later, the "
    "engineers found her drawings in the archive and used them almost without "
    "change, adjusting only the height of the lantern room to clear a headland "
    "that had eroded in the meantime. She was not invited to the lighting "
    "ceremony, having by then moved inland to run a school, but a former pupil "
    "sent her a photograph of the beam sweeping the reef, and she wrote on the "
    "back, in the same cramped hand she had used for the manifests: 'received "
    "in good condition, no damage to contents.' The photograph is in the "
    "museum now, and the note is what people photograph.",
]
for p in prompts:
    out = llm.generate([p], SamplingParams(temperature=0.0, max_tokens=4))
    o = out[0]
    print(f"prompt_tokens={len(o.prompt_token_ids)} output={o.outputs[0].text!r}",
          flush=True)
EOF

echo "--- capture summary ---"
for d in "$out"/pid*/; do
  echo "$d: $(ls "$d" | grep -c cubin) cubins, $(wc -l < "$d/launches.jsonl") launches"
done
