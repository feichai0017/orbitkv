#!/usr/bin/env python3
"""vLLM 0.28 reference outputs for the Qwen3.8-27B bring-up (the oracle).

Same backend pins as the mining capture (TRITON_ATTN + triton GDN prefill,
enforce_eager), greedy, one prompt at a time (bs=1: batching changes GEMM
M and therefore cuBLAS algorithm choice, which is exactly the kind of
ulp-level difference we are trying to keep out of the oracle).

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/qwen38_ref.py [out.json]
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import AttentionConfig  # noqa: E402

HUB = os.path.expanduser("~/.cache/huggingface/hub")
TARGET = f"{HUB}/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

PROMPTS = [
    "The harbor master kept a ledger of every ship that wintered in the bay, "
    "noting cargo, crew, and the state of each hull in a cramped, looping hand. "
    "In the spring of the third year he found",
    "When the observatory finally reopened after the renovation, the docents "
    "discovered that the old refractor had been quietly recollimated by a "
    "retired machinist who lived nearby. He left no note, only",
    "The floodplain census took three summers to complete. In the first summer "
    "the crews mapped oxbow lakes and counted heron rookeries from canoes, "
    "losing two clipboards and one outboard motor to the river. In the second",
    "Explain, in plain language and with one concrete example each, why a "
    "bridge expands in summer, why a kettle sings before it boils, and why the "
    "far side of the moon is never seen from Earth.",
    "The lighthouse keeper's daughter learned to read from shipping manifests "
    "and weather logs, so her first stories were inventories: forty barrels of "
    "salt cod, a crate of oranges gone soft, one bishop travelling incognito. "
    "By twelve she had",
]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "docs/qwen38/ref.json"
    max_tokens = int(os.environ.get("MAX_TOKENS", "400"))
    # PROMPTS_FILE: JSON list of prompts instead of the built-in five.
    # MAX_NUM_BATCHED_TOKENS: force vLLM's own chunked prefill (chunk-invariance check).
    prompts = json.load(open(os.environ["PROMPTS_FILE"])) if "PROMPTS_FILE" in os.environ else PROMPTS
    extra = {}
    if "MAX_NUM_BATCHED_TOKENS" in os.environ:
        extra = {"max_num_batched_tokens": int(os.environ["MAX_NUM_BATCHED_TOKENS"]),
                 "max_num_seqs": 1, "enable_chunked_prefill": True}
    llm = LLM(model=TARGET, tokenizer=TARGET, dtype="bfloat16",
              tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=0.6, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "video": 0},
              attention_config=AttentionConfig(backend="TRITON_ATTN"),
              additional_config={"gdn_prefill_backend": "triton"}, **extra)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    results = []
    for p in prompts:
        t0 = time.time()
        o = llm.generate([p], sp)[0]
        dt = time.time() - t0
        c = o.outputs[0]
        results.append({
            "prompt": p,
            "prompt_token_ids": list(o.prompt_token_ids),
            "output_token_ids": list(c.token_ids),
            "text": c.text,
            "finish_reason": c.finish_reason,
            "seconds": round(dt, 2),
        })
        print(f"[ref] prompt_tokens={len(o.prompt_token_ids)} "
              f"out_tokens={len(c.token_ids)} finish={c.finish_reason} "
              f"{len(c.token_ids)/dt:.1f} tok/s (eager bs=1)", flush=True)
        print(c.text[:300].replace("\n", " ") + " ...", flush=True)
    meta = {"engine": "vllm 0.28.0", "backend": "TRITON_ATTN + gdn_prefill_backend=triton",
            "enforce_eager": True, "sampling": "greedy", "max_tokens": max_tokens,
            "model": TARGET, **extra}
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1, ensure_ascii=False)
    print(f"[ref] wrote {out_path}")


if __name__ == "__main__":
    main()
