#!/usr/bin/env python3
"""Top-2 logprob margin of vLLM's greedy choice at every step of the five
reference prompts (same engine setup as qwen38_ref.py), to tell a near-tie
flip from a real numerical bug when two configurations diverge: at a
position where the top two candidates are within a few hundredths of a nat,
any 1-ulp difference in the arithmetic picks the other one.

    CUDA_VISIBLE_DEVICES=0 MAX_TOKENS=200 .venv/bin/python tools/qwen38_margins.py docs/qwen38/margins.json
"""
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
sys.path.insert(0, os.path.dirname(__file__))
from qwen38_ref import PROMPTS, TARGET  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import AttentionConfig  # noqa: E402


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "docs/qwen38/margins.json"
    max_tokens = int(os.environ.get("MAX_TOKENS", "200"))
    llm = LLM(model=TARGET, tokenizer=TARGET, dtype="bfloat16", tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=0.6, enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0},
              attention_config=AttentionConfig(backend="TRITON_ATTN"), additional_config={"gdn_prefill_backend": "triton"})
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=2)
    results = []
    for p in PROMPTS:
        o = llm.generate([p], sp)[0].outputs[0]
        margins = []
        for tok, lp in zip(o.token_ids, o.logprobs):
            vals = sorted((v.logprob for v in lp.values()), reverse=True)
            margins.append(round(vals[0] - vals[1], 4) if len(vals) > 1 else None)
        results.append({"output_token_ids": list(o.token_ids), "margin": margins})
    json.dump({"max_tokens": max_tokens, "results": results}, open(out_path, "w"))
    for i, r in enumerate(results):
        m = [x for x in r["margin"] if x is not None]
        tight = [k for k, x in enumerate(r["margin"]) if x is not None and x < 0.05]
        print(f"[{i}] {len(m)} steps, margin<0.05 at {tight[:20]}{'...' if len(tight) > 20 else ''}")


if __name__ == "__main__":  # vLLM spawns workers: the module must import cleanly
    main()
