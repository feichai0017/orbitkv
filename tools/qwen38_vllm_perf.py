#!/usr/bin/env python3
"""vLLM bs=1 throughput reference for the Stage 3 table: plain decode with
CUDA graphs, and DFlash2 speculative decoding (acceptance + tok/s), on the
five reference prompts x 400 tokens (greedy).

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/qwen38_vllm_perf.py plain|spec [pinned] [out.json]

`pinned` uses the Stage 1 backend pins (TRITON_ATTN + triton GDN prefill);
without it vLLM picks its default backends.
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

from vllm import LLM, SamplingParams  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen38_ref import PROMPTS, TARGET  # noqa: E402

DRAFT = (os.path.expanduser("~/.cache/huggingface/hub") + "/models--incoai--Qwen3.8-27B-DFlash2/snapshots/"
         "dedf8df68adfb1afeaf7b7480c0a0243108177b4")


def main():
    mode = sys.argv[1]
    pinned = "pinned" in sys.argv[2:]
    out = next((a for a in sys.argv[2:] if a.endswith(".json")), f"docs/qwen38/vllm-perf-{mode}{'-pinned' if pinned else ''}.json")
    kw = dict(model=TARGET, tokenizer=TARGET, dtype="bfloat16", tensor_parallel_size=1,
              max_model_len=4096, gpu_memory_utilization=0.6, enforce_eager=False,
              limit_mm_per_prompt={"image": 0, "video": 0}, disable_log_stats=False)
    if pinned:
        from vllm.config import AttentionConfig
        kw["attention_config"] = AttentionConfig(backend="TRITON_ATTN")
        kw["additional_config"] = {"gdn_prefill_backend": "triton"}
    if mode == "spec":
        kw["speculative_config"] = {"method": "dflash", "model": DRAFT, "num_speculative_tokens": 7}
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=400)
    llm.generate([PROMPTS[0]], SamplingParams(temperature=0.0, max_tokens=8))  # warm-up
    results = []
    for p in PROMPTS:
        t0 = time.time()
        o = llm.generate([p], sp)[0]
        dt = time.time() - t0
        n = len(o.outputs[0].token_ids)
        results.append({"prompt_tokens": len(o.prompt_token_ids), "out_tokens": n,
                        "seconds": round(dt, 3), "tok_s": round(n / dt, 1),
                        "output_token_ids": list(o.outputs[0].token_ids)})
        print(f"[{mode}] out={n} {n/dt:.1f} tok/s (bs=1, graph{', pinned' if pinned else ''})", flush=True)
    metrics = {}
    try:
        for m in llm.get_metrics():
            if any(k in m.name for k in ("spec", "accept", "draft")):
                metrics[m.name] = getattr(m, "value", None) or getattr(m, "values", None) or str(m)
    except Exception as e:  # noqa: BLE001
        metrics["error"] = str(e)
    print("[metrics]", json.dumps(metrics, default=str)[:2000], flush=True)
    json.dump({"mode": mode, "pinned": pinned, "results": results, "metrics": metrics,
               "mean_tok_s": round(sum(r["tok_s"] for r in results) / len(results), 1)},
              open(out, "w"), indent=1, default=str)
    print("wrote", out)


if __name__ == "__main__":
    main()
