#!/usr/bin/env python3
"""Per-layer activation probe of vLLM (eager, TRITON_ATTN + triton GDN) for
one prompt: embedding output, every decoder layer's output hidden_states
(= MLP output, kern's `y`) and residual, and the logits, for the prefill
forward and two decode steps.  Pairs with `kern run --probe-dir`.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/qwen38_probe_vllm.py [prompt_index] [out.pt]
"""

import json
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import AttentionConfig  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen38_ref import TARGET  # noqa: E402

STEPS = {}   # step -> {"embed": t, "y": {i: t}, "res": {i: t}, "logits": t}
CUR = [-1]


def rec(step, key, val, idx=None):
    s = STEPS.setdefault(step, {"y": {}, "res": {}})
    if idx is None:
        s[key] = val
    else:
        s[key][idx] = val


def install(model):
    def embed_hook(mod, inp, out):
        CUR[0] += 1
        rec(CUR[0], "embed", out.detach().clone().cpu())

    def layer_hook(i):
        def hook(mod, inp, out):
            h, r = out
            rec(CUR[0], "y", h.detach().clone().cpu(), i)
            rec(CUR[0], "res", r.detach().clone().cpu(), i)
        return hook

    def logits_hook(mod, inp, out):
        rec(CUR[0], "logits", out.detach().clone().cpu())

    # Qwen3_5ForConditionalGeneration nests the text model; find the module
    # that owns embed_tokens + layers, and the logits processor, by walking.
    inner = next(m for _, m in model.named_modules()
                 if hasattr(m, "embed_tokens") and hasattr(m, "layers"))
    lp = next(m for n, m in model.named_modules() if n.endswith("logits_processor"))
    inner.embed_tokens.register_forward_hook(embed_hook)
    for i, layer in enumerate(inner.layers):
        layer.register_forward_hook(layer_hook(i))
    lp.register_forward_hook(logits_hook)

    # PROBE_LAYER=<i>: also record that layer's submodule outputs (and the
    # gated norm's input) under "fine"; matches kern run's KERN_PROBE_LAYER.
    fl = os.environ.get("PROBE_LAYER")
    if fl is not None:
        layer = inner.layers[int(fl)]

        def fine_hook(name, pick=lambda o: o):
            def hook(mod, inp, out):
                t = pick(out)
                STEPS.setdefault(CUR[0], {"y": {}, "res": {}}).setdefault("fine", {})[name] = t.detach().clone().cpu()
            return hook
        first = lambda o: o[0] if isinstance(o, tuple) else o  # noqa: E731
        subs = {"post_attn_norm": layer.post_attention_layernorm,
                "gate_up": layer.mlp.gate_up_proj, "silu_mul": layer.mlp.act_fn, "down_proj": layer.mlp.down_proj}
        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            subs.update({"in_proj_qkvz": la.in_proj_qkvz, "in_proj_ba": la.in_proj_ba,
                         "gated_norm": la.norm, "out_proj": la.out_proj})
            la.norm.register_forward_pre_hook(
                lambda mod, inp: STEPS.setdefault(CUR[0], {"y": {}, "res": {}}).setdefault("fine", {}).__setitem__(
                    "core_attn_out", inp[0].detach().clone().cpu()))
        else:
            sa = layer.self_attn
            subs.update({"qkv_proj": sa.qkv_proj, "o_proj": sa.o_proj})
        for name, mod in subs.items():
            mod.register_forward_hook(fine_hook(name, first))
    return f"hooked {len(inner.layers)} layers of {type(inner).__name__}"


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out = sys.argv[2] if len(sys.argv) > 2 else "dumped-kernels/probe-vllm.pt"
    ref = json.load(open("docs/qwen38/ref.json"))
    prompt = ref["results"][idx]["prompt"]
    llm = LLM(model=TARGET, tokenizer=TARGET, dtype="bfloat16",
              tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=0.6, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "video": 0},
              attention_config=AttentionConfig(backend="TRITON_ATTN"),
              additional_config={"gdn_prefill_backend": "triton"})
    print(llm.apply_model(install), flush=True)
    o = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=3))[0]
    ids = list(o.outputs[0].token_ids)
    print("prompt ids", len(o.prompt_token_ids), "generated", ids, flush=True)
    assert ids == ref["results"][idx]["output_token_ids"][:3], "probe run disagrees with the reference run"
    torch.save({"steps": STEPS, "prompt_token_ids": list(o.prompt_token_ids), "generated": ids}, out)
    print("wrote", out, {k: (v["embed"].shape, len(v["y"])) for k, v in STEPS.items()})


if __name__ == "__main__":
    main()
