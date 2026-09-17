#!/usr/bin/env python3
"""Compare the per-dispatch dumps of one layer (kern run --probe-dir with
KERN_PROBE_LAYER=<i>) with the vLLM submodule hooks (PROBE_LAYER=<i>), in
dispatch order, to find the first op whose output differs.

    .venv/bin/python tools/qwen38_probe_fine.py dumped-kernels/probe-kern-fine dumped-kernels/probe-vllm-fine.pt [layer]
"""

import pathlib
import sys

import numpy as np
import torch

kern = pathlib.Path(sys.argv[1])
vl = torch.load(sys.argv[2])
L = int(sys.argv[3]) if len(sys.argv) > 3 else 0

# kern dump suffix -> vLLM fine key (dispatch order)
PAIRS = [
    ("in_proj_qkvz.qkvz", "in_proj_qkvz"),
    ("in_proj_ba.ba", "in_proj_ba"),
    ("chunk_o.core_attn_out", "core_attn_out"),      # prefill: FLA output, pre-norm
    ("recurrent.core_attn_out", "core_attn_out"),    # decode: recurrent output, pre-norm
    ("gated_norm.core_attn_out", "gated_norm"),
    ("out_proj.y", "out_proj"),
    ("post_attn_norm.x", "post_attn_norm"),
    ("gate_up.gate_up", "gate_up"),
    ("silu_mul.act", "silu_mul"),
    ("down_proj.y", "down_proj"),
]


def bf16(path):
    raw = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)


for step, tag in {0: "chunk", 1: "decode0", 2: "decode1"}.items():
    fine = vl["steps"][step].get("fine", {})
    print(f"== {tag}")
    for suffix, key in PAIRS:
        p = kern / f"{tag}.l{L}.{suffix}.bin"
        if not p.exists() or key not in fine:
            continue
        a = bf16(p).float()
        b = fine[key].to(torch.bfloat16).reshape(-1).float()
        if a.numel() != b.numel():
            print(f"   {suffix:28s} SIZE MISMATCH kern {a.numel()} vs vllm {b.numel()} {tuple(fine[key].shape)}")
            continue
        n = (a != b).sum().item()
        d = (a - b).abs()
        rel = (d / (b.abs() + 1e-6)).max().item()
        print(f"   {suffix:28s} {n:8d}/{a.numel():8d} differ  max|d| {d.max().item():.4g}  max rel {rel:.3g}"
              + ("" if n else "  EXACT"))
