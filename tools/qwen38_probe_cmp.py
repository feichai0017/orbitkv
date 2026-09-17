#!/usr/bin/env python3
"""Compare kern run --probe-dir dumps with the vLLM probe (.pt), layer by layer.

    .venv/bin/python tools/qwen38_probe_cmp.py dumped-kernels/probe-kern dumped-kernels/probe-vllm.pt
"""

import pathlib
import sys

import numpy as np
import torch

kern = pathlib.Path(sys.argv[1])
vl = torch.load(sys.argv[2])
HIDDEN = 5120


def bf16(path, rows):
    raw = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16).reshape(rows, -1)


def cmp(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    n = (a != b).sum().item()
    return n, (a - b).abs().max().item(), a.numel()


tags = {0: "chunk", 1: "decode0", 2: "decode1"}
for step, tag in tags.items():
    s = vl["steps"][step]
    rows = s["embed"].shape[0]
    n, mx, tot = cmp(bf16(kern / f"{tag}.embed.bin", rows), s["embed"])
    print(f"== {tag}: {rows} tokens; embed mismatch {n}/{tot} (max {mx:.3g})")
    first = None
    for i in range(len(s["y"])):
        n, mx, tot = cmp(bf16(kern / f"{tag}.l{i}.bin", rows), s["y"][i])
        kind = "attn" if (i + 1) % 4 == 0 else "gdn "
        if n and first is None:
            first = i
        if n or i in (0, 3, 63):
            print(f"   layer {i:2d} {kind} y: {n:8d}/{tot} differ, max |d| {mx:.4g}")
    lk = bf16(kern / f"{tag}.logits.bin", 1)
    lv = s["logits"].reshape(1, -1)
    n, mx, tot = cmp(lk, lv[:, :lk.shape[1]].to(torch.bfloat16))
    nt = int(np.fromfile(kern / f"{tag}.next_token.bin", dtype=np.int64)[0])
    print(f"   logits: {n}/{tot} differ (max |d| {mx:.4g}; vllm dtype {lv.dtype}); "
          f"kern argmax {nt} vs vllm argmax {int(lv.float().argmax())}; first differing layer: {first}")
