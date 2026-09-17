#!/usr/bin/env python3
"""Tests for the handwritten DFlash2 kernels against the ATen chains they
replace (qwen3_dflash2.py in vLLM's model zoo) and the GDN advance helpers:

- kern_dflash_conv_bf16   vs DFlashGroupedConv.prepare / finish
- kern_topk16_bf16        vs torch.topk(16) (set equality + values)
- kern_dflash_select      vs the selector's bf16 bilinear scoring + greedy walk
- kern_mask_row0          row 0 of every group -> -inf, the rest copied
- kern_conv_shift         conv line tokens a..a+2 -> 0..2, line read from column a

Launched through the CUDA driver API the way kern-runtime launches them.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/test_kernels_dflash2.py <cubin_dir>
"""

import ctypes
import pathlib
import sys

import torch
from cuda.bindings import driver as cu

C_PTR, C_INT = ctypes.c_void_p, ctypes.c_int
BF = torch.bfloat16


def check(r):
    err = r[0]
    assert err == cu.CUresult.CUDA_SUCCESS, err
    return r[1] if len(r) == 2 else r[1:]


def launch(fn, grid, block, smem, args):
    grid = tuple(grid) + (1,) * (3 - len(grid))
    block = tuple(block) + (1,) * (3 - len(block))
    vals = tuple(a for a, _ in args)
    types = tuple(t for _, t in args)
    stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
    check(cu.cuLaunchKernel(fn, *grid, *block, smem, stream, (vals, types), 0))
    torch.cuda.synchronize()


def ptr(t):
    return (t.data_ptr(), C_PTR)


def i32(v):
    return (v, C_INT)


def i64(v):
    return (v, ctypes.c_longlong)


def same(name, a, b):
    ok = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if not ok:
        diff = (a != b).sum().item() if a.shape == b.shape else -1
        print(f"  FAIL {name}: {diff} of {a.numel()} elements differ")
    return ok


# --- references (the ATen chains, in bf16 like vLLM runs them)
def ref_conv(x, delta, base, side, groups):
    T, D = x.shape
    gsz = D // groups
    coef = base.view(2, 2, D)[side][None] + delta.view(T, 2, 2, groups)[:, side].repeat_interleave(gsz, -1)
    prev = torch.cat([torch.zeros_like(x[:1]), x[:-1]], 0)
    mask = (torch.arange(T, device=x.device) % 8 != 0).to(BF)[:, None]
    return coef[:, 0] * x + coef[:, 1] * prev * mask


def ref_select(cand, unary, hidden_r, succ_cb, pred_cb, anchor):
    steps = cand.shape[0]
    out, prev = [], anchor
    for l in range(steps):
        pred = pred_cb[prev]                                # [rank]
        ph = (pred * hidden_r[l]).to(BF)                    # bf16 product
        succ = succ_cb[cand[l]]                             # [16, rank]
        edge = (succ.float() @ ph.float()).to(BF).float()   # bf16 matmul result
        s = unary[l] + edge
        best = int(torch.argmax(s).item())                  # first max
        prev = int(cand[l, best].item())
        out.append(prev)
    return torch.tensor(out, dtype=torch.int64, device=cand.device)


def main():
    cubins = pathlib.Path(sys.argv[1])
    torch.zeros(1, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(4321)
    mods = {k: check(cu.cuModuleLoad(str(cubins / f"{k}.cubin").encode()))
            for k in ("dflash_conv", "topk_row", "dflash_select", "gdn_advance")}
    fn = lambda m, s: check(cu.cuModuleGetFunction(mods[m], s.encode()))  # noqa: E731
    k_conv = fn("dflash_conv", "kern_dflash_conv_bf16")
    k_topk = fn("topk_row", "kern_topk16_bf16")
    k_sel = fn("dflash_select", "kern_dflash_select")
    k_mask = fn("gdn_advance", "kern_mask_row0")
    k_shift = fn("gdn_advance", "kern_conv_shift")
    fails = 0
    D, GROUPS, PROJ = 5120, 320, 1280

    # grouped conv, both sides, T = 8 and T = 16 (block boundary mask)
    for T in (8, 16, 5):
        for side in (0, 1):
            x = (torch.randn(T, D, device="cuda", generator=gen) * 2).to(BF)
            delta = (torch.randn(T, PROJ, device="cuda", generator=gen) * 0.1).to(BF)
            base = (torch.randn(4, D, device="cuda", generator=gen)).to(BF)
            want = ref_conv(x, delta, base, side, GROUPS)
            got = x.clone()
            launch(k_conv, [D // 256], [256], 0,
                   [ptr(got), ptr(delta), ptr(base), i32(T), i32(D), i32(GROUPS), i32(PROJ), i32(side)])
            fails += not same(f"dflash_conv T={T} side={side}", got, want)

    # top-16: the id set and values must match torch.topk (tie order aside)
    V = 248320
    for rows in (7, 1):
        logits = (torch.randn(rows, V, device="cuda", generator=gen) * 4).to(BF)
        logits[0, 12345] = logits[0].max() + 1  # a clear winner
        ids = torch.empty(rows, 16, dtype=torch.int64, device="cuda")
        vals = torch.empty(rows, 16, dtype=torch.float32, device="cuda")
        launch(k_topk, [rows], [1024], 0, [ptr(logits), ptr(ids), ptr(vals), i32(V), i32(V)])
        tv, ti = torch.topk(logits.float(), 16, dim=-1)
        ok = torch.equal(vals, tv) and all(set(ids[r].tolist()) == set(ti[r].tolist()) for r in range(rows))
        ok = ok and torch.equal(logits.gather(1, ids).float(), vals)
        ok = ok and all(ids[r].tolist() == sorted(ids[r].tolist(), key=lambda i, r=r: (-logits[r, i].float().item(), i))
                        for r in range(rows))
        if not ok:
            print(f"  FAIL topk16 rows={rows}")
            fails += 1

    # selector walk on random codebooks (rank 256, 7 steps, 16 candidates):
    # a batch of sequences, 8 draft rows each of which row 0 (the anchor)
    # carries junk candidates the walk must skip
    RANK, STEPS, K, ROWS = 256, 7, 16, 8
    for trial, seqs in enumerate((1, 3, 2)):
        cand = torch.randint(0, V, (seqs * ROWS, K), device="cuda", generator=gen)
        unary = (torch.randn(seqs * ROWS, K, device="cuda", generator=gen) * 3)
        hidden_r = torch.randn(seqs * ROWS, RANK, device="cuda", generator=gen).to(BF)
        succ_cb = (torch.randn(V, RANK, device="cuda", generator=gen) * 0.2).to(BF)
        pred_cb = (torch.randn(V, RANK, device="cuda", generator=gen) * 0.2).to(BF)
        anchors = torch.randint(0, V, (seqs,), device="cuda", generator=gen)
        want = torch.stack([
            ref_select(cand[s * ROWS + 1:(s + 1) * ROWS], unary[s * ROWS + 1:(s + 1) * ROWS],
                       hidden_r[s * ROWS + 1:(s + 1) * ROWS], succ_cb, pred_cb, int(anchors[s].item()))
            for s in range(seqs)])
        # gathered rows, as the manifest feeds them (embedding kernel)
        succ_g = succ_cb[cand.reshape(-1)].contiguous()
        pred_g = pred_cb[cand.reshape(-1)].contiguous()
        pred_anchor = pred_cb[anchors].contiguous()
        out = torch.empty(seqs, STEPS, dtype=torch.int64, device="cuda")
        launch(k_sel, [seqs], [RANK], K * RANK * 4,
               [ptr(cand), ptr(unary), ptr(hidden_r), ptr(succ_g), ptr(pred_g), ptr(pred_anchor), ptr(out),
                i32(STEPS), i32(RANK), i32(RANK), i32(ROWS)])
        fails += not same(f"dflash_select trial={trial} seqs={seqs}", out, want)

    # mask_row0: three 8-row groups, row 0 of each -> -inf, others copied
    rows, per, cols = 24, 8, 48
    src = torch.randn(rows, cols, device="cuda", generator=gen).to(BF)
    dst = torch.zeros_like(src)
    launch(k_mask, [rows], [64], 0, [ptr(dst), ptr(src), i32(rows), i32(per), i32(cols)])
    want = src.clone()
    want[::per] = float("-inf")
    fails += not same("mask_row0", dst, want)

    # conv_shift: lines [state_len=10][dim] bf16; seq s keeps its line in
    # column a_s of an 8-wide line-table cell; a = 0 (no-op) / 2 / 5, plus a
    # seq whose cell holds no line (null 0) at a = 3.
    dim, state_len, width = 10240, 10, 8
    state = torch.randn(4, state_len, dim, device="cuda", generator=gen).to(BF)
    orig = state.clone()
    a_of = [0, 2, 5, 3]
    line_of = [1, 2, 3, 0]
    idx = torch.zeros(len(a_of) * width, dtype=torch.int32, device="cuda")
    for s_, (a, line) in enumerate(zip(a_of, line_of)):
        idx[s_ * width + a] = line
    nacc = torch.tensor([a + 1 for a in a_of], dtype=torch.int32, device="cuda")
    tok_bytes, line_bytes = dim * 2, state_len * dim * 2
    launch(k_shift, [len(a_of), tok_bytes // 16 // 256], [256], 0,
           [ptr(idx), i32(width), ptr(state), ptr(nacc), i64(line_bytes), i64(tok_bytes)])
    want = orig.clone()
    for a, line in zip(a_of, line_of):
        if a > 0 and line > 0:
            want[line, 0:3] = orig[line, a:a + 3]
    fails += not same("conv_shift", state, want)

    print("all passed" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
