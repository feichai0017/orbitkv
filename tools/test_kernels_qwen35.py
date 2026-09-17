#!/usr/bin/env python3
"""Bit-exactness tests for the handwritten Qwen3.5/3.8 glue kernels.

Oracle = vLLM's own eager ops (the exact code the reference run executes):
GemmaRMSNorm.forward_native (hidden norm, fused-add norm, per-head q/k norm)
and `attn * torch.sigmoid(gate)`.  The cubins are launched through the CUDA
driver API exactly as kern-runtime launches them.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/test_kernels_qwen35.py <cubin_dir>
"""

import ctypes
import pathlib
import sys

import torch
from cuda.bindings import driver as cu

C_PTR, C_INT, C_F32 = ctypes.c_void_p, ctypes.c_int, ctypes.c_float


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


def same(name, a, b):
    ok = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if not ok:
        diff = (a.view(torch.int16) != b.view(torch.int16)).sum().item() if a.shape == b.shape else -1
        print(f"  FAIL {name}: {diff} of {a.numel()} elements differ")
    return ok


def rand_rows(rows, n, gen):
    scale = torch.pow(2.0, torch.empty(rows, 1, device="cuda").uniform_(-4, 4, generator=gen))
    return (torch.randn(rows, n, device="cuda", generator=gen) * scale).to(torch.bfloat16)


def main():
    cubins = pathlib.Path(sys.argv[1])
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    torch.zeros(1, device="cuda")  # primary context
    gen = torch.Generator(device="cuda").manual_seed(1234)
    mods = {k: check(cu.cuModuleLoad(str(cubins / f"{k}.cubin").encode()))
            for k in ("gemma_rms_norm", "sigmoid_mul", "copy_rows")}
    fn = lambda m, s: check(cu.cuModuleGetFunction(mods[m], s.encode()))  # noqa: E731
    k_norm = fn("gemma_rms_norm", "kern_gemma_rms_norm_bf16")
    k_fused = fn("gemma_rms_norm", "kern_gemma_fused_add_rms_norm_bf16")
    k_sig = fn("sigmoid_mul", "kern_sigmoid_mul_bf16")
    k_copy = fn("copy_rows", "kern_copy_rows_bf16")
    k_last = fn("copy_rows", "kern_last_row_bf16")
    EPS = 1e-6
    fails = 0

    with set_current_vllm_config(VllmConfig()):
        norm_h = GemmaRMSNorm(5120, EPS).cuda()
        norm_h.weight.data = (torch.randn(5120, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
        norm_q = GemmaRMSNorm(256, EPS).cuda()
        norm_q.weight.data = (torch.randn(256, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    w1_h = (norm_h.weight.float() + 1.0).contiguous()
    w1_q = (norm_q.weight.float() + 1.0).contiguous()

    # --- hidden norm, plain + fused-add, all the row counts that change ATen's block width
    rows_list = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 250, 511, 512, 513, 1024, 2047, 2048]
    smem = (2 * 5120 + 512) * 4
    for rows in rows_list:
        x = rand_rows(rows, 5120, gen)
        ref = norm_h.forward_native(x)
        out = torch.empty_like(x)
        launch(k_norm, (rows,), (512,), smem,
               [ptr(out), ptr(x), ptr(w1_h), (5120, C_INT), (rows, C_INT), (1, C_INT),
                (5120, C_INT), (0, C_INT), (5120, C_INT), (0, C_INT), (EPS, C_F32)])
        fails += not same(f"rms_norm rows={rows}", out, ref)

        res = rand_rows(rows, 5120, gen)
        ref_out, ref_res = norm_h.forward_native(x.clone(), res.clone())
        res_k = res.clone()
        launch(k_fused, (rows,), (512,), smem,
               [ptr(out), ptr(x), ptr(res_k), ptr(w1_h), (5120, C_INT), (rows, C_INT),
                (5120, C_INT), (5120, C_INT), (EPS, C_F32)])
        fails += not same(f"fused_add_rms_norm rows={rows} out", out, ref_out)
        fails += not same(f"fused_add_rms_norm rows={rows} residual", res_k, ref_res)

    # --- per-head q/k norm reading the strided qkv projection view
    smem_q = (2 * 256 + 512) * 4
    for tokens in [1, 2, 3, 4, 5, 8, 17, 250, 2048]:
        qkv = rand_rows(tokens, 14336, gen)
        q_view = qkv[:, :12288].view(tokens, 24, 512)[:, :, :256]  # [T, 24, 256] strided
        k_view = qkv[:, 12288:13312].view(tokens, 4, 256)
        for name, view, heads, hs, off in (("q", q_view, 24, 512, 0), ("k", k_view, 4, 256, 12288)):
            ref = norm_q.forward_native(view).reshape(tokens, heads * 256)
            out = torch.empty(tokens, heads * 256, dtype=torch.bfloat16, device="cuda")
            rows = tokens * heads
            launch(k_norm, (rows,), (512,), smem_q,
                   [ptr(out), (qkv.data_ptr() + off * 2, C_PTR), ptr(w1_q), (256, C_INT), (rows, C_INT),
                    (heads, C_INT), (14336, C_INT), (hs, C_INT), (heads * 256, C_INT), (256, C_INT),
                    (EPS, C_F32)])
            fails += not same(f"{name}_norm tokens={tokens}", out, ref)

    # --- gated attention output
    for tokens in [1, 7, 250, 2048]:
        qkv = rand_rows(tokens, 14336, gen)
        attn = rand_rows(tokens, 6144, gen)
        gate = qkv[:, :12288].view(tokens, 24, 512)[:, :, 256:].reshape(tokens, 6144)
        ref = attn * torch.sigmoid(gate)
        out = torch.empty_like(attn)
        launch(k_sig, (tokens,), (256,), 0,
               [ptr(out), ptr(attn), (qkv.data_ptr() + 256 * 2, C_PTR), (24, C_INT), (256, C_INT),
                (14336, C_INT), (512, C_INT)])
        fails += not same(f"sigmoid_mul tokens={tokens}", out, ref)

    # --- copies
    for tokens in [1, 5, 250]:
        qkvz = rand_rows(tokens, 16384, gen)
        z = torch.empty(tokens, 6144, dtype=torch.bfloat16, device="cuda")
        launch(k_copy, (tokens,), (256,), 0,
               [ptr(z), (qkvz.data_ptr() + 10240 * 2, C_PTR), (6144, C_INT), (16384, C_INT), (6144, C_INT)])
        fails += not same(f"copy_rows tokens={tokens}", z, qkvz[:, 10240:].contiguous())
        last = torch.empty(1, 5120, dtype=torch.bfloat16, device="cuda")
        h = rand_rows(tokens, 5120, gen)
        launch(k_last, (1,), (256,), 0,
               [ptr(last), ptr(h), (5120, C_INT), (5120, C_INT), (tokens, C_INT)])
        fails += not same(f"last_row tokens={tokens}", last, h[tokens - 1:tokens])

    print("FAILURES:", fails) if fails else print("all bit-exact")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
