#!/usr/bin/env python3
"""Generate examples/qwen3.8-27b.json from the vLLM capture of Qwen3.8-27B.

Qwen3.8-27B is a Qwen3.5-family hybrid: 64 layers, 48 gated-delta-net
(GDN, "linear attention") layers and 16 full-attention layers (every 4th),
gated attention output, Gemma-style norms, mrope.  bs=1 manifest, two
programs sized by the `tokens` symbol:

- `prefill` (tokens ∈ [1, CHUNK_MAX]): the vLLM chunked-prefill forward —
  KV writes for the attention layers, conv/SSM state carry for the GDN
  layers — *plus* the final norm / lm_head / argmax on the last row.  Unlike
  the qwen3-4b manifests, prefill emits `next_token`: the GDN prefill path
  (FLA chunk kernels) and the decode path (recurrent kernel) are different
  arithmetic, and vLLM runs the last prompt token through the chunk path.
  Putting it through `decode` instead would change bits.  The driver
  (kern run) sees the output buffer in the program and prefills all prompt
  tokens.
- `decode` (tokens = 1): one token, recurrent GDN kernels, split-KV
  attention, logits + argmax.

Data source: the TRITON_ATTN + `gdn_prefill_backend=triton` capture (the
only flat-ABI kernel set: every kernel is a plain CUDA/Triton launch).  The
generator takes launch geometry, scalar literals and instance identity from
the capture, and asserts the wiring it hand-writes against the captured
pointers (buffer identities between consecutive kernels) and the grid
formulas against four prefill forwards of different lengths.

What the runtime cannot express and had to be handwritten (all bit-exact
against vLLM's own ops, tools/test_kernels_qwen35.py):

- `gemma_rms_norm.cu`: GemmaRMSNorm is a chain of ATen ops in vLLM (pow /
  mean / rsqrt / mul); the kernel reproduces ATen's reduction order.
- `sigmoid_mul.cu`: `attn_out * sigmoid(gate)` (two ATen ops).
- `copy_rows.cu`: the strided z-gate copy (`layer_norm_fwd_kernel` bakes
  `%16` assumptions into its strides, so the [T,16384] view cannot be fed
  directly) and the last-row select for the prefill tail (`tokens-1` is not
  in the manifest expression set).

Triton instances: same-symbol kernels compiled for different
specializations (an int arg == 1, or % 16) have identical param layouts, so
the runtime's layout matching cannot pick one.  Every Triton step is pinned
to a dump module by sha256; the module is chosen by matching the dump
against the Triton cache and reading the `.ttir` signature — the least
specialized instance (most runtime args, fewest divisibility attributes)
among those with the launch's register count.  A generic instance computes
the same values as a specialized one; it just makes fewer assumptions.

State layout (baked into the kernels by vLLM's constexprs):

- GDN: one 3211264-B line per layer = [conv state 10240×3 bf16 | SSM state
  48×128×128 f32 | pad]; line 0 is the null line (kernels skip index <= 0),
  layer i uses line i+1; indices come from a constant table via byte
  offsets.  `bytes_fixed` state (per sequence, not per token) — the one
  schema extension this bring-up needed.
- KV: BLOCK_SIZE = 784 (vLLM aligned it to the mamba page), k/v interleaved
  per head, layers interleaved per page: [page][16][784][4][K|V][256],
  bytes_per_token = 65536.

Usage: python3 tools/gen_qwen35.py [dump_dir] [out.json]
"""

import hashlib
import json
import os
import pathlib
import re
import struct
import subprocess
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from qwen_weights import qwen38  # noqa: E402
from kern_manifest import normalize, program, resolve_constants, SCHEMA_VERSION  # noqa: E402
from handwritten import hw  # tools/handwritten.py: build + pin handwritten cubins

# --- model geometry (config.json; asserted against the capture below)
HIDDEN = 5120
LAYERS = 64
FFN = 17408
VOCAB = 248320
HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
Q_DIM = HEADS * HEAD_DIM                 # 6144
KV_DIM = KV_HEADS * HEAD_DIM             # 1024
QKV_DIM = HEADS * 2 * HEAD_DIM + 2 * KV_DIM   # 14336: per head [q | gate], then k, v
GATE_OFF = HEAD_DIM                      # gate half inside a [q | gate] head pair
ROT_HALF = 32                            # rotary_dim 64
MAX_POS = 262144                         # the model's max_position_embeddings: rope table rows, KV pages per sequence
# GDN
GDN_V_HEADS = 48
GDN_K_HEADS = 16
GDN_D = 128
GDN_Q = GDN_K_HEADS * GDN_D              # 2048
GDN_V = GDN_V_HEADS * GDN_D              # 6144
CONV_DIM = 2 * GDN_Q + GDN_V             # 10240 (q, k, v go through the conv)
QKVZ_DIM = CONV_DIM + GDN_V              # 16384 (+ z)
Z_OFF = CONV_DIM                         # z columns start after q|k|v
BA_DIM = 2 * GDN_V_HEADS                 # 96: [b | a]
FLA_CHUNK = 64
CONV_BLOCK_M = 8
POST_CONV_BLOCK = 16
LN_ROWS_PER_BLOCK = 4                    # prefill layer_norm instance
BF16 = 2
ATTN_LAYERS = [i for i in range(LAYERS) if (i + 1) % 4 == 0]
GDN_LAYERS = [i for i in range(LAYERS) if (i + 1) % 4 != 0]
# prefill chunk bound (`tokens.max`): sizes the FLA chunk-state workspace
# `h` (3 MiB per chunk), the conv index tables and, downstream, the
# TRTLLM-GEN Q / O tensormaps; 8192 prefills ~7% faster than 2048 on a 36k
# prompt (GB300, 2026-09-09) and is where the per-chunk gain flattens
CHUNK_MAX = 8192
NT_MAX = CHUNK_MAX // FLA_CHUNK          # 128 chunks
# attention
BLOCK_SIZE = 784                         # vLLM block_size (constexpr in the kernels)
BLOCK_Q = 5                              # unified 2D: query rows per block
NUM_SEGMENTS = 16                        # unified 3D: grid.z
BLOCK_TABLE_LEN = -(-MAX_POS // BLOCK_SIZE)   # block_table row = pages a sequence can reach (335)
CAPTURE_TABLE_LEN = 8                    # the capture's block_table_stride (vLLM max_model_len 6272)
BLOCK_ELEMS_PER_LAYER = BLOCK_SIZE * KV_HEADS * 2 * HEAD_DIM   # 1605632
LAYER_KV_BYTES = BLOCK_ELEMS_PER_LAYER * BF16                  # 3211264
BLOCK_STRIDE = len(ATTN_LAYERS) * BLOCK_ELEMS_PER_LAYER        # elems
KV_BYTES_PER_TOKEN = len(ATTN_LAYERS) * KV_HEADS * 2 * HEAD_DIM * BF16  # 65536
V_BYTE_OFF = HEAD_DIM * BF16             # v = k + 256 elems inside a head
# GDN state lines
CONV_STATE_BYTES = CONV_DIM * 3 * BF16   # 61440
SSM_STATE_BYTES = GDN_V_HEADS * GDN_D * GDN_D * 4   # 3145728
GDN_LINE_BYTES = 3211264                 # conv + ssm + 4096 pad (vLLM page)
# Serving layout: the GDN state is per sequence (`bytes_per_seq`), one line
# per GDN layer; the line table `gdn.line_index[layer, seq]` = slot × 48 +
# layer. How many slots exist is the runtime's business: it starts with
# MAX_SEQS + 2 (slot 0 = null lines, one spare for a batched caller's
# padding) and grows them out of its state budget, so nothing here may
# bound a slot number. The conv kernels take vLLM's `num_cache_lines` only
# to mask a line index past its cache; they get the largest i32 and the
# mask never bites (a manifest that baked MAX_SEQS + 2 in here silently
# dropped the conv state of every slot past 129).
MAX_SEQS = 128                           # `seqs` bound: decode_batch rows
GDN_SEQ_BYTES = len(GDN_LAYERS) * GDN_LINE_BYTES
GDN_LINES_BOUND = 2**31 - 1              # num_cache_lines the conv kernels see

# --- DFlash2 speculative decoding (examples/qwen3.8-27b-dflash2.json, --spec)
SPEC_BLOCK = 8                           # draft block: anchor + 7 masks = verify rows
DRAFT_TOKENS = SPEC_BLOCK - 1
MASK_TOKEN = 248070                      # dflash_config.mask_token_id
TAPS = {5: 0, 19: 1, 33: 2, 47: 3, 61: 4}   # target_layer_ids -> fc column block
# GDN state under speculation: vLLM's spec-decode kernels take a 10-wide
# conv line (3 history + 7 drafts, token-major) and read the SSM state /
# conv history at index `num_accepted - 1` of a per-sequence 8-entry line
# list (`gdn.line_index[layer, seq, 8]`). vLLM keeps 8 checkpoint pages per
# layer; kern keeps one page per layer per sequence and recomputes instead
# ("The Mamba in the Llama", Wang et al. 2024): verify resumes from the
# committed state (num_accepted = 1, the line in entry 0 — its row 0, the
# anchor, stores the after-anchor state back), then `advance` re-runs the
# saved rows with the anchor masked out from that state and stores after
# row a (the line in entry a, num_accepted = a + 1), shifting the conv
# history down by a.
SPEC_PAGE_BYTES = 3407872
SPEC_CONV_BYTES = CONV_DIM * (3 + DRAFT_TOKENS) * BF16   # 204800
SPEC_SSM_OFF = SPEC_CONV_BYTES
SPEC_SEQ_BYTES = len(GDN_LAYERS) * SPEC_PAGE_BYTES       # 164 MB per sequence
SPEC_ROWS_MAX = MAX_SEQS * SPEC_BLOCK                    # verify / draft rows of the widest batch
# draft geometry (5 Qwen3 layers, non-causal over the block, sliding window 2048)
D_LAYERS = 5
D_HEADS = 32
D_KV_HEADS = 8
D_HEAD_DIM = 128
D_Q = D_HEADS * D_HEAD_DIM               # 4096
D_KV = D_KV_HEADS * D_HEAD_DIM           # 1024
D_QKV = D_Q + 2 * D_KV                   # 6144
D_GROUPS = 320                           # grouped conv: 5120 / conv_group_size 16
D_CONV_PROJ = 2 * 2 * D_GROUPS           # kernel_projection out: (side, tap, group)
SEL_RANK = 256
SEL_K = 16
D_BLOCK = 16                             # draft KV page: constexpr of the borrowed non-causal instance
D_BLOCK_ELEMS_PER_LAYER = D_BLOCK * D_KV_HEADS * 2 * D_HEAD_DIM   # 32768
D_LAYER_KV_BYTES = D_BLOCK_ELEMS_PER_LAYER * BF16                # 65536
D_BLOCK_STRIDE = D_LAYERS * D_BLOCK_ELEMS_PER_LAYER              # elems per page
D_KV_BYTES_PER_TOKEN = D_LAYERS * D_KV_HEADS * 2 * D_HEAD_DIM * BF16   # 20480
D_V_BYTE_OFF = D_HEAD_DIM * BF16
D_TABLE_LEN = MAX_POS // D_BLOCK
D_KV_FLAT = D_LAYERS * 2 * D_KV          # fused context-KV GEMM width 10240
D_ATTN_SCALE = D_HEAD_DIM ** -0.5
SPEC_SYMS = {
    "conv_update_spec": "_causal_conv1d_update_kernel",
    "recurrent_spec": "fused_sigmoid_gating_delta_rule_update_kernel",
    "d_rms_norm": "_ZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi2ELb1EEEvPT_PKS3_lllllS6_lfii",
    "d_head_norm": "_ZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi3ELb1EEEvPT_PKS3_lllllS6_lfii",
    "d_fused_norm": "_ZN4vllm25fused_add_rms_norm_kernelIN3c108BFloat16ELi8ELb1EEENSt9enable_ifIXaagtT0_Li0Esr4vllm12_typeConvertIT_EE6existsEvE4typeEPS4_lS7_PKS4_fiil",
    "d_rope": "_ZN4vllm23rotary_embedding_kernelIN3c108BFloat16ES2_Lb1EEEvPKlPT_S6_PKT0_illliiilb",
    "d_cache": "_ZN4vllm30reshape_and_cache_flash_kernelI13__nv_bfloat16S1_LNS_18Fp8KVCacheDataTypeE0EEEvPKT_S5_PT0_S7_PKlllllliiiPKfSB_i",
}

TRITON = {
    "conv_fwd": "_causal_conv1d_fwd_kernel",
    "conv_update": "_causal_conv1d_update_kernel",
    "post_conv": "_fused_post_conv_kernel",
    "cumsum": "chunk_local_cumsum_scalar_kernel",
    "kkt": "chunk_scaled_dot_kkt_fwd_kernel",
    "solve_tril": "merge_16x16_to_64x64_inverse_kernel",
    "recompute": "recompute_w_u_fwd_kernel",
    "chunk_h": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_o": "chunk_fwd_kernel_o",
    "recurrent": "fused_recurrent_gated_delta_rule_packed_decode_kernel",
    "layer_norm": "layer_norm_fwd_kernel",
    "mrope": "_triton_mrope_forward",
    "cache": "reshape_and_cache_kernel_flash",
    "unified": "kernel_unified_attention",
    "reduce": "reduce_segments",
}


# ----------------------------------------------------------------- capture
def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def forwards(recs):
    """Forward = launches up to and including the sampler's ArgMax reduce."""
    out, start = [], 0
    for i, r in enumerate(recs):
        if "ArgMaxOps" in r["symbol"]:
            out.append(recs[start:i + 1])
            start = i + 1
    return out


def tokens_of(fwd):
    for r in fwd:
        if r["symbol"] == TRITON["mrope"]:
            return r["grid"][0]
    return None


def group(fwd):
    by = {k: [] for k in TRITON}
    by["silu"] = []
    for r in fwd:
        s = r["symbol"]
        if not isinstance(r.get("params"), list):
            continue
        if "act_and_mul" in s:
            by["silu"].append(r)
            continue
        for tag, sym in TRITON.items():
            if s == sym:
                by[tag].append(r)
                break
    return by


def spec_launches(recs):
    """One bs=1 launch record per speculative-path kernel (tools/capture_qwen38_spec.sh
    dump): the IS_SPEC_DECODING conv update (13 params, batch 1), the T=8
    recurrent kernel (N=1), and the draft's vLLM CUDA kernels."""
    want = {
        "conv_update_spec": lambda r: len(r["params"]) == 13 and pv(r, 7) == 1,
        "recurrent_spec": lambda r: r["grid"][2] == GDN_V_HEADS and pv(r, 16) == 1 and pv(r, 17) == SPEC_BLOCK,
        "d_rms_norm": lambda r: r["block"][0] == HIDDEN // 8,
        "d_head_norm": lambda r: pv(r, 2) == D_HEAD_DIM,
        "d_fused_norm": lambda r: pv(r, 6) == HIDDEN,
        "d_rope": lambda r: pv(r, 4) == D_HEAD_DIM,
        "d_cache": lambda r: pv(r, 10) == D_KV_HEADS,
    }
    out = {}
    for r in recs:
        for tag, sym in SPEC_SYMS.items():
            if tag not in out and r["symbol"] == sym and isinstance(r.get("params"), list) and want[tag](r):
                out[tag] = r
    missing = [t for t in SPEC_SYMS if t not in out]
    assert not missing, f"spec capture lacks launches for {missing}"
    return out


def pv(rec, i):
    return int.from_bytes(bytes.fromhex(rec["params"][i]["data"]), "little")


def pf(rec, i):
    return struct.unpack("<f", bytes.fromhex(rec["params"][i]["data"]))[0]


def cdiv_i(a, b):
    return -(-a // b)


# ------------------------------------------------------ instance selection
def cuobjdump():
    return str(pathlib.Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "cuobjdump")


def module_functions(mod):
    """{function: regs} for a cubin."""
    out = subprocess.run([cuobjdump(), "-res-usage", str(mod)],
                         capture_output=True, text=True).stdout
    fns, cur = {}, None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Function "):
            cur = s.split()[1].rstrip(":")
        elif cur and "REG:" in s:
            fns[cur] = int(s.split("REG:")[1].split()[0])
            cur = None
    return fns


def triton_cache_by_sha():
    root = pathlib.Path(os.environ.get("TRITON_CACHE_DIR", pathlib.Path.home() / ".triton" / "cache"))
    idx = {}
    for cb in root.glob("*/*.cubin"):
        idx[hashlib.sha256(cb.read_bytes()).hexdigest()] = cb
    return idx


def ttir_signature(ttir_path, symbol):
    """(runtime int args, int args carrying specialization attributes)."""
    text = ttir_path.read_text()
    m = re.search(r"tt\.func public @%s\((.*?)\)\s*attributes" % re.escape(symbol), text, re.S)
    assert m, f"no tt.func signature in {ttir_path}"
    sig = re.sub(r'loc\("[^"]*"\([^)]*\)\)', "", m.group(1))
    sig = re.sub(r"loc\([^)]*\)", "", sig)
    ints, attrs, ptrs = 0, 0, 0
    for arg in re.split(r",\s*(?=%)", sig):
        arg = arg.strip()
        if not arg:
            continue
        if "!tt.ptr" in arg:
            ptrs += 1
            continue
        ints += 1
        if "{" in arg:
            attrs += 1
    return ints, attrs, ptrs


class Pinner:
    """Pick the dump module for a Triton symbol: register count from the
    launch narrows to the constexpr instance, the Triton cache's .ttir picks
    the least specialized runtime-arg instance among the survivors."""

    def __init__(self, dump_dir):
        self.dump = pathlib.Path(dump_dir)
        self.mods = {}
        for mod in sorted(self.dump.glob("module_*.cubin")):
            fns = module_functions(mod)
            if fns:
                self.mods[mod] = fns
        self.shas = {m: hashlib.sha256(m.read_bytes()).hexdigest() for m in self.mods}
        self.cache = triton_cache_by_sha()

    def pin(self, symbol, regs, nparams=None):
        """`nparams` (the launch's param count) separates instances whose
        constexpr choice changes the pointer list (IS_SPEC_DECODING adds two
        pointers to the conv update) but not the register count."""
        cands = {}
        for mod, fns in self.mods.items():
            if fns.get(symbol) == regs:
                cands.setdefault(self.shas[mod], mod)  # dedupe repeated loads
        assert cands, f"{symbol} REG={regs}: no dump module"
        if len(cands) == 1:
            sha, mod = next(iter(cands.items()))
            return mod.name, sha
        scored = []
        for sha, mod in cands.items():
            cb = self.cache.get(sha)
            assert cb, f"{symbol}: dump module {mod.name} not in the Triton cache, cannot read its signature"
            ints, attrs, ptrs = ttir_signature(cb.with_suffix(".ttir"), symbol)
            if nparams is not None and ints + ptrs != nparams - 2:   # minus Triton's 2 scratch pointers
                continue
            scored.append(((-ints, attrs), sha, mod))
        scored.sort()
        assert scored, f"{symbol}: no instance with {nparams} launch params"
        assert len(scored) == 1 or scored[0][0] != scored[1][0], f"{symbol}: instances {scored} not separable"
        return scored[0][2].name, scored[0][1]

    def ttir(self, sha):
        cb = self.cache.get(sha)
        assert cb, f"{sha[:12]}: not in the Triton cache"
        return cb.with_suffix(".ttir").read_text().splitlines()

    def pin_nearest(self, symbol, regs, ref_sha):
        """Among the instances with `regs`, the one whose .ttir differs least
        from a reference instance (same specialization, different baked
        constants — e.g. the conv_fwd of the speculative state layout, whose
        only change is the page stride)."""
        import difflib
        ref = self.ttir(ref_sha)
        best = None
        for mod, fns in self.mods.items():
            if fns.get(symbol) != regs:
                continue
            sha = self.shas[mod]
            n = sum(1 for l in difflib.unified_diff(ref, self.ttir(sha), lineterm="", n=0)
                    if not l.startswith(("---", "+++", "@@")))
            if best is None or n < best[0]:
                best = (n, mod.name, sha)
        assert best, f"{symbol} REG={regs}: no dump module"
        return best[1], best[2]


# ------------------------------------------------------------ verification
def check_prefill(by, T):
    """Falsify the hand-written GDN/attention wiring against one prefill
    forward's pointers and scalars."""
    n_gdn, n_attn = len(GDN_LAYERS), len(ATTN_LAYERS)
    for tag in ("conv_fwd", "post_conv", "cumsum", "kkt", "solve_tril", "recompute",
                "chunk_h", "chunk_o", "layer_norm"):
        assert len(by[tag]) == n_gdn, (tag, len(by[tag]))
    for tag in ("mrope", "cache", "unified"):
        assert len(by[tag]) == n_attn, (tag, len(by[tag]))
    assert len(by["silu"]) == LAYERS and len(by["reduce"]) == 0
    assert len(by["unified"][0]["params"]) == 28, "not the 2D prefill instance"
    for i in range(n_gdn):
        conv, pc, cs, kkt = by["conv_fwd"][i], by["post_conv"][i], by["cumsum"][i], by["kkt"][i]
        st, rc, ch, co, ln = by["solve_tril"][i], by["recompute"][i], by["chunk_h"][i], by["chunk_o"][i], by["layer_norm"][i]
        assert conv["grid"] == [cdiv_i(T, CONV_BLOCK_M), CONV_DIM // 256, 1], conv["grid"]
        assert [pv(conv, 10), pv(conv, 11)] == [QKVZ_DIM, CONV_DIM], "conv strides"
        assert pv(pc, 0) == pv(conv, 8), "post_conv input is not the conv output"
        assert pc["grid"] == [cdiv_i(T, POST_CONV_BLOCK), GDN_V_HEADS + GDN_K_HEADS, 1], pc["grid"]
        assert [pv(pc, j) for j in range(10, 17)] == \
            [CONV_DIM, GDN_V_HEADS, GDN_V_HEADS, GDN_Q, GDN_Q, GDN_V, T], "post_conv strides/L"
        nt = cdiv_i(T, FLA_CHUNK)
        for r in (cs, kkt, st, rc):
            assert r["grid"] == [nt, GDN_V_HEADS, 1], (r["symbol"], r["grid"])
        assert ch["grid"] == [4, GDN_V_HEADS, 1] and co["grid"] == [2, nt, GDN_V_HEADS]
        assert pv(cs, 0) == pv(pc, 8) and pv(kkt, 1) == pv(pc, 9), "g/beta not from post_conv"
        assert pv(kkt, 0) == pv(pc, 6) == pv(rc, 0) == pv(ch, 0) == pv(co, 1), "k drift"
        assert pv(kkt, 2) == pv(cs, 1) == pv(rc, 6) == pv(ch, 4) == pv(co, 4), "cumsum g drift"
        assert pv(st, 0) == pv(kkt, 3) and pv(rc, 5) == pv(st, 1), "A / Ai chain"
        assert pv(rc, 4) == pv(kkt, 3), "vLLM aliases u onto A (same size); we give u its own buffer"
        assert pv(rc, 1) == pv(pc, 7), "recompute v is not post_conv v"
        assert pv(ch, 1) == pv(rc, 4) and pv(ch, 2) == pv(rc, 3), "chunk_h v/w"
        assert pv(co, 0) == pv(pc, 5) and pv(co, 2) == pv(ch, 3) and pv(co, 3) == pv(ch, 5), "chunk_o q/v_new/h"
        assert pv(ch, 6) != pv(ch, 7), "vLLM h0/ht are temporaries; kern aliases both onto the state line"
        for r, j in ((cs, 2), (kkt, 4), (st, 2), (rc, 7), (ch, 8), (co, 6), (conv, 5)):
            assert pv(r, j) == pv(cs, 2), "cu_seqlens drift"
        for r, j in ((cs, 3), (kkt, 5), (st, 3), (rc, 8), (co, 7)):
            assert pv(r, j) == pv(cs, 3), "chunk_indices drift"
        for r, j in ((cs, 4), (kkt, 6), (st, 4), (rc, 9), (ch, 10), (co, 9)):
            assert pv(r, j) == T, "T arg"
        assert pv(ln, 0) == pv(ln, 1), "layer_norm not in place"
        assert [pv(ln, j) for j in (5, 6, 7, 8)] == [GDN_D, GDN_D, GDN_D, T * GDN_V_HEADS]
        assert ln["grid"] == [cdiv_i(T * GDN_V_HEADS, LN_ROWS_PER_BLOCK), 1, 1]
    for i in range(n_attn):
        mr, ca, un = by["mrope"][i], by["cache"][i], by["unified"][i]
        assert mr["grid"] == [T, 1, 1] and ca["grid"] == [T, 1, 1]
        assert pv(mr, 4) == T, "mrope num_tokens"
        assert pv(ca, 0) == pv(mr, 1), "cache key is not the roped k"
        assert pv(un, 1) == pv(mr, 0), "attention query is not the roped q"
        assert pv(ca, 3) - pv(ca, 2) == V_BYTE_OFF and pv(un, 2) == pv(ca, 2) and pv(un, 3) == pv(ca, 3)
        assert [pv(ca, j) for j in range(7, 16)] == \
            [KV_DIM, QKV_DIM, BLOCK_ELEMS_PER_LAYER, 2 * HEAD_DIM, 0, 0, KV_HEADS * 2 * HEAD_DIM, 0, 0]
        assert un["grid"] == [T // BLOCK_Q + 1, KV_HEADS, 1], un["grid"]
        assert pv(un, 17) == pv(un, 5), "rswa_prefix_lens is not the seq_lens dup"
        assert [pv(un, j) for j in range(11, 17)] == [CAPTURE_TABLE_LEN, Q_DIM, HEAD_DIM, Q_DIM, HEAD_DIM, 0]
        assert [pv(un, j) for j in range(18, 24)] == [BLOCK_ELEMS_PER_LAYER, KV_HEADS * 2 * HEAD_DIM, 2 * HEAD_DIM] * 2
        assert pv(un, 25) == 1 and pv(un, 7) == pv(ca, 5) and pv(un, 8) == pv(ca, 6)
    assert len(by["silu"][0]["params"]) == 6 and pv(by["silu"][0], 2) == FFN


def check_decode(by):
    n_gdn, n_attn = len(GDN_LAYERS), len(ATTN_LAYERS)
    for tag in ("conv_update", "recurrent", "layer_norm"):
        assert len(by[tag]) == n_gdn, (tag, len(by[tag]))
    for tag in ("mrope", "cache", "unified", "reduce"):
        assert len(by[tag]) == n_attn, (tag, len(by[tag]))
    assert len(by["unified"][0]["params"]) == 31 and len(by["reduce"][0]["params"]) == 12
    for i in range(n_gdn):
        cu, rec, ln = by["conv_update"][i], by["recurrent"][i], by["layer_norm"][i]
        assert pv(cu, 0) == pv(cu, 4) == pv(rec, 0), "conv update not in place on the qkvz row"
        assert [pv(cu, j) for j in (5, 7, 8)] == [1, 1, 1]
        assert pv(cu, 3) == pv(rec, 8), "conv/ssm index tables differ"
        assert pv(rec, 6) == pv(rec, 7), "decode h0 != ht"
        assert pv(rec, 1) - pv(rec, 2) == GDN_V_HEADS * BF16, "ba layout is not [b | a]"
        assert pv(ln, 0) == pv(ln, 1) == pv(rec, 5), "layer_norm not in place on the recurrent output"
        assert pv(ln, 3) == pv(rec, 0) + Z_OFF * BF16, "z is not the qkvz row tail"
        assert [pv(ln, j) for j in (5, 6, 7, 8)] == [GDN_D, GDN_D, GDN_D, GDN_V_HEADS]
        assert ln["grid"] == [GDN_V_HEADS, 1, 1]
    for i in range(n_attn):
        un, rd = by["unified"][i], by["reduce"][i]
        assert un["grid"] == [1, KV_HEADS, NUM_SEGMENTS]
        assert [pv(rd, j) for j in (1, 2, 3)] == [pv(un, j) for j in (26, 27, 28)]
        assert pv(rd, 0) == pv(un, 0) and pv(rd, 4) == pv(un, 5) and pv(rd, 9) == pv(un, 24)
        assert [pv(rd, j) for j in (6, 7, 8)] == [Q_DIM, HEAD_DIM, CAPTURE_TABLE_LEN]
    assert len(by["mrope"][0]["params"]) == 6, "decode mrope should be the num_tokens==1 instance"


# ---------------------------------------------------------------- builders
def var(s):
    return {"var": s}


def _e(e):
    """Expression form of a var arg: shapes/grids name a var bare."""
    return e["var"] if isinstance(e, dict) and "var" in e else e


def mul(e, c):
    return {"mul": [_e(e), c]}


def cdiv(e, c):
    return {"ceil_div": [_e(e), c]}


def expr(e):
    return {"expr": e}


def buf(n, off=0):
    return {"buf": n, "offset": off} if off else {"buf": n}


def state(n, off=0):
    return {"state": n, "offset": off} if off else {"state": n}


def i32(v):
    return {"i32": v}


def i64(v):
    return {"i64": v}


def f32(v):
    return {"f32": v}


def u8(v):
    return {"u8": v}


def d(label, op, args):
    return {"label": label, "op": op, "args": args}


def a(i):
    return {"param": i}


def scr(name):
    return {"scratch": name}


def step(symbol, params, block, grid, args, shared_mem=None, cubin=None, sha256=None):
    s = {"entry": symbol, "params": params, "block": block, "grid": [_e(g) for g in grid], "args": args}
    if shared_mem is not None:
        s["shared_mem"] = shared_mem
    if cubin is not None:
        s["cubin"] = cubin
    if sha256 is not None:
        s["sha256"] = sha256
    return s


def single(symbol, params, block, grid, shared_mem=None, cubin=None, sha256=None):
    return {"params": params,
            "impl": {"launches": [step(symbol, params, block, grid, [a(i) for i in range(len(params))],
                                    shared_mem, cubin, sha256)]}}


TOKEN_DOMAIN = {"index_into": "model.embed_tokens.weight"}
DOMAINS = {
    "token_ids": TOKEN_DOMAIN,
    "positions": {"index_into": "rope.cos"},
    "slot_mapping": {"index_into": "kv"},
    "block_table": {"index_into": "kv", "stride": BLOCK_SIZE},
    "gdn.line_index": {"index_into": "gdn", "stride": GDN_LINE_BYTES},
    "seq_lens": {"min": 1},
    "cu_seqlens_q": {"min": 0, "max": "tokens", "monotone": True},
    "next_token": TOKEN_DOMAIN,
    "kv_scales": {"min": 0.0},
}

I2 = ["i64", "i64"]   # Triton's trailing global/profile scratch pointers (always 0)


def build(pre, dec, pins, eps, attn_scale, gdn_scale, silu_sym, spec=None):
    """`spec` (from --spec): {"src": bs=1 launch records of the speculative
    path, "pins": {tag: (cubin, sha)}, "attn_draft": DSpark's pinned
    non-causal unified-attention step} — adds the DFlash2 programs."""
    T = var("tokens")
    S = var("seqs")
    # GDN state layout: one page per layer per sequence either way; the
    # speculative page is wider (10-token conv line) and the kernels' page
    # stride is a constexpr, so the two layouts pin different instances
    if spec:
        page_bytes, ssm_off, seq_bytes = SPEC_PAGE_BYTES, SPEC_SSM_OFF, SPEC_SEQ_BYTES
    else:
        page_bytes, ssm_off, seq_bytes = GDN_LINE_BYTES, CONV_STATE_BYTES, GDN_SEQ_BYTES
    n_pages = GDN_LINES_BOUND
    # entries per (layer, sequence) cell of the line table: the spec
    # kernels take an 8-entry list per sequence
    line_w = SPEC_BLOCK if spec else 1
    DOMAINS["gdn.line_index"] = {"index_into": "gdn", "stride": page_bytes}
    buffers = {
        "token_ids": {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "token"},
        "positions": {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "position"},
        "slot_mapping": {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "slot"},
        # one row per sequence of a decode_batch step (prefill / decode are
        # seqs=1: row 0 / the first two cu_seqlens entries)
        "block_table": {"dtype": "i32", "shape": ["seqs", BLOCK_TABLE_LEN], "kind": "input"},
        "seq_lens": {"dtype": "i32", "shape": ["seqs"], "kind": "input", "fill": "seq_len"},
        "cu_seqlens_q": {"dtype": "i32", "shape": [MAX_SEQS + 1], "kind": "input", "fill": "cu_seqlens"},
        "logits": {"dtype": "bf16", "shape": ["seqs", VOCAB], "kind": "workspace"},
        "next_token": {"dtype": "i64", "shape": ["seqs"], "kind": "output", "fill": "tokens"},
    }
    ws = {
        "residual": ["tokens", HIDDEN], "x": ["tokens", HIDDEN], "y": ["tokens", HIDDEN],
        "final_x": [1, HIDDEN],
        # GDN
        "qkvz": ["tokens", QKVZ_DIM], "ba": ["tokens", BA_DIM], "conv_out": ["tokens", CONV_DIM],
        "gdn_q": ["tokens", GDN_Q], "gdn_k": ["tokens", GDN_Q], "gdn_v": ["tokens", GDN_V],
        "Ai": ["tokens", GDN_V_HEADS, FLA_CHUNK],
        "w": ["tokens", GDN_V], "u": ["tokens", GDN_V], "v_new": ["tokens", GDN_V],
        "h": [NT_MAX, GDN_V_HEADS, GDN_D, GDN_D],
        "core_attn_out": ["tokens", GDN_V], "z_c": ["tokens", GDN_V],
        # attention
        "qkv": ["tokens", QKV_DIM], "q_n": ["tokens", Q_DIM], "k_n": ["tokens", KV_DIM],
        "cos_g": ["tokens", ROT_HALF], "sin_g": ["tokens", ROT_HALF],
        "attn_out": ["tokens", Q_DIM], "gated": ["tokens", Q_DIM],
        # MLP
        "gate_up": ["tokens", 2 * FFN], "act": ["tokens", FFN],
    }
    for name, shape in ws.items():
        buffers[name] = {"dtype": "bf16", "shape": shape, "kind": "workspace"}
    for name, shape in {"g": ["tokens", GDN_V_HEADS], "beta": ["tokens", GDN_V_HEADS],
                        "g_cum": ["tokens", GDN_V_HEADS], "A": ["tokens", GDN_V_HEADS, FLA_CHUNK],
                        # the sequence's SSM state line, gathered for the FLA
                        # chunk kernel (h0/ht by pointer) and scattered back
                        "h0": [GDN_V_HEADS, GDN_D, GDN_D]}.items():
        buffers[name] = {"dtype": "f32", "shape": shape, "kind": "workspace"}

    def weight(name, shape, dtype="bf16"):
        buffers[name] = {"dtype": dtype, "shape": shape, "kind": "weight"}

    weight("model.embed_tokens.weight", [VOCAB, HIDDEN])
    weight("lm_head.weight", [VOCAB, HIDDEN])
    weight("model.norm.weight_p1", [HIDDEN], "f32")
    weight("rope.cos", [MAX_POS, ROT_HALF])
    weight("rope.sin", [MAX_POS, ROT_HALF])
    weight("kv_scales", [2], "f32")
    weight("fla.chunk_indices", [NT_MAX, 2], "i32")
    weight("fla.chunk_offsets", [2], "i64")
    weight("conv.batch_ptr", [CHUNK_MAX // CONV_BLOCK_M], "i32")
    weight("conv.token_chunk_offset", [CHUNK_MAX // CONV_BLOCK_M], "i32")
    # line table over the per-sequence GDN state: row = GDN layer, column =
    # sequence of the batch (runtime-filled from the lease; a prefill or
    # bs=1 decode reads column 0)
    buffers["gdn.line_index"] = {"dtype": "i32", "kind": "input",
                                 "shape": [len(GDN_LAYERS), "seqs"] + ([SPEC_BLOCK] if spec else [])}
    weight("gdn.has_initial", [16], "u8")
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        weight(p + "input_layernorm.weight_p1", [HIDDEN], "f32")
        weight(p + "post_attention_layernorm.weight_p1", [HIDDEN], "f32")
        weight(p + "mlp.gate_up_proj.weight", [2 * FFN, HIDDEN])
        weight(p + "mlp.down_proj.weight", [HIDDEN, FFN])
        if i in ATTN_LAYERS:
            weight(p + "self_attn.qkv_proj.weight", [QKV_DIM, HIDDEN])
            weight(p + "self_attn.q_norm.weight_p1", [HEAD_DIM], "f32")
            weight(p + "self_attn.k_norm.weight_p1", [HEAD_DIM], "f32")
            weight(p + "self_attn.o_proj.weight", [HIDDEN, Q_DIM])
        else:
            weight(p + "linear_attn.in_proj_qkvz.weight", [QKVZ_DIM, HIDDEN])
            weight(p + "linear_attn.in_proj_ba.weight", [BA_DIM, HIDDEN])
            weight(p + "linear_attn.conv1d.weight", [CONV_DIM, 4])
            weight(p + "linear_attn.A_log", [GDN_V_HEADS], "f32")
            weight(p + "linear_attn.dt_bias", [GDN_V_HEADS])
            weight(p + "linear_attn.norm.weight", [GDN_D])
            weight(p + "linear_attn.out_proj.weight", [HIDDEN, GDN_V])

    if spec:
        buffers.update({
            "draft_block_table": {"dtype": "i32", "shape": ["seqs", D_TABLE_LEN], "kind": "input"},
            # each sequence's first row: the caller stages it, the round
            # splices its own draft and verify rows from it
            "anchor_token": {"dtype": "i64", "shape": ["seqs"], "kind": "input", "fill": "token"},
            # target hidden states at the 5 taps, projected by fc: written by
            # prefill / verify, read by draft_precompute
            "fc_out": {"dtype": "bf16", "shape": ["tokens", HIDDEN], "kind": "carry"},
            # what a round hands back: verify's greedy token after every row,
            # and how many of them the sequence takes (accepted drafts + 1)
            "verify_tokens": {"dtype": "i64", "shape": ["seqs", SPEC_BLOCK], "kind": "output", "fill": "tokens"},
            "nacc_adv": {"dtype": "i32", "shape": ["seqs"], "kind": "output", "fill": "count"},
            "draft_tokens": {"dtype": "i64", "shape": ["seqs", DRAFT_TOKENS], "kind": "output"},
            # the round's device-written rows: draft's ids ([anchor, mask x7]),
            # verify's ids (spliced from draft's output), the count a step's
            # GDN resumes from (1: the committed state) and advance's line
            # table (the line in entry `accepted`)
            "draft_ids": {"dtype": "i64", "shape": ["tokens"], "kind": "carry"},
            "verify_ids": {"dtype": "i64", "shape": ["tokens"], "kind": "carry"},
            "nacc_one": {"dtype": "i32", "shape": ["seqs"], "kind": "carry"},
            "line_adv": {"dtype": "i32", "shape": [len(GDN_LAYERS), "seqs", SPEC_BLOCK], "kind": "carry"},
            "logits_blk": {"dtype": "bf16", "shape": [SPEC_ROWS_MAX, VOCAB], "kind": "workspace"},
            "cand_ids": {"dtype": "i64", "shape": [SPEC_ROWS_MAX, SEL_K], "kind": "workspace"},
            "cand_vals": {"dtype": "f32", "shape": [SPEC_ROWS_MAX, SEL_K], "kind": "workspace"},
            # verify's post-conv k / v and gates a / b per GDN layer, re-read
            # by advance
            "k_save": {"dtype": "bf16", "shape": [len(GDN_LAYERS), SPEC_ROWS_MAX, GDN_Q], "kind": "carry"},
            "v_save": {"dtype": "bf16", "shape": [len(GDN_LAYERS), SPEC_ROWS_MAX, GDN_V], "kind": "carry"},
            "a_save": {"dtype": "bf16", "shape": [len(GDN_LAYERS), SPEC_ROWS_MAX, GDN_V_HEADS], "kind": "carry"},
            "b_save": {"dtype": "bf16", "shape": [len(GDN_LAYERS), SPEC_ROWS_MAX, GDN_V_HEADS], "kind": "carry"},
        })
        for name, shape in {
            "a_c": ["tokens", GDN_V_HEADS], "b_c": ["tokens", GDN_V_HEADS],   # advance's masked a/b
            "kv_flat": ["tokens", D_KV_FLAT],
            "d_qkv": ["tokens", D_QKV], "d_q": ["tokens", D_Q], "d_coef": ["tokens", D_CONV_PROJ],
            "d_attn": ["tokens", D_Q],
            "hidden_r": [SPEC_ROWS_MAX, SEL_RANK], "succ_g": [SPEC_ROWS_MAX * SEL_K, SEL_RANK],
            "pred_g": [SPEC_ROWS_MAX * SEL_K, SEL_RANK], "pred_anchor": ["seqs", SEL_RANK],
        }.items():
            buffers[name] = {"dtype": "bf16", "shape": shape, "kind": "workspace"}
        for j in range(5):
            weight(f"draft.fc.{j}.weight", [HIDDEN, HIDDEN])
        weight("draft.hidden_norm.weight", [HIDDEN])
        weight("draft.norm.weight", [HIDDEN])
        weight("draft.fused_kv.weight", [D_KV_FLAT, HIDDEN])
        weight("draft.rope.cos_sin_cache", [MAX_POS, D_HEAD_DIM])
        weight("draft.kv_scales", [2 * D_LAYERS], "f32")
        weight("draft.selector.hidden_projection.weight", [SEL_RANK, HIDDEN])
        weight("draft.selector.predecessor", [VOCAB, SEL_RANK])
        weight("draft.selector.successor", [VOCAB, SEL_RANK])
        for l in range(D_LAYERS):
            p = f"draft.layers.{l}."
            weight(p + "input_layernorm.weight", [HIDDEN])
            weight(p + "post_attention_layernorm.weight", [HIDDEN])
            weight(p + "self_attn.qkv_proj.weight", [D_QKV, HIDDEN])
            weight(p + "self_attn.q_norm.weight", [D_HEAD_DIM])
            weight(p + "self_attn.k_norm.weight", [D_HEAD_DIM])
            weight(p + "self_attn.o_proj.weight", [HIDDEN, D_Q])
            weight(p + "mlp.gate_up_proj.weight", [2 * FFN, HIDDEN])
            weight(p + "mlp.down_proj.weight", [HIDDEN, FFN])
            for c in ("attention_conv", "mlp_conv"):
                weight(p + c + ".base_kernel", [4, HIDDEN])
                weight(p + c + ".kernel_projection.weight", [D_CONV_PROJ, HIDDEN])

    def blk(tag, src=pre):
        return src[tag][0]["block"]

    def smem(tag, src=pre):
        return src[tag][0]["dynamic_shared_mem_bytes"]

    def spec_kernel(tag, params, grid, **kw):
        """Single-step kernel from the speculative capture, pinned."""
        r = spec["src"][tag]
        cubin, sha = spec["pins"][tag]
        return single(SPEC_SYMS[tag], params, r["block"], grid,
                      shared_mem=r["dynamic_shared_mem_bytes"] or None, cubin=cubin, sha256=sha, **kw)

    def tri(tag, params, grid, src=pre, **kw):
        """Single-step Triton kernel pinned to its dump module."""
        cubin, sha = pins[(tag, src is pre)]
        return single(TRITON[tag], params, blk(tag, src), grid,
                      shared_mem=smem(tag, src) or None, cubin=cubin, sha256=sha, **kw)

    GEMMA_PARAMS = ["out buffer<bf16>", "in buffer<bf16>", "in buffer<f32>",
                    "i32", "i32", "i32", "i32", "i32", "i32", "i32", "f32"]
    GEMMA_FUSED_PARAMS = ["out buffer<bf16>", "in buffer<bf16>", "inout buffer<bf16>",
                          "in buffer<f32>", "i32", "i32", "i32", "i32", "f32"]
    GEMMA_SMEM = (2 * HIDDEN + 512) * 4
    GEMMA_HEAD_SMEM = (2 * HEAD_DIM + 512) * 4
    LN_PARAMS = ["inout buffer<bf16>", "out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                 "out buffer<f32>", "i32", "i32", "i32", "i32", "f32"] + I2
    LN_IFACE = LN_PARAMS[:4] + LN_PARAMS[5:]
    UNIFIED_PARAMS = [
        # `kernel_unified_attention` only loads the KV pages: its two stores go
        # to the output and the segment scratch. `reshape_and_cache` is the
        # writer. Declaring the state `in` keeps a profiler from snapshotting
        # the whole cache around every isolated call.
        "out buffer<bf16>", "in buffer<bf16>", "in state", "in state",
        "in buffer<i32>", "in buffer<i32>", "f32",
        "in buffer<f32>", "in buffer<f32>", "f32", "f32",
        "i64", "i64", "i64", "i64", "i64", "i64",
        "in buffer<i32>", "i64", "i64", "i64", "i64", "i64", "i64",
        "in buffer<i32>", "i32", "out buffer<f32>", "out buffer<f32>", "out buffer<f32>"] + I2
    ATTN_IFACE = UNIFIED_PARAMS[:26] + UNIFIED_PARAMS[29:]
    REDUCE_PARAMS = ["out buffer<bf16>", "in buffer<f32>", "in buffer<f32>", "in buffer<f32>",
                     "in buffer<i32>", "f32", "i64", "i64", "i64", "in buffer<i32>"] + I2

    def layer_norm_kernel(src, grid, rows_shape):
        """Gated RMSNorm (FLA layer_norm_fwd) with its Rstd side output as
        impl scratch."""
        cubin, sha = pins[("layer_norm", src is pre)]
        return {
            "params": LN_IFACE,
            "impl": {
                "scratch": {"rstd": {"dtype": "f32", "shape": rows_shape}},
                "launches": [step(TRITON["layer_norm"], LN_PARAMS, blk("layer_norm", src), grid,
                               [a(0), a(1), a(2), a(3), scr("rstd")] + [a(i) for i in range(4, 11)],
                               cubin=cubin, sha256=sha)],
            },
        }

    kernels = {
        "embedding": single("kern_embedding_i64_bf16",
                            ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
                            [256, 1, 1], [T, 1, 1], **hw("embedding")),
        "gemm": single("extern:cublaslt_bf16_tn",
                       ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
                       [1, 1, 1], [1, 1, 1]),
        # Gemma norms: one block per row; rows/grid differ per use (ATen's
        # reduction width depends on the row count, so `rows` is an arg).
        "gemma_norm": single("kern_gemma_rms_norm_bf16", GEMMA_PARAMS, [512, 1, 1], [T, 1, 1],
                             shared_mem=GEMMA_SMEM, **hw("gemma_rms_norm")),
        "gemma_norm_qhead": single("kern_gemma_rms_norm_bf16", GEMMA_PARAMS, [512, 1, 1],
                                   [mul(T, HEADS), 1, 1], shared_mem=GEMMA_HEAD_SMEM,
                                   **hw("gemma_rms_norm")),
        "gemma_norm_khead": single("kern_gemma_rms_norm_bf16", GEMMA_PARAMS, [512, 1, 1],
                                   [mul(T, KV_HEADS), 1, 1], shared_mem=GEMMA_HEAD_SMEM,
                                   **hw("gemma_rms_norm")),
        "gemma_fused_norm": single("kern_gemma_fused_add_rms_norm_bf16", GEMMA_FUSED_PARAMS,
                                   [512, 1, 1], [T, 1, 1], shared_mem=GEMMA_SMEM,
                                   **hw("gemma_rms_norm")),
        "silu_mul": single(silu_sym, ["out buffer<bf16>", "in buffer<bf16>", "i32", "i32", "f32", "i32"],
                           blk("silu"), [T, 1, 1], cubin=pins[("silu", True)][0], sha256=pins[("silu", True)][1]),
        "copy_rows": single("kern_copy_rows_bf16",
                            ["out buffer<bf16>", "in buffer<bf16>", "i32", "i32", "i32"],
                            [256, 1, 1], [T, 1, 1], **hw("copy_rows")),
        "last_row": single("kern_last_row_bf16",
                           ["out buffer<bf16>", "in buffer<bf16>", "i32", "i32", "i32"],
                           [256, 1, 1], [1, 1, 1], **hw("copy_rows")),
        "sigmoid_mul": single("kern_sigmoid_mul_bf16",
                              ["out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                               "i32", "i32", "i32", "i32"],
                              [256, 1, 1], [T, 1, 1], **hw("sigmoid_mul")),
        "argmax_row": {
            "params": ["in buffer<bf16>", "out buffer<i64>", "i32"],
            "impl": {
                "scratch": {"pmax": {"dtype": "f32", "shape": [1, 64]},
                            "pidx": {"dtype": "i32", "shape": [1, 64]}},
                "launches": [
                    step("kern_argmax_partial_bf16",
                         ["in buffer<bf16>", "out buffer<f32>", "out buffer<i32>", "i32"],
                         [1024, 1, 1], [1, 64, 1], [a(0), scr("pmax"), scr("pidx"), a(2)],
                         **hw("argmax")),
                    step("kern_argmax_final_i64",
                         ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32"],
                         [64, 1, 1], [1, 1, 1], [scr("pmax"), scr("pidx"), a(1), i32(64)],
                         **hw("argmax")),
                ],
            },
        },
        # --- GDN prefill chain (vLLM triton backend, FLA chunk kernels)
        "conv_fwd": tri("conv_fwd",
                        ["in buffer<bf16>", "in buffer<bf16>", "inout state", "in buffer<i32>",
                         "in buffer<u8>", "in buffer<i32>", "in buffer<i32>", "in buffer<i32>",
                         "out buffer<bf16>", "i32", "i64", "i64"] + I2,
                        [cdiv(T, CONV_BLOCK_M), CONV_DIM // 256, 1]),
        "post_conv": tri("post_conv",
                         ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>",
                          "in buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>",
                          "out buffer<f32>", "out buffer<f32>",
                          "i32", "i32", "i32", "i32", "i32", "i32", "i32"] + I2,
                         [cdiv(T, POST_CONV_BLOCK), GDN_V_HEADS + GDN_K_HEADS, 1]),
        "cumsum": tri("cumsum", ["in buffer<f32>", "out buffer<f32>", "in buffer<i32>",
                                 "in buffer<i32>", "i32"] + I2,
                      [cdiv(T, FLA_CHUNK), GDN_V_HEADS, 1]),
        "kkt": tri("kkt", ["in buffer<bf16>", "in buffer<f32>", "in buffer<f32>", "out buffer<f32>",
                           "in buffer<i32>", "in buffer<i32>", "i32"] + I2,
                   [cdiv(T, FLA_CHUNK), GDN_V_HEADS, 1]),
        # writes only the lower/diagonal 16x16 tiles of each 64x64 block; the
        # upper tiles are the zeros the buffer was allocated with
        "solve_tril": tri("solve_tril", ["in buffer<f32>", "out buffer<bf16>", "in buffer<i32>",
                                         "in buffer<i32>", "i32"] + I2,
                          [cdiv(T, FLA_CHUNK), GDN_V_HEADS, 1]),
        "recompute": tri("recompute",
                         ["in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>", "out buffer<bf16>",
                          "out buffer<bf16>", "in buffer<bf16>", "in buffer<f32>", "in buffer<i32>",
                          "in buffer<i32>", "i32"] + I2,
                         [cdiv(T, FLA_CHUNK), GDN_V_HEADS, 1]),
        # h0/ht are one SSM state line: each program loads its h0 tile
        # first and stores ht last, so in-place is race-free. The kernel
        # takes them by pointer, so the per-sequence layout gathers the
        # sequence's line into `h0` and scatters it back (vLLM does the
        # same).
        "chunk_h": tri("chunk_h",
                       ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>",
                        "in buffer<f32>", "out buffer<bf16>", "inout buffer<f32>", "inout buffer<f32>",
                        "in buffer<i32>", "in buffer<i64>", "i32"] + I2,
                       [GDN_D // 32, GDN_V_HEADS, 1]),
        "line_gather": single("kern_line_gather",
                              ["in buffer<i32>", "inout state", "out buffer<f32>", "i64", "i64", "i64"],
                              [256, 1, 1], [SSM_STATE_BYTES // 16 // 256, 1, 1], **hw("line_copy")),
        "line_scatter": single("kern_line_scatter",
                               ["in buffer<i32>", "inout state", "in buffer<f32>", "i64", "i64", "i64"],
                               [256, 1, 1], [SSM_STATE_BYTES // 16 // 256, 1, 1], **hw("line_copy")),
        "chunk_o": tri("chunk_o",
                       ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                        "in buffer<f32>", "out buffer<bf16>", "in buffer<i32>", "in buffer<i32>",
                        "f32", "i32"] + I2,
                       [GDN_D // 64, cdiv(T, FLA_CHUNK), GDN_V_HEADS]),
        "gated_norm": layer_norm_kernel(pre, [mul(T, GDN_V_HEADS // LN_ROWS_PER_BLOCK), 1, 1],
                                        ["tokens", GDN_V_HEADS]),
        # --- GDN decode: vLLM's decode kernels are batch-native — one row
        # per sequence, the state line from `state_indices[seq]`; grids from
        # their launchers (causal_conv1d_update: (batch, cdiv(dim, BLOCK_N));
        # fused_recurrent_..._packed_decode: (NV, B·HV); layer_norm_fwd:
        # cdiv(rows, ROWS_PER_BLOCK) with ROWS_PER_BLOCK = 1 in the bs=1
        # instance). `batch` is a plain runtime arg of the conv kernel.
        "conv_update": tri("conv_update",
                           ["in buffer<bf16>", "in buffer<bf16>", "inout state", "in buffer<i32>",
                            "out buffer<bf16>", "i32", "i32", "i64", "i64"] + I2,
                           [S, CONV_DIM // 256, 1], src=dec),
        "recurrent": tri("recurrent",
                         ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>",
                          "in buffer<bf16>", "out buffer<bf16>", "inout state", "inout state",
                          "in buffer<i32>", "f32"] + I2,
                         [GDN_D // 32, mul(S, GDN_V_HEADS), 1], src=dec),
        "gated_norm_decode": layer_norm_kernel(dec, [mul(S, GDN_V_HEADS), 1, 1], ["seqs", GDN_V_HEADS]),
        # --- attention
        # one generic instance for both programs (decode's launch in vLLM is
        # the num_tokens==1 specialization: same arithmetic, one arg fewer)
        "mrope": tri("mrope", ["inout buffer<bf16>", "inout buffer<bf16>", "in buffer<bf16>",
                               "in buffer<bf16>", "i32"] + I2, [T, 1, 1]),
        "reshape_and_cache": tri("cache",
                                 ["in buffer<bf16>", "in buffer<bf16>", "inout state", "inout state",
                                  "in buffer<i64>", "in buffer<f32>", "in buffer<f32>"] + ["i64"] * 9,
                                 [T, 1, 1]),
        "attn_prefill": tri("unified", ATTN_IFACE, [cdiv(T, BLOCK_Q), KV_HEADS, 1]),
        # batched decode attention: the same 2D causal instance, one query
        # row per sequence (vLLM itself decodes through it past its
        # num_seqs threshold). grid.x covers vLLM's q-block index space
        # tokens//BLOCK_Q + num_seqs; no two-var sum in the expression set,
        # so ceil((BLOCK_Q+1)·tokens/BLOCK_Q) ≥ tokens//BLOCK_Q + seqs
        # (seqs ≤ tokens); the extra blocks return early in the kernel.
        "attn_batch": tri("unified", ATTN_IFACE, [cdiv(mul(T, BLOCK_Q + 1), BLOCK_Q), KV_HEADS, 1]),
        # row argmax over `tokens` rows (decode_batch / verify)
        "argmax": {
            "params": ["in buffer<bf16>", "out buffer<i64>", "i32"],
            "impl": {
                "scratch": {"pmax": {"dtype": "f32", "shape": ["tokens", 64]},
                            "pidx": {"dtype": "i32", "shape": ["tokens", 64]}},
                "launches": [
                    step("kern_argmax_partial_bf16",
                         ["in buffer<bf16>", "out buffer<f32>", "out buffer<i32>", "i32"],
                         [1024, 1, 1], [T, 64, 1], [a(0), scr("pmax"), scr("pidx"), a(2)],
                         **hw("argmax")),
                    step("kern_argmax_final_i64",
                         ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32"],
                         [64, 1, 1], [T, 1, 1], [scr("pmax"), scr("pidx"), a(1), i32(64)],
                         **hw("argmax")),
                ],
            },
        },
        "attn": {
            "params": ATTN_IFACE,
            "impl": {
                "scratch": {
                    "segm_out": {"dtype": "f32", "shape": [1, HEADS, NUM_SEGMENTS, HEAD_DIM]},
                    "segm_max": {"dtype": "f32", "shape": [1, HEADS, NUM_SEGMENTS]},
                    "segm_expsum": {"dtype": "f32", "shape": [1, HEADS, NUM_SEGMENTS]},
                },
                "launches": [
                    step(TRITON["unified"], UNIFIED_PARAMS, blk("unified", dec),
                         [1, KV_HEADS, NUM_SEGMENTS],
                         [a(i) for i in range(26)]
                         + [scr("segm_out"), scr("segm_max"), scr("segm_expsum"), a(26), a(27)],
                         shared_mem=smem("unified", dec), cubin=pins[("unified", False)][0],
                         sha256=pins[("unified", False)][1]),
                    step(TRITON["reduce"], REDUCE_PARAMS, blk("reduce", dec), [1, HEADS, 1],
                         [a(0), scr("segm_out"), scr("segm_max"), scr("segm_expsum"), a(5),
                          f32(1.0), i64(Q_DIM), i64(HEAD_DIM), i64(BLOCK_TABLE_LEN), a(24),
                          i64(0), i64(0)],
                         shared_mem=smem("reduce", dec), cubin=pins[("reduce", False)][0],
                         sha256=pins[("reduce", False)][1]),
                ],
            },
        },
    }

    Z2 = [i64(0), i64(0)]

    if spec:
        RMS_PARAMS = ["out buffer<bf16>", "in buffer<bf16>", "i64", "i64", "i64", "i64", "i64",
                      "in buffer<bf16>", "i64", "f32", "i32", "i32"]
        ad = spec["attn_draft"]
        D_BLOCK_Q = ad["grid"][0]["ceil_div"][1]   # the borrowed instance's query rows per block
        kernels.update({
            "gemm_acc": single("extern:cublaslt_bf16_tn_acc",
                               ["in buffer<bf16>", "in buffer<bf16>", "inout buffer<bf16>", "i32", "i32", "i32"],
                               [1, 1, 1], [1, 1, 1]),
            # --- target GDN under speculation (vLLM's spec-decode kernels)
            # conv update: seqlen 8 / state_len 10 are constexprs; reads the
            # history taps at conv-line offset num_accepted-1, writes the
            # new 10-wide window (in place on the qkvz rows)
            "conv_update_spec": spec_kernel("conv_update_spec",
                                            ["in buffer<bf16>", "in buffer<bf16>", "inout state", "in buffer<i32>",
                                             "in buffer<i32>", "in buffer<i32>", "out buffer<bf16>",
                                             "i32", "i32", "i64", "i64"] + I2,
                                            [S, CONV_DIM // 256, 1]),
            # recurrent delta rule over each sequence's rows with the sigmoid
            # gating fused: initial state from entry num_accepted-1 of the
            # sequence's line-table cell, the state after row i stored to
            # entry i wherever that entry is non-null
            "recurrent_spec": spec_kernel("recurrent_spec",
                                          ["in buffer<f32>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                                           "f32", "f32", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                                           "out buffer<bf16>", "inout state", "inout state", "in buffer<i32>",
                                           "in buffer<i32>", "in buffer<i32>", "f32", "i64", "i64"] + I2,
                                          [1, GDN_D // 32, mul(S, GDN_V_HEADS)]),
            # the advance pass's own two steps (tools/kernels-src/gdn_advance.cu)
            "mask_row0": single("kern_mask_row0",
                                ["out buffer<bf16>", "in buffer<bf16>", "i32", "i32", "i32"],
                                [64, 1, 1], [T, 1, 1], **hw("gdn_advance")),
            "conv_shift": single("kern_conv_shift",
                                 ["in buffer<i32>", "i32", "inout state", "in buffer<i32>", "i64", "i64"],
                                 [256, 1, 1], [S, CONV_DIM * BF16 // 16 // 256, 1], **hw("gdn_advance")),
            # the round's glue (tools/kernels-src/spec_round.cu): draft's ids
            # from the anchor and the mask, verify's from draft's output, the
            # count of rows taken, the line table entry advance reads, and
            # the constant 1 a step's GDN resumes from
            "splice_draft": single("kern_splice_draft",
                                   ["in buffer<i64>", "out buffer<i64>", "i32", "i64"],
                                   [32, 1, 1], [S, 1, 1], **hw("spec_round")),
            "splice_verify": single("kern_splice_verify",
                                    ["in buffer<i64>", "in buffer<i64>", "out buffer<i64>", "i32", "i32"],
                                    [32, 1, 1], [S, 1, 1], **hw("spec_round")),
            "spec_count": single("kern_spec_count",
                                 ["in buffer<i64>", "in buffer<i64>", "out buffer<i32>", "i32", "i32"],
                                 [32, 1, 1], [S, 1, 1], **hw("spec_round")),
            "spec_lines": single("kern_spec_lines",
                                 ["in buffer<i32>", "in buffer<i32>", "out buffer<i32>", "i32", "i32", "i32"],
                                 [64, 1, 1], [S, 1, 1], **hw("spec_round")),
            "ones_i32": single("kern_ones_i32", ["out buffer<i32>", "i32"], [128, 1, 1], [cdiv(S, 128), 1, 1],
                               **hw("spec_round")),
            # verify's attention: the 2D causal instance over SPEC_BLOCK rows
            # per sequence; q-block index space tokens//BLOCK_Q + seqs
            "attn_verify": tri("unified", ATTN_IFACE,
                               [cdiv(mul(T, BLOCK_Q + SPEC_BLOCK), BLOCK_Q * SPEC_BLOCK), KV_HEADS, 1]),
            # --- draft (vLLM CUDA kernels + the borrowed non-causal attention instance)
            "d_rms_norm": spec_kernel("d_rms_norm", RMS_PARAMS, [T, 1, 1]),
            "d_head_norm": spec_kernel("d_head_norm", RMS_PARAMS, [mul(T, D_HEADS), 1, 1]),
            "d_kv_norm": spec_kernel("d_head_norm", RMS_PARAMS, [mul(T, D_KV_HEADS), 1, 1]),
            "d_fused_norm": spec_kernel("d_fused_norm",
                                        ["inout buffer<bf16>", "i64", "inout buffer<bf16>", "in buffer<bf16>",
                                         "f32", "i32", "i32", "i64"], [T, 1, 1]),
            "d_rope": spec_kernel("d_rope",
                                  ["in buffer<i64>", "inout buffer<bf16>", "inout buffer<bf16>", "in buffer<bf16>",
                                   "i32", "i64", "i64", "i64", "i32", "i32", "i32", "i64", "u8"], [T, 1, 1]),
            "d_cache": spec_kernel("d_cache",
                                   ["in buffer<bf16>", "in buffer<bf16>", "inout state", "inout state", "in buffer<i64>",
                                    "i64", "i64", "i64", "i64", "i64", "i32", "i32", "i32",
                                    "in buffer<f32>", "in buffer<f32>", "i32"], [T, 1, 1]),
            "attn_draft": single("kernel_unified_attention", ATTN_IFACE, ad["block"],
                                 [cdiv(mul(T, D_BLOCK_Q + SPEC_BLOCK), D_BLOCK_Q * SPEC_BLOCK), D_KV_HEADS, 1],
                                 shared_mem=ad.get("shared_mem"), cubin=ad["cubin"], sha256=ad["sha256"]),
            "dflash_conv": single("kern_dflash_conv_bf16",
                                  ["inout buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                                   "i32", "i32", "i32", "i32", "i32"],
                                  [256, 1, 1], [HIDDEN // 256, 1, 1], **hw("dflash_conv")),
            "topk16": single("kern_topk16_bf16",
                             ["in buffer<bf16>", "out buffer<i64>", "out buffer<f32>", "i32", "i32"],
                             [1024, 1, 1], [T, 1, 1], **hw("topk_row")),
            "dflash_select": single("kern_dflash_select",
                                    ["in buffer<i64>", "in buffer<f32>", "in buffer<bf16>", "in buffer<bf16>",
                                     "in buffer<bf16>", "in buffer<bf16>", "out buffer<i64>", "i32", "i32", "i32", "i32"],
                                    [SEL_RANK, 1, 1], [S, 1, 1], shared_mem=SEL_K * SEL_RANK * 4,
                                    **hw("dflash_select")),
            "gather_cands": single("kern_embedding_i64_bf16",
                                   ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
                                   [256, 1, 1], [mul(T, SEL_K), 1, 1], **hw("embedding")),
            "gather_row": single("kern_embedding_i64_bf16",
                                 ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
                                 [256, 1, 1], [S, 1, 1], **hw("embedding")),
        })

    def gemm(label, ab, w, c, m, n, k):
        return d(label, "gemm", [buf(ab), buf(w), buf(c), m, i32(n), i32(k)])

    def fused(label, x_in, w):
        return d(label, "gemma_fused_norm",
                 [buf("x"), buf(x_in), buf("residual"), buf(w), i32(HIDDEN), T,
                  i32(HIDDEN), i32(HIDDEN), f32(eps)])

    def saved(g):
        """GDN layer g's rows of verify's saved k / v / a / b (spec)."""
        return (buf("k_save", g * SPEC_ROWS_MAX * GDN_Q * BF16), buf("v_save", g * SPEC_ROWS_MAX * GDN_V * BF16),
                buf("a_save", g * SPEC_ROWS_MAX * GDN_V_HEADS * BF16), buf("b_save", g * SPEC_ROWS_MAX * GDN_V_HEADS * BF16))

    def gdn_layer(i, decode, nacc=None, batch=False):
        """decode=False: the chunked FLA prefill chain.  decode=True: the
        recurrent chain — the packed decode kernels over `seqs` rows, or,
        when `nacc` names the num_accepted_tokens buffer, vLLM's speculative
        kernels over the `tokens` rows (SSM checkpoint per row, resume from
        slot nacc-1)."""
        p = f"model.layers.{i}.linear_attn."
        l = f"l{i}."
        g = GDN_LAYERS.index(i)
        # this layer's row of the line table: one cell per sequence of the
        # batch (a prefill / bs=1 decode reads cell 0; the spec kernels an
        # 8-entry cell)
        idx = buf("gdn.line_index", 4 * line_w * MAX_SEQS * g)
        h0 = ht = buf("h0")
        line_args = [i64(page_bytes), i64(ssm_off), i64(SSM_STATE_BYTES)]
        gather = [d(l + "h0_gather", "line_gather", [idx, state("gdn"), buf("h0")] + line_args)]
        scatter = [d(l + "ht_scatter", "line_scatter", [idx, state("gdn"), buf("h0")] + line_args)]
        ds = [
            gemm(l + "in_proj_qkvz", "x", p + "in_proj_qkvz.weight", "qkvz", T, QKVZ_DIM, HIDDEN),
            gemm(l + "in_proj_ba", "x", p + "in_proj_ba.weight", "ba", T, BA_DIM, HIDDEN),
        ]
        a_ = buf("ba", GDN_V_HEADS * BF16)   # ba row = [b | a]
        b_ = buf("ba")
        if not decode and nacc is None:
            ds += [
                d(l + "conv", "conv_fwd",
                  [buf("qkvz"), buf(p + "conv1d.weight"), state("gdn"), idx, buf("gdn.has_initial"),
                   buf("cu_seqlens_q"), buf("conv.batch_ptr"), buf("conv.token_chunk_offset"),
                   buf("conv_out"), i32(n_pages), i64(QKVZ_DIM), i64(CONV_DIM)] + Z2),
                # a/b are strided views into ba (vLLM copies them contiguous;
                # the kernel takes row strides, so no copy here)
                d(l + "post_conv", "post_conv",
                  [buf("conv_out"), a_, b_, buf(p + "A_log"), buf(p + "dt_bias"),
                   buf("gdn_q"), buf("gdn_k"), buf("gdn_v"), buf("g"), buf("beta"),
                   i32(CONV_DIM), i32(BA_DIM), i32(BA_DIM), i32(GDN_Q), i32(GDN_Q), i32(GDN_V), T] + Z2),
                d(l + "cumsum", "cumsum",
                  [buf("g"), buf("g_cum"), buf("cu_seqlens_q"), buf("fla.chunk_indices"), T] + Z2),
                d(l + "kkt", "kkt",
                  [buf("gdn_k"), buf("beta"), buf("g_cum"), buf("A"), buf("cu_seqlens_q"),
                   buf("fla.chunk_indices"), T] + Z2),
                d(l + "solve_tril", "solve_tril",
                  [buf("A"), buf("Ai"), buf("cu_seqlens_q"), buf("fla.chunk_indices"), T] + Z2),
                d(l + "recompute_wu", "recompute",
                  [buf("gdn_k"), buf("gdn_v"), buf("beta"), buf("w"), buf("u"), buf("Ai"),
                   buf("g_cum"), buf("cu_seqlens_q"), buf("fla.chunk_indices"), T] + Z2),
            ] + gather + [
                d(l + "chunk_h", "chunk_h",
                  [buf("gdn_k"), buf("u"), buf("w"), buf("v_new"), buf("g_cum"), buf("h"), h0, ht,
                   buf("cu_seqlens_q"), buf("fla.chunk_offsets"), T] + Z2),
            ] + scatter + [
                d(l + "chunk_o", "chunk_o",
                  [buf("gdn_q"), buf("gdn_k"), buf("v_new"), buf("h"), buf("g_cum"),
                   buf("core_attn_out"), buf("cu_seqlens_q"), buf("fla.chunk_indices"),
                   f32(gdn_scale), T] + Z2),
                d(l + "z_copy", "copy_rows",
                  [buf("z_c"), buf("qkvz", Z_OFF * BF16), i32(GDN_V), i32(QKVZ_DIM), i32(GDN_V)]),
                d(l + "gated_norm", "gated_norm",
                  [buf("core_attn_out"), buf("core_attn_out"), buf(p + "norm.weight"), buf("z_c"),
                   i32(GDN_D), i32(GDN_D), i32(GDN_D), expr(mul(T, GDN_V_HEADS)), f32(eps)] + Z2),
            ]
        elif nacc is None:
            ds += [
                d(l + "conv_update", "conv_update",
                  [buf("qkvz"), buf(p + "conv1d.weight"), state("gdn"), idx, buf("qkvz"),
                   S, i32(n_pages), i64(1), i64(1)] + Z2),
                d(l + "recurrent", "recurrent",
                  [buf("qkvz"), a_, b_, buf(p + "A_log"), buf(p + "dt_bias"), buf("core_attn_out"),
                   state("gdn", ssm_off), state("gdn", ssm_off), idx,
                   f32(gdn_scale)] + Z2),
            ]
            # the gated norm views x/z as [tokens·heads, 128] rows of one
            # stride: z must be contiguous per token — a copy out of the
            # fused qkvz row, except at bs=1 where the row's 48 heads are
            # the whole tensor
            if batch:
                ds.append(d(l + "z_copy", "copy_rows",
                            [buf("z_c"), buf("qkvz", Z_OFF * BF16), i32(GDN_V), i32(QKVZ_DIM), i32(GDN_V)]))
            ds.append(d(l + "gated_norm", "gated_norm_decode",
                        [buf("core_attn_out"), buf("core_attn_out"), buf(p + "norm.weight"),
                         buf("z_c") if batch else buf("qkvz", Z_OFF * BF16),
                         i32(GDN_D), i32(GDN_D), i32(GDN_D), expr(mul(S, GDN_V_HEADS)), f32(eps)] + Z2))
        else:
            # vLLM's spec-decode GDN path over the batch: conv update
            # resuming from history offset num_accepted-1 (seqlen 8 baked:
            # it reads/writes 8 qkvz rows per sequence), post_conv split
            # into this layer's saved k/v (advance re-reads them), a/b
            # copied contiguous into their saved rows, the recurrent kernel
            # over the rows loading the state from entry num_accepted-1 of
            # the sequence's cell and storing wherever an entry is non-null.
            ks, vs_, as_, bs_ = saved(g)
            ds += [
                d(l + "conv_update", "conv_update_spec",
                  [buf("qkvz"), buf(p + "conv1d.weight"), state("gdn"), idx, nacc, buf("cu_seqlens_q"),
                   buf("qkvz"), S, i32(n_pages), i64(QKVZ_DIM), i64(QKVZ_DIM)] + Z2),
                d(l + "post_conv", "post_conv",
                  [buf("qkvz"), a_, b_, buf(p + "A_log"), buf(p + "dt_bias"),
                   buf("gdn_q"), ks, vs_, buf("g"), buf("beta"),
                   i32(QKVZ_DIM), i32(BA_DIM), i32(BA_DIM), i32(GDN_Q), i32(GDN_Q), i32(GDN_V), T] + Z2),
                d(l + "a_copy", "copy_rows", [as_, a_, i32(GDN_V_HEADS), i32(BA_DIM), i32(GDN_V_HEADS)]),
                d(l + "b_copy", "copy_rows", [bs_, b_, i32(GDN_V_HEADS), i32(BA_DIM), i32(GDN_V_HEADS)]),
                d(l + "recurrent", "recurrent_spec",
                  [buf(p + "A_log"), as_, bs_, buf(p + "dt_bias"), f32(1.0), f32(20.0),
                   buf("gdn_q"), ks, vs_, buf("core_attn_out"),
                   state("gdn", ssm_off), state("gdn", ssm_off), buf("cu_seqlens_q"),
                   idx, nacc, f32(gdn_scale), S, T] + Z2),
                d(l + "z_copy", "copy_rows",
                  [buf("z_c"), buf("qkvz", Z_OFF * BF16), i32(GDN_V), i32(QKVZ_DIM), i32(GDN_V)]),
                d(l + "gated_norm", "gated_norm",
                  [buf("core_attn_out"), buf("core_attn_out"), buf(p + "norm.weight"), buf("z_c"),
                   i32(GDN_D), i32(GDN_D), i32(GDN_D), expr(mul(T, GDN_V_HEADS)), f32(eps)] + Z2),
            ]
        ds.append(gemm(l + "out_proj", "core_attn_out", p + "out_proj.weight", "y", T, HIDDEN, GDN_V))
        return ds

    def advance_layer(i, nacc_buf="num_accepted_tokens", table="gdn.line_index"):
        """Commit GDN layer i's accepted rows after a verify (spec): a/b
        with row 0 of every sequence (the anchor, already in the state)
        masked to -inf, the recurrent kernel over verify's saved rows
        loading the after-anchor state from entry a = num_accepted-1 of the
        cell and storing after row a, the conv history shifted down by a.
        `nacc_buf` / `table`: host-staged inputs, or the round's twins."""
        p = f"model.layers.{i}.linear_attn."
        l = f"l{i}."
        g = GDN_LAYERS.index(i)
        idx = buf(table, 4 * line_w * MAX_SEQS * g)
        nacc = buf(nacc_buf)
        ks, vs_, as_, bs_ = saved(g)
        return [
            d(l + "mask_a", "mask_row0", [buf("a_c"), as_, T, i32(SPEC_BLOCK), i32(GDN_V_HEADS)]),
            d(l + "mask_b", "mask_row0", [buf("b_c"), bs_, T, i32(SPEC_BLOCK), i32(GDN_V_HEADS)]),
            # q is not part of the state update (it only shapes the output,
            # which advance discards), so k stands in for it
            d(l + "recurrent", "recurrent_spec",
              [buf(p + "A_log"), buf("a_c"), buf("b_c"), buf(p + "dt_bias"), f32(1.0), f32(20.0),
               ks, ks, vs_, buf("core_attn_out"),
               state("gdn", ssm_off), state("gdn", ssm_off), buf("cu_seqlens_q"),
               idx, nacc, f32(gdn_scale), S, T] + Z2),
            d(l + "conv_shift", "conv_shift",
              [idx, i32(SPEC_BLOCK), state("gdn"), nacc, i64(SPEC_PAGE_BYTES), i64(CONV_DIM * BF16)]),
        ]

    def attn_layer(i, decode, batch=False, verify=False):
        p = f"model.layers.{i}.self_attn."
        l = f"l{i}."
        koff = ATTN_LAYERS.index(i) * LAYER_KV_BYTES
        ks, vs = buf("kv_scales"), buf("kv_scales", 4)
        kv_k, kv_v = state("kv", koff), state("kv", koff + V_BYTE_OFF)
        return [
            gemm(l + "qkv_proj", "x", p + "qkv_proj.weight", "qkv", T, QKV_DIM, HIDDEN),
            d(l + "q_norm", "gemma_norm_qhead",
              [buf("q_n"), buf("qkv"), buf(p + "q_norm.weight_p1"), i32(HEAD_DIM),
               expr(mul(T, HEADS)), i32(HEADS), i32(QKV_DIM), i32(2 * HEAD_DIM), i32(Q_DIM),
               i32(HEAD_DIM), f32(eps)]),
            d(l + "k_norm", "gemma_norm_khead",
              [buf("k_n"), buf("qkv", HEADS * 2 * HEAD_DIM * BF16), buf(p + "k_norm.weight_p1"),
               i32(HEAD_DIM), expr(mul(T, KV_HEADS)), i32(KV_HEADS), i32(QKV_DIM), i32(HEAD_DIM),
               i32(KV_DIM), i32(HEAD_DIM), f32(eps)]),
            d(l + "rope", "mrope", [buf("q_n"), buf("k_n"), buf("cos_g"), buf("sin_g"), i32(0)] + Z2),
            d(l + "kv_write", "reshape_and_cache",
              [buf("k_n"), buf("qkv", (HEADS * 2 * HEAD_DIM + KV_DIM) * BF16), kv_k, kv_v,
               buf("slot_mapping"), ks, vs,
               i64(KV_DIM), i64(QKV_DIM), i64(BLOCK_STRIDE), i64(2 * HEAD_DIM), i64(0), i64(0),
               i64(KV_HEADS * 2 * HEAD_DIM), i64(0), i64(0)]),
            d(l + "attn", "attn_verify" if verify else "attn_batch" if batch else "attn" if decode else "attn_prefill",
              [buf("attn_out"), buf("q_n"), kv_k, kv_v, buf("block_table"), buf("seq_lens"),
               f32(attn_scale), ks, vs, f32(1.0), f32(0.0),
               i64(BLOCK_TABLE_LEN), i64(Q_DIM), i64(HEAD_DIM), i64(Q_DIM), i64(HEAD_DIM), i64(0),
               buf("seq_lens"),
               i64(BLOCK_STRIDE), i64(KV_HEADS * 2 * HEAD_DIM), i64(2 * HEAD_DIM),
               i64(BLOCK_STRIDE), i64(KV_HEADS * 2 * HEAD_DIM), i64(2 * HEAD_DIM),
               buf("cu_seqlens_q"), S if (batch or verify) else i32(1)] + Z2),
            d(l + "gate", "sigmoid_mul",
              [buf("gated"), buf("attn_out"), buf("qkv", GATE_OFF * BF16), i32(HEADS), i32(HEAD_DIM),
               i32(QKV_DIM), i32(2 * HEAD_DIM)]),
            gemm(l + "o_proj", "gated", p + "o_proj.weight", "y", T, HIDDEN, Q_DIM),
        ]

    def forward(decode, taps=False, nacc=None, tail=None, batch=False, ids="token_ids"):
        """Target forward.  decode: recurrent GDN + split-KV attention (else
        chunked FLA + 2D attention over the `tokens` rows); batch: recurrent
        GDN + 2D attention, one row per sequence (tokens = seqs).  taps: the
        five fc GEMMs into `fc_out` at the DFlash taps (after layers
        5/19/33/47/61: residual = hidden + residual there).  tail: "prefill"
        (last row -> next_token), "decode" (row 0 -> next_token), "batch"
        (every row -> next_token), "verify" (all rows -> verify_tokens).
        ids: the token buffer the embedding reads."""
        tail = tail or ("batch" if batch else "decode" if decode else "prefill")
        ds = [
            d("embed", "embedding",
              [buf(ids), buf("model.embed_tokens.weight"), buf("residual"), T, i32(HIDDEN)]),
            # rope tables gathered by position; mrope gets num_tokens=0 so its
            # three (t/h/w) planes alias this one table — text-only positions
            d("rope_cos", "embedding", [buf("positions"), buf("rope.cos"), buf("cos_g"), T, i32(ROT_HALF)]),
            d("rope_sin", "embedding", [buf("positions"), buf("rope.sin"), buf("sin_g"), T, i32(ROT_HALF)]),
            d("l0.input_norm", "gemma_norm",
              [buf("x"), buf("residual"), buf("model.layers.0.input_layernorm.weight_p1"),
               i32(HIDDEN), T, i32(1), i32(HIDDEN), i32(0), i32(HIDDEN), i32(0), f32(eps)]),
        ]
        for i in range(LAYERS):
            p = f"model.layers.{i}."
            l = f"l{i}."
            ds += attn_layer(i, decode, batch, tail == "verify") if i in ATTN_LAYERS else gdn_layer(i, decode, nacc, batch)
            ds += [
                fused(l + "post_attn_norm", "y", p + "post_attention_layernorm.weight_p1"),
                gemm(l + "gate_up", "x", p + "mlp.gate_up_proj.weight", "gate_up", T, 2 * FFN, HIDDEN),
                d(l + "silu_mul", "silu_mul",
                  [buf("act"), buf("gate_up"), i32(FFN), i32(0), f32(1.0), i32(0)]),
                gemm(l + "down_proj", "act", p + "mlp.down_proj.weight", "y", T, HIDDEN, FFN),
            ]
            last = i + 1 == LAYERS
            ds.append(fused(l + ("final_norm" if last else "next_input_norm"), "y",
                            "model.norm.weight_p1" if last else f"model.layers.{i + 1}.input_layernorm.weight_p1"))
            if taps and i in TAPS:
                # residual now holds hidden + residual = the input of layer
                # i+1, vLLM's aux hidden state; fc's column block j, β=1
                # accumulating after the first (no concat)
                j = TAPS[i]
                ds.append(d(l + "fc_tap", "gemm" if j == 0 else "gemm_acc",
                            [buf("residual"), buf(f"draft.fc.{j}.weight"), buf("fc_out"),
                             T, i32(HIDDEN), i32(HIDDEN)]))
        # The final norm runs over all rows like vLLM (ATen's reduction width
        # depends on the row count); lm_head only needs the last one.
        if tail == "decode":
            ds.append(gemm("lm_head", "x", "lm_head.weight", "logits", i32(1), VOCAB, HIDDEN))
            ds.append(d("sample", "argmax_row", [buf("logits"), buf("next_token"), i32(VOCAB)]))
        elif tail == "batch":
            ds.append(gemm("lm_head", "x", "lm_head.weight", "logits", S, VOCAB, HIDDEN))
            ds.append(d("sample", "argmax", [buf("logits"), buf("next_token"), i32(VOCAB)]))
        elif tail == "verify":
            ds.append(gemm("lm_head", "x", "lm_head.weight", "logits_blk", T, VOCAB, HIDDEN))
            ds.append(d("sample", "argmax", [buf("logits_blk"), buf("verify_tokens"), i32(VOCAB)]))
        else:
            ds.append(d("last_row", "last_row", [buf("final_x"), buf("x"), i32(HIDDEN), i32(HIDDEN), T]))
            ds.append(gemm("lm_head", "final_x", "lm_head.weight", "logits", i32(1), VOCAB, HIDDEN))
            ds.append(d("sample", "argmax_row", [buf("logits"), buf("next_token"), i32(VOCAB)]))
        return ds

    def d_fused(label, x, w):
        return d(label, "d_fused_norm",
                 [buf(x), i64(HIDDEN), buf("residual"), buf(w), f32(eps), T, i32(HIDDEN), i64(HIDDEN)])

    def d_conv(label, x, c, side):
        return d(label, "dflash_conv",
                 [buf(x), buf("d_coef"), buf(c + ".base_kernel"), T, i32(HIDDEN), i32(D_GROUPS),
                  i32(D_CONV_PROJ), i32(side)])

    def draft(ids="draft_ids"):
        """One non-causal pass over the 8-row block [anchor, mask x7] at
        positions pos..pos+7 (env tokens=8), then top-16 + selector walk ->
        7 draft tokens.  Draft hidden size = target's, so the target's
        activation buffers are reused; the two grouped convs of every layer
        wrap attention and MLP (prepare before, finish after, coefficients
        from one GEMM of the pre-attention / pre-MLP normed state)."""
        mp = "draft."
        ds = [
            d("embed", "embedding",
              [buf(ids), buf("model.embed_tokens.weight"), buf("residual"), T, i32(HIDDEN)]),
            d("l0.input_norm", "d_rms_norm",
              [buf("x"), buf("residual"), i64(HIDDEN), i64(0), i64(0), i64(0), i64(0),
               buf(mp + "layers.0.input_layernorm.weight"), i64(0), f32(eps), T, i32(HIDDEN)]),
        ]
        for j in range(D_LAYERS):
            p = f"{mp}layers.{j}."
            l = f"l{j}."
            koff = j * D_LAYER_KV_BYTES
            ks, vs = buf("draft.kv_scales", j * 8), buf("draft.kv_scales", j * 8 + 4)
            last = j + 1 == D_LAYERS
            ds += [
                gemm(l + "attn_conv_proj", "x", p + "attention_conv.kernel_projection.weight", "d_coef",
                     T, D_CONV_PROJ, HIDDEN),
                d_conv(l + "attn_conv_pre", "x", p + "attention_conv", 0),
                gemm(l + "qkv_proj", "x", p + "self_attn.qkv_proj.weight", "d_qkv", T, D_QKV, HIDDEN),
                d(l + "q_norm", "d_head_norm",
                  [buf("d_q"), buf("d_qkv"), i64(D_HEAD_DIM), i64(D_QKV), i64(0), i64(D_HEADS), i64(0),
                   buf(p + "self_attn.q_norm.weight"), i64(0), f32(eps), expr(mul(T, D_HEADS)), i32(D_HEAD_DIM)]),
                d(l + "k_norm", "d_kv_norm",
                  [buf("k_n"), buf("d_qkv", D_Q * BF16), i64(D_HEAD_DIM), i64(D_QKV), i64(0), i64(D_KV_HEADS),
                   i64(0), buf(p + "self_attn.k_norm.weight"), i64(0), f32(eps), expr(mul(T, D_KV_HEADS)),
                   i32(D_HEAD_DIM)]),
                d(l + "rope", "d_rope",
                  [buf("positions"), buf("d_q"), buf("k_n"), buf("draft.rope.cos_sin_cache"), i32(D_HEAD_DIM),
                   i64(D_Q), i64(D_KV), i64(D_HEAD_DIM), i32(D_HEADS), i32(D_KV_HEADS), i32(D_HEAD_DIM),
                   i64(0), u8(0)]),
                d(l + "kv_write", "d_cache",
                  [buf("k_n"), buf("d_qkv", (D_Q + D_KV) * BF16),
                   state("draft_kv", koff), state("draft_kv", koff + D_V_BYTE_OFF), buf("slot_mapping"),
                   i64(D_BLOCK_STRIDE), i64(D_KV_HEADS * 2 * D_HEAD_DIM), i64(2 * D_HEAD_DIM),
                   i64(D_KV), i64(D_QKV), i32(D_KV_HEADS), i32(D_HEAD_DIM), i32(D_BLOCK), ks, vs, i32(0)]),
                d(l + "attn", "attn_draft",
                  [buf("d_attn"), buf("d_q"), state("draft_kv", koff), state("draft_kv", koff + D_V_BYTE_OFF),
                   buf("draft_block_table"), buf("seq_lens"), f32(D_ATTN_SCALE), ks, vs, f32(1.0), f32(0.0),
                   i64(D_TABLE_LEN), i64(D_Q), i64(D_HEAD_DIM), i64(D_Q), i64(D_HEAD_DIM), i64(0),
                   buf("seq_lens"),
                   i64(D_BLOCK_STRIDE), i64(D_KV_HEADS * 2 * D_HEAD_DIM), i64(2 * D_HEAD_DIM),
                   i64(D_BLOCK_STRIDE), i64(D_KV_HEADS * 2 * D_HEAD_DIM), i64(2 * D_HEAD_DIM),
                   buf("cu_seqlens_q"), S] + Z2),
                gemm(l + "o_proj", "d_attn", p + "self_attn.o_proj.weight", "y", T, HIDDEN, D_Q),
                d_conv(l + "attn_conv_post", "y", p + "attention_conv", 1),
                d_fused(l + "post_attn_norm", "y", p + "post_attention_layernorm.weight"),
                gemm(l + "mlp_conv_proj", "y", p + "mlp_conv.kernel_projection.weight", "d_coef",
                     T, D_CONV_PROJ, HIDDEN),
                d_conv(l + "mlp_conv_pre", "y", p + "mlp_conv", 0),
                gemm(l + "gate_up", "y", p + "mlp.gate_up_proj.weight", "gate_up", T, 2 * FFN, HIDDEN),
                d(l + "silu_mul", "silu_mul", [buf("act"), buf("gate_up"), i32(FFN), i32(0), f32(1.0), i32(0)]),
                gemm(l + "down_proj", "act", p + "mlp.down_proj.weight", "x", T, HIDDEN, FFN),
                d_conv(l + "mlp_conv_post", "x", p + "mlp_conv", 1),
                d_fused(l + ("final_norm" if last else "next_input_norm"), "x",
                        mp + "norm.weight" if last else f"{mp}layers.{j + 1}.input_layernorm.weight"),
            ]
        # every row -> shared lm_head -> top-16 candidates (the anchor rows
        # ride along, 1/8 of the work, and are skipped by the walk); the
        # rank-256 selector scores predecessor/successor codebook rows of
        # the candidates (gathered with the embedding kernel) and walks each
        # sequence's 7 mask rows greedily
        ds += [
            d("lm_head", "gemm", [buf("x"), buf("lm_head.weight"), buf("logits_blk"), T, i32(VOCAB), i32(HIDDEN)]),
            d("sel.topk", "topk16", [buf("logits_blk"), buf("cand_ids"), buf("cand_vals"), i32(VOCAB), i32(VOCAB)]),
            d("sel.hidden_proj", "gemm", [buf("x"), buf("draft.selector.hidden_projection.weight"),
                                          buf("hidden_r"), T, i32(SEL_RANK), i32(HIDDEN)]),
            d("sel.succ", "gather_cands", [buf("cand_ids"), buf("draft.selector.successor"), buf("succ_g"),
                                           expr(mul(T, SEL_K)), i32(SEL_RANK)]),
            d("sel.pred", "gather_cands", [buf("cand_ids"), buf("draft.selector.predecessor"), buf("pred_g"),
                                           expr(mul(T, SEL_K)), i32(SEL_RANK)]),
            d("sel.pred_anchor", "gather_row", [buf("anchor_token"), buf("draft.selector.predecessor"),
                                                buf("pred_anchor"), S, i32(SEL_RANK)]),
            d("sel.walk", "dflash_select", [buf("cand_ids"), buf("cand_vals"), buf("hidden_r"), buf("succ_g"),
                                            buf("pred_g"), buf("pred_anchor"), buf("draft_tokens"),
                                            i32(DRAFT_TOKENS), i32(SEL_RANK), i32(SEL_RANK), i32(SPEC_BLOCK)]),
        ]
        return ds

    def draft_precompute():
        """Target taps -> draft context KV (DSpark's shape, docs/spec-decode.md):
        hidden_norm(fc_out) -> fused KV GEMM [n, 10240] -> per layer k_norm,
        K-only rope (key = the same buffer with 0 kv heads), cache write.
        Runs at env tokens = valid rows of fc_out; positions / slot_mapping
        are still those of the forward that produced them."""
        ds = [
            d("hidden_norm", "d_rms_norm",
              [buf("x"), buf("fc_out"), i64(HIDDEN), i64(0), i64(0), i64(0), i64(0),
               buf("draft.hidden_norm.weight"), i64(0), f32(eps), T, i32(HIDDEN)]),
            gemm("fused_kv", "x", "draft.fused_kv.weight", "kv_flat", T, D_KV_FLAT, HIDDEN),
        ]
        for j in range(D_LAYERS):
            koff = j * D_LAYER_KV_BYTES
            ks, vs = buf("draft.kv_scales", j * 8), buf("draft.kv_scales", j * 8 + 4)
            ds += [
                d(f"ctx{j}.k_norm", "d_kv_norm",
                  [buf("k_n"), buf("kv_flat", j * 2 * D_KV * BF16), i64(D_HEAD_DIM), i64(D_KV_FLAT), i64(0),
                   i64(D_KV_HEADS), i64(0), buf(f"draft.layers.{j}.self_attn.k_norm.weight"), i64(0), f32(eps),
                   expr(mul(T, D_KV_HEADS)), i32(D_HEAD_DIM)]),
                d(f"ctx{j}.rope", "d_rope",
                  [buf("positions"), buf("k_n"), buf("k_n"), buf("draft.rope.cos_sin_cache"), i32(D_HEAD_DIM),
                   i64(D_KV), i64(0), i64(D_HEAD_DIM), i32(D_KV_HEADS), i32(0), i32(D_HEAD_DIM), i64(0), u8(0)]),
                d(f"ctx{j}.kv_write", "d_cache",
                  [buf("k_n"), buf("kv_flat", (j * 2 * D_KV + D_KV) * BF16),
                   state("draft_kv", koff), state("draft_kv", koff + D_V_BYTE_OFF), buf("slot_mapping"),
                   i64(D_BLOCK_STRIDE), i64(D_KV_HEADS * 2 * D_HEAD_DIM), i64(2 * D_HEAD_DIM),
                   i64(D_KV), i64(D_KV_FLAT), i32(D_KV_HEADS), i32(D_HEAD_DIM), i32(D_BLOCK), ks, vs, i32(0)]),
            ]
        return ds

    states = {"kv": {"bytes_per_token": KV_BYTES_PER_TOKEN},
              "gdn": {"bytes_per_seq": seq_bytes}}
    head = {"schema_version": SCHEMA_VERSION, "model": "qwen3.8-27b"}
    if spec:
        # A step's GDN resumes from the committed state: the count it reads
        # is the constant 1 the step writes first.
        nacc = buf("nacc_one")
        ones = [d("ones", "ones_i32", [nacc, S])]

        def pre(prefix, calls):
            return [dict(c, label=f"{prefix}.{c['label']}") for c in calls]
        programs = {
            "prefill": program(forward(False, taps=True), groups=1, rows="tokens"),
            # the one-row step on the speculative state layout (the spec
            # kernels at tokens=1): the in-manifest oracle for the round
            "decode": program(ones + forward(True, nacc=nacc), groups=1, rows=1, graph=True),
            # round = one speculative round per sequence as one program (one
            # CUDA graph, one host sync): draft's rows spliced from the
            # anchor the caller staged, draft, verify's ids spliced from
            # draft's output, verify (2D attention, spec GDN resuming from
            # the committed state), every row's tap into the draft KV, the
            # count of rows taken and the line entry advance reads, advance
            # (the accepted rows committed into the GDN state, see
            # SPEC_SEQ_BYTES). Rows per sequence are the same in draft and
            # verify, so positions / slot_mapping / seq_lens / cu_seqlens_q
            # are staged once; the caller reads verify_tokens and nacc_adv.
            "round": program(
                ones
                + [d("splice_draft", "splice_draft",
                     [buf("anchor_token"), buf("draft_ids"), i32(SPEC_BLOCK), i64(MASK_TOKEN)])]
                + pre("draft", draft())
                + [d("splice_verify", "splice_verify",
                     [buf("anchor_token"), buf("draft_tokens"), buf("verify_ids"), i32(SPEC_BLOCK),
                      i32(DRAFT_TOKENS)])]
                + pre("verify", forward(False, taps=True, nacc=nacc, tail="verify", ids="verify_ids"))
                + pre("precompute", draft_precompute())
                + [d("count", "spec_count",
                     [buf("draft_tokens"), buf("verify_tokens"), buf("nacc_adv"), i32(SPEC_BLOCK),
                      i32(DRAFT_TOKENS)]),
                   d("lines", "spec_lines",
                     [buf("gdn.line_index"), buf("nacc_adv"), buf("line_adv"), i32(SPEC_BLOCK),
                      i32(len(GDN_LAYERS)), i32(MAX_SEQS)])]
                + pre("advance", [c for i in GDN_LAYERS for c in advance_layer(i, "nacc_adv", "line_adv")]),
                groups=MAX_SEQS, rows=SPEC_BLOCK, graph=True),
        }
        states["draft_kv"] = {"bytes_per_token": D_KV_BYTES_PER_TOKEN}
        head = {"schema_version": SCHEMA_VERSION, "model": "qwen3.8-27b-dflash2"}
        DOMAINS.update({
            "draft_block_table": {"index_into": "draft_kv", "stride": D_BLOCK},
            "anchor_token": TOKEN_DOMAIN, "verify_tokens": TOKEN_DOMAIN, "draft_tokens": TOKEN_DOMAIN,
            "draft_ids": TOKEN_DOMAIN, "verify_ids": TOKEN_DOMAIN,
            "nacc_adv": {"min": 1, "max": SPEC_BLOCK}, "nacc_one": {"min": 1, "max": 1},
            "line_adv": DOMAINS["gdn.line_index"],
            "cand_ids": {"index_into": "draft.selector.successor"},
            "draft.kv_scales": {"min": 0.0},
        })
    else:
        programs = {
            "prefill": program(forward(False), groups=1, rows="tokens"),
            "decode": program(forward(True), groups=1, rows=1, graph=True),
            "decode_batch": program(forward(True, batch=True), groups=MAX_SEQS, rows=1, graph=True),
        }
    for name, dom in DOMAINS.items():
        if name in buffers:
            buffers[name]["domain"] = dom
    # the runtime rejects dead kernels / buffers (a manifest describes only
    # what runs): the spec layout leaves Stage 1's packed decode chain unused
    used_k = {c["op"] for p in programs.values() for c in p["calls"]}
    used_b = {a["buf"] for p in programs.values() for c in p["calls"] for a in c["args"] if "buf" in a}
    kernels = {k: v for k, v in kernels.items() if k in used_k}
    buffers = {k: v for k, v in buffers.items() if k in used_b}
    # normalize: hoist inline cubin/sha256 into `modules`, fold the ABI
    # constants every call repeats into the impls, default identity wiring
    return normalize(head | {
        # bs=1; prefill per chunk (tokens <= CHUNK_MAX) over *all* prompt
        # tokens (it emits next_token), decode at tokens=1
        "vars": {"tokens": {"max": CHUNK_MAX}, "seqs": {"max": MAX_SEQS}},
        "states": states,
        "buffers": buffers,
        "ops": kernels,
        "programs": programs,
    })


def main():
    repo = pathlib.Path(__file__).resolve().parent.parent
    args = sys.argv[1:]
    spec_dump = None
    if "--spec" in args:
        i = args.index("--spec")
        spec_dump = pathlib.Path(args[i + 1])
        del args[i:i + 2]
    dump = pathlib.Path(args[0] if args else repo / "dumped-kernels" / "pid1898802")
    out = pathlib.Path(args[1] if len(args) > 1 else
                       repo / "examples" / ("qwen3.8-27b-dflash2.json" if spec_dump else "qwen3.8-27b.json"))

    recs = load(dump / "launches.jsonl")
    fwds = forwards(recs)
    ts = [tokens_of(f) for f in fwds]
    # real prefill forwards (multi-token, after the profiling passes) + the
    # decode forward that follows the longest one
    prefills = [(t, i) for i, t in enumerate(ts) if t and 1 < t <= CHUNK_MAX and i > ts.index(1)]
    assert len(prefills) >= 3, f"need several prefill forwards to fit grids, got {prefills}"
    t_ref, i_ref = max(prefills)
    pre = group(fwds[i_ref])
    dec = group(fwds[i_ref + 1])
    assert tokens_of(fwds[i_ref + 1]) == 1
    for t, i in prefills:
        check_prefill(group(fwds[i]), t)
    check_decode(dec)
    print(f"capture: prefill forwards T={sorted(t for t, _ in prefills)} verified, "
          f"decode after T={t_ref} verified", file=sys.stderr)

    eps = pf(pre["layer_norm"][0], 9)
    attn_scale = pf(pre["unified"][0], 6)
    gdn_scale = pf(pre["chunk_o"][0], 8)
    assert pf(dec["recurrent"][0], 9) == gdn_scale and pf(dec["unified"][0], 6) == attn_scale
    assert pf(dec["layer_norm"][0], 9) == eps
    assert abs(eps - 1e-6) < 1e-12 and attn_scale == 0.0625
    silu_sym = pre["silu"][0]["symbol"]

    # A pinned step's `cubin` is a label (the runtime resolves the sha256),
    # so name Triton instances by what they are, not by dump module number.
    pinner = Pinner(dump)
    pins = {}
    for tag in TRITON:
        for src, is_pre in ((pre, True), (dec, False)):
            if src[tag]:
                _, sha = pinner.pin(TRITON[tag], src[tag][0]["attributes"]["num_regs"])
                pins[(tag, is_pre)] = (f"{tag}.cubin", sha)
    # the one mined CUDA kernel (vLLM's activation_kernels.cu) pins its module too
    pins[("silu", True)] = ("vllm_activation.cubin",
                            pinner.pin(silu_sym, pre["silu"][0]["attributes"]["num_regs"])[1])
    for (tag, is_pre), (mod, sha) in sorted(pins.items()):
        print(f"  pin {TRITON.get(tag, silu_sym)[:52]:<52} {'prefill' if is_pre else 'decode '} -> {mod} {sha[:12]}",
              file=sys.stderr)

    spec = None
    if spec_dump:
        # Two dumps feed one manifest (Stage 1's target instances + the
        # speculative path's); extract_kernels.sh resolves every pin by
        # sha256 across dumps, labels stay by tag.
        src = spec_launches(load(spec_dump / "launches.jsonl"))
        sp = Pinner(spec_dump)
        spins = {}
        for tag, r in src.items():
            spins[tag] = (f"{tag}.cubin", sp.pin(SPEC_SYMS[tag], r["attributes"]["num_regs"], nparams=len(r["params"]))[1])
        # conv_fwd of the spec state layout: Stage 1's instance with the page stride rebaked
        pins[("conv_fwd", True)] = ("conv_fwd.cubin",
                                    sp.pin_nearest(TRITON["conv_fwd"], pre["conv_fwd"][0]["attributes"]["num_regs"],
                                                   pinner.pin(TRITON["conv_fwd"], pre["conv_fwd"][0]["attributes"]["num_regs"])[1])[1])
        assert pf(src["recurrent_spec"], 15) == gdn_scale and pf(src["recurrent_spec"], 4) == 1.0
        assert pf(src["recurrent_spec"], 5) == 20.0, "softplus threshold"
        assert pv(src["conv_update_spec"], 9) == pv(src["conv_update_spec"], 10), "conv update not in place"
        assert pf(src["d_rms_norm"], 9) == eps and pf(src["d_fused_norm"], 4) == eps
        dspark = resolve_constants(json.loads((repo / "examples" / "qwen3-4b-dspark.json").read_text()))
        ad = dict(dspark["ops"]["attn_draft"]["impl"]["launches"][0])
        ad_mod = dspark["modules"][ad["module"]]
        ad["cubin"], ad["sha256"] = ad_mod["source"], ad_mod["sha256"]
        assert ad["entry"] == TRITON["unified"] and ad["cubin"] == "unified_noncausal.cubin"
        for tag, (mod, sha) in sorted(spins.items()):
            print(f"  spec pin {SPEC_SYMS[tag][:52]:<52} -> {mod} {sha[:12]}", file=sys.stderr)
        print(f"  spec pin conv_fwd (page stride rebaked) -> {pins[('conv_fwd', True)][0]}", file=sys.stderr)
        spec = {"src": src, "pins": spins, "attn_draft": ad}

    m = qwen38(build(pre, dec, pins, eps, attn_scale, gdn_scale, silu_sym, spec))
    out.write_text(json.dumps(m, indent=1) + "\n")
    n_calls = {k: len(v["calls"]) for k, v in m["programs"].items()}
    print(f"wrote {out}: {len(m['buffers'])} buffers, {len(m['ops'])} ops, {len(m['modules'])} modules, "
          f"calls {n_calls}", file=sys.stderr)


if __name__ == "__main__":
    main()
