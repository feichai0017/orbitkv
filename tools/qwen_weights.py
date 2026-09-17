"""Weights straight from the checkpoint: a Qwen manifest's weight buffers
`bind` the Hugging Face tensors they are made of, and whatever is not in
the checkpoint (Gemma norms' `weight + 1`, f32 copies, rope tables, the
constant index tables the captured Triton kernels take as pointers) is a
`carry` buffer the `load` program computes once after load with
tools/kernels-src/weight_prep.cu. No exported artifact, no torch: `kern run
--weights <HF snapshot dir>`.

Two families, both with their draft:

- `qwen38`: Qwen3.8-27B (`model.language_model.*` in the checkpoint) and
  its DFlash2 draft (`layers.*`, `fc.weight`, `candidate_selector.*`).
- `qwen3`: Qwen3-4B (`model.*`, tied lm_head) and its DSpark draft.

A pass is a pure function of the resolved manifest; the generators call it
last, and the CLI reapplies it to a checked-in example:

    python tools/qwen_weights.py qwen38 --input examples/qwen3.8-27b.json
"""
import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from handwritten import hw  # noqa: E402
from kern_manifest import normalize, program, resolve_constants  # noqa: E402
from qwen_constants import name_constants  # noqa: E402

BLOCK = 256


def seg(tensor, rows=None, cols=None):
    s = {"tensor": tensor}
    if rows is not None:
        s["rows"] = list(rows)
    if cols is not None:
        s["cols"] = list(cols)
    return s


def bind(*tensors):
    """Whole tensors laid end to end (a merged projection)."""
    return [seg(t) for t in tensors]


def buf(n, off=0):
    return {"buf": n, "offset": off} if off else {"buf": n}


class Once:
    """The `load` program under construction: carries plus the calls that
    fill them. Ops are shared; each op's grid covers its widest call."""

    def __init__(self, m):
        self.m = m
        self.calls = []
        self.widest = {}

    def call(self, op, args, n):
        self.widest[op] = max(self.widest.get(op, 0), n)
        self.calls.append({"label": f"load.{op}.{len(self.calls)}", "op": op, "args": args})

    def derive(self, name, raw=None, dtype="bf16"):
        """Turn weight buffer `name` into a carry; when the value comes from
        a checkpoint tensor, add `raw` bound to it (same shape, `dtype`)."""
        b = self.m["buffers"][name]
        b["kind"] = "carry"
        b.pop("bind", None)
        if raw is not None:
            rname, tensor = raw
            self.m["buffers"][rname] = {"dtype": dtype, "shape": list(b["shape"]), "kind": "weight", "bind": bind(tensor)}
        return b

    def elems(self, name):
        n = 1
        for d in self.m["buffers"][name]["shape"]:
            n *= d
        return n

    def p1(self, name, tensor):
        """`name` (f32, `<x>.weight_p1`) = float(<x>.weight) + 1."""
        raw = name[: -len("_p1")]
        self.derive(name, (raw, tensor))
        self.call("weight_p1", [buf(raw), buf(name), {"i32": self.elems(name)}], self.elems(name))

    def cast(self, name, tensor):
        raw = name + ".bf16"
        self.derive(name, (raw, tensor))
        self.call("cast_f32", [buf(raw), buf(name), {"i32": self.elems(name)}], self.elems(name))

    def fill(self, name, value):
        b = self.derive(name)
        n = self.elems(name)
        op = {"f32": "fill_f32", "i32": "fill_i32", "i64": "fill_i64", "u8": "fill_u8"}[b["dtype"]]
        self.call(op, [buf(name), {"i32": n}, {b["dtype"] if b["dtype"] != "u8" else "i32": value}], n)

    def iota(self, name, n, start=0, step=1, stride=1, off=0):
        self.call("iota_i32", [buf(name, off), {"i32": n}, {"i32": start}, {"i32": step}, {"i32": stride}], n)

    def rope(self, cos, sin, rows, half, stride, base):
        """vLLM's cos/sin cache: `cos`/`sin` are (buffer, byte offset)."""
        for name, _ in (cos, sin):
            if self.m["buffers"][name]["kind"] == "weight":
                self.derive(name)
        self.call("rope_table", [buf(*cos), buf(*sin), {"i32": rows}, {"i32": half}, {"i32": stride}, {"f32": base}], rows)

    def finish(self):
        pin = hw("weight_prep")
        blocks = {op: (n + BLOCK - 1) // BLOCK for op, n in self.widest.items()}
        signatures = {
            "weight_p1": ("kern_weight_p1_bf16_f32", ["in buffer<bf16>", "out buffer<f32>", "i32"]),
            "cast_f32": ("kern_cast_bf16_f32", ["in buffer<bf16>", "out buffer<f32>", "i32"]),
            "fill_f32": ("kern_fill_f32", ["out buffer<f32>", "i32", "f32"]),
            "fill_i32": ("kern_fill_i32", ["out buffer<i32>", "i32", "i32"]),
            "fill_i64": ("kern_fill_i64", ["out buffer<i64>", "i32", "i64"]),
            "fill_u8": ("kern_fill_u8", ["out buffer<u8>", "i32", "i32"]),
            "iota_i32": ("kern_iota_i32", ["out buffer<i32>", "i32", "i32", "i32", "i32"]),
            # one block per table row; a row is at most 64 wide
            "rope_table": ("kern_rope_table_bf16", ["out buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32", "f32"]),
        }
        for op, n in self.widest.items():
            entry, params = signatures[op]
            grid = [n, 1, 1] if op == "rope_table" else [blocks[op], 1, 1]
            block = [64, 1, 1] if op == "rope_table" else [BLOCK, 1, 1]
            assert op not in self.m["ops"], op
            self.m["ops"][op] = {"params": params, "impl": {"launches": [
                {"entry": entry, "params": params, "block": block, "grid": grid,
                 "args": [{"param": i} for i in range(len(params))], **pin}]}}
        assert "load" not in self.m["programs"]
        self.m["programs"]["load"] = program(self.calls, once=True)
        return self.m


def weights(m):
    return [n for n, b in m["buffers"].items() if b["kind"] == "weight"]


def draft_layers(m):
    """The draft's layer count, read off its per-layer buffers: the fused KV
    buffer is every layer's k_proj then v_proj, in layer order."""
    return sum(1 for n in m["buffers"] if re.fullmatch(r"draft\.layers\.\d+\.self_attn\.o_proj\.weight", n))


def qwen38(m):
    """Qwen3.8-27B: mrope theta 1e7 over 64 of 256 dims (32-wide cos / sin
    tables), Gemma norms, GDN layers with f32 A_log; the DFlash2 draft's
    5-tap combiner is column blocks of one `fc.weight`, its KV projections
    one fused buffer, its rope a full 128-wide cos | sin cache."""
    m = resolve_constants(m)
    once = Once(m)
    P = "model.language_model."
    for name in weights(m):
        b = m["buffers"][name]
        if name == "model.embed_tokens.weight":
            b["bind"] = bind(P + "embed_tokens.weight")
        elif name == "lm_head.weight":
            b["bind"] = bind("lm_head.weight")
        elif name == "model.norm.weight_p1":
            once.p1(name, P + "norm.weight")
        elif name == "rope.cos":
            rows, half = b["shape"]
            once.rope(("rope.cos", 0), ("rope.sin", 0), rows, half, half, 1e7)
        elif name in ("rope.sin", "kv_scales", "draft.kv_scales"):
            if name != "rope.sin":
                once.fill(name, 1.0)
        elif name == "fla.chunk_indices":
            nt = b["shape"][0]
            once.fill(name, 0)
            once.iota(name, nt, stride=2, off=4)   # (0, i) pairs: one sequence, chunk i
        elif name in ("fla.chunk_offsets", "conv.batch_ptr"):
            once.fill(name, 0)
        elif name == "conv.token_chunk_offset":
            once.fill(name, 0)
            once.iota(name, b["shape"][0])
        elif name == "gdn.has_initial":
            once.fill(name, 1)
        elif (lm := re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)):
            i, rest = lm.groups()
            q = f"{P}layers.{i}."
            if rest.endswith(".weight_p1"):
                once.p1(name, q + rest[: -len("_p1")])
            elif rest == "mlp.gate_up_proj.weight":
                b["bind"] = bind(q + "mlp.gate_proj.weight", q + "mlp.up_proj.weight")
            elif rest == "self_attn.qkv_proj.weight":
                b["bind"] = bind(*(q + f"self_attn.{x}_proj.weight" for x in "qkv"))
            elif rest == "linear_attn.in_proj_qkvz.weight":
                b["bind"] = bind(q + "linear_attn.in_proj_qkv.weight", q + "linear_attn.in_proj_z.weight")
            elif rest == "linear_attn.in_proj_ba.weight":
                b["bind"] = bind(q + "linear_attn.in_proj_b.weight", q + "linear_attn.in_proj_a.weight")
            elif rest == "linear_attn.A_log":
                once.cast(name, q + rest)
            else:
                b["bind"] = bind(q + rest)
        elif name == "draft.rope.cos_sin_cache":
            rows, width = b["shape"]
            once.rope((name, 0), (name, width), rows, width // 2, width, 1e7)
        elif (dm := re.fullmatch(r"draft\.fc\.(\d)\.weight", name)):
            j = int(dm.group(1))
            h = b["shape"][1]
            b["bind"] = [seg("fc.weight", cols=(j * h, (j + 1) * h))]
        elif name == "draft.fused_kv.weight":
            b["bind"] = bind(*(f"layers.{l}.self_attn.{x}_proj.weight" for l in range(draft_layers(m)) for x in "kv"))
        elif name.startswith("draft.selector."):
            leaf = name[len("draft.selector."):]
            b["bind"] = bind("candidate_selector." + {"predecessor": "predecessor_codebook",
                                                      "successor": "successor_codebook"}.get(leaf, leaf))
        elif name in ("draft.hidden_norm.weight", "draft.norm.weight"):
            b["bind"] = bind(name[len("draft."):])
        elif (dl := re.fullmatch(r"draft\.layers\.(\d+)\.(.+)", name)):
            l, rest = dl.groups()
            q = f"layers.{l}."
            if rest == "mlp.gate_up_proj.weight":
                b["bind"] = bind(q + "mlp.gate_proj.weight", q + "mlp.up_proj.weight")
            elif rest == "self_attn.qkv_proj.weight":
                b["bind"] = bind(*(q + f"self_attn.{x}_proj.weight" for x in "qkv"))
            else:
                b["bind"] = bind(q + rest)
        else:
            raise ValueError(f"no binding for weight buffer `{name}`")
    return name_constants(normalize(once.finish()))


def qwen3(m):
    """Qwen3-4B: rope theta 1e6 over the whole 128-dim head (one 128-wide
    cos | sin cache), plain norms, tied lm_head; the DSpark draft has its
    own embedding and lm_head, a 5-tap `fc.weight` and a Markov head."""
    m = resolve_constants(m)
    once = Once(m)
    for name in weights(m):
        b = m["buffers"][name]
        if name == "lm_head.weight":
            b["bind"] = bind("model.embed_tokens.weight")
        elif name == "rope.cos_sin_cache":
            rows, width = b["shape"]
            once.rope((name, 0), (name, width), rows, width // 2, width, 1e6)
        elif name in ("kv_scales", "draft.kv_scales"):
            once.fill(name, 1.0)
        elif name.startswith("model."):
            lm = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
            rest = lm.group(2) if lm else None
            q = f"model.layers.{lm.group(1)}." if lm else ""
            if rest == "mlp.gate_up_proj.weight":
                b["bind"] = bind(q + "mlp.gate_proj.weight", q + "mlp.up_proj.weight")
            elif rest == "self_attn.qkv_proj.weight":
                b["bind"] = bind(*(q + f"self_attn.{x}_proj.weight" for x in "qkv"))
            else:
                b["bind"] = bind(name)
        elif (dm := re.fullmatch(r"draft\.fc\.(\d)\.weight", name)):
            j = int(dm.group(1))
            h = b["shape"][1]
            b["bind"] = [seg("fc.weight", cols=(j * h, (j + 1) * h))]
        elif name == "draft.fused_kv.weight":
            b["bind"] = bind(*(f"layers.{l}.self_attn.{x}_proj.weight" for l in range(draft_layers(m)) for x in "kv"))
        elif name == "draft.markov_w1":
            b["bind"] = bind("markov_head.markov_w1.weight")
        elif name == "draft.markov_w2.weight":
            b["bind"] = bind("markov_head.markov_w2.weight")
        elif (dl := re.fullmatch(r"draft\.layers\.(\d+)\.(.+)", name)):
            l, rest = dl.groups()
            q = f"layers.{l}."
            if rest == "mlp.gate_up_proj.weight":
                b["bind"] = bind(q + "mlp.gate_proj.weight", q + "mlp.up_proj.weight")
            elif rest == "self_attn.qkv_proj.weight":
                b["bind"] = bind(*(q + f"self_attn.{x}_proj.weight" for x in "qkv"))
            else:
                b["bind"] = bind(q + rest)
        elif name.startswith("draft."):
            b["bind"] = bind(name[len("draft."):])
        else:
            raise ValueError(f"no binding for weight buffer `{name}`")
    return name_constants(normalize(once.finish()))


FAMILIES = {"qwen38": qwen38, "qwen3": qwen3}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("family", choices=FAMILIES)
    p.add_argument("--input", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, help="default: rewrite --input")
    args = p.parse_args()
    m = FAMILIES[args.family](json.loads(args.input.read_text()))
    (args.output or args.input).write_text(json.dumps(m, indent=1) + "\n")


if __name__ == "__main__":
    main()
