"""Manifest post-pass shared by the generators: turn an explicit, verbose
manifest into the normalized wire form (schema_version 4).

A generator writes everything out longhand — every launch with its full
ABI and wiring, every call with every ABI scalar the mined kernel takes,
each launch naming its artifact inline as ``{"cubin": ..., "sha256": ...}``.
`normalize` is the linker pass that makes the manifest minimal without
changing what runs:

1. **hoist modules** — inline ``cubin``/``sha256`` pairs become entries of
   the top-level ``modules`` table (the manifest's dependency list); each
   launch keeps only ``"module": <name>``.
2. **fold constants** — an interface scalar that every call of an op passes
   as the same literal is not part of the contract, it is the impl's ABI
   constant (a mined kernel's strides, flags, eps). It leaves the interface
   and becomes a literal in each launch's wiring; every call drops it.
3. **default the identity** — a launch whose ABI equals the op's params
   omits ``params``; wiring that forwards the params in order omits ``args``.
4. **extern launches have no geometry** — ``block``/``grid``/``shared_mem``
   are dropped from ``extern:`` entries.

Keys are emitted in a canonical order so diffs stay readable.
"""

import copy
import hashlib
import os
import pathlib
import re
import subprocess

SCHEMA_VERSION = 5
SCALARS = ("i32", "i64", "f32", "u8")

_TOP = ["schema_version", "model", "constants", "vars", "topology", "states", "buffers", "modules", "ops", "programs"]
_BUFFER = ["dtype", "shape", "kind", "fill", "domain", "bind"]
_PROGRAM = ["batch", "once", "graph", "calls"]
_LAUNCH = ["module", "entry", "params", "block", "grid", "shared_mem", "cluster", "pdl", "args"]
_CALL = ["label", "op", "args"]


def program(calls, groups=None, rows=None, span=None, once=False, graph=False):
    """A program object of the wire form: a forward of `groups` sequences of
    `rows` rows each (rows a constant or the name of the var fed per call;
    `span` the var one sequence's run of rows is sized by), a
    once-after-load program, or a plain one. `graph`: the runtime drives
    it through a CUDA graph captured per call shape (a fixed-shape step,
    never a variable-row prefill)."""
    p = {}
    if groups is not None:
        p["batch"] = {"groups": groups, "rows": rows, **({"span": span} if span else {})}
    if once:
        p["once"] = True
    if graph:
        p["graph"] = True
    p["calls"] = calls
    return p


def _order(d, keys):
    return {k: d[k] for k in keys if k in d} | {k: v for k, v in d.items() if k not in keys}


def _is_literal(arg):
    return isinstance(arg, dict) and len(arg) == 1 and next(iter(arg)) in SCALARS


def module_name(source):
    """Human name for a module from its source: the repo of a registry ref,
    else the file stem minus any ``-<sha12>`` suffix."""
    if source.startswith("hf:"):
        return source[3:].split("@")[0].split("/")[1]
    stem = source.rsplit("/", 1)[-1]
    stem = stem[: -len(".cubin")] if stem.endswith(".cubin") else stem.rsplit(".", 1)[0]
    return re.sub(r"-[0-9a-f]{12}$", "", stem)


def hoist_modules(m):
    modules = dict(m.get("modules", {}))
    by_sha = {v["sha256"]: k for k, v in modules.items()}
    for op in m["ops"].values():
        for launch in op["impl"]["launches"]:
            cubin, sha = launch.pop("cubin", None), launch.pop("sha256", None)
            if cubin is None:
                assert sha is None, f"{launch['entry']}: sha256 without cubin"
                continue
            assert sha, f"{launch['entry']}: cubin `{cubin}` without sha256"
            if sha not in by_sha:
                name = module_name(cubin)
                if name in modules:
                    name = f"{name}-{sha[:8]}"
                assert name not in modules
                modules[name] = {"source": cubin, "sha256": sha}
                by_sha[sha] = name
            launch["module"] = by_sha[sha]
    m["modules"] = dict(sorted(modules.items()))


def _materialize(op):
    """Make every launch's params/args explicit against the op's interface."""
    for launch in op["impl"]["launches"]:
        launch.setdefault("params", list(op["params"]))
        launch.setdefault("args", [{"param": i} for i in range(len(op["params"]))])


def fold_constants(m):
    calls_by_op = {}
    for p in m["programs"].values():
        for c in p["calls"]:
            calls_by_op.setdefault(c["op"], []).append(c)
    for oname, op in m["ops"].items():
        calls = calls_by_op.get(oname)
        if not calls:
            continue
        params = op["params"]
        folded = {}
        for i, p in enumerate(params):
            if p not in SCALARS:
                continue
            first = calls[0]["args"][i]
            if _is_literal(first) and all(c["args"][i] == first for c in calls):
                folded[i] = first
        if not folded:
            continue
        _materialize(op)
        keep = [i for i in range(len(params)) if i not in folded]
        renumber = {old: new for new, old in enumerate(keep)}
        def fold(a, keep_keys=()):
            # a launch arg or a pack field: a folded param becomes the call's literal
            # (a pack field keeps its offset/width), any other param is renumbered;
            # a tensormap field renumbers the buffer it describes
            if "tensormap" in a:
                return {**a, "tensormap": {**a["tensormap"], "param": renumber[a["tensormap"]["param"]]}}
            if "param" not in a:
                return a
            if a["param"] in folded:
                return {**{k: a[k] for k in keep_keys if k in a}, **folded[a["param"]]}
            return {**a, "param": renumber[a["param"]]}

        for launch in op["impl"]["launches"]:
            args = []
            for a in launch["args"]:
                if "pack" in a:
                    a = {"pack": {**a["pack"], "fields": [fold(f, ("at", "width")) for f in a["pack"]["fields"]]}}
                else:
                    a = fold(a)
                args.append(a)
            launch["args"] = args
        op["params"] = [params[i] for i in keep]
        for c in calls:
            c["args"] = [c["args"][i] for i in keep]


def default_identity(m):
    for op in m["ops"].values():
        n = len(op["params"])
        for launch in op["impl"]["launches"]:
            if launch.get("params") == op["params"]:
                del launch["params"]
            if launch.get("args") == [{"param": i} for i in range(n)]:
                del launch["args"]


def normalize(m):
    m = copy.deepcopy(m)
    assert m.get("schema_version") == SCHEMA_VERSION, m.get("schema_version")
    for op in m["ops"].values():
        for launch in op["impl"]["launches"]:
            for a in launch.get("args", []):
                if "scratch" in a:
                    assert a.pop("offset", 0) == 0, "scratch offsets are gone; declare two scratches"
    hoist_modules(m)
    fold_constants(m)
    default_identity(m)
    for op in m["ops"].values():
        launches = []
        for launch in op["impl"]["launches"]:
            if launch["entry"].startswith("extern:"):
                for k in ("module", "block", "grid", "shared_mem"):
                    launch.pop(k, None)
            launches.append(_order(launch, _LAUNCH))
        op["impl"]["launches"] = launches
    m["buffers"] = {k: _order(v, _BUFFER) for k, v in m["buffers"].items()}
    m["programs"] = {
        k: _order({**v, "calls": [_order(c, _CALL) for c in v["calls"]]}, _PROGRAM) for k, v in m["programs"].items()
    }
    return _order(m, _TOP)


# ------------------------------------------------------------- dump index
def cuobjdump():
    return str(pathlib.Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "cuobjdump")


def module_functions(mod):
    """{function: register count} for one cubin (``cuobjdump -res-usage``)."""
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


class DumpIndex:
    """Every ``module_*.cubin`` of one or more capture dumps, by content:
    ``pin(symbol, regs)`` names the one module that defines ``symbol`` with
    that register count, so a generator can pin every mined launch — the
    manifest's ``modules`` table is the complete dependency list."""

    def __init__(self, *dump_dirs):
        self.mods = {}   # sha -> (path, {function: regs})
        for d in dump_dirs:
            for mod in sorted(pathlib.Path(d).glob("module_*.cubin")):
                sha = hashlib.sha256(mod.read_bytes()).hexdigest()
                if sha not in self.mods:
                    fns = module_functions(mod)
                    if fns:
                        self.mods[sha] = (mod, fns)

    def param_sizes(self, sha, symbol):
        """Kernel parameter sizes in ordinal order, from the cubin's
        ``.nv.info.<symbol>`` (EIATTR_KPARAM_INFO) — what the runtime reads
        back with cuFuncGetParamInfo."""
        mod, _ = self.mods[sha]
        out = subprocess.run([cuobjdump(), "-elf", str(mod)], capture_output=True, text=True).stdout
        # the section body (not its line in the section table): a header
        # line that is exactly the section name, up to the next section
        hdr = re.search(r"^\.nv\.info\.%s\s*$" % re.escape(symbol), out, re.M)
        assert hdr, f"{mod.name}: no .nv.info.{symbol}"
        nxt = re.search(r"^\.[A-Za-z]", out[hdr.end():], re.M)
        sect = out[hdr.end():hdr.end() + nxt.start() if nxt else None]
        params = {int(o, 16): int(sz, 16) for o, sz in
                  re.findall(r"Ordinal\s*:\s*0x([0-9a-f]+)\s+Offset\s*:\s*0x[0-9a-f]+\s+Size\s*:\s*0x([0-9a-f]+)", sect)}
        return [params[i] for i in range(len(params))]

    def pin(self, symbol, regs=None, sizes=None):
        """sha256 of the unique dump module defining ``symbol`` — at ``regs``
        registers, and (when the register count does not separate two
        constexpr instances) with parameter layout ``sizes``."""
        hits = {sha: mod for sha, (mod, fns) in self.mods.items()
                if symbol in fns and (regs is None or fns[symbol] == regs)}
        assert hits, f"{symbol} REG={regs}: no dump module defines it"
        if len(hits) > 1 and sizes is not None:
            hits = {sha: mod for sha, mod in hits.items() if self.param_sizes(sha, symbol) == list(sizes)}
        assert len(hits) == 1, f"{symbol} REG={regs} params={sizes}: {len(hits)} dump modules match: {sorted(m.name for m in hits.values())}"
        return next(iter(hits))


def resolve_constants(manifest):
    """Expand named numbers before a generator reuses a manifest's ABI.

    Follow the published schema so a buffer/module/var name that happens to
    equal a constant stays a name. The Rust parser performs final validation.
    """
    import json
    m = copy.deepcopy(manifest)
    constants = m.pop("constants", {})
    if not constants:
        return m
    schema = json.loads((pathlib.Path(__file__).resolve().parent.parent /
                         "schema/manifest-v5.schema.json").read_text())

    def alternatives(s):
        if "$ref" in s:
            return [s] + alternatives(schema["$defs"][s["$ref"].rsplit("/", 1)[1]])
        if "anyOf" in s or "oneOf" in s:
            return [s] + [a for branch in s.get("anyOf", s.get("oneOf")) for a in alternatives(branch)]
        return [s]

    def expand(value, schemas):
        choices = [a for s in schemas for a in alternatives(s)]
        if isinstance(value, str):
            return constants.get(value, value) if any(any(t in ("integer", "number") for t in (s["type"] if isinstance(s.get("type"), list) else [s.get("type")])) for s in choices) else value
        if isinstance(value, dict):
            return {k: expand(v, [child for s in choices
                                 if isinstance(child := s.get("properties", {}).get(k, s.get("additionalProperties")), dict)])
                    for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v, [child for s in choices
                               if isinstance(child := (s["prefixItems"][i] if i < len(s.get("prefixItems", []))
                                                        else s.get("items")), dict)]) for i, v in enumerate(value)]
        return value
    return {k: v if k == "schema_version" else expand(v, [schema["properties"].get(k, {})]) for k, v in m.items()}
