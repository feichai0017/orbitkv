#!/usr/bin/env python3
"""Cross-configuration consistency of kern run outputs: every
docs/qwen38/compare-<tag>.json given must hold byte-identical generated ids
per prompt (chunk=1 / chunk=512 / chunk=2048 / eager / graph must not change
the arithmetic).  Also prints the agreement length vs vLLM per prompt.

    tools/qwen38_consistency.py docs/qwen38/compare-{eager-c512,graph-c512,eager-c1,graph-c2048}.json
"""

import json
import sys

runs = {p: json.load(open(p)) for p in sys.argv[1:]}
base_path, base = next(iter(runs.items()))
ok = True
for r in base["results"]:
    i = r["index"]
    line = f"[{i}] vs vLLM: {r['match']}/{r['ref_len']} tokens agree"
    for p, run in runs.items():
        if p == base_path:
            continue
        o = next(x for x in run["results"] if x["index"] == i)
        if o["generated"] != r["generated"]:
            n = next((k for k, (a, b) in enumerate(zip(o["generated"], r["generated"])) if a != b), min(len(o["generated"]), len(r["generated"])))
            line += f"  | {p}: DIFFERS from {base_path} at token {n}"
            ok = False
    print(line)
print("configs:", ", ".join(runs))
print("ALL CONFIGS BYTE-IDENTICAL TO EACH OTHER" if ok else "CONFIG MISMATCH")
sys.exit(0 if ok else 1)
