#!/usr/bin/env python3
"""Run kern run on every prompt of docs/qwen38/ref.json and compare the
generated token ids with vLLM's, token by token.

    tools/qwen38_compare.py --gpu 1 [--chunk 512] [--eager] [--steps 400]
                            [--target qwen3.8-27b-dflash2 --spec]
                            [--out docs/qwen38/compare-<tag>.json]

`KERN_BIN` overrides the binary (default target/release/kern).

Prints one line per prompt (match length / first divergence) and a summary;
exit status 1 unless every prompt matches to the full length.
"""

import os
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent


def run_one(args, prompt, steps):
    cmd = [os.environ.get("KERN_BIN", str(REPO / "target/release/kern")), "run", args.target,
           "--gpu", str(args.gpu), "--capacity", str(args.capacity), "--chunk", str(args.chunk),
           "--steps", str(steps), "--prompt", prompt]
    if args.spec:
        cmd += ["--rows", "8"]
    if args.eager:
        cmd.append("--eager")
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    err = p.stderr
    if p.returncode != 0:
        sys.exit(f"kern run failed ({p.returncode}):\n{err[-4000:]}")
    ids = lambda key: json.loads(re.search(key + r": (?:\d+ tokens )?(\[.*?\])", err).group(1))  # noqa: E731
    info = {
        "prompt_ids": ids("prompt"),
        "generated": ids("generated ids"),
        "prefill": re.search(r"prefill: .*", err).group(0) if "prefill:" in err else "",
        "decode": re.search(r"\d+ tokens generated(?: in .*|, .*)|spec: .*", err).group(0),
        "wall_s": round(wall, 1),
        "stdout": p.stdout,
    }
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--target", default="qwen3.8-27b",
                    help="kern.toml target: manifest, kernels, the HF checkpoint dirs and tokenizer come from it")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--capacity", type=int, default=4704)
    ap.add_argument("--only", type=int, default=None, help="run a single prompt index")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=str(REPO / "docs/qwen38/ref.json"),
                    help="vLLM reference (qwen38_ref.py) or a kern compare JSON (then kern-vs-kern)")
    ap.add_argument("--spec", action="store_true", help="kern run --rows 8 (draft/verify rounds)")
    args = ap.parse_args()

    ref = json.load(open(args.ref))
    if "results" in ref and ref["results"] and "generated" in ref["results"][0]:
        # a kern compare JSON as the oracle: greedy spec decode must reproduce
        # plain decode token for token
        base = json.load(open(REPO / "docs/qwen38/ref.json"))
        ref = {"results": [{"prompt": b["prompt"], "prompt_token_ids": b["prompt_token_ids"],
                            "output_token_ids": r["generated"]}
                           for r, b in zip(ref["results"], base["results"])]}
    results = []
    all_ok = True
    for i, r in enumerate(ref["results"]):
        if args.only is not None and i != args.only:
            continue
        info = run_one(args, r["prompt"], min(args.steps, len(r["output_token_ids"])))
        want = r["output_token_ids"][:args.steps]
        got = info["generated"]
        n = 0
        while n < min(len(got), len(want)) and got[n] == want[n]:
            n += 1
        full = n == len(want) and len(got) == len(want)
        all_ok &= full and info["prompt_ids"] == r["prompt_token_ids"]
        div = "" if full else f"  DIVERGES at {n}: kern {got[n:n + 3]} vs ref {want[n:n + 3]}"
        pid = "" if info["prompt_ids"] == r["prompt_token_ids"] else "  PROMPT IDS DIFFER"
        print(f"[{i}] match {n}/{len(want)} tokens{div}{pid} | {info['prefill']} | {info['decode']}")
        results.append({"index": i, "match": n, "ref_len": len(want), "byte_identical": full,
                        "prefill": info["prefill"], "decode": info["decode"], "wall_s": info["wall_s"],
                        "generated": got, "text": info["stdout"]})
    print("ALL BYTE-IDENTICAL" if all_ok else "MISMATCH")
    if args.out:
        json.dump({"config": vars(args), "results": results}, open(args.out, "w"), indent=1)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
