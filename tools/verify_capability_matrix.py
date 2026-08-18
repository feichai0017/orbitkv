from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs/capability-matrix.md"
PUBLIC_SURFACES = (
    ROOT / "README.md",
    ROOT / "results/README.md",
    ROOT / "docs/sglang-e2e.md",
    ROOT / "website/src/data/site.ts",
    ROOT / "website/src/pages/index.astro",
    ROOT / "website/src/pages/docs/index.astro",
    ROOT / "website/src/pages/evidence.astro",
)
STALE_CLAIMS = (
    "SGLang export and hydration are not yet",
    "Capsule persistence is host-qualified; SGLang export and hydration",
    'name: "Capsule engine path"',
    "No radix/prefix-cache, speculative, overlap, or CUDA Graph qualification",
)


def main() -> None:
    matrix = MATRIX.read_text(encoding="utf-8")
    for level in (
        "L1 Compiler",
        "L2 Reference Runtime",
        "L3 GPU Primitive",
        "L4 Engine E2E",
        "L5 Production",
    ):
        if level not in matrix:
            raise RuntimeError(f"Capability Matrix is missing {level}")

    for relative in re.findall(r"`(results/[^`]+)`", matrix):
        if not (ROOT / relative).exists():
            raise RuntimeError(f"Capability Matrix evidence does not exist: {relative}")

    for path in PUBLIC_SURFACES:
        text = path.read_text(encoding="utf-8")
        if "capability-matrix" not in text.lower():
            raise RuntimeError(f"public capability surface does not link the matrix: {path}")
        for stale in STALE_CLAIMS:
            if stale in text:
                raise RuntimeError(f"stale capability claim in {path}: {stale}")

    print(
        f"verified Capability Matrix: 5 levels, "
        f"{len(PUBLIC_SURFACES)} linked public surfaces"
    )


if __name__ == "__main__":
    main()
