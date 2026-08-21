from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs/capability-matrix.md"
PUBLIC_SURFACES = (
    ROOT / "README.md",
    ROOT / "results/README.md",
    ROOT / "docs/standalone-kv-manager-architecture.md",
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
    "frozen e610",
    "Frozen e610",
    "+0.5639%",
    "+5.0041%",
    "ABI5 host-qualified",
    "Compact batch-only ABI5 is host-qualified",
    "The checked-in ABI5 wire no longer matches",
)
MATRIX_REQUIRED_CLAIMS = (
    "| Typed C ABI6 wire | L2 GO |",
    "| ABI6 Python FFI/runtime | L2 GO |",
    "| SGLang `OrbitKVPrefixCache` | L2 GO |",
    "Exactly 23 batch-only symbols",
    "`9233c06d…`",
    "Same-capacity intrinsic KV-memory reduction is **0%**",
    "`performance_go=false`",
)


def main() -> None:
    matrix = MATRIX.read_text(encoding="utf-8")
    for level in (
        "L1 Compiler",
        "L2 Host/ABI",
        "L3 GPU Primitive",
        "L4 Engine E2E",
        "L5 Production",
    ):
        if level not in matrix:
            raise RuntimeError(f"Capability Matrix is missing {level}")

    for claim in MATRIX_REQUIRED_CLAIMS:
        if claim not in matrix:
            raise RuntimeError(f"Capability Matrix is missing current boundary: {claim}")

    for relative in re.findall(r"`(results/[^`]+)`", matrix):
        if not (ROOT / relative).exists():
            raise RuntimeError(f"Capability Matrix evidence does not exist: {relative}")

    for path in PUBLIC_SURFACES:
        text = path.read_text(encoding="utf-8")
        if "capability-matrix" not in text.lower():
            raise RuntimeError(f"public capability surface does not link the matrix: {path}")
        if "ABI6" not in text:
            raise RuntimeError(f"public capability surface omits live ABI6 status: {path}")
        if "histor" not in text.lower():
            raise RuntimeError(f"public capability surface omits historical boundary: {path}")
        for stale in STALE_CLAIMS:
            if stale in text:
                raise RuntimeError(f"stale capability claim in {path}: {stale}")

    print(
        f"verified Capability Matrix: 5 levels, "
        f"{len(PUBLIC_SURFACES)} linked public surfaces"
    )


if __name__ == "__main__":
    main()
