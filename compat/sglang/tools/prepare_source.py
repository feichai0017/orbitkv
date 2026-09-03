#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


_TOOLS_ROOT = Path(__file__).resolve().parent
_SGLANG_ROOT = _TOOLS_ROOT.parent
_REPOSITORY_ROOT = _TOOLS_ROOT.parents[2]
_DEFAULT_SOURCE = _SGLANG_ROOT / "source"
sys.path.insert(0, str(_SGLANG_ROOT / "bridge/src"))

from orbitkv_sglang.pinned import (  # noqa: E402
    apply_reviewed_patch,
    pinned_source_contract,
    validate_base_checkout,
    validate_patched_checkout,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply or verify the reviewed OrbitKV manager overlay on a complete "
            "pinned SGLang source checkout."
        )
    )
    parser.add_argument("action", choices=("check-base", "apply", "verify"))
    parser.add_argument("--sglang-root", type=Path, default=_DEFAULT_SOURCE)
    arguments = parser.parse_args()

    if arguments.action == "check-base":
        checkout = validate_base_checkout(arguments.sglang_root)
    elif arguments.action == "apply":
        contract = pinned_source_contract()
        checkout = apply_reviewed_patch(
            arguments.sglang_root, _REPOSITORY_ROOT / contract["patch_path"]
        )
    else:
        checkout = validate_patched_checkout(arguments.sglang_root)
    print(
        "OrbitKV full-source SGLang overlay "
        f"{arguments.action} passed: {checkout}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
