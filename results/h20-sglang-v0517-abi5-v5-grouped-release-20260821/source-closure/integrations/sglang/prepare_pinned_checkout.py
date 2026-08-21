#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


INTEGRATION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(INTEGRATION_ROOT / "src"))

from orbitkv_sglang.pinned import (  # noqa: E402
    apply_reviewed_patch,
    validate_base_checkout,
    validate_patched_checkout,
)


PATCH_FILE = INTEGRATION_ROOT / "patches/v0.5.17-orbitkv-fail-closed.patch"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply or verify the reviewed OrbitKV patch for pinned SGLang."
    )
    parser.add_argument("action", choices=("check-base", "apply", "verify"))
    parser.add_argument("--sglang-root", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.action == "check-base":
        checkout = validate_base_checkout(arguments.sglang_root)
    elif arguments.action == "apply":
        checkout = apply_reviewed_patch(arguments.sglang_root, PATCH_FILE)
    else:
        checkout = validate_patched_checkout(arguments.sglang_root)
    print(f"OrbitKV pinned SGLang {arguments.action} passed: {checkout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
