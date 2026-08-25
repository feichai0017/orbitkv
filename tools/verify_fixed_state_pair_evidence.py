#!/usr/bin/env python3
"""Verify fixed-state pair evidence with the selected profile.

This capability-oriented facade delegates to a legacy, manifest-bound
verifier.  The generic name does not broaden the evidence claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


_LEGACY_PATH = Path(__file__).resolve().with_name(
    "verify_qwen35_h20_pair_evidence.py"
)


def _load_legacy() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_orbitkv_legacy_fixed_state_pair_evidence",
        _LEGACY_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load fixed-state pair-evidence verifier: {_LEGACY_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy()


def verify_archive(path: Path) -> dict[str, Any]:
    """Verify a fixed-state pair-evidence archive."""

    return _legacy.verify_archive(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive",
        type=Path,
        help="fixed-state pair-evidence archive directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected fixed-state pair-evidence CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = verify_archive(args.archive)
    except RuntimeError as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
