#!/usr/bin/env python3
"""Verify a sealed token-relocation archive with the selected profile.

This capability-oriented facade delegates to a legacy, manifest-bound seal
verifier.  The generic name does not broaden the qualification claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


_LEGACY_PATH = Path(__file__).resolve().with_name(
    "verify_token_relocation_h20_seal.py"
)


def _load_legacy() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_orbitkv_legacy_token_relocation_seal",
        _LEGACY_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load token-relocation seal verifier: {_LEGACY_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy()


def verify_sealed_archive(root: Path) -> dict[str, Any]:
    """Verify a sealed token-relocation archive with the selected verifier."""

    return _legacy.verify_sealed_archive(root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive",
        type=Path,
        help="sealed token-relocation archive directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = verify_sealed_archive(args.archive)
    except RuntimeError as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
