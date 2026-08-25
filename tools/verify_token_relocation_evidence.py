#!/usr/bin/env python3
"""Verify token-relocation diagnostic evidence with the selected profile.

This capability-oriented facade delegates to a legacy, manifest-bound
verifier.  The generic name does not broaden the verifier's evidence scope.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


_LEGACY_PATH = Path(__file__).resolve().with_name(
    "verify_token_relocation_h20_evidence.py"
)


def _load_legacy() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_orbitkv_legacy_token_relocation_evidence",
        _LEGACY_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load token-relocation evidence verifier: {_LEGACY_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy()


def verify_evidence(
    root: Path,
    *,
    expected_harness_sha256: str | None = None,
    expected_adapter: dict[str, Any] | None = None,
    expected_adapter_source_root: Path | None = None,
) -> dict[str, Any]:
    """Verify raw token-relocation records with the selected verifier."""

    return _legacy.verify_evidence(
        root,
        expected_harness_sha256=expected_harness_sha256,
        expected_adapter=expected_adapter,
        expected_adapter_source_root=expected_adapter_source_root,
    )


def verify_sealed_archive(root: Path) -> dict[str, Any]:
    """Verify a sealed token-relocation archive through the legacy API."""

    return _legacy.verify_sealed_archive(root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "evidence_root",
        type=Path,
        help="directory containing the token-relocation evidence records",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected token-relocation evidence CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = verify_evidence(args.evidence_root)
    except RuntimeError as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
