#!/usr/bin/env python3
"""Capability-oriented token-relocation qualification CLI facade.

The selected implementation remains manifest-bound.  This generic entry point
does not broaden the qualification scope or its claims.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import qualify_token_relocation_h20 as _legacy


_ACTIONS = ("preflight", "run", "component", "seal", "verify-seal")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        choices=_ACTIONS,
        help="qualification lifecycle action; defaults to preflight",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the currently selected token-relocation qualification profile."""
    raw = sys.argv[1:] if argv is None else argv
    if raw and raw[0] in {"-h", "--help"}:
        build_parser().print_help()
        return 0
    return _legacy.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
