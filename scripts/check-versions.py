#!/usr/bin/env python3
"""Require one version across Rust, Python, release tooling and an optional tag."""

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Release tag, including the v prefix")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    versions = {}
    for path, section in (
        ("Cargo.toml", "workspace.package"),
        ("python/pyproject.toml", "project"),
        ("python/pyproject.toml", "tool.commitizen"),
    ):
        source = (root / path).read_text()
        table = re.search(rf"(?ms)^\[{re.escape(section)}\]\n(.*?)(?:^\[|\Z)", source)
        version = (
            re.search(r'^version = "([^"]+)"$', table[1], re.MULTILINE)
            if table
            else None
        )
        if version is None:
            raise SystemExit(f"missing version in {path} [{section}]")
        versions[f"{path} [{section}]"] = version[1]
    if len(set(versions.values())) != 1:
        raise SystemExit(f"version mismatch: {versions}")
    version = next(iter(versions.values()))
    if args.tag is not None and args.tag != f"v{version}":
        raise SystemExit(f"release tag {args.tag!r} must match v{version}")
    print(version)


if __name__ == "__main__":
    main()
