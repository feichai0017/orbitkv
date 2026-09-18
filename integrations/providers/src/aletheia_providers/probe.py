"""Verify that Aletheia kernel modules use the pinned source tree."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[4]
EXPECTED = {
    "flashinfer": (ROOT / "third-party" / "flashinfer", "flashinfer-python"),
    "deep_gemm": (ROOT / "third-party" / "deepgemm", "sgl-deep-gemm"),
    "sgl_kernel": (
        ROOT / "third-party" / "sglang" / "python" / "sglang" / "kernels" / "aot",
        "sglang-kernel",
    ),
}


def probe(module_name: str) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    source = Path(module.__file__).resolve()
    expected, distribution_name = EXPECTED[module_name]
    expected = expected.resolve()
    provenance = "editable" if source.is_relative_to(expected) else _wheel_provenance(distribution_name, expected)
    return {
        "module": module_name,
        "source": str(source),
        "expected_root": str(expected),
        "from_pinned_source": provenance is not None,
        "provenance": provenance,
        "version": getattr(module, "__version__", None),
    }


def _wheel_provenance(distribution_name: str, expected: Path) -> str | None:
    try:
        direct_url = importlib.metadata.distribution(distribution_name).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not direct_url:
        return None
    url = json.loads(direct_url).get("url")
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    installed_from = Path(unquote(parsed.path)).resolve()
    return "local_wheel" if installed_from.is_relative_to(expected) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("modules", nargs="*", default=list(EXPECTED))
    parser.add_argument("--require-source", action="store_true")
    args = parser.parse_args()
    reports = [probe(module) for module in args.modules]
    print(json.dumps({"schema_version": 1, "providers": reports}, indent=2, sort_keys=True))
    if args.require_source and not all(report["from_pinned_source"] for report in reports):
        raise SystemExit("one or more providers were not imported from the pinned source checkout")


if __name__ == "__main__":
    main()
