from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


SUPPORTED_SGLANG_RELEASE = "v0.5.17"
SUPPORTED_SGLANG_REVISION = "29481685462732237d80d86076d6563e1f658102"
PATCHED_SOURCE_PATH = "python/sglang/srt/plugins/__init__.py"
PYTHON_SOURCE_TREE = "python/sglang"
BASE_SOURCE_SHA256 = (
    "3a975a73f1a7887e68c81ea7a2530250597ac8ae978efc0b0f70f038a99a3164"
)
PATCHED_SOURCE_SHA256 = (
    "1fc2e2472e8fd55f564826509b2afa1f8f0d86a4b2ee3a3986c3209e3c09c934"
)
PATCH_DIFF_SHA256 = (
    "6de7acab246b299386b5d6557154a7aa237c8bdb498b8b885bed1c72e849745d"
)
_EXPECTED_PATCHED_STATUS = f" M {PATCHED_SOURCE_PATH}\0".encode()


def validate_base_checkout(root: Path | str) -> Path:
    """Require the pristine pinned tree that the OrbitKV patch applies to."""

    checkout = _checkout(root)
    _validate_revision(checkout)
    status = _python_tree_status(checkout)
    if status:
        raise RuntimeError("pinned SGLang Python sources are not pristine")
    if _source_sha256(checkout) != BASE_SOURCE_SHA256:
        raise RuntimeError("pinned SGLang plugin loader has an unexpected base hash")
    return checkout


def validate_patched_checkout(root: Path | str) -> Path:
    """Require exactly the reviewed OrbitKV loader patch and no other edits."""

    checkout = _checkout(root)
    _validate_revision(checkout)
    status = _python_tree_status(checkout)
    if status != _EXPECTED_PATCHED_STATUS:
        raise RuntimeError(
            "pinned SGLang Python sources must contain exactly the reviewed "
            "OrbitKV loader patch"
        )
    if _source_sha256(checkout) != PATCHED_SOURCE_SHA256:
        raise RuntimeError("patched SGLang plugin loader has an unexpected hash")
    diff = _git(
        checkout,
        "diff",
        "--no-ext-diff",
        "--no-color",
        "--full-index",
        "--binary",
        "--unified=3",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "--",
        PATCHED_SOURCE_PATH,
    )
    if hashlib.sha256(diff).hexdigest() != PATCH_DIFF_SHA256:
        raise RuntimeError("SGLang loader diff is not the reviewed OrbitKV patch")
    return checkout


def apply_reviewed_patch(root: Path | str, patch_file: Path | str) -> Path:
    """Apply the one reviewed loader patch to a pristine pinned checkout."""

    checkout = validate_base_checkout(root)
    patch = Path(patch_file).expanduser().resolve(strict=True)
    if not patch.is_file():
        raise RuntimeError("OrbitKV loader patch is not a regular file")
    patch_bytes = patch.read_bytes()
    if hashlib.sha256(patch_bytes).hexdigest() != PATCH_DIFF_SHA256:
        raise RuntimeError("OrbitKV loader patch artifact has an unexpected hash")
    _git(checkout, "apply", "--check", "--whitespace=error-all", str(patch))
    _git(checkout, "apply", "--whitespace=error-all", str(patch))
    return validate_patched_checkout(checkout)


def _checkout(root: Path | str) -> Path:
    try:
        checkout = Path(root).expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"invalid SGLang checkout {root}: {error}") from error
    if not checkout.is_dir() or not (checkout / "python/sglang/__init__.py").is_file():
        raise RuntimeError("path is not an SGLang source checkout")
    return checkout


def _validate_revision(checkout: Path) -> None:
    revision = _git(checkout, "rev-parse", "HEAD").decode("ascii").strip()
    if revision != SUPPORTED_SGLANG_REVISION:
        raise RuntimeError(
            f"SGLang revision {revision!r} is not pinned revision "
            f"{SUPPORTED_SGLANG_REVISION}"
        )


def _python_tree_status(checkout: Path) -> bytes:
    return _git(
        checkout,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        PYTHON_SOURCE_TREE,
    )


def _source_sha256(checkout: Path) -> str:
    source = checkout / PATCHED_SOURCE_PATH
    if not source.is_file() or source.is_symlink():
        raise RuntimeError("pinned SGLang plugin loader is missing or is a symlink")
    try:
        return hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeError("cannot read the pinned SGLang plugin loader") from error


def _git(checkout: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail = ": " + error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot verify or patch pinned SGLang checkout{detail}") from error
    return completed.stdout
