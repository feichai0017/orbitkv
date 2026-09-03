from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any


_CHUNK_BYTES = 1024 * 1024
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"value is not canonical-JSON serializable: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def parse_strict_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("strict JSON input must be text")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON number {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number {value}")
        return parsed

    try:
        value = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except (TypeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON object: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("strict JSON value must be an object")
    return value


def canonical_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("path must be a non-empty string")
    if "\\" in value:
        raise ValueError(f"path contains a backslash: {value!r}")
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise ValueError(f"path contains a forbidden control character: {value!r}")

    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError(f"path must be relative: {value!r}")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"path contains an unsafe component: {value!r}")
    if path.as_posix() != value:
        raise ValueError(f"path is not canonical POSIX form: {value!r}")
    return path


def require_exact_keys(
    value: object, expected: Iterable[str], label: str
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    try:
        expected_keys = set(expected)
    except TypeError as error:
        raise ValueError(f"{label} expected keys are invalid") from error
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted((repr(key) for key in expected_keys - actual_keys))
        extra = sorted((repr(key) for key in actual_keys - expected_keys))
        raise ValueError(
            f"{label} keys differ: missing={missing} extra={extra}"
        )


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} is not a canonical SHA-256 digest")
    return value


def parse_sha256sums(text: str) -> dict[str, str]:
    if not isinstance(text, str):
        raise ValueError("SHA256SUMS input must be text")

    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or any(not line for line in lines):
        raise ValueError("SHA256SUMS must contain a non-empty table without blank lines")

    result: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError(
                f"SHA256SUMS line {line_number} does not have canonical format"
            )
        digest = require_sha256(
            line[:64], f"SHA256SUMS line {line_number} digest"
        )
        path = canonical_relative_path(line[66:])
        name = path.as_posix()
        if name == "SHA256SUMS":
            raise ValueError("SHA256SUMS must not contain itself")
        if name in result:
            raise ValueError(f"SHA256SUMS contains duplicate path {name!r}")
        result[name] = digest
    return result


__all__ = [
    "canonical_json_sha256",
    "canonical_relative_path",
    "parse_sha256sums",
    "parse_strict_json_object",
    "require_exact_keys",
    "require_sha256",
    "sha256_file",
]
