from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

from orbitkv_sglang.qualification_primitives import (  # noqa: E402
    canonical_json_sha256,
    canonical_relative_path,
    parse_sha256sums,
    parse_strict_json_object,
    require_exact_keys,
    require_sha256,
    sha256_file,
)


def test_sha256_file_handles_more_than_one_chunk(tmp_path: Path) -> None:
    payload = b"a" * (1024 * 1024) + b"boundary"
    target = tmp_path / "payload.bin"
    target.write_bytes(payload)
    assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_hashes_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "empty"
    target.write_bytes(b"")
    assert sha256_file(target) == hashlib.sha256(b"").hexdigest()


def test_canonical_json_sha256_uses_the_required_encoding() -> None:
    value = {"unicode": "caf\u00e9", "items": [3, True, None], "a": 1}
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    assert canonical_json_sha256(value) == hashlib.sha256(encoded).hexdigest()
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256(
        {"a": 1, "b": 2}
    )


def test_canonical_json_sha256_normalizes_serialization_errors() -> None:
    with pytest.raises(ValueError):
        canonical_json_sha256({"unsupported": object()})


def test_parse_strict_json_object_accepts_nested_objects() -> None:
    assert parse_strict_json_object('{"z":2,"nested":{"x":1}}') == {
        "z": 2,
        "nested": {"x": 1},
    }


@pytest.mark.parametrize(
    "text",
    [
        '{"x":1,"x":2}',
        '{"nested":{"x":1,"x":2}}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":-Infinity}',
        '{"x":1e10000}',
        "[]",
        "null",
        "1",
        "{",
    ],
)
def test_parse_strict_json_object_rejects_non_strict_input(text: str) -> None:
    with pytest.raises(ValueError):
        parse_strict_json_object(text)


def test_parse_strict_json_object_rejects_non_text() -> None:
    with pytest.raises(ValueError):
        parse_strict_json_object(b"{}")  # type: ignore[arg-type]


def test_canonical_relative_path_accepts_canonical_posix_paths() -> None:
    assert canonical_relative_path("artifacts/run 1/result.json") == PurePosixPath(
        "artifacts/run 1/result.json"
    )
    assert canonical_relative_path("caf\u00e9") == PurePosixPath("caf\u00e9")


@pytest.mark.parametrize(
    "value",
    [
        None,
        PurePosixPath("artifact"),
        "",
        "/absolute",
        "//absolute",
        "back\\slash",
        "nul\x00byte",
        "carriage\rreturn",
        "line\nfeed",
        ".",
        "..",
        "./artifact",
        "artifact/.",
        "artifact/..",
        "artifact/../other",
        "artifact//other",
        "artifact/",
    ],
)
def test_canonical_relative_path_rejects_unsafe_or_noncanonical_values(
    value: object,
) -> None:
    with pytest.raises(ValueError):
        canonical_relative_path(value)


def test_require_exact_keys_accepts_only_the_exact_dict_shape() -> None:
    assert require_exact_keys({"a": 1, "b": 2}, ("b", "a"), "record") is None
    with pytest.raises(ValueError, match="record must be an object"):
        require_exact_keys([], {"a"}, "record")
    with pytest.raises(ValueError, match="missing"):
        require_exact_keys({"a": 1}, {"a", "b"}, "record")
    with pytest.raises(ValueError, match="extra"):
        require_exact_keys({"a": 1, "b": 2}, {"a"}, "record")


def test_require_sha256_accepts_only_lowercase_hex() -> None:
    digest = "0123456789abcdef" * 4
    assert require_sha256(digest, "artifact") == digest


@pytest.mark.parametrize(
    "value",
    [None, 0, "0" * 63, "0" * 65, "A" * 64, "g" * 64, "0" * 63 + "\n"],
)
def test_require_sha256_rejects_noncanonical_values(value: object) -> None:
    with pytest.raises(ValueError, match="artifact"):
        require_sha256(value, "artifact")


@pytest.mark.parametrize("trailing_newline", [False, True])
def test_parse_sha256sums_accepts_canonical_table(trailing_newline: bool) -> None:
    first = "0" * 64
    second = "abcdef0123456789" * 4
    text = f"{first}  artifact.bin\n{second}  nested/file name.json"
    if trailing_newline:
        text += "\n"
    assert parse_sha256sums(text) == {
        "artifact.bin": first,
        "nested/file name.json": second,
    }


@pytest.mark.parametrize(
    "text",
    [
        "",
        "\n",
        f"{'0' * 64} artifact",
        f"{'0' * 64}\tartifact",
        f"{'0' * 64} *artifact",
        f"{'A' * 64}  artifact",
        f"{'0' * 63}  artifact",
        f"{'0' * 64}  /absolute",
        f"{'0' * 64}  ./artifact",
        f"{'0' * 64}  artifact\r\n",
        f"{'0' * 64}  artifact\n\n{'1' * 64}  other",
        f"{'0' * 64}  artifact\n{'1' * 64}  artifact",
        f"{'0' * 64}  SHA256SUMS",
    ],
)
def test_parse_sha256sums_rejects_noncanonical_tables(text: str) -> None:
    with pytest.raises(ValueError):
        parse_sha256sums(text)


def test_parse_sha256sums_rejects_non_text() -> None:
    with pytest.raises(ValueError):
        parse_sha256sums(b"")  # type: ignore[arg-type]
