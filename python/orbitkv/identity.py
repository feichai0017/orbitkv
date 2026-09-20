"""Stable computation and byte-format identities shared by engine adapters.

Identity is resolved once when an engine connects. Query and publish carry
native prefix hashes within this domain; no model files are read on that path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

_WEIGHTS = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
_ARTIFACTS = _WEIGHTS | {
    ".json",
    ".model",
    ".tiktoken",
    ".txt",
    ".py",
    ".jinja",
    ".vocab",
    ".merges",
}
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def canonical_digest(domain: str, value: Any) -> str:
    """Hash a JSON value with an explicit schema domain and unambiguous encoding."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    return hashlib.sha256(domain.encode() + b"\0" + encoded).hexdigest()


def _file_stamp(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _file_digest(path: Path, stamp: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    if _file_stamp(path) != stamp:
        raise ValueError(f"model artifact changed while OrbitKV fingerprinted it: {path}")
    return digest.hexdigest()


def artifact_identity(model: str, revision: str | None = None) -> dict[str, str]:
    """Fingerprint local artifact contents, or require an immutable Hub revision.

    Each resolution reads the contents: equal-sized writes can share a file
    stamp on coarse clocks. Stamps only detect changes while hashing and never
    define the identity, so identical copies remain interchangeable.
    """
    path = Path(model).expanduser()
    if path.exists():
        files = (
            [path]
            if path.is_file()
            else sorted(
                item
                for item in path.rglob("*")
                if item.is_file()
                and item.suffix in _ARTIFACTS
                and not any(part.startswith(".") for part in item.relative_to(path).parts)
            )
        )
        if not files:
            raise ValueError(f"no model/tokenizer artifacts to fingerprint in {model}")
        manifest = [
            (
                item.name if path.is_file() else item.relative_to(path).as_posix(),
                _file_digest(item.resolve(), _file_stamp(item)),
            )
            for item in files
        ]
        return {"content": canonical_digest("orbitkv.artifacts.v1", manifest)}
    if revision is None or not _COMMIT.fullmatch(revision):
        raise ValueError(
            f"OrbitKV requires local model artifacts or a full immutable Hub revision for {model!r}; "
            "pin --revision to a commit or set ORBITKV_MODEL_FINGERPRINT to the deployment's "
            "SHA-256 artifact digest"
        )
    return {"repository": model, "revision": revision}


def model_identity(
    model: str,
    *,
    revision: str | None = None,
    tokenizer: str | None = None,
    tokenizer_revision: str | None = None,
) -> dict[str, Any]:
    """Identify the deployed weights, model code, tokenizer and processor.

    An explicit fingerprint asserts the entire deployment's artifact identity;
    operators must change it when any covered artifact changes.
    """
    fingerprint = os.environ.get("ORBITKV_MODEL_FINGERPRINT")
    if fingerprint is not None:
        if not _DIGEST.fullmatch(fingerprint):
            raise ValueError(
                "ORBITKV_MODEL_FINGERPRINT must be a lowercase 64-digit SHA-256 digest"
            )
        return {"deployment": fingerprint}
    identity: dict[str, Any] = {"model": artifact_identity(model, revision)}
    if tokenizer and (tokenizer != model or tokenizer_revision not in (None, revision)):
        identity["tokenizer"] = artifact_identity(tokenizer, tokenizer_revision)
    return identity


def state_namespace(
    *,
    engine: str,
    engine_version: str,
    model: dict[str, Any],
    computation: dict[str, Any],
    representation: dict[str, Any],
) -> str:
    """Bind native hashes to one model computation and storage representation."""
    return "orbitkv:v1:" + canonical_digest(
        "orbitkv.state-identity.v1",
        {
            "engine": engine,
            "engine_version": engine_version,
            "model": model,
            "computation": computation,
            "representation": representation,
            "scope": os.environ.get("ORBITKV_CACHE_SCOPE", ""),
        },
    )


def model_config_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Drop artifact locations from HF config; their contents are fingerprinted."""
    return {
        key: model_config_identity(value) if isinstance(value, dict) else value
        for key, value in config.items()
        if key not in {"_name_or_path", "name_or_path", "_commit_hash"}
    }
