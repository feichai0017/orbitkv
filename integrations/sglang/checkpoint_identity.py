from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(model_path: Path, load_format: str) -> dict:
    config_path = model_path / "config.json"
    index_paths = sorted(model_path.glob("*.safetensors.index.json"))
    weight_paths = sorted(model_path.glob("*.safetensors"))
    indexed_weight_names = []
    indexed_weight_bytes = None
    if index_paths:
        index = json.loads(index_paths[0].read_text(encoding="utf-8"))
        indexed_weight_names = sorted(set(index.get("weight_map", {}).values()))
        indexed_weight_bytes = index.get("metadata", {}).get("total_size")
    indexed_weight_paths = [model_path / name for name in indexed_weight_names]
    missing_indexed_weights = [
        path.name for path in indexed_weight_paths if not path.is_file()
    ]
    observed_indexed_weight_bytes = sum(
        path.stat().st_size for path in indexed_weight_paths if path.is_file()
    )
    return {
        "load_format": load_format,
        "config_sha256": (
            sha256_file(config_path) if config_path.is_file() else None
        ),
        "index_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in index_paths
        ],
        "weight_files": [
            {"name": path.name, "bytes": path.stat().st_size}
            for path in weight_paths
        ],
        "weight_bytes": sum(path.stat().st_size for path in weight_paths),
        "indexed_weight_files": indexed_weight_names,
        "indexed_weight_bytes": indexed_weight_bytes,
        "observed_indexed_weight_bytes": observed_indexed_weight_bytes,
        "indexed_weight_container_overhead_bytes": (
            observed_indexed_weight_bytes - indexed_weight_bytes
            if indexed_weight_bytes is not None
            else None
        ),
        "missing_indexed_weights": missing_indexed_weights,
        "indexed_weights_complete": (
            not missing_indexed_weights
            and (
                indexed_weight_bytes is None
                or observed_indexed_weight_bytes >= indexed_weight_bytes
            )
        ),
    }
