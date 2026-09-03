from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def attention_state_from_manager_plan(
    manager_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate a token-manager test fixture into compiler input.

    The runtime has no compatibility loader for manager plans. Tests which need
    a small synthetic layout still describe that layout conveniently as manager
    classes, then pass the equivalent attention-state input through the real
    Rust compiler before loading it.
    """

    page_tokens = manager_plan.get("page_tokens")
    classes = manager_plan.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("manager test plan must contain non-empty classes")

    states: list[dict[str, Any]] = []
    for index, raw_class in enumerate(classes):
        if not isinstance(raw_class, Mapping):
            raise ValueError(f"manager test class {index} must be an object")
        storage_kind = raw_class.get("storage", "token_kv")
        components = raw_class.get("components")
        total_bytes = raw_class.get("bytes_per_token_per_layer")
        component_bytes: dict[str, Any]
        if components is None:
            if (
                not isinstance(total_bytes, int)
                or isinstance(total_bytes, bool)
                or total_bytes < 2
            ):
                raise ValueError(
                    f"manager test class {index} needs at least two bytes per token"
                )
            first = total_bytes // 2
            component_bytes = {
                "key": first,
                "value": total_bytes - first,
            }
        else:
            if not isinstance(components, list):
                raise ValueError(
                    f"manager test class {index} components must be a list"
                )
            component_bytes = {
                component["name"]: component["bytes_per_token_per_layer"]
                for component in components
            }

        if storage_kind == "token_kv":
            expected = {"key", "value"}
            fields = (
                ("key_bytes_per_token_per_layer", component_bytes.get("key")),
                ("value_bytes_per_token_per_layer", component_bytes.get("value")),
            )
        elif storage_kind == "latent_kv":
            expected = {"latent", "rope"}
            fields = (
                (
                    "latent_bytes_per_token_per_layer",
                    component_bytes.get("latent"),
                ),
                ("rope_bytes_per_token_per_layer", component_bytes.get("rope")),
            )
        else:
            raise ValueError(
                f"manager test class {index} has unsupported storage {storage_kind!r}"
            )
        if set(component_bytes) != expected or any(value is None for _, value in fields):
            raise ValueError(
                f"manager test class {index} has invalid {storage_kind} components"
            )
        if sum(component_bytes.values()) != total_bytes:
            raise ValueError(
                f"manager test class {index} component bytes do not match its total"
            )

        storage = {
            "kind": storage_kind,
            **dict(fields),
            "retention": raw_class.get("retention"),
            "window_tokens": raw_class.get("window_tokens"),
        }
        states.append(
            {
                "name": raw_class.get("name"),
                "layers": raw_class.get("layers"),
                "storage": storage,
            }
        )
    return {"page_tokens": page_tokens, "states": states}


def compile_runtime_manifest(
    directory: Path,
    *,
    manager_plan: Mapping[str, Any] | Path | None = None,
    attention_state: Mapping[str, Any] | Path | None = None,
    stem: str = "runtime",
) -> Path:
    """Compile one canonical manifest with the repository's real compiler."""

    if (manager_plan is None) == (attention_state is None):
        raise ValueError(
            "exactly one of manager_plan or attention_state must be provided"
        )
    source = attention_state
    if manager_plan is not None:
        raw_manager = _read_object(manager_plan, "manager_plan")
        source = attention_state_from_manager_plan(raw_manager)
    raw_source = _read_object(source, "attention_state")

    source_path = directory / f"{stem}-attention-state.json"
    manifest_path = directory / f"{stem}-runtime-manifest.json"
    source_path.write_text(json.dumps(raw_source), encoding="utf-8")
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "orbitkv",
            "--",
            "compile-runtime-manifest",
            str(source_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    manifest = json.loads(completed.stdout)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def runtime_manifest_environment(
    directory: Path,
    library: Path,
    *,
    manager_plan: Mapping[str, Any] | Path | None = None,
    attention_state: Mapping[str, Any] | Path | None = None,
    stem: str = "runtime",
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    manifest = compile_runtime_manifest(
        directory,
        manager_plan=manager_plan,
        attention_state=attention_state,
        stem=stem,
    )
    environment = {
        "ORBITKV_RUNTIME_MANIFEST": str(manifest),
        "ORBITKV_LIBRARY": str(library),
    }
    if extra is not None:
        environment.update(extra)
    return environment


def _read_object(value: Mapping[str, Any] | Path | None, name: str) -> dict[str, Any]:
    if isinstance(value, Path):
        value = json.loads(value.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object or path")
    return dict(value)
