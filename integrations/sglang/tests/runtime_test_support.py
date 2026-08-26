from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import pytest


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = INTEGRATION_ROOT.parents[1]

from orbitkv_sglang.config import load_config
from orbitkv_sglang.ffi import CtypesManagerFactory
from orbitkv_sglang.ffi.manager import CtypesManager
from orbitkv_sglang.runtime import (
    ArenaRegistration,
    CanonicalRuntime,
    ClassTokenDispositionUpdate,
    ManagerCreateSettings,
    TokenDisposition,
    TokenDispositionKind,
)


@pytest.fixture(scope="session")
def ffi_library() -> Path:
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            str(REPOSITORY_ROOT / "crates/orbitkv-ffi/Cargo.toml"),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    return REPOSITORY_ROOT / "crates/orbitkv-ffi/target/release/liborbitkv_ffi.so"


def _runtime(
    tmp_path: Path,
    library: Path,
    *,
    hybrid: bool = True,
    pure_sliding: bool = False,
    window_tokens: int = 18,
    requests: int = 16,
) -> tuple[Any, CtypesManager, CanonicalRuntime]:
    if pure_sliding and hybrid:
        raise ValueError("pure_sliding and hybrid are mutually exclusive")
    classes = [] if pure_sliding else [
        {
            "name": "full",
            "layers": [0],
            "retention": "full",
            "bytes_per_token_per_layer": 128,
            "window_tokens": None,
        }
    ]
    if hybrid or pure_sliding:
        classes.append(
            {
                "name": "swa",
                "layers": [1] if hybrid else [0],
                "retention": "sliding",
                "bytes_per_token_per_layer": 128,
                "window_tokens": window_tokens,
            }
        )
    plan = tmp_path / f"plan-{hybrid}-{pure_sliding}-{window_tokens}.json"
    plan.write_text(json.dumps({"page_tokens": 16, "classes": classes}))
    config = load_config(
        {"ORBITKV_PLAN": str(plan), "ORBITKV_LIBRARY": str(library)}
    )
    arenas = tuple(
        ArenaRegistration(
            item.class_id, item.pool_id, item.backend_domain, 64, 0
        )
        for item in config.classes
    )
    manager = CtypesManagerFactory().create(
        config,
        ManagerCreateSettings(
            requests,
            4,
            requests,
            64 * len(arenas),
            64,
        ),
        arenas,
    )
    assert isinstance(manager, CtypesManager)
    return config, manager, CanonicalRuntime(config, manager)


class ReadyEvent:
    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


def _step_batch(
    runtime: CanonicalRuntime,
    values: Sequence[tuple[Any, int]],
    *,
    domain: int = 1,
) -> Any:
    batch, plans = runtime.prepare_batch(tuple(values))
    assert len(plans) == len(values)
    runtime.mark_lowered(batch)
    submitted = runtime.submit_batch(batch)
    assert len(submitted) == len(values)
    runtime.mark_forward(batch)
    runtime.register_event(batch, ReadyEvent(), domain)
    runtime.poll()
    return batch


def _policy_updates() -> tuple[ClassTokenDispositionUpdate, ...]:
    return tuple(
        ClassTokenDispositionUpdate(
            0,
            token_id,
            TokenDisposition(TokenDispositionKind.POLICY_EVICTED, 7, 1, 99),
        )
        for token_id in range(48)
        if token_id % 16 >= 8
    )
