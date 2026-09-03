from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = INTEGRATION_ROOT / "bridge/src"
sys.path.insert(0, str(SOURCE_ROOT))

import orbitkv_sglang.bridge.validation as validation  # noqa: E402
import orbitkv_sglang.bridge.state as state  # noqa: E402
from orbitkv_sglang.config import (  # noqa: E402
    ClassConfig,
    RuntimeConfig,
    TokenReclamationConfig,
)


@pytest.fixture(autouse=True)
def _reset_bridge_state() -> None:
    state._install_test_state()
    yield
    state._install_test_state()


def _config(mode: str) -> RuntimeConfig:
    return RuntimeConfig(
        library_path=Path("liborbitkv_ffi.so"),
        plan_json=b"{}",
        plan_fingerprint="sha256:backend-capability-test",
        page_tokens=16,
        classes=(
            ClassConfig(
                class_id=0,
                pool_id=1,
                backend_domain=1,
                name="full",
                layers=(0,),
                retention="full",
                bytes_per_token_per_layer=128,
                window_tokens=None,
                period_blocks=None,
            ),
        ),
        token_reclamation=TokenReclamationConfig(
            mode=mode,
            trigger_tokens=48,
            retained_per_page=8,
            policy_id=7,
            policy_version=1,
            quality_contract=99,
            fragmentation_threshold_milli=250,
            maximum_source_pages=3,
            evacuation_headroom_pages=2,
        ),
    )


def _configurator(backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        server_args=SimpleNamespace(
            get_attention_backends=lambda: (backend, backend)
        )
    )


def test_naive_rejects_page_granular_backend_for_sparse_retained_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation, "_config", lambda: _config("naive"))

    with pytest.raises(
        RuntimeError,
        match=(
            "page-granular attention backend.*cannot represent "
            "retained_per_page=8 within page_tokens=16"
        ),
    ):
        validation._validate_token_reclamation_backend_contract(
            _configurator("fa3")
        )


def test_runtime_preflight_rejects_incompatible_naive_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(validation, "_config", lambda: _config("naive"))
    monkeypatch.setattr(
        validation, "_validate_gdn_fixed_state_backend_contract", lambda _: None
    )
    monkeypatch.setattr(
        validation, "_validate_radix_cache_contract", lambda _: None
    )
    monkeypatch.setattr(
        validation, "_validate_attention_backend_contract", lambda _: None
    )
    configurator = _configurator("fa3")
    configurator.server_args.cuda_graph_config = SimpleNamespace()

    with pytest.raises(RuntimeError, match="cannot represent"):
        validation._validate_configurator(
            lambda *_args, **_kwargs: pytest.fail(
                "pool construction ran despite an incompatible backend"
            ),
            configurator,
        )


@pytest.mark.parametrize(
    ("backend", "mode"),
    (("flashinfer", "naive"), ("fa3", "relocate")),
)
def test_reclamation_accepts_representable_backend_mode_pairs(
    monkeypatch: pytest.MonkeyPatch, backend: str, mode: str
) -> None:
    monkeypatch.setattr(validation, "_config", lambda: _config(mode))

    validation._validate_token_reclamation_backend_contract(
        _configurator(backend)
    )
