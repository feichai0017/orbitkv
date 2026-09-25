"""Fixed transfer modes must be validated before native registration."""

import pytest

from orbitkv.sglang.config import resolve_transfer_backend


@pytest.mark.parametrize(
    ("configured", "expected"), [(None, "direct"), ("direct", "direct"), ("kernel", "kernel")]
)
def test_transfer_backend_preserves_default_and_explicit_modes(monkeypatch, configured, expected):
    monkeypatch.delenv("ORBITKV_TRANSFER_BACKEND", raising=False)
    if configured is not None:
        monkeypatch.setenv("ORBITKV_TRANSFER_BACKEND", configured)
    assert resolve_transfer_backend() == expected


@pytest.mark.parametrize("configured", ["", "auto", "Kernel"])
def test_transfer_backend_rejects_unknown_modes(monkeypatch, configured):
    monkeypatch.setenv("ORBITKV_TRANSFER_BACKEND", configured)
    with pytest.raises(ValueError, match="ORBITKV_TRANSFER_BACKEND must be direct or kernel"):
        resolve_transfer_backend()
