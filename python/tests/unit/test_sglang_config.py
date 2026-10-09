"""Fixed transfer modes must be validated before native registration."""

from types import SimpleNamespace

import pytest

from orbitkv.sglang.config import (
    resolve_static_loras,
    resolve_transfer_backend,
    validate_lora_request,
)


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


@pytest.mark.parametrize(
    ("uid", "extra_key", "accepted"),
    [
        (None, None, True),
        ("static-uid", "static-uid", True),
        (None, "static-uid", False),
        ("static-uid", "userstatic-uid", False),
        ("dynamic-uid", "dynamic-uid", False),
    ],
)
def test_static_lora_request_identity_rejects_native_extra_key_alias(uid, extra_key, accepted):
    req = SimpleNamespace(lora_id=uid, extra_key=extra_key)
    static = frozenset({"static-uid"})
    if accepted:
        validate_lora_request(req, static)
    else:
        with pytest.raises(ValueError):
            validate_lora_request(req, static)


def test_static_sglang_requires_explicit_fixed_refs_and_no_session_bypass(monkeypatch):
    ref = SimpleNamespace(lora_id="fixed", lora_name="fixed", lora_path="/adapter")
    args = SimpleNamespace(enable_lora=True, lora_paths=[ref], enable_session_radix_cache=False)
    monkeypatch.delenv("ORBITKV_STATIC_LORA", raising=False)
    with pytest.raises(ValueError, match="dynamic LoRA"):
        resolve_static_loras(args, args)
    monkeypatch.setenv("ORBITKV_STATIC_LORA", "1")
    assert resolve_static_loras(args, args) == (ref,)
    args.enable_session_radix_cache = True
    with pytest.raises(ValueError, match="session"):
        resolve_static_loras(args, args)
    args.enable_session_radix_cache = False
    args.lora_paths = [ref, ref]
    with pytest.raises(ValueError, match="unique"):
        resolve_static_loras(args, args)


@pytest.mark.parametrize("uid", [None, "a", "b"])
def test_static_lora_rejects_streaming_session_before_it_can_reuse_adapter_pages(uid):
    req = SimpleNamespace(lora_id=uid, extra_key=uid, session=object())
    with pytest.raises(ValueError, match="streaming sessions"):
        validate_lora_request(req, frozenset({"a", "b"}))


def test_static_sglang_uses_resolved_lora_state_when_raw_enable_is_unspecified(monkeypatch):
    monkeypatch.setenv("ORBITKV_STATIC_LORA", "1")
    args = SimpleNamespace(enable_lora=None, lora_paths=["fixed=/adapter"])
    ref = SimpleNamespace(lora_id="resolved-uid", lora_name="fixed", lora_path="/adapter")
    resolved = SimpleNamespace(enable_lora=True, lora_paths=[ref])
    assert resolve_static_loras(args, resolved) == (ref,)
    validate_lora_request(
        SimpleNamespace(lora_id=ref.lora_id, extra_key=ref.lora_id), frozenset({ref.lora_id})
    )
