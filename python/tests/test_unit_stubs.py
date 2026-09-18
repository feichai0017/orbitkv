from __future__ import annotations

import sys
import types

from . import unit_stubs


def test_connector_unit_stubs_do_not_replace_existing_modules(monkeypatch):
    torch = types.ModuleType("torch")
    vllm = types.ModuleType("vllm")
    extension = types.ModuleType("orbitkv.orbitkv")

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "orbitkv.orbitkv", extension)

    unit_stubs.install_connector_unit_stubs()

    assert sys.modules["torch"] is torch
    assert sys.modules["vllm"] is vllm
    assert sys.modules["orbitkv.orbitkv"] is extension


def test_connector_unit_stubs_do_not_shadow_importable_modules(monkeypatch):
    importable = {"torch", "vllm", "orbitkv.orbitkv"}
    for name in list(sys.modules):
        if name in importable or name.startswith("torch.") or name.startswith("vllm."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, "orbitkv.orbitkv", raising=False)

    def fake_find_spec(name: str):
        if name in importable:
            return object()
        return None

    monkeypatch.setattr(unit_stubs.importlib_util, "find_spec", fake_find_spec)

    unit_stubs.install_connector_unit_stubs()

    assert "torch" not in sys.modules
    assert "vllm" not in sys.modules
    assert "orbitkv.orbitkv" not in sys.modules
