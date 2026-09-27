from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from tests.support import unit_stubs


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


def test_noop_connector_is_test_only_and_importable() -> None:
    unit_stubs.install_connector_unit_stubs()
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from tests.support.noop_vllm_connector import NoopKVConnector

    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(kv_lora_rank=None)),
    )
    connector = NoopKVConnector(
        config,
        KVConnectorRole.WORKER,
        SimpleNamespace(kv_cache_groups=()),
    )

    assert connector.get_num_new_matched_tokens(None, 0) == (0, False)
