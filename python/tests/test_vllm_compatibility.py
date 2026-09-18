"""Compatibility tests for the historical vLLM connector import path."""

from __future__ import annotations

import importlib

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from orbitkv import connector, vllm  # noqa: E402


def test_legacy_connector_exports_canonical_connector_classes():
    assert connector.OrbitKVConnector is vllm.OrbitKVConnector
    assert connector.NoopKVConnector is vllm.NoopKVConnector
    assert connector.KVConnectorRole is vllm.KVConnectorRole
    assert connector.__all__ == vllm.__all__


def test_legacy_connector_submodules_alias_canonical_modules():
    for module_name in (
        "common",
        "connector_metrics",
        "scheduler",
        "state_manager",
        "tp_shards",
        "worker",
    ):
        legacy = importlib.import_module(f"orbitkv.connector.{module_name}")
        canonical = importlib.import_module(f"orbitkv.vllm.{module_name}")
        assert legacy is canonical
