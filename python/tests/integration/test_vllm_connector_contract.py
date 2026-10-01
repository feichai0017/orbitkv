"""Released vLLM consumers of OrbitKV completion and delivery contracts.

Run when upgrading vLLM or changing connector completion/composition callbacks.
Requires official vLLM 0.30.0. Model composition uses test_vllm_native_pd_e2e.py.
"""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration


def test_native_multi_connector_keeps_cache_best_effort_and_handoff_reliable():
    pytest.importorskip("vllm")
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlConnector,
    )

    from orbitkv.vllm.connector import OrbitKVConnector

    cache = OrbitKVConnector.__new__(OrbitKVConnector)
    cache._kv_transfer_config = SimpleNamespace(is_kv_producer=True)
    prefill = NixlConnector.__new__(NixlConnector)
    prefill._kv_transfer_config = SimpleNamespace(is_kv_producer=True)
    multi = MultiConnector.__new__(MultiConnector)
    multi._connectors = [cache]
    assert not multi.requires_kv_delivery
    multi._connectors.append(prefill)
    assert multi.requires_kv_delivery
    assert not multi.supports_divergent_local_hybrid_hits
