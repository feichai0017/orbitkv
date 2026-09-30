"""Released vLLM consumers of OrbitKV completion and delivery contracts.

Run when upgrading vLLM or changing connector completion/composition callbacks.
Requires the pinned engine package; transport completion is controlled locally.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("composed", [False, True], ids=["standalone", "multi-connector"])
def test_native_worker_consumes_failed_receive_and_blocks_in_one_snapshot(composed):
    pytest.importorskip("vllm")
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )
    from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

    from orbitkv.vllm.pd import PdDecodeConnector
    from orbitkv.vllm.pd.decode_worker import DecodeWorker, _DecodeWaitState
    from orbitkv.vllm.pd.metadata import WaitReqMeta
    from orbitkv.vllm.pd.metrics import PdMetricsTracker

    metrics = PdMetricsTracker()
    worker = DecodeWorker.__new__(DecodeWorker)
    worker._state = _DecodeWaitState(metrics)
    worker.transfer = Mock()
    worker._state.register_wait(
        "decode",
        WaitReqMeta(
            local_block_ids=([3, 7],),
            remote_request_id="prefill",
            done_request_id="decode",
            prompt_token_ids=(1, 2),
            prefill_url="",
        ),
    )
    worker._state.mark_wait_failed("decode")
    connector = PdDecodeConnector.__new__(PdDecodeConnector)
    connector._worker = worker
    connector._metrics = metrics
    connector._connector_metadata = None
    if composed:
        multi = MultiConnector.__new__(MultiConnector)
        multi._connectors = [connector]
        multi._extra_async_saves = {}
        connector = multi

    active = ActiveKVConnector.__new__(ActiveKVConnector)
    active.kv_connector = connector
    active._disabled = False
    active._pending_load_kwargs = None
    output = active.post_forward(set(), wait_for_save=False)
    assert output.finished_recving == {"decode"}
    assert output.failed_recving == {"decode"}
    assert output.invalid_block_ids == {3, 7}
    worker.transfer.close_request.assert_called_once_with("decode")

    drained = active.post_forward(set(), wait_for_save=False)
    assert not drained.finished_recving
    assert not drained.failed_recving
    assert not drained.invalid_block_ids


def test_native_multi_connector_keeps_cache_best_effort_and_handoff_reliable():
    pytest.importorskip("vllm")
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )

    from orbitkv.vllm.connector import OrbitKVConnector
    from orbitkv.vllm.pd import PdPrefillConnector

    cache = OrbitKVConnector.__new__(OrbitKVConnector)
    cache._kv_transfer_config = SimpleNamespace(is_kv_producer=True)
    prefill = PdPrefillConnector.__new__(PdPrefillConnector)
    prefill._kv_transfer_config = SimpleNamespace(is_kv_producer=True)
    multi = MultiConnector.__new__(MultiConnector)
    multi._connectors = [cache]
    assert not multi.requires_kv_delivery
    multi._connectors.append(prefill)
    assert multi.requires_kv_delivery
    assert not multi.supports_divergent_local_hybrid_hits
