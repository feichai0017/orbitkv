"""The native P/D adapter returns success only for a complete TENT batch."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from orbitkv.vllm.transport import TentTransferEngine


@pytest.mark.parametrize("result", [12, 11, RuntimeError("transfer failed")])
def test_native_payload_batch_completion(monkeypatch, result):
    native = Mock(endpoint="tent://instance")
    native.write.return_value = result
    if isinstance(result, Exception):
        native.write.side_effect = result
    module = ModuleType("orbitkv.orbitkv")
    module.MooncakeTransferEngine = Mock(return_value=native)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("MC_FORCE_TCP", "1")
    engine = TentTransferEngine(hostname="127.0.0.1", protocol="tcp", device_name="")
    assert engine.batch_register_memory([16], [12]) == 0
    native.register_memory.assert_called_once_with([{"addr": 16, "len": 12, "location": "*"}])
    assert engine.batch_transfer_sync_write("peer", [16, 24], [32, 40], [8, 4]) == (
        0 if result == 12 else -1
    )
    native.write.assert_called_once_with("peer", [(16, 32, 8), (24, 40, 4)], timeout_s=30.0)
    if result == 12:
        native.invalidate_segment.assert_not_called()
    else:
        native.invalidate_segment.assert_called_once_with("peer")


@pytest.mark.parametrize(
    "protocol,force_tcp",
    [("tcp", "0"), ("rdma", "1"), ("rdma", "0"), ("rdma", ""), ("unknown", "0")],
)
def test_transport_configuration_is_checked_before_native_initialization(
    monkeypatch, protocol, force_tcp
):
    monkeypatch.setenv("MC_FORCE_TCP", force_tcp)
    with pytest.raises(ValueError, match="(?:MC_FORCE_TCP|mooncake_protocol)"):
        TentTransferEngine(hostname="127.0.0.1", protocol=protocol, device_name="")
