import sys
from types import ModuleType, SimpleNamespace

import pytest

from orbitkv.sglang.pd import (
    SGLangTentTransferEngine,
    install_sglang_tent_backend,
    validate_pd_cache_transport,
)


@pytest.fixture(autouse=True)
def _clear_sglang_pd_environment(monkeypatch):
    for name in (
        "MC_FORCE_TCP",
        "ORBITKV_SGLANG_TENT",
        "ORBITKV_SGLANG_TENT_TIMEOUT_S",
        "SGLANG_ENABLE_FAILED_SESSION_PROBE",
    ):
        monkeypatch.delenv(name, raising=False)


class _NativeTent:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.endpoint = "10.0.0.1:15123"
        self.registered = []
        self.unregistered = []
        self.writes = []
        self.invalidated = []
        self.stats = [("mlx5_0", 4096, 12.5)]
        self.partial = False
        self.fail = False

    def register_memory(self, regions):
        self.registered.append(regions)

    def unregister_memory(self, addresses):
        self.unregistered.append(addresses)

    def write(self, endpoint, slices, timeout_s):
        self.writes.append((endpoint, slices, timeout_s))
        if self.fail:
            raise RuntimeError("transfer failed")
        total = sum(length for _, _, length in slices)
        return total - 1 if self.partial else total

    def invalidate_segment(self, endpoint):
        self.invalidated.append(endpoint)

    def nic_load_stats(self):
        return self.stats


def _install_modules(monkeypatch, *, active=None):
    native = ModuleType("orbitkv.orbitkv")
    native.MooncakeTransferEngine = _NativeTent
    monkeypatch.setitem(sys.modules, "orbitkv.orbitkv", native)

    engine_module = ModuleType(
        "sglang.srt.distributed.device_communicators.mooncake_transfer_engine"
    )
    engine_module.MooncakeTransferEngine = object
    engine_module.get_mooncake_transfer_engine = lambda: active
    engine_module.get_ib_devices_for_gpu = lambda value, _gpu_id: value

    device_communicators = ModuleType("sglang.srt.distributed.device_communicators")
    device_communicators.mooncake_transfer_engine = engine_module
    distributed = ModuleType("sglang.srt.distributed")
    distributed.device_communicators = device_communicators
    srt = ModuleType("sglang.srt")
    srt.distributed = distributed
    sglang = ModuleType("sglang")
    sglang.srt = srt

    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.distributed.device_communicators",
        device_communicators,
    )
    monkeypatch.setitem(sys.modules, engine_module.__name__, engine_module)
    return engine_module


def test_install_is_explicit_and_precedes_engine_init(monkeypatch):
    engine_module = _install_modules(monkeypatch)
    monkeypatch.delenv("ORBITKV_SGLANG_TENT", raising=False)

    assert not install_sglang_tent_backend()
    assert engine_module.MooncakeTransferEngine is object

    monkeypatch.setenv("ORBITKV_SGLANG_TENT", "1")
    assert install_sglang_tent_backend()
    assert engine_module.MooncakeTransferEngine is SGLangTentTransferEngine


def test_install_rejects_mixed_runtime(monkeypatch):
    _install_modules(monkeypatch, active=SimpleNamespace())
    monkeypatch.setenv("ORBITKV_SGLANG_TENT", "true")

    with pytest.raises(RuntimeError, match="before SGLang initializes"):
        install_sglang_tent_backend()


def test_install_rejects_invalid_toggle(monkeypatch):
    _install_modules(monkeypatch)
    monkeypatch.setenv("ORBITKV_SGLANG_TENT", "sometimes")

    with pytest.raises(ValueError, match="ORBITKV_SGLANG_TENT must be one of"):
        install_sglang_tent_backend()


def test_install_rejects_unavailable_peer_probe(monkeypatch):
    _install_modules(monkeypatch)
    monkeypatch.setenv("ORBITKV_SGLANG_TENT", "1")
    monkeypatch.setenv("SGLANG_ENABLE_FAILED_SESSION_PROBE", "1")

    with pytest.raises(RuntimeError, match="stable TENT peer-liveness ABI"):
        install_sglang_tent_backend()


def test_cache_composition_requires_tent_mooncake(monkeypatch):
    monkeypatch.delenv("ORBITKV_SGLANG_TENT", raising=False)
    validate_pd_cache_transport("null", "nixl")

    with pytest.raises(ValueError, match="ORBITKV_SGLANG_TENT=1"):
        validate_pd_cache_transport("prefill", "mooncake")

    monkeypatch.setenv("ORBITKV_SGLANG_TENT", "1")
    with pytest.raises(ValueError, match="disaggregation-transfer-backend mooncake"):
        validate_pd_cache_transport("decode", "nixl")

    validate_pd_cache_transport("prefill", "mooncake")
    validate_pd_cache_transport("decode", "mooncake")


def test_adapter_uses_rust_tent_for_registration_and_batches(monkeypatch):
    _install_modules(monkeypatch)
    monkeypatch.setenv("ORBITKV_SGLANG_TENT_TIMEOUT_S", "12.5")
    _NativeTent.last_kwargs = None

    engine = SGLangTentTransferEngine(
        hostname="10.0.0.1",
        gpu_id=3,
        ib_device="mlx5_0, mlx5_1",
    )
    assert _NativeTent.last_kwargs == {
        "bind_host": "10.0.0.1",
        "nics": ["mlx5_0", "mlx5_1"],
    }
    assert engine.get_session_id() == "10.0.0.1:15123"
    assert engine.get_ib_device() == "mlx5_0, mlx5_1"

    assert engine.batch_register([0x1000, 0x2000], [128, 256]) == 0
    assert engine._engine.registered == [
        [
            {"addr": 0x1000, "len": 128, "location": "*"},
            {"addr": 0x2000, "len": 256, "location": "*"},
        ]
    ]
    assert (
        engine.batch_transfer_sync(
            "10.0.0.2:15124",
            [0x1000, 0x2000],
            [0x3000, 0x4000],
            [128, 256],
        )
        == 0
    )
    assert engine._engine.writes == [
        (
            "10.0.0.2:15124",
            [(0x1000, 0x3000, 128), (0x2000, 0x4000, 256)],
            12.5,
        )
    ]
    assert engine.batch_deregister([0x1000, 0x2000]) == 0
    assert engine._engine.unregistered == [[0x1000, 0x2000]]
    assert engine.nic_load_stats() == [("mlx5_0", 4096, 12.5)]


def test_adapter_forces_tcp_without_nic_filter(monkeypatch):
    _install_modules(monkeypatch)
    monkeypatch.setenv("MC_FORCE_TCP", "1")
    _NativeTent.last_kwargs = None

    SGLangTentTransferEngine("127.0.0.1", gpu_id=0, ib_device="mlx5_0")

    assert _NativeTent.last_kwargs == {"bind_host": "127.0.0.1", "nics": []}


@pytest.mark.parametrize("failure", ["partial", "exception", "invalid_vectors"])
def test_adapter_fails_closed_and_invalidates_peer(monkeypatch, failure):
    _install_modules(monkeypatch)
    engine = SGLangTentTransferEngine("10.0.0.1")
    if failure == "partial":
        engine._engine.partial = True
        buffers = [0x1000]
        remotes = [0x2000]
    elif failure == "exception":
        engine._engine.fail = True
        buffers = [0x1000]
        remotes = [0x2000]
    else:
        buffers = [0x1000, 0x2000]
        remotes = [0x3000]

    assert engine.batch_transfer_sync("10.0.0.2:15124", buffers, remotes, [64]) == -1
    assert engine._engine.invalidated == ["10.0.0.2:15124"]


def test_registration_rejects_shape_and_zero_values(monkeypatch):
    _install_modules(monkeypatch)
    engine = SGLangTentTransferEngine("10.0.0.1")

    with pytest.raises(ValueError, match="equal size"):
        engine.batch_register([0x1000], [])
    with pytest.raises(ValueError, match="positive address and length"):
        engine.batch_register([0], [64])
