"""Bind SGLang's native P/D control plane to OrbitKV's TENT runtime."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_ENABLE_ENV = "SGLANG_MOONCAKE_TRANSFER_ENGINE"
_TIMEOUT_ENV = "ORBITKV_SGLANG_TENT_TIMEOUT_S"
_UPSTREAM_PROBE_ENV = "SGLANG_ENABLE_FAILED_SESSION_PROBE"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class SGLangTentTransferEngine:
    """SGLang's synchronous transfer contract over the Rust TENT owner.

    SGLang continues to own its P/D bootstrap rooms and request state machine.
    This adapter owns no request state: memory registration, batch completion,
    timeout cancellation, terminal draining, and segment lifetime stay in Rust.
    """

    def __init__(
        self,
        hostname: str,
        gpu_id: int | None = None,
        ib_device: str | None = None,
    ) -> None:
        from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
            get_ib_devices_for_gpu,
        )

        from orbitkv.orbitkv import MooncakeTransferEngine

        self.hostname = hostname
        self.gpu_id = int(gpu_id or 0)
        force_tcp = os.getenv("MC_FORCE_TCP") == "1"
        resolved_nics = None if force_tcp else get_ib_devices_for_gpu(ib_device, self.gpu_id)
        self.ib_device = resolved_nics or ""
        nics = _split_nics(resolved_nics)
        self._timeout_s = _positive_float(_TIMEOUT_ENV, 30.0)
        self._engine = MooncakeTransferEngine(bind_host=hostname, nics=nics)
        self.session_id = str(self._engine.endpoint)
        logger.info(
            "SGLang P/D TENT ready: gpu=%d nics=%s endpoint=%s",
            self.gpu_id,
            nics,
            self.session_id,
        )

    def register(self, ptr: int, length: int) -> None:
        self.batch_register([ptr], [length])

    def deregister(self, ptr: int) -> None:
        self.batch_deregister([ptr])

    def batch_register(self, ptrs: list[int], lengths: list[int]) -> int:
        regions = _memory_regions(ptrs, lengths)
        if regions:
            self._engine.register_memory(regions)
        return 0

    def batch_deregister(self, ptrs: list[int]) -> int:
        addresses = [int(ptr) for ptr in ptrs]
        if addresses:
            self._engine.unregister_memory(addresses)
        return 0

    def transfer_sync(
        self,
        session_id: str,
        buffer: int,
        peer_buffer_address: int,
        length: int,
    ) -> int:
        return self.batch_transfer_sync(
            session_id,
            [buffer],
            [peer_buffer_address],
            [length],
        )

    def batch_transfer_sync(
        self,
        session_id: str,
        buffers: list[int],
        peer_buffer_addresses: list[int],
        lengths: list[int],
    ) -> int:
        try:
            slices = _transfer_slices(buffers, peer_buffer_addresses, lengths)
            if not slices:
                return 0
            transferred = self._engine.write(
                session_id,
                slices,
                timeout_s=self._timeout_s,
            )
            expected = sum(length for _, _, length in slices)
            if transferred != expected:
                logger.error(
                    "TENT reported a partial SGLang P/D batch: "
                    "endpoint=%s expected=%d transferred=%d",
                    session_id,
                    expected,
                    transferred,
                )
                self._engine.invalidate_segment(session_id)
                return -1
            return 0
        except Exception:
            self._engine.invalidate_segment(session_id)
            logger.exception("SGLang P/D TENT batch failed: endpoint=%s", session_id)
            return -1

    def get_session_id(self) -> str:
        return self.session_id

    def get_ib_device(self) -> str:
        return self.ib_device

    def nic_load_stats(self) -> list[tuple[str, int, float]]:
        return self._engine.nic_load_stats()


def register_sglang_tent_backend() -> bool:
    """Select TENT through SGLang's explicit payload-engine factory."""

    if not sglang_tent_enabled():
        return False
    if _enabled(_UPSTREAM_PROBE_ENV):
        raise RuntimeError(
            "SGLang failed-session probing requires a stable TENT peer-liveness ABI; "
            f"unset {_UPSTREAM_PROBE_ENV}"
        )

    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
        register_mooncake_transfer_engine_factory,
    )

    register_mooncake_transfer_engine_factory("orbitkv", SGLangTentTransferEngine)
    logger.info(
        "OrbitKV registered the Rust TENT payload engine for SGLang P/D; "
        "SGLang retains bootstrap and request-state ownership"
    )
    return True


def validate_pd_cache_transport(disaggregation_mode: str, transfer_backend: str) -> None:
    """Require the project-wide TENT payload policy for composed P/D caching."""

    if disaggregation_mode not in {"prefill", "decode"}:
        return
    if transfer_backend != "mooncake":
        raise ValueError(
            "OrbitKV SGLang P/D cache composition requires "
            "--disaggregation-transfer-backend mooncake"
        )
    if not sglang_tent_enabled():
        raise ValueError(
            "OrbitKV SGLang P/D cache composition requires SGLANG_MOONCAKE_TRANSFER_ENGINE=orbitkv"
        )


def sglang_tent_enabled() -> bool:
    return os.getenv(_ENABLE_ENV, "mooncake") == "orbitkv"


def _enabled(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {raw!r}")


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _split_nics(value: str | None) -> list[str]:
    if not value:
        return []
    return [nic.strip() for nic in value.split(",") if nic.strip()]


def _memory_regions(ptrs: list[int], lengths: list[int]) -> list[dict[str, int | str]]:
    if len(ptrs) != len(lengths):
        raise ValueError("SGLang P/D registration pointers and lengths must have equal size")
    regions = []
    for ptr, length in zip(ptrs, lengths, strict=True):
        address = int(ptr)
        size = int(length)
        if address <= 0 or size <= 0:
            raise ValueError("SGLang P/D memory regions require positive address and length")
        regions.append({"addr": address, "len": size, "location": "*"})
    return regions


def _transfer_slices(
    buffers: list[int],
    peer_buffer_addresses: list[int],
    lengths: list[int],
) -> list[tuple[int, int, int]]:
    if not (len(buffers) == len(peer_buffer_addresses) == len(lengths)):
        raise ValueError("SGLang P/D transfer vectors must have equal size")
    slices = []
    for local, remote, length in zip(
        buffers,
        peer_buffer_addresses,
        lengths,
        strict=True,
    ):
        local_address = int(local)
        remote_address = int(remote)
        size = int(length)
        if local_address <= 0 or remote_address <= 0 or size <= 0:
            raise ValueError("SGLang P/D transfers require positive addresses and length")
        slices.append((local_address, remote_address, size))
    return slices
