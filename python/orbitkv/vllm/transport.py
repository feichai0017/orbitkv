"""TENT payloads for vLLM's native Mooncake P/D lifecycle."""

from __future__ import annotations

import os

from orbitkv.logging_utils import get_connector_logger

logger = get_connector_logger()


class TentTransferEngine:
    """Own the Rust payload engine consumed by native MooncakeConnector workers."""

    def __init__(self, *, hostname: str, protocol: str, device_name: str) -> None:
        if protocol not in {"tcp", "rdma"}:
            raise ValueError("TENT payloads require mooncake_protocol=tcp or rdma")
        force_tcp = os.getenv("MC_FORCE_TCP")
        if (protocol == "tcp" and force_tcp != "1") or (
            protocol == "rdma" and force_tcp is not None
        ):
            raise ValueError(
                "mooncake_protocol=tcp requires MC_FORCE_TCP=1; rdma requires it unset"
            )

        from orbitkv.orbitkv import MooncakeTransferEngine

        nics = [nic.strip() for nic in device_name.split(",") if nic.strip()]
        self._engine = MooncakeTransferEngine(bind_host=hostname, nics=nics)
        self.endpoint = str(self._engine.endpoint)
        logger.info("vLLM native P/D TENT ready: endpoint=%s", self.endpoint)

    def batch_register_memory(self, addresses: list[int], lengths: list[int]) -> int:
        regions = [
            {"addr": address, "len": length, "location": "*"}
            for address, length in zip(addresses, lengths, strict=True)
        ]
        if regions:
            self._engine.register_memory(regions)
        return 0

    def batch_transfer_sync_write(
        self,
        endpoint: str,
        sources: list[int],
        destinations: list[int],
        lengths: list[int],
    ) -> int:
        try:
            slices = list(zip(sources, destinations, lengths, strict=True))
            if not slices:
                return 0
            transferred = self._engine.write(endpoint, slices, timeout_s=30.0)
            expected = sum(lengths)
            if transferred != expected:
                raise RuntimeError(f"TENT transferred {transferred} of {expected} bytes")
            return 0
        except Exception:
            self._engine.invalidate_segment(endpoint)
            logger.exception("vLLM native P/D TENT write failed: endpoint=%s", endpoint)
            return -1
