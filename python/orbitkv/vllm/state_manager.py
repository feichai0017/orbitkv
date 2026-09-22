"""
Simple service state management for OrbitKV connector fault tolerance.

Current implementation scope:
- scheduler query path can bypass remote lookup while service is unavailable
- worker load/save paths do not consult this state
"""

import threading
from typing import TYPE_CHECKING

from orbitkv.logging_utils import get_connector_logger

if TYPE_CHECKING:
    from orbitkv import CacheManagerClient

logger = get_connector_logger()


class ServiceStateManager:
    """
    Simple service state manager with health checking.

    When service becomes unavailable:
    - Scheduler query bypasses remote lookup
    - Background thread checks health periodically
    - Service recovers when health check passes
    """

    def __init__(
        self,
        client: "CacheManagerClient",
        health_check_interval: float = 10.0,
    ):
        self._client = client
        self._interval = health_check_interval
        self._available = True
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def is_available(self) -> bool:
        """Check if service is available. Thread-safe."""
        with self._lock:
            return self._available

    def mark_unavailable(self, reason: str) -> None:
        """Mark service unavailable and start health check. Thread-safe."""
        with self._lock:
            if not self._available:
                # Already unavailable, skip
                return
            self._available = False
            logger.warning("[OrbitKVConnector] Service unavailable: %s", reason)

        # Start health check thread
        self._start_health_check()

    def _start_health_check(self) -> None:
        """Start background health check thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="OrbitKVHealthCheck",
        )
        self._thread.start()
        logger.debug(
            "[OrbitKVConnector] Started health check (interval=%.1fs)",
            self._interval,
        )

    def _health_check_loop(self) -> None:
        """Background thread: periodically check health until recovered."""
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._interval):
                break

            try:
                ok, _ = self._client.health()
                if ok:
                    with self._lock:
                        self._available = True
                    logger.debug("[OrbitKVConnector] Service recovered")
                    return
            except Exception as e:
                logger.debug("[OrbitKVConnector] Health check failed: %s", e)

    def shutdown(self) -> None:
        """Stop health check thread. Call on connector shutdown."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)


__all__ = ["ServiceStateManager"]
