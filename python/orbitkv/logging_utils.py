"""Logging utilities for OrbitKV connector.

This module provides logger configuration.
"""

import json
import logging
import os
import time

_TRACE_TRANSFERS = os.environ.get("ORBITKV_TRACE_TRANSFERS") == "1"


def trace_transfer(stage: str, request_id: str, **fields) -> None:
    """Opt-in, request-correlated observations; never log tokens or cache keys."""
    if _TRACE_TRANSFERS:
        get_connector_logger().info(
            "cache_timeline %s",
            json.dumps(
                {
                    "stage": stage,
                    "request_id": request_id,
                    "pid": os.getpid(),
                    "at_unix_ns": time.time_ns(),
                    "monotonic_ns": time.monotonic_ns(),
                    **fields,
                },
                separators=(",", ":"),
            ),
        )


def get_connector_logger() -> logging.Logger:
    """Get a logger for the connector module."""
    connector_logger = logging.getLogger("orbitkv.vllm")
    connector_logger.setLevel(logging.INFO)
    if not connector_logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(logging.NOTSET)
        handler.setFormatter(logging.Formatter("%(message)s"))
        connector_logger.addHandler(handler)
        connector_logger.propagate = False
    return connector_logger


__all__ = ["get_connector_logger", "trace_transfer"]
