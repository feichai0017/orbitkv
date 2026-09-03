from __future__ import annotations

from typing import Any

from orbitkv_sglang.runtime import (
    CacheSharingPolicy,
    ManagerCreateSettings,
    ManagerError,
    SessionCreateSettings,
)

from . import layouts as L
from .codec import uint


def require_session_create_settings(
    value: Any,
) -> tuple[ManagerCreateSettings, CacheSharingPolicy]:
    if type(value) is not SessionCreateSettings:
        raise ManagerError(
            "runtime session create settings must be SessionCreateSettings"
        )
    manager = value.manager
    if type(manager) is not ManagerCreateSettings:
        raise ManagerError(
            "runtime session manager settings must be ManagerCreateSettings"
        )
    policy = value.cache_sharing_policy
    if type(policy) is not CacheSharingPolicy:
        raise ManagerError(
            "runtime session cache sharing policy must be CacheSharingPolicy"
        )
    return manager, policy


def session_create_config(
    settings: ManagerCreateSettings,
    policy: CacheSharingPolicy,
    plan_format: int,
    total_pages: int,
) -> L.SessionCreateConfigLayout:
    maximum_requests = uint("maximum requests", settings.maximum_requests, 32)
    maximum_operations = uint("maximum operations", settings.maximum_operations, 32)
    maximum_reclamations = uint(
        "maximum reclamations", settings.maximum_reclamations, 32
    )
    maximum_step_tokens = uint(
        "maximum step tokens", settings.maximum_step_tokens, 32
    )
    if not maximum_requests or not maximum_operations or not maximum_step_tokens:
        raise ManagerError(
            "session request, operation, and step-token bounds must be positive"
        )
    if maximum_reclamations < total_pages:
        raise ManagerError(
            "maximum_reclamations must cover all physical pages"
        )
    manager = L.ManagerConfigLayout(
        maximum_requests,
        maximum_operations,
        uint("maximum prefixes", settings.maximum_prefixes, 32),
        maximum_reclamations,
        maximum_step_tokens,
        uint("manager plan format", plan_format, 32),
        0,
    )
    return L.SessionCreateConfigLayout(
        manager, uint("cache sharing policy", int(policy), 32), 0
    )


__all__ = ["require_session_create_settings", "session_create_config"]
