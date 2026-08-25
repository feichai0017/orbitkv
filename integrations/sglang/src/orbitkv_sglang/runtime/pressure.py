from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from functools import wraps
from typing import Any, Mapping, Sequence


PRESSURE_ENV = "ORBITKV_PRESSURE_TELEMETRY"
PRESSURE_SCHEMA = "orbitkv.runtime-pressure.v1"

_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_VALUES = frozenset(("0", "false", "no", "off"))


def pressure_enabled_from_environment(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return the explicit pressure-telemetry opt-in.

    Absence means disabled.  An invalid value is rejected instead of silently
    enabling an observer that performs native census calls on every event.
    """

    source = os.environ if environ is None else environ
    raw = source.get(PRESSURE_ENV)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{PRESSURE_ENV} must be one of "
        "1/true/yes/on or 0/false/no/off"
    )


def _nonnegative(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class PressureClassDescriptor:
    """Engine-neutral byte geometry for one independently sized arena."""

    class_id: int
    name: str
    page_tokens: int
    page_count: int
    bytes_per_token: int

    def __post_init__(self) -> None:
        _nonnegative("class id", self.class_id)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("class name must be a nonempty string")
        _positive("page tokens", self.page_tokens)
        _positive("page count", self.page_count)
        _positive("bytes per token", self.bytes_per_token)

    @property
    def page_bytes(self) -> int:
        return self.page_tokens * self.bytes_per_token


@dataclass(frozen=True, slots=True)
class ArenaPressureSample:
    """One validated protocol census projected into pressure phases."""

    class_id: int
    page_count: int
    free_pages: int
    reserved_pages: int
    writing_pages: int
    active_pages: int
    retiring_pages: int
    quarantined_pages: int
    exhausted_pages: int

    def __post_init__(self) -> None:
        _nonnegative("class id", self.class_id)
        _positive("arena page count", self.page_count)
        values = (
            self.free_pages,
            self.reserved_pages,
            self.writing_pages,
            self.active_pages,
            self.retiring_pages,
            self.quarantined_pages,
            self.exhausted_pages,
        )
        for value in values:
            _nonnegative("arena page phase", value)
        if sum(values) != self.page_count:
            raise ValueError("arena page phases do not sum to capacity")

    @property
    def consumed_pages(self) -> int:
        # Every non-free phase consumes admission capacity, including reserved
        # pages that do not contain data and exhausted pages that cannot return.
        return self.page_count - self.free_pages

    @property
    def resident_data_pages(self) -> int:
        # Retiring and quarantined data remain resident for safety.  Reserved
        # pages have no published/written data; exhausted pages are capacity
        # loss rather than a claim that useful request data remains resident.
        return (
            self.writing_pages
            + self.active_pages
            + self.retiring_pages
            + self.quarantined_pages
        )


def semantic_live_tokens_for_request(
    *,
    boundary: int,
    retention: str,
    window_tokens: int | None,
    active_kv_length: int | None = None,
) -> int:
    """Compute one request-private class's live KV token count.

    ``active_kv_length`` is the exact post-policy token view when token
    virtualization is active.  Otherwise Full retains the whole boundary and
    sliding attention retains the ``window_tokens - 1`` past tokens required
    by the next query.
    """

    boundary = _nonnegative("request boundary", boundary)
    if active_kv_length is not None:
        active = _nonnegative("active KV length", active_kv_length)
        if active > boundary:
            raise ValueError("active KV length exceeds the request boundary")
        return active
    if retention == "full":
        if window_tokens is not None:
            raise ValueError("Full retention cannot carry a window")
        return boundary
    if retention == "sliding":
        window = _positive("sliding window", window_tokens)
        return min(boundary, window - 1)
    raise ValueError(f"unsupported retention kind {retention!r}")


def retention_amplification_milli(
    resident_data_bytes: int, semantic_live_bytes: int
) -> int | None:
    """Return floor(physical resident / semantic live * 1000).

    A zero semantic denominator is reported as undefined rather than zero or
    infinity.  Retiring and quarantined bytes belong in the numerator because
    they are the observable safety cost of delayed reuse.
    """

    resident = _nonnegative("resident data bytes", resident_data_bytes)
    semantic = _nonnegative("semantic live bytes", semantic_live_bytes)
    if semantic == 0:
        return None
    return resident * 1000 // semantic


@dataclass(slots=True)
class _HighWater:
    sample_count: int = 0
    min_free_pages: int | None = None
    min_free_bytes: int | None = None
    high_water_consumed_pages: int = 0
    high_water_consumed_bytes: int = 0
    high_water_resident_data_pages: int = 0
    high_water_resident_data_bytes: int = 0
    high_water_request_reachable_unique_pages: int = 0
    high_water_request_reachable_unique_bytes: int = 0
    high_water_semantic_live_bytes: int = 0
    high_water_retention_amplification_milli: int | None = None

    def observe(
        self,
        *,
        free_pages: int,
        free_bytes: int,
        consumed_pages: int,
        consumed_bytes: int,
        resident_data_pages: int,
        resident_data_bytes: int,
        request_reachable_unique_pages: int,
        request_reachable_unique_bytes: int,
        semantic_live_bytes: int,
        retention_amplification: int | None,
    ) -> None:
        self.sample_count += 1
        self.min_free_pages = (
            free_pages
            if self.min_free_pages is None
            else min(self.min_free_pages, free_pages)
        )
        self.min_free_bytes = (
            free_bytes
            if self.min_free_bytes is None
            else min(self.min_free_bytes, free_bytes)
        )
        self.high_water_consumed_pages = max(
            self.high_water_consumed_pages, consumed_pages
        )
        self.high_water_consumed_bytes = max(
            self.high_water_consumed_bytes, consumed_bytes
        )
        self.high_water_resident_data_pages = max(
            self.high_water_resident_data_pages, resident_data_pages
        )
        self.high_water_resident_data_bytes = max(
            self.high_water_resident_data_bytes, resident_data_bytes
        )
        self.high_water_request_reachable_unique_pages = max(
            self.high_water_request_reachable_unique_pages,
            request_reachable_unique_pages,
        )
        self.high_water_request_reachable_unique_bytes = max(
            self.high_water_request_reachable_unique_bytes,
            request_reachable_unique_bytes,
        )
        self.high_water_semantic_live_bytes = max(
            self.high_water_semantic_live_bytes, semantic_live_bytes
        )
        if retention_amplification is not None:
            self.high_water_retention_amplification_milli = max(
                retention_amplification,
                self.high_water_retention_amplification_milli or 0,
            )


class PressureTelemetry:
    """Event-driven high-water aggregation with no engine dependency.

    This first schema is deliberately request-private.  Callers must reject
    Prefix sharing and request forks rather than supplying double-counted
    semantic-live bytes.
    """

    def __init__(self, classes: Sequence[PressureClassDescriptor]):
        descriptors = tuple(classes)
        if not descriptors or len({item.class_id for item in descriptors}) != len(
            descriptors
        ):
            raise ValueError("pressure classes must be nonempty and unique")
        self._classes = descriptors
        self._class_water = {item.class_id: _HighWater() for item in descriptors}
        self._global_water = _HighWater()
        self._event_counts: Counter[str] = Counter()
        self._last_event: str | None = None
        self._active_requests = 0
        self._max_active_requests = 0
        self._current_classes: dict[int, dict[str, int | None | str]] = {}
        self._current_global: dict[str, int | None] = {}

    @property
    def sample_count(self) -> int:
        return self._global_water.sample_count

    def sample(
        self,
        event: str,
        *,
        active_requests: int,
        arenas: Sequence[ArenaPressureSample],
        request_reachable_unique_pages: Mapping[int, int],
        semantic_live_tokens: Mapping[int, int],
    ) -> None:
        if not isinstance(event, str) or not event:
            raise ValueError("pressure event must be a nonempty string")
        active = _nonnegative("active requests", active_requests)
        samples = tuple(arenas)
        arena_ids = tuple(item.class_id for item in samples)
        expected_ids = tuple(item.class_id for item in self._classes)
        if arena_ids != expected_ids:
            raise ValueError("pressure arena order differs from class order")
        if set(request_reachable_unique_pages) != set(expected_ids):
            raise ValueError("request-reachable census omits or adds a class")
        if set(semantic_live_tokens) != set(expected_ids):
            raise ValueError("semantic-live census omits or adds a class")

        current: dict[int, dict[str, int | None | str]] = {}
        global_values = {
            "capacity_pages": 0,
            "free_pages": 0,
            "consumed_pages": 0,
            "resident_data_pages": 0,
            "request_reachable_unique_pages": 0,
            "capacity_bytes": 0,
            "free_bytes": 0,
            "consumed_bytes": 0,
            "resident_data_bytes": 0,
            "request_reachable_unique_bytes": 0,
            "semantic_live_tokens": 0,
            "semantic_live_bytes": 0,
        }
        for descriptor, arena in zip(self._classes, samples, strict=True):
            if arena.page_count != descriptor.page_count:
                raise ValueError("pressure arena capacity changed")
            reachable = _nonnegative(
                "request-reachable unique pages",
                request_reachable_unique_pages[descriptor.class_id],
            )
            semantic_tokens = _nonnegative(
                "semantic-live tokens", semantic_live_tokens[descriptor.class_id]
            )
            if reachable > arena.resident_data_pages:
                raise ValueError(
                    "request-reachable pages exceed protocol resident data"
                )
            if semantic_tokens > reachable * descriptor.page_tokens:
                raise ValueError(
                    "semantic-live tokens exceed request-reachable slots"
                )
            page_bytes = descriptor.page_bytes
            semantic_bytes = semantic_tokens * descriptor.bytes_per_token
            resident_bytes = arena.resident_data_pages * page_bytes
            amplification = retention_amplification_milli(
                resident_bytes, semantic_bytes
            )
            values: dict[str, int | None | str] = {
                "class_id": descriptor.class_id,
                "name": descriptor.name,
                "capacity_pages": arena.page_count,
                "free_pages": arena.free_pages,
                "consumed_pages": arena.consumed_pages,
                "resident_data_pages": arena.resident_data_pages,
                "request_reachable_unique_pages": reachable,
                "capacity_bytes": arena.page_count * page_bytes,
                "free_bytes": arena.free_pages * page_bytes,
                "consumed_bytes": arena.consumed_pages * page_bytes,
                "resident_data_bytes": resident_bytes,
                "request_reachable_unique_bytes": reachable * page_bytes,
                "semantic_live_tokens": semantic_tokens,
                "semantic_live_bytes": semantic_bytes,
                "retention_amplification_milli": amplification,
            }
            water = self._class_water[descriptor.class_id]
            water.observe(
                free_pages=arena.free_pages,
                free_bytes=arena.free_pages * page_bytes,
                consumed_pages=arena.consumed_pages,
                consumed_bytes=arena.consumed_pages * page_bytes,
                resident_data_pages=arena.resident_data_pages,
                resident_data_bytes=resident_bytes,
                request_reachable_unique_pages=reachable,
                request_reachable_unique_bytes=reachable * page_bytes,
                semantic_live_bytes=semantic_bytes,
                retention_amplification=amplification,
            )
            current[descriptor.class_id] = values
            for name in global_values:
                if name == "semantic_live_tokens":
                    global_values[name] += semantic_tokens
                elif name in values:
                    value = values[name]
                    assert isinstance(value, int)
                    global_values[name] += value

        global_amplification = retention_amplification_milli(
            global_values["resident_data_bytes"],
            global_values["semantic_live_bytes"],
        )
        self._global_water.observe(
            free_pages=global_values["free_pages"],
            free_bytes=global_values["free_bytes"],
            consumed_pages=global_values["consumed_pages"],
            consumed_bytes=global_values["consumed_bytes"],
            resident_data_pages=global_values["resident_data_pages"],
            resident_data_bytes=global_values["resident_data_bytes"],
            request_reachable_unique_pages=global_values[
                "request_reachable_unique_pages"
            ],
            request_reachable_unique_bytes=global_values[
                "request_reachable_unique_bytes"
            ],
            semantic_live_bytes=global_values["semantic_live_bytes"],
            retention_amplification=global_amplification,
        )
        global_values["retention_amplification_milli"] = global_amplification
        self._current_classes = current
        self._current_global = global_values
        self._active_requests = active
        self._max_active_requests = max(self._max_active_requests, active)
        self._event_counts[event] += 1
        self._last_event = event

    @staticmethod
    def _with_water(
        current: Mapping[str, int | None | str], water: _HighWater
    ) -> dict[str, int | None | str]:
        return {
            **current,
            "min_free_pages": water.min_free_pages,
            "min_free_bytes": water.min_free_bytes,
            "high_water_consumed_pages": water.high_water_consumed_pages,
            "high_water_consumed_bytes": water.high_water_consumed_bytes,
            "high_water_resident_data_pages": (
                water.high_water_resident_data_pages
            ),
            "high_water_resident_data_bytes": (
                water.high_water_resident_data_bytes
            ),
            "high_water_request_reachable_unique_pages": (
                water.high_water_request_reachable_unique_pages
            ),
            "high_water_request_reachable_unique_bytes": (
                water.high_water_request_reachable_unique_bytes
            ),
            "high_water_semantic_live_bytes": (
                water.high_water_semantic_live_bytes
            ),
            "high_water_retention_amplification_milli": (
                water.high_water_retention_amplification_milli
            ),
            "sample_count": water.sample_count,
        }

    def report(self) -> dict[str, object]:
        """Return a JSON-safe copy of the current values and high waters."""

        classes = [
            self._with_water(
                self._current_classes.get(
                    descriptor.class_id,
                    {
                        "class_id": descriptor.class_id,
                        "name": descriptor.name,
                    },
                ),
                self._class_water[descriptor.class_id],
            )
            for descriptor in self._classes
        ]
        return {
            "schema": PRESSURE_SCHEMA,
            "enabled": True,
            "mode": "event_driven_high_water",
            "scope": {
                "state_ownership": "request_private",
                "shared_prefix_deduplication": "unsupported_fail_closed",
                "request_fork": "unsupported_fail_closed",
                "consumed_pages": "capacity_pages - free_pages",
                "resident_data_pages": (
                    "writing_pages + active_pages + retiring_pages + "
                    "quarantined_pages"
                ),
                "retention_amplification": (
                    "resident_data_bytes / request_private_semantic_live_bytes"
                ),
                "retiring_and_quarantined_are_safety_cost": True,
                "fixed_state_bytes": "excluded",
            },
            "sample_count": self.sample_count,
            "last_event": self._last_event,
            "event_counts": dict(sorted(self._event_counts.items())),
            "active_requests": self._active_requests,
            "max_active_requests": self._max_active_requests,
            "global": self._with_water(
                self._current_global, self._global_water
            ),
            "classes": classes,
        }


class PressureRuntimeMixin:
    """Opt-in lifecycle instrumentation mixed into the host journal."""

    def _initialize_pressure(
        self, classes: Sequence[Any], *, pressure_enabled: bool | None
    ) -> None:
        enabled = (
            pressure_enabled_from_environment()
            if pressure_enabled is None
            else pressure_enabled
        )
        if not isinstance(enabled, bool):
            raise TypeError("pressure_enabled must be boolean or None")
        self._pressure = (
            PressureTelemetry(
                tuple(
                    PressureClassDescriptor(
                        class_id=class_config.class_id,
                        name=class_config.name,
                        page_tokens=self.page_tokens,
                        page_count=arena.page_count,
                        bytes_per_token=(
                            class_config.bytes_per_token_per_layer
                            * len(class_config.layers)
                        ),
                    )
                    for class_config, arena in zip(
                        classes, self.arenas, strict=True
                    )
                )
            )
            if enabled
            else None
        )
        self._pressure_checkpoint("runtime_initialized")

    _POST_EVENTS = {
        "request_acquire_batch": "request_acquired",
        "prepare_batch": "step_prepared",
        "release_batch": "request_released",
        "submit_batch": "step_submitted",
        "abort_unobserved": "step_aborted",
        "_complete_group": "step_completed",
        "mark_token_dispositions": "token_dispositions_published",
        "relocate_tokens_batch": "relocation_published",
        "acknowledge_relocation_batch": "relocation_reclaimed",
    }
    _PRIVATE_ONLY = {
        "request_fork_batch": "request fork",
        "prefix_lookup_batch": "Prefix lookup",
        "prefix_attach_batch": "Prefix attach",
        "prefix_publish_batch": "Prefix publication",
        "prefix_publish_release_batch": "Prefix publish-release",
        "prefix_evict_batch": "Prefix eviction",
        "prefix_recycle_batch": "Prefix recycling",
    }

    @staticmethod
    def _wrap_post(original: Any, event: str) -> Any:
        @wraps(original)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            if getattr(self, "_pressure", None) is None:
                return original(self, *args, **kwargs)
            output = original(self, *args, **kwargs)
            self._pressure_checkpoint(event)
            return output

        return wrapped

    @staticmethod
    def _wrap_private(original: Any, operation: str) -> Any:
        @wraps(original)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            self._pressure_require_request_private(operation)
            return original(self, *args, **kwargs)

        return wrapped

    @classmethod
    def register(cls, runtime_class: type) -> type:
        """Decorate the journal class while preserving its public identity."""

        original_init = runtime_class.__init__

        def initialized(
            self: Any, config: Any, manager: Any, *,
            pressure_enabled: bool | None = None,
        ) -> None:
            original_init(self, config, manager)
            cls._initialize_pressure(
                self, self._pressure_options, pressure_enabled=pressure_enabled
            )
            del self._pressure_options

        runtime_class.__init__ = initialized
        for name, event in cls._POST_EVENTS.items():
            setattr(runtime_class, name, cls._wrap_post(getattr(runtime_class, name), event))
        for name, operation in cls._PRIVATE_ONLY.items():
            setattr(
                runtime_class, name,
                cls._wrap_private(getattr(runtime_class, name), operation),
            )
        return runtime_class


pressure_instrumented = PressureRuntimeMixin.register


def disabled_pressure_report() -> dict[str, object]:
    """Return the stable readback used when event sampling is disabled."""

    return {
        "schema": PRESSURE_SCHEMA,
        "enabled": False,
        "mode": "event_driven_high_water",
        "sample_count": 0,
    }


__all__ = [
    "ArenaPressureSample",
    "PRESSURE_ENV",
    "PRESSURE_SCHEMA",
    "PressureClassDescriptor",
    "PressureRuntimeMixin",
    "PressureTelemetry",
    "disabled_pressure_report",
    "pressure_enabled_from_environment",
    "pressure_instrumented",
    "retention_amplification_milli",
    "semantic_live_tokens_for_request",
]
