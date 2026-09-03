"""Non-synchronizing validation for physical-location tensors."""

from __future__ import annotations

import os
from collections.abc import Mapping
from numbers import Integral
from typing import Any, Sequence


PHYSICAL_LOCATION_VALIDATION_ENV = "ORBITKV_VALIDATE_PHYSICAL_LOCATIONS"


def physical_location_validation_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if environ is None else environ
    raw = source.get(PHYSICAL_LOCATION_VALIDATION_ENV)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in ("1", "true"):
        return True
    if value in ("0", "false"):
        return False
    raise RuntimeError(
        f"{PHYSICAL_LOCATION_VALIDATION_ENV} must be 0/1/false/true"
    )


# Parse once when the bridge process initializes. The production forward path
# must not repeatedly inspect or parse process environment state.
VALIDATE_PHYSICAL_LOCATIONS = physical_location_validation_enabled()


def _expected_step_locations(
    step: Any, class_id: int, page_tokens: int
) -> tuple[int, ...]:
    spec = step.by_class[class_id]
    previous = int(spec.previous_layout_boundary)
    target = int(spec.target_layout_boundary)
    remaining = target - previous
    values: list[int] = []
    if previous % page_tokens:
        tail_count = min(remaining, page_tokens - previous % page_tokens)
        values.extend(
            range(int(spec.last_location) + 1, int(spec.last_location) + 1 + tail_count)
        )
        remaining -= tail_count
    for page_id in spec.exact_new_pages:
        count = min(remaining, page_tokens)
        values.extend(
            range(int(page_id) * page_tokens, int(page_id) * page_tokens + count)
        )
        remaining -= count
    if remaining != 0:
        raise RuntimeError(
            "session plan cannot account for every lowered token location"
        )
    return tuple(values)


def validate_locations(
    locations: Any,
    lowered: Any,
    class_ids: Sequence[int],
    page_tokens: int,
    execution_context: Any | None = None,
) -> None:
    """Validate host metadata, optionally followed by diagnostic values."""

    import torch

    class_ids = tuple(class_ids)
    if not isinstance(locations, Mapping) or set(locations) != set(class_ids):
        raise RuntimeError(
            "session tensor lowering did not return every compiled KV class"
        )
    if execution_context is not None:
        execution_context.validate_current()
        expected_device = torch.device(execution_context.device)
    else:
        expected_device = None
    for class_id in class_ids:
        expected_numel = sum(
            int(step.by_class[class_id].target_layout_boundary)
            - int(step.by_class[class_id].previous_layout_boundary)
            for step in lowered.steps
        )
        value = locations[class_id]
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                "session tensor lowering returned a non-tensor location result"
            )
        if value.ndim != 1:
            raise RuntimeError(
                "session tensor lowering returned a non-vector location result"
            )
        if int(value.numel()) != expected_numel:
            raise RuntimeError(
                f"session class {class_id} tensor location cardinality "
                "differs from the plan"
            )
        try:
            torch.iinfo(value.dtype)
        except TypeError as error:
            raise RuntimeError(
                "session tensor lowering returned a non-integer location dtype"
            ) from error
        if value.dtype == torch.bool:
            raise RuntimeError(
                "session tensor lowering returned a non-integer location dtype"
            )
        if expected_device is not None:
            actual_device = value.device
            if actual_device.type != expected_device.type or (
                expected_device.index is not None
                and actual_device.index != expected_device.index
            ):
                raise RuntimeError(
                    f"session class {class_id} tensor locations are on a "
                    "foreign device"
                )
        if not VALIDATE_PHYSICAL_LOCATIONS:
            continue
        try:
            actual_values = value.detach().to(device="cpu").tolist()
        except Exception as error:
            raise RuntimeError(
                "session tensor lowering result is not readable"
            ) from error
        if any(
            isinstance(item, bool) or not isinstance(item, Integral)
            for item in actual_values
        ):
            raise RuntimeError(
                "session tensor lowering returned non-integer locations"
            )
        expected = tuple(
            location
            for step in lowered.steps
            for location in _expected_step_locations(step, class_id, page_tokens)
        )
        actual = tuple(int(item) for item in actual_values)
        if actual != expected:
            raise RuntimeError(
                f"session class {class_id} tensor locations differ from the plan"
            )


__all__ = [
    "PHYSICAL_LOCATION_VALIDATION_ENV",
    "VALIDATE_PHYSICAL_LOCATIONS",
    "physical_location_validation_enabled",
    "validate_locations",
]
