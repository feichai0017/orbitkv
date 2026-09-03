from __future__ import annotations

from types import SimpleNamespace

import pytest

from orbitkv_sglang.bridge import location_validation


@pytest.mark.parametrize(
    ("raw", "expected"),
    ((None, False), ("0", False), ("false", False), ("1", True), ("true", True)),
)
def test_diagnostic_environment_is_strict_boolean(
    raw: str | None, expected: bool
) -> None:
    environ = {} if raw is None else {
        location_validation.PHYSICAL_LOCATION_VALIDATION_ENV: raw
    }
    assert location_validation.physical_location_validation_enabled(environ) is expected


def test_diagnostic_environment_rejects_invalid_value() -> None:
    with pytest.raises(RuntimeError, match="must be 0/1/false/true"):
        location_validation.physical_location_validation_enabled(
            {location_validation.PHYSICAL_LOCATION_VALIDATION_ENV: "yes"}
        )


def test_default_validation_never_reads_tensor_contents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    class HostReadForbiddenTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, value):
            return torch.Tensor._make_subclass(cls, value, False)

        def cpu(self, *_args, **_kwargs):
            raise AssertionError("location contents must not be read on CPU")

        def tolist(self):
            raise AssertionError("location contents must not be materialized")

        def item(self):
            raise AssertionError("location scalars must not be materialized")

        def to(self, *args, **kwargs):
            destination = kwargs.get("device", args[0] if args else None)
            if destination is not None and str(destination) == "cpu":
                raise AssertionError("location contents must not be copied to CPU")
            return super().to(*args, **kwargs)

    spec = SimpleNamespace(
        previous_layout_boundary=4,
        target_layout_boundary=5,
        last_location=35,
        exact_new_pages=(),
    )
    lowered = SimpleNamespace(steps=(SimpleNamespace(by_class={0: spec}),))
    value = HostReadForbiddenTensor(torch.tensor([36], dtype=torch.int64))
    monkeypatch.setattr(location_validation, "VALIDATE_PHYSICAL_LOCATIONS", False)

    location_validation.validate_locations({0: value}, lowered, (0,), 16)
