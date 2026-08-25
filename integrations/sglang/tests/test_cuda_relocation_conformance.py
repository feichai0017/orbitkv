from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from relocation_conformance import (
    TOKEN_BYTES,
    assert_batch_prepare_failure_atomic,
    opaque_payload_bytes,
)

if importlib.util.find_spec("torch") is not None:
    import torch

    from relocation_conformance import run_cuda_relocation_conformance
else:
    torch = None
from test_multi_arena_ffi import ffi_library


@pytest.fixture(scope="session", autouse=True)
def _record_cuda_device_identity(record_testsuite_property: object) -> None:
    """Bind JUnit CUDA evidence to the device that executed the suite."""

    available = torch is not None and torch.cuda.is_available()
    record_testsuite_property(
        "orbitkv.cuda.available", str(available).lower()
    )
    if not available:
        return
    assert torch is not None
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    uuid = str(properties.uuid)
    if not uuid.startswith("GPU-"):
        uuid = f"GPU-{uuid}"
    record_testsuite_property("orbitkv.cuda.device_name", properties.name)
    record_testsuite_property("orbitkv.cuda.device_uuid", uuid)
    record_testsuite_property(
        "orbitkv.cuda.runtime_version", str(torch.version.cuda)
    )
    record_testsuite_property("orbitkv.torch.version", torch.__version__)


def _cycles() -> int:
    raw = os.environ.get("ORBITKV_CUDA_RELOCATION_CYCLES", "2")
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "ORBITKV_CUDA_RELOCATION_CYCLES must be at least two"
        ) from error
    if value < 2:
        raise ValueError(
            "ORBITKV_CUDA_RELOCATION_CYCLES must be at least two"
        )
    return value


def test_opaque_payload_coordinates_are_embedded_without_collisions() -> None:
    coordinates = tuple(
        (cycle, request_index, token_id)
        for cycle in range(2)
        for request_index in range(4)
        for token_id in range(65)
    )
    payloads = tuple(opaque_payload_bytes(*coordinate) for coordinate in coordinates)

    assert all(len(payload) == TOKEN_BYTES for payload in payloads)
    assert len(set(payloads)) == len(coordinates)
    assert payloads[0][:8] == b"ORBITKV1"


@pytest.mark.parametrize("batch_size", (1, 4, 32))
def test_stale_member_makes_append_and_relocation_prepare_failure_atomic(
    tmp_path: Path, ffi_library: Path, batch_size: int
) -> None:
    assert_batch_prepare_failure_atomic(
        tmp_path, ffi_library, batch_size=batch_size
    )


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="requires a real CUDA device for relocation stream/event conformance",
)
@pytest.mark.parametrize("batch_size", (1, 4, 32))
def test_real_cuda_opaque_token_relocation_conformance(
    tmp_path: Path, ffi_library: Path, batch_size: int
) -> None:
    assert torch is not None
    cycles = _cycles()
    result = run_cuda_relocation_conformance(
        tmp_path,
        ffi_library,
        batch_size=batch_size,
        cycles=cycles,
        device=torch.device("cuda", torch.cuda.current_device()),
    )

    assert TOKEN_BYTES == 257
    assert result.batch_size == batch_size
    assert result.cycles == cycles
    assert result.copied_tokens == batch_size * cycles * 24
    assert result.reused_source_pages == batch_size * cycles * 3


__all__ = ["ffi_library"]
