from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

from orbitkv_runtime import (
    AdapterCapabilities,
    ArenaRegistration,
    BackendPageAddress,
    BackendTokenAddress,
    CompletionEvidence,
    CompletionFence,
    DataPlaneOperation,
    KvDataPlaneAdapter,
    OperationContext,
    PageLease,
    ReclamationCertificate,
    ReclamationLease,
    RelocationLease,
    RequestLease,
    StepLease,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"


def _page() -> PageLease:
    return PageLease(7, 11, 3, 101, 5)


def _context(operation: DataPlaneOperation) -> OperationContext:
    transaction_type = (
        StepLease if operation is DataPlaneOperation.APPEND else RelocationLease
    )
    return OperationContext(
        RequestLease(7, 0, 1), transaction_type(7, 0, 1), operation
    )


def test_public_identity_and_address_contract_is_generation_bearing() -> None:
    page = _page()
    address = BackendTokenAddress(page, 2, 13, 9, 4)

    assert address.page_address == BackendPageAddress(page, 2, 13, 9)
    assert (
        address.page.engine_epoch,
        address.page.pool_epoch,
        address.page.pool_id,
        address.page.page_id,
        address.page.generation,
    ) == (7, 11, 5, 101, 3)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PageLease(0, 1, 1, 1, 1),
        lambda: PageLease(1, 1, 0, 1, 1),
        lambda: BackendTokenAddress(_page(), 0, 13, 9, -1),
        lambda: CompletionFence("adapter", 7, 0, 1, 1),
        lambda: ArenaRegistration(7, 11, 5, 2, 13, 0, 16, 32),
        lambda: PageLease(1 << 64, 1, 1, 1, 1),
        lambda: PageLease(1, 1, 1, 1 << 32, 1),
        lambda: BackendTokenAddress(_page(), 1 << 16, 13, 9, 0),
        lambda: CompletionFence("adapter", 7, 1, 1 << 64, 1),
        lambda: OperationContext(
            RequestLease(7, 0, 1),
            StepLease(8, 0, 1),
            DataPlaneOperation.APPEND,
        ),
    ],
)
def test_contract_values_fail_closed_on_invalid_unsigned_fields(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_reclamation_certificate_retains_exact_span_and_completion_point() -> None:
    certificate = ReclamationCertificate(
        ReclamationLease(7, 0, 2),
        _page(),
        2,
        13,
        6,
        9,
        96,
        112,
        4,
        17,
    )

    assert certificate.page_address == BackendPageAddress(_page(), 2, 13, 9)
    assert (certificate.token_begin, certificate.token_end_exclusive) == (96, 112)
    assert (certificate.completion_domain, certificate.completion_value) == (4, 17)


class _StructurallyCompleteAdapter:
    capabilities = AdapterCapabilities()

    def resolve_pages(self, addresses):
        return ()

    def resolve_tokens(self, addresses):
        return ()

    def append(self, writes, *, cow_copies=(), completion_domain=1):
        raise NotImplementedError

    def relocate(self, moves, *, completion_domain=1):
        raise NotImplementedError

    def query_completion(self, fence):
        return None

    def wait_completion(self, fence):
        return CompletionEvidence(fence, ())

    def record_completion(self, pages, *, completion_domain=1):
        return CompletionFence("fake", 1, completion_domain, 1, 1)

    def update_mirrors(
        self, page_updates, token_updates, *, after, completion_domain=1
    ):
        raise NotImplementedError

    def prepare_reuse(
        self, certificates, *, last_use, mirror_cleanup
    ):
        raise NotImplementedError

    def note_reuse_acknowledged(self, evidence):
        return None

    def poison(self, reason):
        return None


def test_protocol_is_runtime_checkable_without_engine_base_class() -> None:
    assert isinstance(_StructurallyCompleteAdapter(), KvDataPlaneAdapter)


def test_neutral_package_has_no_engine_or_torch_imports() -> None:
    forbidden = {"sglang", "torch", "cuda"}
    imported = set()
    for path in (SOURCE_ROOT / "orbitkv_runtime").glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
    assert imported.isdisjoint(forbidden)


def test_neutral_package_import_does_not_load_engine_or_torch() -> None:
    script = """
import sys
import orbitkv_runtime
assert 'torch' not in sys.modules
assert not any(name == 'sglang' or name.startswith('sglang.') for name in sys.modules)
assert orbitkv_runtime.AdapterCapabilities().cpu_arenas
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_zero_based_lease_slots_are_valid_and_abi_widths_are_enforced() -> None:
    assert ReclamationLease(7, 0, 1).slot == 0
    assert RequestLease(7, 0, 1).slot == 0
    assert StepLease(7, 0, 1).slot == 0
    assert RelocationLease(7, 0, 1).slot == 0

    with pytest.raises(ValueError, match="uint16"):
        BackendPageAddress(_page(), 1 << 16, 13, 9)
    with pytest.raises(ValueError, match="uint32"):
        ReclamationLease(7, 1 << 32, 1)
    with pytest.raises(ValueError, match="uint64"):
        CompletionFence("adapter", 7, 1, 1 << 64, 1)
