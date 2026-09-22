"""Exercise the public compiled demand API without engine or GPU dependencies."""

import pytest

pytestmark = pytest.mark.integration


def test_declared_model_rules_produce_exact_page_demand():
    from orbitkv import RecoveryContract

    contract = RecoveryContract(
        "weights/layout", 64, [(0, "attention", 0), (1, "window", 256), (2, "recurrent", 0)]
    )
    assert contract.required_ranges("weights/layout", 1024, 8192) == [
        (0, 1024, 8192),
        (1, 7936, 8192),
        (2, 8128, 8192),
    ]
    groups = [
        (0, list(range(1088, 8193, 64))),
        (1, [8000, 8064, 8128, 8192]),
        (2, [8192]),
    ]
    assert contract.restorable_boundaries("weights/layout", 1024, 8192, groups) == [8192]
    # A plan names demand even when its final checkpoint is unavailable.
    groups[2] = (2, [])
    assert contract.restorable_boundaries("weights/layout", 1024, 8192, groups) == []
    for namespace, start, end in [("other", 1024, 8192), ("weights/layout", 1025, 8192)]:
        with pytest.raises(ValueError):
            contract.required_ranges(namespace, start, end)
