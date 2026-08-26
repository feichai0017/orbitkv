from __future__ import annotations

import pytest

from orbitkv_sglang.ffi import layouts as L
from orbitkv_sglang.ffi import conversions
from orbitkv_sglang.runtime import (
    ManagerError,
    PageLease,
    PrefixLease,
    ReclamationCertificate,
    ReclamationLease,
    RelocationLease,
    RequestLease,
    RequestView,
    SnapshotLease,
    StepLease,
    SubmissionLease,
)


@pytest.mark.parametrize(
    ("lease", "layout", "decode"),
    (
        (RequestLease(7, 11, 13), L.RequestLeaseLayout, conversions.request),
        (SnapshotLease(7, 11, 13), L.SnapshotLeaseLayout, conversions.snapshot),
        (StepLease(7, 11, 13), L.StepLeaseLayout, conversions.step),
        (SubmissionLease(7, 11, 13), L.SubmissionLeaseLayout, conversions.submission),
        (ReclamationLease(7, 11, 13), L.ReclamationLeaseLayout, conversions.reclamation),
        (RelocationLease(7, 11, 13), L.RelocationLeaseLayout, conversions.relocation),
        (PrefixLease(7, 11, 13), L.PrefixLeaseLayout, conversions.prefix),
    ),
)
def test_shared_lease_conversion_preserves_type_and_value(
    lease: object, layout: type[object], decode: object
) -> None:
    encoded = conversions.lease_to_c(lease)

    assert type(encoded) is layout
    assert (encoded.engine_epoch, encoded.slot, encoded.generation) == (7, 11, 13)
    assert decode(encoded) == lease


@pytest.mark.parametrize("value", (True, -1, 1 << 16))
def test_shared_uint_rejects_non_uint16_values(value: object) -> None:
    with pytest.raises(ManagerError, match="outside uint16_t"):
        conversions.uint("field", value, 16)


def test_shared_uint_accepts_exact_upper_boundary() -> None:
    assert conversions.uint("field", (1 << 16) - 1, 16) == (1 << 16) - 1


def test_shared_lease_conversion_rejects_unknown_dto() -> None:
    with pytest.raises(ManagerError, match="value is not an ABI8 lease DTO"):
        conversions.lease_to_c(object())


def test_shared_page_conversion_round_trips_exact_identity() -> None:
    lease = PageLease(7, 17, 19, 23, 29)

    encoded = conversions.page_to_c(lease)

    assert type(encoded) is L.PageLeaseLayout
    assert conversions.page(encoded) == lease


def test_shared_request_view_conversion_checks_reserved_field() -> None:
    encoded = L.RequestViewLayout(
        L.RequestLeaseLayout(7, 2, 3),
        L.SnapshotLeaseLayout(7, 5, 11),
        13,
        17,
        19,
        0,
    )
    assert conversions.request_view(encoded) == RequestView(
        RequestLease(7, 2, 3), SnapshotLease(7, 5, 11), 13, 17, 19
    )

    encoded.reserved = 1
    with pytest.raises(ManagerError, match="request view reserved field is nonzero"):
        conversions.request_view(encoded)


def test_shared_reclamation_certificate_conversion_checks_reserved_field() -> None:
    encoded = L.ReclamationCertificateLayout(
        L.ReclamationLeaseLayout(7, 2, 3),
        L.PageLeaseLayout(7, 17, 19, 23, 29),
        31,
        37,
        0,
        41,
        43,
        47,
        53,
        59,
        61,
    )
    assert conversions.reclamation_certificate(encoded) == ReclamationCertificate(
        ReclamationLease(7, 2, 3),
        PageLease(7, 17, 19, 23, 29),
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
    )

    encoded.reserved32 = 1
    with pytest.raises(
        ManagerError, match="reclamation certificate reserved field is nonzero"
    ):
        conversions.reclamation_certificate(encoded)
