from __future__ import annotations

from typing import Any

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

from . import layouts as L


def uint(name: str, value: int, bits: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << bits:
        raise ManagerError(f"{name} is outside uint{bits}_t")
    return value


def lease_c(layout: Any, value: Any) -> Any:
    return layout(
        uint("lease engine epoch", value.engine_epoch, 64),
        uint("lease slot", value.slot, 32),
        uint("lease generation", value.generation, 32),
    )


def lease_to_c(value: Any) -> Any:
    layouts = (
        (RequestLease, L.RequestLeaseLayout),
        (SnapshotLease, L.SnapshotLeaseLayout),
        (StepLease, L.StepLeaseLayout),
        (SubmissionLease, L.SubmissionLeaseLayout),
        (ReclamationLease, L.ReclamationLeaseLayout),
        (RelocationLease, L.RelocationLeaseLayout),
        (PrefixLease, L.PrefixLeaseLayout),
    )
    layout = next((item for kind, item in layouts if isinstance(value, kind)), None)
    if layout is None:
        raise ManagerError("value is not an ABI8 lease DTO")
    return lease_c(layout, value)


def request(value: Any) -> RequestLease:
    return RequestLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def snapshot(value: Any) -> SnapshotLease:
    return SnapshotLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def step(value: Any) -> StepLease:
    return StepLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def submission(value: Any) -> SubmissionLease:
    return SubmissionLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def reclamation(value: Any) -> ReclamationLease:
    return ReclamationLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def relocation(value: Any) -> RelocationLease:
    return RelocationLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def prefix(value: Any) -> PrefixLease:
    return PrefixLease(int(value.engine_epoch), int(value.slot), int(value.generation))


def page_to_c(value: PageLease) -> L.PageLeaseLayout:
    return L.PageLeaseLayout(
        uint("page engine epoch", value.engine_epoch, 64),
        uint("page pool epoch", value.pool_epoch, 64),
        uint("page generation", value.generation, 64),
        uint("page id", value.page_id, 32),
        uint("page pool id", value.pool_id, 32),
    )


def page(value: Any) -> PageLease:
    return PageLease(
        int(value.engine_epoch),
        int(value.pool_epoch),
        int(value.generation),
        int(value.page_id),
        int(value.pool_id),
    )


def request_view(value: Any) -> RequestView:
    if int(value.reserved) != 0:
        raise ManagerError("request view reserved field is nonzero")
    return RequestView(
        request(value.request),
        snapshot(value.snapshot),
        int(value.view_version),
        int(value.boundary),
        int(value.resident_count),
    )


def reclamation_certificate(value: Any) -> ReclamationCertificate:
    if int(value.reserved32) != 0:
        raise ManagerError("reclamation certificate reserved field is nonzero")
    return ReclamationCertificate(
        reclamation(value.reclamation),
        page(value.page),
        int(value.class_id),
        int(value.backend_domain),
        int(value.logical_ordinal),
        int(value.backend_index),
        int(value.token_begin),
        int(value.token_end_exclusive),
        int(value.completion_domain),
        int(value.completion_value),
    )
