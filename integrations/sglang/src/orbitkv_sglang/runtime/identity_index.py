from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .identity import FailStopped, RequestLease, SnapshotLease, StepLease
from .records import RequestRecord


class IdentityIndexMixin:
    """Incremental ABA guards for the online request lifecycle."""

    def _initialize_identity_indexes(self) -> None:
        self._request_leases: set[RequestLease] = set()
        self._snapshot_leases: set[SnapshotLease] = set()
        self._step_leases: set[StepLease] = set()

    def _identity_index_failed(self, message: str) -> None:
        self.fail_stop(message)
        raise FailStopped(self._failure or message)

    def _require_indexed_record(self, record: RequestRecord) -> None:
        valid = record.lease in self._request_leases and record.head in self._snapshot_leases
        if record.pending is not None:
            valid = valid and (
                record.pending.prepared.target_snapshot in self._snapshot_leases
                and record.pending.prepared.step in self._step_leases
            )
        if not valid:
            self._identity_index_failed("request identity index changed unexpectedly")

    def _register_acquired(self, records: Iterable[RequestRecord]) -> None:
        values = tuple(records)
        requests = {record.lease for record in values}
        snapshots = {record.head for record in values}
        if (
            len(requests) != len(values)
            or len(snapshots) != len(values)
            or requests & self._request_leases
            or snapshots & self._snapshot_leases
        ):
            self._identity_index_failed("acquired identity index is not unique")
        self._request_leases.update(requests)
        self._snapshot_leases.update(snapshots)

    def _register_prepared(self, pending: Any) -> None:
        snapshot = pending.prepared.target_snapshot
        step = pending.prepared.step
        if snapshot in self._snapshot_leases or step in self._step_leases:
            self._identity_index_failed("prepared identity index is not unique")
        self._snapshot_leases.add(snapshot)
        self._step_leases.add(step)

    def _remove_exact(self, index: set[Any], value: Any, kind: str) -> None:
        if value not in index:
            self._identity_index_failed(f"{kind} identity disappeared before commit")
        index.remove(value)

    def _replace_heads(
        self, old: Sequence[SnapshotLease], new: Sequence[SnapshotLease]
    ) -> None:
        for snapshot in old:
            self._remove_exact(self._snapshot_leases, snapshot, "snapshot")
        self._snapshot_leases.update(new)

    def _abort_prepared_identity(self, pending: Any) -> None:
        self._remove_exact(
            self._snapshot_leases, pending.prepared.target_snapshot, "snapshot"
        )
        self._remove_exact(self._step_leases, pending.prepared.step, "step")

    def _complete_prepared_identity(self, pending: Any) -> None:
        self._remove_exact(
            self._snapshot_leases, pending.prepared.base_snapshot, "snapshot"
        )
        self._remove_exact(self._step_leases, pending.prepared.step, "step")

    def _drop_request_identity(self, record: RequestRecord) -> None:
        self._remove_exact(self._request_leases, record.lease, "request")
        self._remove_exact(self._snapshot_leases, record.head, "snapshot")

    def _identity_indexes_live(self) -> bool:
        return bool(self._request_leases or self._snapshot_leases or self._step_leases)


__all__ = ["IdentityIndexMixin"]
