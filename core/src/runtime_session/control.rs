use std::collections::BTreeSet;

use serde::Serialize;

use crate::kv_manager::{
    AttachedPrefix, CancelAttachedRequestItem, ForkedRequest, PrefixAttachItem, PrefixLease,
    PrefixLookupHint, PrefixPublishItem, PrefixSemanticKey, ReclamationCertificate,
    ReclamationReceipt, RequestForkItem, SnapshotPage,
};

use super::{
    EngineControlId, EnginePrefixId, EngineRequestId, PendingCanceledRequest, RequestPhase,
    RuntimeSession, RuntimeSessionError, SessionRequest, was_issued,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EnginePrefixLookup {
    pub key: PrefixSemanticKey,
    pub candidate: Option<EnginePrefixId>,
    pub resident_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EnginePublishedPrefix {
    pub prefix_id: EnginePrefixId,
    pub key: PrefixSemanticKey,
    pub resident_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineMaterializedRequest {
    pub request_id: EngineRequestId,
    pub view_version: crate::kv_manager::ViewVersion,
    pub boundary: u64,
    pub resident_count: u32,
    pub pages: Box<[SnapshotPage]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineMaterializationPlan {
    pub control_id: EngineControlId,
    pub requests: Box<[EngineMaterializedRequest]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EnginePendingAttachCancel {
    pub control_id: EngineControlId,
    pub request_id: EngineRequestId,
    pub prefix_id: EnginePrefixId,
    pub view_version: crate::kv_manager::ViewVersion,
    pub boundary: u64,
    pub resident_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum EnginePendingAttachCancelDisposition {
    RecyclePending,
    Finalized,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EnginePendingAttachCancelOutcome {
    pub control_id: EngineControlId,
    pub request_id: EngineRequestId,
    pub prefix_id: EnginePrefixId,
    pub view_version: crate::kv_manager::ViewVersion,
    pub boundary: u64,
    pub resident_count: u32,
    pub disposition: EnginePendingAttachCancelDisposition,
}

impl EnginePendingAttachCancelOutcome {
    const fn identity(self) -> EnginePendingAttachCancel {
        EnginePendingAttachCancel {
            control_id: self.control_id,
            request_id: self.request_id,
            prefix_id: self.prefix_id,
            view_version: self.view_version,
            boundary: self.boundary,
            resident_count: self.resident_count,
        }
    }
}

/// Engine-facing retirement facts with the manager reclamation capability
/// deliberately omitted. The session reattaches it during confirmation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EngineRetirement {
    pub page: crate::kv_manager::PageLease,
    pub class_id: u16,
    pub backend_domain: u16,
    pub logical_ordinal: u64,
    pub backend_index: u64,
    pub token_begin: u64,
    pub token_end_exclusive: u64,
    pub completion_domain: u64,
    pub completion_value: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EngineRetirementEvidence {
    pub page: crate::kv_manager::PageLease,
    pub backend_domain: u16,
    pub acknowledged: bool,
    pub backend_index: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EnginePrefixEvictionPlan {
    pub control_id: EngineControlId,
    pub prefixes: Box<[EnginePrefixId]>,
    pub retirements: Box<[EngineRetirement]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum EngineControlPlan {
    Materialization(EngineMaterializationPlan),
    PrefixEviction(EnginePrefixEvictionPlan),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineControlEvidence {
    pub control_id: EngineControlId,
    pub mirror_updates_confirmed: bool,
    pub reclamation_receipts: Box<[EngineRetirementEvidence]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[must_use = "control confirmation reports the committed operation kind"]
pub enum EngineControlOutcome {
    Materialized,
    Evicted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PrefixPhase {
    Resident,
    AttachPinned(EngineControlId),
    EvictionPrepared(EngineControlId),
    EvictionPending(EngineControlId),
    Quarantined,
}

impl PrefixPhase {
    const fn name(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::AttachPinned(_) => "attach control pending",
            Self::EvictionPrepared(_) => "eviction prepared",
            Self::EvictionPending(_) => "eviction confirmation pending",
            Self::Quarantined => "quarantined",
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct SessionPrefix {
    pub(super) lease: PrefixLease,
    pub(super) key: PrefixSemanticKey,
    pub(super) resident_count: u32,
    pub(super) phase: PrefixPhase,
}

#[derive(Clone, Debug)]
struct AttachReservation {
    target: EngineRequestId,
    prefix: EnginePrefixId,
    hint: EnginePrefixLookup,
}

#[derive(Clone, Copy, Debug)]
struct ForkReservation {
    source: EngineRequestId,
    target: EngineRequestId,
}

#[derive(Clone, Debug)]
enum ControlReservation {
    Attach(Box<[AttachReservation]>),
    Fork(Box<[ForkReservation]>),
    Evict(Box<[EnginePrefixId]>),
}

#[derive(Clone, Debug)]
enum CommittedControlKind {
    Attach {
        items: Box<[(EngineRequestId, EnginePrefixId)]>,
    },
    Fork {
        sources: Box<[EngineRequestId]>,
        targets: Box<[EngineRequestId]>,
    },
    Eviction {
        prefixes: Box<[EnginePrefixId]>,
        leases: Box<[PrefixLease]>,
        retirements: Box<[ReclamationCertificate]>,
    },
}

#[derive(Clone, Debug)]
struct CommittedControl {
    plan: EngineControlPlan,
    kind: CommittedControlKind,
}

#[derive(Clone, Debug)]
pub(super) struct PendingControl(PendingControlState);

#[derive(Clone, Debug)]
enum PendingControlState {
    Prepared(ControlReservation),
    Committed(CommittedControl),
}

impl RuntimeSession {
    /// Resolves semantic keys to session-scoped, non-owning prefix hints.
    ///
    /// # Errors
    ///
    /// Rejects an empty lookup through the canonical manager, propagates
    /// manager lookup failures, and fail-stops the session if manager output
    /// disagrees with its private prefix index.
    pub fn lookup_prefix_batch(
        &mut self,
        keys: &[PrefixSemanticKey],
    ) -> Result<Box<[EnginePrefixLookup]>, RuntimeSessionError> {
        self.ensure_healthy()?;
        self.ensure_prefix_operations_supported()?;
        let hints = self.manager.lookup_prefix_batch(keys)?;
        if hints.len() != keys.len() {
            return Err(self.poison("prefix lookup result cardinality"));
        }
        let mut output = Vec::with_capacity(hints.len());
        for (&key, hint) in keys.iter().zip(hints.iter()) {
            if hint.key != key {
                return Err(self.poison("prefix lookup result ordering"));
            }
            let candidate = if let Some(lease) = hint.candidate {
                let Some(&prefix_id) = self.prefix_leases.get(&lease) else {
                    return Err(self.poison("prefix lookup returned unknown lease"));
                };
                let Some(record) = self.prefixes.get(&prefix_id) else {
                    return Err(self.poison("prefix lease mapping lost record"));
                };
                if record.lease != lease
                    || record.key != key
                    || record.resident_count != hint.resident_count
                    || self.prefix_index.get(&key) != Some(&prefix_id)
                    || matches!(
                        record.phase,
                        PrefixPhase::EvictionPending(_) | PrefixPhase::Quarantined
                    )
                {
                    return Err(self.poison("prefix lookup disagreed with session state"));
                }
                Some(prefix_id)
            } else {
                if self.prefix_index.contains_key(&key) {
                    return Err(self.poison("prefix lookup lost indexed prefix"));
                }
                None
            };
            output.push(EnginePrefixLookup {
                key,
                candidate,
                resident_count: hint.resident_count,
            });
        }
        Ok(output.into_boxed_slice())
    }

    /// Publishes ready request heads and installs opaque prefix identities.
    ///
    /// # Errors
    ///
    /// Rejects empty or duplicate input, non-ready requests, invalid semantic
    /// keys, exhausted identity space, manager failures, or inconsistent
    /// manager output.
    pub fn publish_prefix_batch(
        &mut self,
        items: &[(EngineRequestId, PrefixSemanticKey)],
    ) -> Result<Box<[EnginePublishedPrefix]>, RuntimeSessionError> {
        self.ensure_healthy()?;
        self.ensure_prefix_operations_supported()?;
        if items.is_empty() {
            return Err(RuntimeSessionError::EmptyBatch);
        }
        let mut requests = BTreeSet::new();
        let mut keys = BTreeSet::new();
        let mut records = Vec::with_capacity(items.len());
        for &(request_id, key) in items {
            if !requests.insert(request_id) {
                return Err(RuntimeSessionError::DuplicateRequest(request_id));
            }
            if !keys.insert(key) || self.prefix_index.contains_key(&key) {
                return Err(crate::kv_manager::KvManagerError::DuplicatePrefixKey.into());
            }
            records.push(self.ready_request(request_id)?.clone());
        }
        let count = u64::try_from(items.len())
            .map_err(|_| RuntimeSessionError::IdentityExhausted("prefix"))?;
        let first_sequence = self.next_prefix_sequence;
        let next_sequence = first_sequence
            .checked_add(count)
            .ok_or(RuntimeSessionError::IdentityExhausted("prefix"))?;
        let ids = (first_sequence..next_sequence)
            .map(|sequence| EnginePrefixId {
                session_epoch: self.session_epoch,
                sequence,
            })
            .collect::<Vec<_>>();
        let manager_items = records
            .iter()
            .zip(items)
            .map(|(record, (_, key))| PrefixPublishItem {
                request: record.view.request,
                expected_head: record.view.snapshot,
                key: *key,
            })
            .collect::<Vec<_>>();
        let published = self.manager.publish_prefix_batch(&manager_items)?;
        if published.len() != items.len()
            || published.iter().zip(items).any(|(item, (_, key))| {
                item.key != *key || self.prefix_leases.contains_key(&item.prefix)
            })
        {
            return Err(self.poison("prefix publication result changed"));
        }
        self.next_prefix_sequence = next_sequence;
        let mut output = Vec::with_capacity(published.len());
        for ((published, prefix_id), (_, key)) in published.iter().zip(ids).zip(items) {
            let record = SessionPrefix {
                lease: published.prefix,
                key: *key,
                resident_count: published.resident_count,
                phase: PrefixPhase::Resident,
            };
            self.prefixes.insert(prefix_id, record);
            self.prefix_leases.insert(published.prefix, prefix_id);
            self.prefix_index.insert(*key, prefix_id);
            output.push(EnginePublishedPrefix {
                prefix_id,
                key: *key,
                resident_count: published.resident_count,
            });
        }
        Ok(output.into_boxed_slice())
    }

    /// Reserves empty targets and resident prefixes without invoking manager mutation.
    ///
    /// # Errors
    ///
    /// Rejects empty or duplicate input, non-empty or non-ready targets,
    /// missing, stale, or busy prefix hints, and exhausted control identities.
    ///
    /// # Panics
    ///
    /// Panics only if private request or prefix state changes after collective
    /// preflight, which indicates an internal session invariant violation.
    pub fn prepare_prefix_attach(
        &mut self,
        items: &[(EngineRequestId, EnginePrefixLookup)],
    ) -> Result<EngineControlId, RuntimeSessionError> {
        self.ensure_healthy()?;
        self.ensure_prefix_operations_supported()?;
        if items.is_empty() {
            return Err(RuntimeSessionError::EmptyBatch);
        }
        let mut targets = BTreeSet::new();
        let mut prefixes = BTreeSet::new();
        let mut reserved = Vec::with_capacity(items.len());
        for &(target, hint) in items {
            if !targets.insert(target) {
                return Err(RuntimeSessionError::DuplicateRequest(target));
            }
            let request = self.ready_request(target)?;
            if request.view.boundary != 0 || request.view.resident_count != 0 {
                return Err(crate::kv_manager::KvManagerError::AttachRequiresEmptyRequest.into());
            }
            let prefix = hint
                .candidate
                .ok_or(crate::kv_manager::KvManagerError::PrefixMiss)?;
            let record = self.prefix(prefix)?;
            if record.phase != PrefixPhase::Resident {
                return Err(RuntimeSessionError::PrefixNotReady {
                    prefix_id: prefix,
                    state: record.phase.name(),
                });
            }
            if record.key != hint.key || record.resident_count != hint.resident_count {
                return Err(crate::kv_manager::KvManagerError::PrefixHintStale.into());
            }
            prefixes.insert(prefix);
            reserved.push(AttachReservation {
                target,
                prefix,
                hint,
            });
        }
        let control_id = self.allocate_control_id()?;
        for target in targets {
            self.requests
                .get_mut(&target)
                .expect("attach preflight retained target")
                .phase = RequestPhase::ControlTarget(control_id);
        }
        for prefix in prefixes {
            self.prefixes
                .get_mut(&prefix)
                .expect("attach preflight retained prefix")
                .phase = PrefixPhase::AttachPinned(control_id);
        }
        self.controls.insert(
            control_id,
            PendingControl(PendingControlState::Prepared(ControlReservation::Attach(
                reserved.into_boxed_slice(),
            ))),
        );
        Ok(control_id)
    }

    /// Reserves source and empty target requests without invoking manager mutation.
    ///
    /// # Errors
    ///
    /// Rejects empty or overlapping input, non-ready sources or targets,
    /// non-empty targets, and exhausted control identities.
    ///
    /// # Panics
    ///
    /// Panics only if private request state changes after collective preflight,
    /// which indicates an internal session invariant violation.
    pub fn prepare_request_fork(
        &mut self,
        items: &[(EngineRequestId, EngineRequestId)],
    ) -> Result<EngineControlId, RuntimeSessionError> {
        self.ensure_healthy()?;
        self.ensure_prefix_operations_supported()?;
        if items.is_empty() {
            return Err(RuntimeSessionError::EmptyBatch);
        }
        let sources = items
            .iter()
            .map(|(source, _)| *source)
            .collect::<BTreeSet<_>>();
        let mut targets = BTreeSet::new();
        for &(source, target) in items {
            if !targets.insert(target) || sources.contains(&target) {
                return Err(RuntimeSessionError::DuplicateRequest(target));
            }
            self.ready_request(source)?;
            let target_record = self.ready_request(target)?;
            if target_record.view.boundary != 0 || target_record.view.resident_count != 0 {
                return Err(crate::kv_manager::KvManagerError::AttachRequiresEmptyRequest.into());
            }
        }
        let control_id = self.allocate_control_id()?;
        for source in &sources {
            self.requests
                .get_mut(source)
                .expect("fork preflight retained source")
                .phase = RequestPhase::ControlSource(control_id);
        }
        for target in &targets {
            self.requests
                .get_mut(target)
                .expect("fork preflight retained target")
                .phase = RequestPhase::ControlTarget(control_id);
        }
        let reservations = items
            .iter()
            .map(|&(source, target)| ForkReservation { source, target })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.controls.insert(
            control_id,
            PendingControl(PendingControlState::Prepared(ControlReservation::Fork(
                reservations,
            ))),
        );
        Ok(control_id)
    }

    /// Reserves resident prefixes for a later semantic eviction commit.
    ///
    /// # Errors
    ///
    /// Rejects empty or duplicate input, foreign, stale, or non-resident
    /// prefixes, and exhausted control identities.
    ///
    /// # Panics
    ///
    /// Panics only if private prefix state changes after collective preflight,
    /// which indicates an internal session invariant violation.
    pub fn prepare_prefix_evict(
        &mut self,
        prefix_ids: &[EnginePrefixId],
    ) -> Result<EngineControlId, RuntimeSessionError> {
        self.ensure_healthy()?;
        self.ensure_prefix_operations_supported()?;
        if prefix_ids.is_empty() {
            return Err(RuntimeSessionError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        for &prefix_id in prefix_ids {
            if !seen.insert(prefix_id) {
                return Err(RuntimeSessionError::DuplicatePrefix(prefix_id));
            }
            let record = self.prefix(prefix_id)?;
            if record.phase != PrefixPhase::Resident {
                return Err(RuntimeSessionError::PrefixNotReady {
                    prefix_id,
                    state: record.phase.name(),
                });
            }
        }
        let control_id = self.allocate_control_id()?;
        for prefix_id in prefix_ids {
            self.prefixes
                .get_mut(prefix_id)
                .expect("eviction preflight retained prefix")
                .phase = PrefixPhase::EvictionPrepared(control_id);
        }
        self.controls.insert(
            control_id,
            PendingControl(PendingControlState::Prepared(ControlReservation::Evict(
                prefix_ids.to_vec().into_boxed_slice(),
            ))),
        );
        Ok(control_id)
    }

    /// Commits a prepared control exactly once and replays its stored plan.
    ///
    /// # Errors
    ///
    /// Rejects foreign, stale, or unknown controls, propagates canonical
    /// manager errors while preserving the reservation, and fail-stops on an
    /// impossible manager output.
    pub fn commit_control(
        &mut self,
        control_id: EngineControlId,
    ) -> Result<EngineControlPlan, RuntimeSessionError> {
        self.ensure_healthy()?;
        self.ensure_prefix_operations_supported()?;
        self.ensure_control_epoch(control_id)?;
        let pending = self
            .controls
            .get(&control_id)
            .cloned()
            .ok_or_else(|| self.control_id_error(control_id))?;
        match pending.0 {
            PendingControlState::Committed(committed) => Ok(committed.plan),
            PendingControlState::Prepared(ControlReservation::Attach(items)) => {
                self.commit_attach(control_id, &items)
            }
            PendingControlState::Prepared(ControlReservation::Fork(items)) => {
                self.commit_fork(control_id, &items)
            }
            PendingControlState::Prepared(ControlReservation::Evict(prefixes)) => {
                self.commit_evict(control_id, &prefixes)
            }
        }
    }

    /// Cancels a session-only reservation. Committed controls cannot roll back.
    ///
    /// # Errors
    ///
    /// Rejects foreign, stale, unknown, or already committed controls.
    pub fn abort_control(
        &mut self,
        control_id: EngineControlId,
    ) -> Result<(), RuntimeSessionError> {
        self.ensure_healthy()?;
        self.ensure_prefix_operations_supported()?;
        self.ensure_control_epoch(control_id)?;
        let pending = self
            .controls
            .get(&control_id)
            .cloned()
            .ok_or_else(|| self.control_id_error(control_id))?;
        let PendingControlState::Prepared(reservation) = pending.0 else {
            return Err(RuntimeSessionError::ControlAlreadyCommitted(control_id));
        };
        self.release_reservation(control_id, &reservation);
        self.controls.remove(&control_id);
        Ok(())
    }

    /// Confirms post-commit mirror work and closes the control transaction.
    ///
    /// # Errors
    ///
    /// Rejects foreign, stale, unknown, or uncommitted controls, unconfirmed
    /// mirror work, and non-exact reclamation evidence. Manager ACK failures
    /// remain retryable; an unexpected post-ACK prefix recycle failure
    /// fail-stops the session.
    ///
    /// # Panics
    ///
    /// Panics only if private prefix state changes after complete preflight,
    /// which indicates an internal session invariant violation.
    pub fn confirm_control(
        &mut self,
        evidence: &EngineControlEvidence,
    ) -> Result<EngineControlOutcome, RuntimeSessionError> {
        self.ensure_healthy()?;
        self.ensure_prefix_operations_supported()?;
        self.ensure_control_epoch(evidence.control_id)?;
        let pending = self
            .controls
            .get(&evidence.control_id)
            .cloned()
            .ok_or_else(|| self.control_id_error(evidence.control_id))?;
        let PendingControlState::Committed(committed) = pending.0 else {
            return Err(RuntimeSessionError::ControlNotCommitted(
                evidence.control_id,
            ));
        };
        if !evidence.mirror_updates_confirmed {
            return Err(RuntimeSessionError::MirrorUpdatesNotConfirmed);
        }
        match committed.kind {
            CommittedControlKind::Attach { items } => {
                if !evidence.reclamation_receipts.is_empty() {
                    return Err(RuntimeSessionError::ReclamationReceiptMismatch);
                }
                let targets = items.iter().map(|item| item.0).collect::<Vec<_>>();
                let prefixes = items.iter().map(|item| item.1).collect::<Vec<_>>();
                self.finish_materialization(
                    evidence.control_id,
                    &[],
                    &targets,
                    &prefixes,
                    RequestPhase::Ready,
                )?;
                self.controls.remove(&evidence.control_id);
                Ok(EngineControlOutcome::Materialized)
            }
            CommittedControlKind::Fork { sources, targets } => {
                if !evidence.reclamation_receipts.is_empty() {
                    return Err(RuntimeSessionError::ReclamationReceiptMismatch);
                }
                self.finish_materialization(
                    evidence.control_id,
                    &sources,
                    &targets,
                    &[],
                    RequestPhase::Ready,
                )?;
                self.controls.remove(&evidence.control_id);
                Ok(EngineControlOutcome::Materialized)
            }
            CommittedControlKind::Eviction {
                prefixes,
                leases,
                retirements,
            } => {
                let receipts =
                    control_reclamation_receipts(&retirements, &evidence.reclamation_receipts)?;
                if !receipts.is_empty() {
                    self.manager.acknowledge_reclamations_batch(&receipts)?;
                }
                #[cfg(any(test, feature = "test-support"))]
                if self.test_fault == Some(super::RuntimeSessionTestFault::PrefixRecycleFatalOnce) {
                    self.test_fault = None;
                    return Err(
                        self.poison("unexpected prefix recycle failure after acknowledgement")
                    );
                }
                if let Err(_error) = self.manager.recycle_prefixes_batch(&leases) {
                    return Err(
                        self.poison("unexpected prefix recycle failure after acknowledgement")
                    );
                }
                for prefix_id in &prefixes {
                    let record = self
                        .prefixes
                        .remove(prefix_id)
                        .expect("eviction confirmation retained prefix");
                    self.prefix_leases.remove(&record.lease);
                }
                self.controls.remove(&evidence.control_id);
                Ok(EngineControlOutcome::Evicted)
            }
        }
    }

    /// Quarantines a committed control whose mirror outcome is ambiguous.
    ///
    /// # Errors
    ///
    /// Rejects foreign, stale, unknown, or uncommitted controls and fail-stops
    /// if private reservation state no longer matches the committed control.
    ///
    /// # Panics
    ///
    /// Panics only if private state changes after complete preflight, which
    /// indicates an internal session invariant violation.
    pub fn quarantine_control(
        &mut self,
        control_id: EngineControlId,
    ) -> Result<(), RuntimeSessionError> {
        self.ensure_healthy()?;
        self.ensure_prefix_operations_supported()?;
        self.ensure_control_epoch(control_id)?;
        let pending = self
            .controls
            .get(&control_id)
            .cloned()
            .ok_or_else(|| self.control_id_error(control_id))?;
        let PendingControlState::Committed(committed) = pending.0 else {
            return Err(RuntimeSessionError::ControlNotCommitted(control_id));
        };
        match committed.kind {
            CommittedControlKind::Attach { items } => {
                let targets = items.iter().map(|item| item.0).collect::<Vec<_>>();
                let prefixes = items.iter().map(|item| item.1).collect::<Vec<_>>();
                self.finish_materialization(
                    control_id,
                    &[],
                    &targets,
                    &prefixes,
                    RequestPhase::Quarantined,
                )?;
            }
            CommittedControlKind::Fork { sources, targets } => self.finish_materialization(
                control_id,
                &sources,
                &targets,
                &[],
                RequestPhase::Quarantined,
            )?,
            CommittedControlKind::Eviction { prefixes, .. } => {
                for &prefix_id in &prefixes {
                    self.control_prefix(prefix_id, PrefixPhase::EvictionPending(control_id))?;
                }
                for prefix_id in prefixes {
                    let record = self
                        .prefixes
                        .get_mut(&prefix_id)
                        .expect("eviction quarantine retained prefix");
                    record.phase = PrefixPhase::Quarantined;
                }
            }
        }
        self.controls.remove(&control_id);
        Ok(())
    }

    /// Atomically contains one committed rowless attached materialization.
    ///
    /// This is valid only for a committed prefix-attach control that owns
    /// exactly one target request, no fork sources, and exactly one attached
    /// prefix provenance. The cancel result is tombstoned by control id and
    /// replayed through [`RuntimeSession::finalize_pending_attach_cancel`].
    /// Finalized tombstones remain in a bounded replay window and may be
    /// evicted only when a later control needs their capacity.
    ///
    /// # Errors
    ///
    /// Rejects foreign controls; unknown or stale controls without a retained
    /// tombstone; uncommitted controls; and committed controls whose shape is
    /// not a singleton attached-prefix materialization target.
    ///
    /// # Panics
    ///
    /// Panics only if private request, Prefix, or control state changes after
    /// complete preflight, which indicates an internal session invariant
    /// violation.
    pub fn cancel_pending_attach(
        &mut self,
        expected: EnginePendingAttachCancel,
    ) -> Result<EnginePendingAttachCancelOutcome, RuntimeSessionError> {
        self.ensure_healthy()?;
        self.ensure_prefix_operations_supported()?;
        let control_id = expected.control_id;
        self.ensure_control_epoch(control_id)?;
        if let Some(pending) = self.canceled_requests.get(&control_id) {
            if pending.outcome.identity() != expected {
                return Err(RuntimeSessionError::PendingAttachCancelMismatch(control_id));
            }
            return Ok(pending.outcome);
        }
        let pending = self
            .controls
            .get(&control_id)
            .cloned()
            .ok_or_else(|| self.control_id_error(control_id))?;
        let PendingControlState::Committed(committed) = pending.0 else {
            return Err(RuntimeSessionError::ControlNotCommitted(control_id));
        };
        let CommittedControlKind::Attach { items } = committed.kind else {
            return Err(RuntimeSessionError::ControlNotCancelable(control_id));
        };
        let EngineControlPlan::Materialization(plan) = committed.plan else {
            return Err(self.poison("committed control kind disagreed with plan"));
        };
        if plan.control_id != control_id || plan.requests.len() != 1 || items.len() != 1 {
            return Err(RuntimeSessionError::ControlNotCancelable(control_id));
        }
        let (target_id, prefix_id) = items[0];
        let materialized = &plan.requests[0];
        let actual = EnginePendingAttachCancel {
            control_id,
            request_id: target_id,
            prefix_id,
            view_version: materialized.view_version,
            boundary: materialized.boundary,
            resident_count: materialized.resident_count,
        };
        if actual != expected {
            return Err(RuntimeSessionError::PendingAttachCancelMismatch(control_id));
        }
        if materialized.request_id != target_id {
            return Err(self.poison("committed control plan changed target ordering"));
        }
        let target_record =
            self.control_request(target_id, RequestPhase::ControlTarget(control_id))?;
        let prefix = self.control_prefix(prefix_id, PrefixPhase::AttachPinned(control_id))?;
        if target_record.view.view_version != materialized.view_version
            || target_record.view.boundary != materialized.boundary
            || target_record.view.resident_count != materialized.resident_count
            || usize::try_from(materialized.resident_count).ok() != Some(materialized.pages.len())
            || self.prefix_index.get(&prefix.key) != Some(&prefix_id)
            || self.prefix_leases.get(&prefix.lease) != Some(&prefix_id)
        {
            return Err(self.poison("committed attach provenance changed before cancel"));
        }
        self.manager
            .cancel_attached_request_batch(&[CancelAttachedRequestItem {
                request: target_record.view.request,
                expected_head: target_record.view.snapshot,
                prefix: prefix.lease,
            }])?;
        let outcome = EnginePendingAttachCancelOutcome {
            control_id,
            request_id: target_id,
            prefix_id,
            view_version: plan.requests[0].view_version,
            boundary: plan.requests[0].boundary,
            resident_count: plan.requests[0].resident_count,
            disposition: EnginePendingAttachCancelDisposition::RecyclePending,
        };
        let removed = self
            .requests
            .remove(&target_id)
            .expect("committed request cancel retained target");
        debug_assert_eq!(removed.phase, RequestPhase::ControlTarget(control_id));
        self.prefixes
            .get_mut(&prefix_id)
            .expect("committed request cancel retained prefix")
            .phase = PrefixPhase::Resident;
        self.controls.remove(&control_id);
        let previous = self.canceled_requests.insert(
            control_id,
            PendingCanceledRequest {
                outcome,
                lease: target_record.view.request,
                finalized: false,
            },
        );
        debug_assert!(previous.is_none());
        Ok(outcome)
    }

    /// Finalizes an already canceled committed request by recycling its lease.
    ///
    /// # Errors
    ///
    /// Rejects foreign controls, missing canceled tombstones, and ordinary
    /// manager recycling failures. A completed finalize retains an idempotent
    /// tombstone until session close, so a lost successful return can replay.
    ///
    /// # Panics
    ///
    /// Panics only if the retained cancellation tombstone disappears after
    /// successful request recycling, which indicates an internal session
    /// invariant violation.
    pub fn finalize_pending_attach_cancel(
        &mut self,
        expected: EnginePendingAttachCancel,
    ) -> Result<EnginePendingAttachCancelOutcome, RuntimeSessionError> {
        self.ensure_healthy()?;
        self.ensure_prefix_operations_supported()?;
        let control_id = expected.control_id;
        self.ensure_control_epoch(control_id)?;
        let pending = self
            .canceled_requests
            .get(&control_id)
            .copied()
            .ok_or(RuntimeSessionError::CanceledRequestNotPending(control_id))?;
        if pending.outcome.identity() != expected {
            return Err(RuntimeSessionError::PendingAttachCancelMismatch(control_id));
        }
        if !pending.finalized {
            self.manager.recycle_requests_batch(&[pending.lease])?;
            let retained = self
                .canceled_requests
                .get_mut(&control_id)
                .expect("canceled request tombstone survived recycle");
            retained.finalized = true;
            retained.outcome.disposition = EnginePendingAttachCancelDisposition::Finalized;
        }
        Ok(self
            .canceled_requests
            .get(&control_id)
            .expect("finalized request retained its replay tombstone")
            .outcome)
    }

    fn commit_attach(
        &mut self,
        control_id: EngineControlId,
        reservations: &[AttachReservation],
    ) -> Result<EngineControlPlan, RuntimeSessionError> {
        #[cfg(any(test, feature = "test-support"))]
        if self.test_fault == Some(super::RuntimeSessionTestFault::ControlCommitManagerOnce) {
            self.test_fault = None;
            return Err(crate::kv_manager::KvManagerError::ArenaExhausted("snapshot").into());
        }
        let mut items = Vec::with_capacity(reservations.len());
        for reservation in reservations {
            let request =
                self.control_request(reservation.target, RequestPhase::ControlTarget(control_id))?;
            let prefix =
                self.control_prefix(reservation.prefix, PrefixPhase::AttachPinned(control_id))?;
            items.push(PrefixAttachItem {
                request: request.view.request,
                expected_empty_head: request.view.snapshot,
                hint: PrefixLookupHint {
                    key: reservation.hint.key,
                    candidate: Some(prefix.lease),
                    resident_count: reservation.hint.resident_count,
                },
            });
        }
        let output = self.manager.attach_prefix_batch(&items)?;
        #[cfg(any(test, feature = "test-support"))]
        let output = self.apply_attach_test_fault(output);
        if output.len() != reservations.len() {
            return Err(self.poison("prefix attach result cardinality"));
        }
        let requests = self.accept_attach(control_id, reservations, &output)?;
        let plan = EngineControlPlan::Materialization(EngineMaterializationPlan {
            control_id,
            requests,
        });
        self.controls.insert(
            control_id,
            PendingControl(PendingControlState::Committed(CommittedControl {
                plan: plan.clone(),
                kind: CommittedControlKind::Attach {
                    items: reservations
                        .iter()
                        .map(|item| (item.target, item.prefix))
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                },
            })),
        );
        Ok(plan)
    }

    fn accept_attach(
        &mut self,
        control_id: EngineControlId,
        reservations: &[AttachReservation],
        output: &[AttachedPrefix],
    ) -> Result<Box<[EngineMaterializedRequest]>, RuntimeSessionError> {
        let targets = reservations
            .iter()
            .map(|item| item.target)
            .collect::<Vec<_>>();
        self.preflight_control_targets(&targets, control_id)?;
        let mut accepted = Vec::with_capacity(output.len());
        for (reservation, attached) in reservations.iter().zip(output) {
            let target = self
                .requests
                .get(&reservation.target)
                .expect("attach retained target");
            let expected_request = target.view.request;
            let expected_prefix = self
                .prefixes
                .get(&reservation.prefix)
                .expect("attach retained prefix")
                .lease;
            let expected_version = target
                .view
                .view_version
                .0
                .checked_add(1)
                .expect("manager rejected exhausted attach view version");
            if attached.prefix != expected_prefix
                || attached.target.view.request != expected_request
                || attached.target.view.boundary != reservation.hint.key.boundary
                || attached.target.view.resident_count != reservation.hint.resident_count
                || usize::try_from(attached.target.view.resident_count).ok()
                    != Some(attached.target.pages.len())
                || attached.target.view.view_version.0 != expected_version
                || attached.target.view.snapshot.engine_epoch != expected_request.engine_epoch
            {
                return Err(self.poison("prefix attach result changed"));
            }
            accepted.push((
                reservation.target,
                attached.target.view,
                engine_materialization(reservation.target, &attached.target),
            ));
        }
        for (target, view, _) in &accepted {
            self.requests
                .get_mut(target)
                .expect("attach retained target")
                .view = *view;
        }
        Ok(accepted
            .into_iter()
            .map(|(_, _, request)| request)
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    fn commit_fork(
        &mut self,
        control_id: EngineControlId,
        reservations: &[ForkReservation],
    ) -> Result<EngineControlPlan, RuntimeSessionError> {
        #[cfg(any(test, feature = "test-support"))]
        if self.test_fault == Some(super::RuntimeSessionTestFault::ControlCommitManagerOnce) {
            self.test_fault = None;
            return Err(crate::kv_manager::KvManagerError::ArenaExhausted("snapshot").into());
        }
        let mut items = Vec::with_capacity(reservations.len());
        for reservation in reservations {
            let source =
                self.control_request(reservation.source, RequestPhase::ControlSource(control_id))?;
            let target =
                self.control_request(reservation.target, RequestPhase::ControlTarget(control_id))?;
            items.push(RequestForkItem {
                source_request: source.view.request,
                expected_source_head: source.view.snapshot,
                target_empty_request: target.view.request,
                expected_target_head: target.view.snapshot,
            });
        }
        let output = self.manager.fork_requests_batch(&items)?;
        #[cfg(any(test, feature = "test-support"))]
        let output = self.apply_fork_test_fault(output);
        if output.len() != reservations.len() {
            return Err(self.poison("request fork result cardinality"));
        }
        let requests = self.accept_fork(control_id, reservations, &output)?;
        let sources = reservations
            .iter()
            .map(|item| item.source)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let targets = reservations
            .iter()
            .map(|item| item.target)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let plan = EngineControlPlan::Materialization(EngineMaterializationPlan {
            control_id,
            requests,
        });
        self.controls.insert(
            control_id,
            PendingControl(PendingControlState::Committed(CommittedControl {
                plan: plan.clone(),
                kind: CommittedControlKind::Fork { sources, targets },
            })),
        );
        Ok(plan)
    }

    fn accept_fork(
        &mut self,
        control_id: EngineControlId,
        reservations: &[ForkReservation],
        output: &[ForkedRequest],
    ) -> Result<Box<[EngineMaterializedRequest]>, RuntimeSessionError> {
        let targets = reservations
            .iter()
            .map(|item| item.target)
            .collect::<Vec<_>>();
        self.preflight_control_targets(&targets, control_id)?;
        let mut accepted = Vec::with_capacity(output.len());
        for (reservation, forked) in reservations.iter().zip(output) {
            let source = self
                .requests
                .get(&reservation.source)
                .expect("fork retained source");
            let target = self
                .requests
                .get(&reservation.target)
                .expect("fork retained target");
            let expected_version = target
                .view
                .view_version
                .0
                .checked_add(1)
                .expect("manager rejected exhausted fork view version");
            if forked.source != source.view.request
                || forked.target.view.request != target.view.request
                || forked.target.view.boundary != source.view.boundary
                || forked.target.view.resident_count != source.view.resident_count
                || usize::try_from(forked.target.view.resident_count).ok()
                    != Some(forked.target.pages.len())
                || forked.target.view.view_version.0 != expected_version
                || forked.target.view.snapshot.engine_epoch != target.view.request.engine_epoch
            {
                return Err(self.poison("request fork result changed"));
            }
            accepted.push((
                reservation.target,
                forked.target.view,
                engine_materialization(reservation.target, &forked.target),
            ));
        }
        for (target, view, _) in &accepted {
            self.requests
                .get_mut(target)
                .expect("fork retained target")
                .view = *view;
        }
        Ok(accepted
            .into_iter()
            .map(|(_, _, request)| request)
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    #[cfg(any(test, feature = "test-support"))]
    fn apply_attach_test_fault(
        &mut self,
        mut output: Box<[AttachedPrefix]>,
    ) -> Box<[AttachedPrefix]> {
        if self.test_fault == Some(super::RuntimeSessionTestFault::AttachSecondOutput) {
            self.test_fault = None;
            if let Some(second) = output.get_mut(1) {
                second.target.view.boundary = second.target.view.boundary.saturating_add(1);
            }
        }
        output
    }

    #[cfg(any(test, feature = "test-support"))]
    fn apply_fork_test_fault(&mut self, mut output: Box<[ForkedRequest]>) -> Box<[ForkedRequest]> {
        if self.test_fault == Some(super::RuntimeSessionTestFault::ForkSecondOutput) {
            self.test_fault = None;
            if let Some(second) = output.get_mut(1) {
                second.target.view.boundary = second.target.view.boundary.saturating_add(1);
            }
        }
        output
    }

    fn commit_evict(
        &mut self,
        control_id: EngineControlId,
        prefix_ids: &[EnginePrefixId],
    ) -> Result<EngineControlPlan, RuntimeSessionError> {
        #[cfg(any(test, feature = "test-support"))]
        if self.test_fault == Some(super::RuntimeSessionTestFault::ControlCommitManagerOnce) {
            self.test_fault = None;
            return Err(crate::kv_manager::KvManagerError::ArenaExhausted("reclamation").into());
        }
        let mut leases = Vec::with_capacity(prefix_ids.len());
        for &prefix_id in prefix_ids {
            leases.push(
                self.control_prefix(prefix_id, PrefixPhase::EvictionPrepared(control_id))?
                    .lease,
            );
        }
        let output = self.manager.evict_prefix_batch(&leases)?;
        if output.evicted.len() != prefix_ids.len()
            || output
                .evicted
                .iter()
                .zip(prefix_ids)
                .any(|(evicted, prefix_id)| {
                    let record = self
                        .prefixes
                        .get(prefix_id)
                        .expect("eviction retained prefix");
                    evicted.prefix != record.lease || evicted.key != record.key
                })
        {
            return Err(self.poison("prefix eviction result changed"));
        }
        for prefix_id in prefix_ids {
            let record = self
                .prefixes
                .get_mut(prefix_id)
                .expect("eviction retained prefix");
            self.prefix_index.remove(&record.key);
            record.phase = PrefixPhase::EvictionPending(control_id);
        }
        let plan = EngineControlPlan::PrefixEviction(EnginePrefixEvictionPlan {
            control_id,
            prefixes: prefix_ids.to_vec().into_boxed_slice(),
            retirements: output
                .retirements
                .iter()
                .map(engine_retirement)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        });
        self.controls.insert(
            control_id,
            PendingControl(PendingControlState::Committed(CommittedControl {
                plan: plan.clone(),
                kind: CommittedControlKind::Eviction {
                    prefixes: prefix_ids.to_vec().into_boxed_slice(),
                    leases: leases.into_boxed_slice(),
                    retirements: output.retirements,
                },
            })),
        );
        Ok(plan)
    }

    fn release_reservation(&mut self, control_id: EngineControlId, control: &ControlReservation) {
        match control {
            ControlReservation::Attach(items) => {
                for item in items {
                    self.requests
                        .get_mut(&item.target)
                        .expect("attach reservation retained target")
                        .phase = RequestPhase::Ready;
                }
                for prefix in items
                    .iter()
                    .map(|item| item.prefix)
                    .collect::<BTreeSet<_>>()
                {
                    let record = self
                        .prefixes
                        .get_mut(&prefix)
                        .expect("attach reservation retained prefix");
                    debug_assert_eq!(record.phase, PrefixPhase::AttachPinned(control_id));
                    record.phase = PrefixPhase::Resident;
                }
            }
            ControlReservation::Fork(items) => {
                for request_id in items
                    .iter()
                    .flat_map(|item| [item.source, item.target])
                    .collect::<BTreeSet<_>>()
                {
                    self.requests
                        .get_mut(&request_id)
                        .expect("fork reservation retained request")
                        .phase = RequestPhase::Ready;
                }
            }
            ControlReservation::Evict(prefixes) => {
                for prefix in prefixes {
                    let record = self
                        .prefixes
                        .get_mut(prefix)
                        .expect("eviction reservation retained prefix");
                    debug_assert_eq!(record.phase, PrefixPhase::EvictionPrepared(control_id));
                    record.phase = PrefixPhase::Resident;
                }
            }
        }
    }

    fn finish_materialization(
        &mut self,
        control_id: EngineControlId,
        sources: &[EngineRequestId],
        targets: &[EngineRequestId],
        prefixes: &[EnginePrefixId],
        target_phase: RequestPhase,
    ) -> Result<(), RuntimeSessionError> {
        for request_id in sources {
            let Some(record) = self.requests.get(request_id) else {
                return Err(self.poison("materialization lost source"));
            };
            if record.phase != RequestPhase::ControlSource(control_id) {
                return Err(self.poison("pending request phase changed"));
            }
        }
        self.preflight_control_targets(targets, control_id)?;
        for prefix_id in prefixes {
            self.control_prefix(*prefix_id, PrefixPhase::AttachPinned(control_id))?;
        }
        for request_id in sources {
            self.requests
                .get_mut(request_id)
                .expect("materialization retained source")
                .phase = RequestPhase::Ready;
        }
        for request_id in targets {
            self.requests
                .get_mut(request_id)
                .expect("materialization retained target")
                .phase = target_phase;
        }
        for prefix_id in prefixes {
            self.prefixes
                .get_mut(prefix_id)
                .expect("materialization retained prefix")
                .phase = PrefixPhase::Resident;
        }
        Ok(())
    }

    fn preflight_control_targets(
        &mut self,
        targets: &[EngineRequestId],
        control_id: EngineControlId,
    ) -> Result<(), RuntimeSessionError> {
        for target in targets {
            let Some(record) = self.requests.get(target) else {
                return Err(self.poison("materialization lost target"));
            };
            if record.phase != RequestPhase::ControlTarget(control_id) {
                return Err(self.poison("pending request phase changed"));
            }
        }
        Ok(())
    }

    fn control_request(
        &mut self,
        request_id: EngineRequestId,
        phase: RequestPhase,
    ) -> Result<SessionRequest, RuntimeSessionError> {
        let Some(record) = self.requests.get(&request_id) else {
            return Err(self.poison("control lost request"));
        };
        if record.phase != phase {
            return Err(self.poison("pending request phase changed"));
        }
        Ok(record.clone())
    }

    fn prefix(&self, prefix_id: EnginePrefixId) -> Result<&SessionPrefix, RuntimeSessionError> {
        self.ensure_prefix_epoch(prefix_id)?;
        self.prefixes
            .get(&prefix_id)
            .ok_or_else(|| self.prefix_id_error(prefix_id))
    }

    fn control_prefix(
        &mut self,
        prefix_id: EnginePrefixId,
        phase: PrefixPhase,
    ) -> Result<SessionPrefix, RuntimeSessionError> {
        self.ensure_prefix_epoch(prefix_id)?;
        let Some(record) = self.prefixes.get(&prefix_id) else {
            return Err(self.prefix_id_error(prefix_id));
        };
        if record.phase != phase {
            return Err(self.poison("pending prefix phase changed"));
        }
        Ok(record.clone())
    }

    fn ensure_prefix_epoch(&self, prefix_id: EnginePrefixId) -> Result<(), RuntimeSessionError> {
        if prefix_id.session_epoch != self.session_epoch {
            return Err(RuntimeSessionError::ForeignPrefix(prefix_id));
        }
        Ok(())
    }

    fn ensure_control_epoch(&self, control_id: EngineControlId) -> Result<(), RuntimeSessionError> {
        if control_id.session_epoch != self.session_epoch {
            return Err(RuntimeSessionError::ForeignControl(control_id));
        }
        Ok(())
    }

    fn prefix_id_error(&self, prefix_id: EnginePrefixId) -> RuntimeSessionError {
        if prefix_id.session_epoch != self.session_epoch {
            return RuntimeSessionError::ForeignPrefix(prefix_id);
        }
        if was_issued(prefix_id.sequence, self.next_prefix_sequence) {
            RuntimeSessionError::StalePrefix(prefix_id)
        } else {
            RuntimeSessionError::UnknownPrefix(prefix_id)
        }
    }

    fn control_id_error(&self, control_id: EngineControlId) -> RuntimeSessionError {
        if control_id.session_epoch != self.session_epoch {
            return RuntimeSessionError::ForeignControl(control_id);
        }
        if was_issued(control_id.sequence, self.next_control_sequence) {
            RuntimeSessionError::StaleControl(control_id)
        } else {
            RuntimeSessionError::UnknownControl(control_id)
        }
    }
}

fn engine_materialization(
    request_id: EngineRequestId,
    value: &crate::kv_manager::MaterializedRequestView,
) -> EngineMaterializedRequest {
    EngineMaterializedRequest {
        request_id,
        view_version: value.view.view_version,
        boundary: value.view.boundary,
        resident_count: value.view.resident_count,
        pages: value.pages.clone(),
    }
}

fn engine_retirement(value: &ReclamationCertificate) -> EngineRetirement {
    EngineRetirement {
        page: value.page,
        class_id: value.class_id,
        backend_domain: value.backend_domain,
        logical_ordinal: value.logical_ordinal,
        backend_index: value.backend_index,
        token_begin: value.token_begin,
        token_end_exclusive: value.token_end_exclusive,
        completion_domain: value.completion_domain,
        completion_value: value.completion_value,
    }
}

fn control_reclamation_receipts(
    certificates: &[ReclamationCertificate],
    evidence: &[EngineRetirementEvidence],
) -> Result<Box<[ReclamationReceipt]>, RuntimeSessionError> {
    if evidence.len() != certificates.len() {
        return Err(RuntimeSessionError::ReclamationReceiptMismatch);
    }
    certificates
        .iter()
        .zip(evidence)
        .map(|(certificate, evidence)| {
            if evidence.page != certificate.page
                || evidence.backend_domain != certificate.backend_domain
                || evidence.backend_index != certificate.backend_index
                || !evidence.acknowledged
            {
                return Err(RuntimeSessionError::ReclamationReceiptMismatch);
            }
            Ok(ReclamationReceipt {
                reclamation: certificate.reclamation,
                page: evidence.page,
                backend_domain: evidence.backend_domain,
                acknowledged: 1,
                reserved8: 0,
                reserved32: 0,
                backend_index: evidence.backend_index,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}
