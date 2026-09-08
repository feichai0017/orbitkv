use serde::Serialize;
use std::collections::BTreeSet;
use thiserror::Error;

use crate::kv_manager::{KvManagerError, PinnedSnapshotPage, SnapshotPage, TailActionKind};

use super::{
    EngineBatchId, EngineBatchPlan, EngineBatchPublication, EngineBindEvidence,
    EngineCompletionEvidence, EngineCopyEvidence, EngineRequestId, EngineStepAbortEvidence,
    EngineStepExecutionEvidence, ExecutionEvidence, RequestPhase, RuntimeSession,
    RuntimeSessionError,
};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ExternalObjectKey {
    pub namespace: [u8; 32],
    pub digest: [u8; 32],
    pub plan_fingerprint: [u8; 32],
    pub boundary: u64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ExternalTransferId {
    session_epoch: u64,
    sequence: u64,
}

impl ExternalTransferId {
    #[must_use]
    pub const fn from_parts(session_epoch: u64, sequence: u64) -> Self {
        Self {
            session_epoch,
            sequence,
        }
    }

    #[must_use]
    pub const fn session_epoch(self) -> u64 {
        self.session_epoch
    }

    #[must_use]
    pub const fn sequence(self) -> u64 {
        self.sequence
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalReplicaTarget {
    pub storage_domain: u64,
    pub object_index: u64,
    pub base_offset: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalExportCopy {
    pub transfer_id: ExternalTransferId,
    pub copy_index: u32,
    pub class_id: u16,
    pub source_backend_domain: u16,
    pub source_backend_index: u64,
    pub destination_storage_domain: u64,
    pub destination_object_index: u64,
    pub destination_offset: u64,
    pub byte_count: u64,
    pub logical_ordinal: u64,
    pub valid_token_count: u32,
    pub visible_token_offset: u32,
    pub visible_token_count: u32,
}

// `source_backend_index` is a logical page index, never a byte address. OrbitKV
// privately binds `copy_index` to the source PageLease generation. The executor
// adapter expands this record through layer/component bindings into
// backend-specific iovecs.

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalExportPlan {
    pub transfer_id: ExternalTransferId,
    pub request_id: EngineRequestId,
    pub key: ExternalObjectKey,
    pub target: ExternalReplicaTarget,
    pub total_bytes: u64,
    pub copies: Box<[ExternalExportCopy]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalExportReceipt {
    pub copy: ExternalExportCopy,
    pub checksum: [u8; 32],
    pub copied: bool,
    pub durable: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalTransferCompletion {
    pub transfer_id: ExternalTransferId,
    pub completion_domain: u64,
    pub completion_value: u64,
    pub confirmed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalExportAbortEvidence {
    pub transfer_id: ExternalTransferId,
    pub backend_unobserved: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalReplicaPage {
    pub copy_index: u32,
    pub class_id: u16,
    pub logical_ordinal: u64,
    pub storage_offset: u64,
    pub byte_count: u64,
    pub valid_token_count: u32,
    pub visible_token_offset: u32,
    pub visible_token_count: u32,
    pub checksum: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalReplica {
    pub key: ExternalObjectKey,
    pub target: ExternalReplicaTarget,
    pub total_bytes: u64,
    pub pages: Box<[ExternalReplicaPage]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalReplicaDeletionEvidence {
    pub key: ExternalObjectKey,
    pub target: ExternalReplicaTarget,
    pub deleted: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalTierStats {
    pub pending_exports: u64,
    pub pending_restores: u64,
    pub quarantined_exports: u64,
    pub quarantined_restores: u64,
    pub replicas: u64,
    pub operation_capacity: u64,
    pub replica_capacity: u64,
    pub pinned_export_pages: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalRestoreCopy {
    pub transfer_id: ExternalTransferId,
    pub copy_index: u32,
    pub class_id: u16,
    pub source_storage_domain: u64,
    pub source_object_index: u64,
    pub source_offset: u64,
    pub destination_backend_domain: u16,
    pub destination_backend_index: u64,
    pub byte_count: u64,
    pub logical_ordinal: u64,
    pub valid_token_count: u32,
    pub visible_token_offset: u32,
    pub visible_token_count: u32,
    pub expected_checksum: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalRestorePlan {
    pub transfer_id: ExternalTransferId,
    pub request_id: EngineRequestId,
    pub key: ExternalObjectKey,
    pub total_bytes: u64,
    pub copies: Box<[ExternalRestoreCopy]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalRestoreReceipt {
    pub copy: ExternalRestoreCopy,
    pub checksum: [u8; 32],
    pub copied: bool,
    pub ordered_before_publish: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalRestoreAbortEvidence {
    pub transfer_id: ExternalTransferId,
    pub backend_unobserved: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalRestoreTicket {
    pub transfer_id: ExternalTransferId,
}

#[derive(Clone, Debug)]
pub(super) struct PendingExternalExport {
    pub request_id: EngineRequestId,
    pub plan: ExternalExportPlan,
    pub pinned: Box<[PinnedSnapshotPage]>,
    phase: ExternalExportPhase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExternalExportPhase {
    Prepared,
    Quarantined,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExternalRestorePhase {
    Prepared,
    Submitted,
    Quarantined,
}

#[derive(Clone, Debug)]
pub(super) struct PendingExternalRestore {
    request_id: EngineRequestId,
    batch_id: EngineBatchId,
    plan: ExternalRestorePlan,
    execution: ExecutionEvidence,
    phase: ExternalRestorePhase,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ExternalTierError {
    #[error(transparent)]
    Session(#[from] RuntimeSessionError),
    #[error(transparent)]
    Manager(#[from] KvManagerError),
    #[error("external object key or target is invalid")]
    InvalidDescriptor,
    #[error("external object is already published or being exported")]
    DuplicateObject,
    #[error("external export capacity is exhausted")]
    CapacityExhausted,
    #[error("unknown external transfer")]
    UnknownTransfer,
    #[error("external transfer belongs to a different runtime session")]
    ForeignTransfer,
    #[error("external transfer is stale")]
    StaleTransfer,
    #[error("external export receipts do not exactly match the copy plan")]
    ReceiptMismatch,
    #[error("external export is not in the required phase")]
    ExportPhaseMismatch,
    #[error("external transfer completion is not confirmed")]
    CompletionNotConfirmed,
    #[error("external export abort requires proof that the backend was unobserved")]
    AbortObservationUnknown,
    #[error("external replica is unknown")]
    UnknownObject,
    #[error("external replica has an active restore")]
    ObjectBusy,
    #[error("external replica deletion is not confirmed")]
    DeletionNotConfirmed,
    #[error("external transfer byte geometry overflowed")]
    ByteGeometryOverflow,
    #[error("external restore requires an empty request and exact replica geometry")]
    RestoreGeometryMismatch,
    #[error("external restore receipts do not exactly match the copy plan")]
    RestoreReceiptMismatch,
    #[error("external restore is not in the required phase")]
    RestorePhaseMismatch,
}

impl RuntimeSession {
    /// Admits public replica metadata produced by another compatible session.
    ///
    /// This is the cross-node catalog boundary. Admission validates the
    /// compiled-plan fingerprint, compact page ordering/geometry, checksums,
    /// capacity, and unique storage-object identity. It grants no local page
    /// authority and performs no I/O.
    ///
    /// # Errors
    ///
    /// Rejects malformed, incompatible, duplicate, or over-capacity metadata.
    pub fn admit_external_replica(
        &mut self,
        replica: ExternalReplica,
    ) -> Result<(), ExternalTierError> {
        self.ensure_healthy()?;
        validate_external_replica(&self.manager, &replica)?;
        if self.external_replicas.contains_key(&replica.key)
            || self
                .external_exports
                .values()
                .any(|pending| pending.plan.key == replica.key)
            || self
                .external_replicas
                .values()
                .any(|current| same_external_object(current.target, replica.target))
            || self
                .external_replicas
                .len()
                .checked_add(self.external_exports.len())
                .is_none_or(|count| count >= self.maximum_external_replicas)
        {
            return Err(ExternalTierError::DuplicateObject);
        }
        self.external_replicas.insert(replica.key, replica);
        Ok(())
    }

    /// Pins and plans one immutable request snapshot for external export.
    ///
    /// # Errors
    ///
    /// Rejects unavailable requests, duplicate/invalid objects, exhausted
    /// capacity, stale pages, or unrepresentable byte geometry.
    pub fn prepare_external_export(
        &mut self,
        request_id: EngineRequestId,
        key: ExternalObjectKey,
        target: ExternalReplicaTarget,
    ) -> Result<ExternalExportPlan, ExternalTierError> {
        self.ensure_healthy()?;
        let record = self.ready_request(request_id)?.clone();
        if key.boundary == 0
            || key.boundary != record.view.boundary
            || key.digest == [0; 32]
            || key.plan_fingerprint != self.manager.plan_fingerprint()
            || target.storage_domain == 0
            || target.object_index == 0
        {
            return Err(ExternalTierError::InvalidDescriptor);
        }
        if self.external_replicas.contains_key(&key)
            || self.external_exports.values().any(|pending| {
                pending.plan.key == key || same_external_object(pending.plan.target, target)
            })
            || self
                .external_replicas
                .values()
                .any(|replica| same_external_object(replica.target, target))
        {
            return Err(ExternalTierError::DuplicateObject);
        }
        if self
            .external_exports
            .len()
            .checked_add(self.external_restores.len())
            .is_none_or(|count| count >= self.maximum_external_operations)
            || self
                .external_replicas
                .len()
                .checked_add(self.external_exports.len())
                .is_none_or(|count| count >= self.maximum_external_replicas)
        {
            return Err(ExternalTierError::CapacityExhausted);
        }
        let next = self
            .next_external_sequence
            .checked_add(1)
            .ok_or(RuntimeSessionError::IdentityExhausted("external transfer"))?;
        let transfer_id = ExternalTransferId {
            session_epoch: self.session_epoch,
            sequence: self.next_external_sequence,
        };
        let pinned = self
            .manager
            .pin_snapshot_for_external_read(record.view.request, record.view.snapshot)?;
        let plan = match external_export_plan(transfer_id, request_id, key, target, &pinned) {
            Ok(plan) => plan,
            Err(error) => {
                self.manager.release_external_read_pins(&pinned)?;
                return Err(error);
            }
        };
        if !self.requests.contains_key(&request_id) {
            self.manager.release_external_read_pins(&pinned)?;
            return Err(self.poison("external export lost request").into());
        }
        self.next_external_sequence = next;
        let Some(request) = self.requests.get_mut(&request_id) else {
            self.manager.release_external_read_pins(&pinned)?;
            return Err(self.poison("external export lost request").into());
        };
        request.phase = RequestPhase::ExternalExportPending(transfer_id);
        self.external_exports.insert(
            transfer_id,
            PendingExternalExport {
                request_id,
                plan: plan.clone(),
                pinned,
                phase: ExternalExportPhase::Prepared,
            },
        );
        Ok(plan)
    }

    /// Publishes a durable external replica after exact per-page receipts.
    ///
    /// # Errors
    ///
    /// Rejects foreign/stale transfers, incomplete or mismatched receipts,
    /// non-advancing completion evidence, or lost internal request state.
    pub fn complete_external_export(
        &mut self,
        completion: ExternalTransferCompletion,
        receipts: &[ExternalExportReceipt],
    ) -> Result<ExternalReplica, ExternalTierError> {
        self.ensure_healthy()?;
        let pending = self.external_export(completion.transfer_id)?.clone();
        if pending.phase != ExternalExportPhase::Prepared {
            return Err(ExternalTierError::ExportPhaseMismatch);
        }
        if !completion.confirmed {
            return Err(ExternalTierError::CompletionNotConfirmed);
        }
        let replica = validate_export_receipts(&pending.plan, receipts)?;
        self.preflight_external_request(&pending)?;
        self.manager.validate_external_completion(
            completion.completion_domain,
            completion.completion_value,
        )?;
        self.manager.validate_external_read_pins(&pending.pinned)?;
        let request = self
            .requests
            .get_mut(&pending.request_id)
            .ok_or(ExternalTierError::UnknownTransfer)?;
        request.phase = RequestPhase::Ready;
        self.manager
            .commit_external_read_pin_release(&pending.pinned);
        self.manager
            .commit_external_completion(completion.completion_domain, completion.completion_value);
        self.external_exports.remove(&completion.transfer_id);
        self.external_replicas
            .insert(pending.plan.key, replica.clone());
        Ok(replica)
    }

    /// Aborts an export only when no backend read could have observed it.
    ///
    /// # Errors
    ///
    /// Rejects foreign/stale transfers, ambiguous backend observation, stale
    /// page pins, or lost internal request state.
    pub fn abort_external_export(
        &mut self,
        evidence: ExternalExportAbortEvidence,
    ) -> Result<(), ExternalTierError> {
        self.ensure_healthy()?;
        if !evidence.backend_unobserved {
            return Err(ExternalTierError::AbortObservationUnknown);
        }
        let pending = self.external_export(evidence.transfer_id)?.clone();
        if pending.phase != ExternalExportPhase::Prepared {
            return Err(ExternalTierError::ExportPhaseMismatch);
        }
        self.preflight_external_request(&pending)?;
        self.manager.validate_external_read_pins(&pending.pinned)?;
        let request = self
            .requests
            .get_mut(&pending.request_id)
            .ok_or(ExternalTierError::UnknownTransfer)?;
        request.phase = RequestPhase::Ready;
        self.manager
            .commit_external_read_pin_release(&pending.pinned);
        self.external_exports.remove(&evidence.transfer_id);
        Ok(())
    }

    /// Fail-stops an export whose backend observation cannot be established.
    ///
    /// Source pins intentionally remain held and the request becomes terminal,
    /// preventing local reuse after an ambiguous remote read.
    ///
    /// # Errors
    ///
    /// Rejects foreign/stale transfers or lost internal request state.
    pub fn quarantine_external_export(
        &mut self,
        transfer_id: ExternalTransferId,
    ) -> Result<(), ExternalTierError> {
        self.ensure_healthy()?;
        let pending = self.external_export(transfer_id)?.clone();
        self.preflight_external_request(&pending)?;
        self.manager.validate_external_read_pins(&pending.pinned)?;
        self.requests
            .get_mut(&pending.request_id)
            .ok_or(ExternalTierError::UnknownTransfer)?
            .phase = RequestPhase::Quarantined;
        self.external_exports
            .get_mut(&transfer_id)
            .ok_or(ExternalTierError::UnknownTransfer)?
            .phase = ExternalExportPhase::Quarantined;
        Ok(())
    }

    /// Allocates manager-owned local pages for one cataloged external replica.
    ///
    /// # Errors
    ///
    /// Rejects unknown/incompatible replicas, non-empty requests, exhausted
    /// operation or page capacity, and any geometry mismatch. A failed
    /// post-prepare validation internally aborts the unobserved native append.
    pub fn prepare_external_restore(
        &mut self,
        request_id: EngineRequestId,
        key: ExternalObjectKey,
    ) -> Result<ExternalRestorePlan, ExternalTierError> {
        self.ensure_healthy()?;
        let replica = self
            .external_replicas
            .get(&key)
            .cloned()
            .ok_or(ExternalTierError::UnknownObject)?;
        let request = self.ready_request(request_id)?;
        if request.view.boundary != 0
            || request.view.resident_count != 0
            || key.plan_fingerprint != self.manager.plan_fingerprint()
        {
            return Err(ExternalTierError::RestoreGeometryMismatch);
        }
        if self
            .external_exports
            .len()
            .checked_add(self.external_restores.len())
            .is_none_or(|count| count >= self.maximum_external_operations)
        {
            return Err(ExternalTierError::CapacityExhausted);
        }
        let next = self
            .next_external_sequence
            .checked_add(1)
            .ok_or(RuntimeSessionError::IdentityExhausted("external transfer"))?;
        let transfer_id = ExternalTransferId {
            session_epoch: self.session_epoch,
            sequence: self.next_external_sequence,
        };
        let batch = self.prepare_append_batch(&[super::EngineAppendIntent {
            request_id,
            target_boundary: key.boundary,
        }])?;
        let prepared_view = match self.prepared_execution_view(batch.batch_id) {
            Ok(view) => view,
            Err(error) => {
                if self
                    .abort_prepared_execution(
                        batch.batch_id,
                        &[EngineStepAbortEvidence {
                            request_id,
                            backend_unobserved: true,
                        }],
                    )
                    .is_err()
                {
                    return Err(self.poison("external restore view cleanup failed").into());
                }
                return Err(error.into());
            }
        };
        let build = build_restore_plan(transfer_id, request_id, &replica, &batch, &prepared_view);
        let (plan, execution) = match build {
            Ok(value) => value,
            Err(error) => {
                self.abort_prepared_execution(
                    batch.batch_id,
                    &[EngineStepAbortEvidence {
                        request_id,
                        backend_unobserved: true,
                    }],
                )?;
                return Err(error);
            }
        };
        self.next_external_sequence = next;
        self.external_restores.insert(
            transfer_id,
            PendingExternalRestore {
                request_id,
                batch_id: batch.batch_id,
                plan: plan.clone(),
                execution,
                phase: ExternalRestorePhase::Prepared,
            },
        );
        Ok(plan)
    }

    /// Validates remote-to-local copies and submits the native append.
    ///
    /// # Errors
    ///
    /// Rejects stale/foreign transfers, wrong phase, reordered or incomplete
    /// receipts, checksum mismatch, and native bind/submission failures.
    pub fn submit_external_restore(
        &mut self,
        transfer_id: ExternalTransferId,
        receipts: &[ExternalRestoreReceipt],
    ) -> Result<ExternalRestoreTicket, ExternalTierError> {
        self.ensure_healthy()?;
        let pending = self.external_restore(transfer_id)?.clone();
        if pending.phase != ExternalRestorePhase::Prepared {
            return Err(ExternalTierError::RestorePhaseMismatch);
        }
        validate_restore_receipts(&pending.plan, receipts)?;
        self.submit_execution(&pending.execution)?;
        let Some(restore) = self.external_restores.get_mut(&transfer_id) else {
            return Err(self.poison("external restore submit lost operation").into());
        };
        restore.phase = ExternalRestorePhase::Submitted;
        Ok(ExternalRestoreTicket { transfer_id })
    }

    /// Publishes a submitted restored request through the native completion path.
    ///
    /// # Errors
    ///
    /// Rejects stale/foreign transfers, wrong phase, or invalid/non-advancing
    /// completion evidence. The returned publication still requires the normal
    /// `confirm_publication` acknowledgement.
    pub fn complete_external_restore(
        &mut self,
        completion: ExternalTransferCompletion,
    ) -> Result<EngineBatchPublication, ExternalTierError> {
        self.ensure_healthy()?;
        let pending = self.external_restore(completion.transfer_id)?.clone();
        if pending.phase != ExternalRestorePhase::Submitted {
            return Err(ExternalTierError::RestorePhaseMismatch);
        }
        let publication = self.complete_execution_by_batch(
            pending.batch_id,
            EngineCompletionEvidence {
                completion_domain: completion.completion_domain,
                completion_value: completion.completion_value,
                confirmed: completion.confirmed,
            },
        )?;
        self.external_restores.remove(&completion.transfer_id);
        Ok(publication)
    }

    /// Aborts only a prepared restore whose backend was not observed.
    ///
    /// # Errors
    ///
    /// Rejects stale/foreign/submitted transfers or ambiguous observation.
    pub fn abort_external_restore(
        &mut self,
        evidence: ExternalRestoreAbortEvidence,
    ) -> Result<(), ExternalTierError> {
        self.ensure_healthy()?;
        let pending = self.external_restore(evidence.transfer_id)?.clone();
        if pending.phase != ExternalRestorePhase::Prepared {
            return Err(ExternalTierError::RestorePhaseMismatch);
        }
        if !evidence.backend_unobserved {
            return Err(ExternalTierError::AbortObservationUnknown);
        }
        self.abort_prepared_execution(
            pending.batch_id,
            &[EngineStepAbortEvidence {
                request_id: pending.request_id,
                backend_unobserved: true,
            }],
        )?;
        self.external_restores.remove(&evidence.transfer_id);
        Ok(())
    }

    /// Fail-stops a restore whose write outcome cannot be established.
    ///
    /// # Errors
    ///
    /// Rejects foreign/stale transfers and propagates native quarantine errors.
    pub fn quarantine_external_restore(
        &mut self,
        transfer_id: ExternalTransferId,
    ) -> Result<(), ExternalTierError> {
        self.ensure_healthy()?;
        let pending = self.external_restore(transfer_id)?.clone();
        match pending.phase {
            ExternalRestorePhase::Prepared => {
                self.quarantine_prepared_execution(pending.batch_id)?;
            }
            ExternalRestorePhase::Submitted => {
                self.quarantine_submitted_execution(pending.batch_id)?;
            }
            ExternalRestorePhase::Quarantined => {
                return Err(ExternalTierError::RestorePhaseMismatch);
            }
        }
        self.external_restores
            .get_mut(&transfer_id)
            .ok_or(ExternalTierError::UnknownTransfer)?
            .phase = ExternalRestorePhase::Quarantined;
        Ok(())
    }

    #[must_use]
    pub fn external_replica(&self, key: ExternalObjectKey) -> Option<&ExternalReplica> {
        self.external_replicas.get(&key)
    }

    #[must_use]
    pub fn external_tier_stats(&self) -> ExternalTierStats {
        ExternalTierStats {
            pending_exports: self
                .external_exports
                .values()
                .filter(|operation| operation.phase == ExternalExportPhase::Prepared)
                .count() as u64,
            pending_restores: self
                .external_restores
                .values()
                .filter(|operation| operation.phase != ExternalRestorePhase::Quarantined)
                .count() as u64,
            quarantined_exports: self
                .external_exports
                .values()
                .filter(|operation| operation.phase == ExternalExportPhase::Quarantined)
                .count() as u64,
            quarantined_restores: self
                .external_restores
                .values()
                .filter(|operation| operation.phase == ExternalRestorePhase::Quarantined)
                .count() as u64,
            replicas: self.external_replicas.len() as u64,
            operation_capacity: self.maximum_external_operations as u64,
            replica_capacity: self.maximum_external_replicas as u64,
            pinned_export_pages: self
                .external_exports
                .values()
                .map(|pending| pending.pinned.len() as u64)
                .sum(),
        }
    }

    /// Removes catalog ownership only after the adapter confirms exact deletion.
    ///
    /// # Errors
    ///
    /// Rejects unknown objects and missing or mismatched deletion evidence.
    pub fn confirm_external_replica_deletion(
        &mut self,
        evidence: ExternalReplicaDeletionEvidence,
    ) -> Result<ExternalReplica, ExternalTierError> {
        self.ensure_healthy()?;
        if !evidence.deleted {
            return Err(ExternalTierError::DeletionNotConfirmed);
        }
        if self
            .external_restores
            .values()
            .any(|restore| restore.plan.key == evidence.key)
        {
            return Err(ExternalTierError::ObjectBusy);
        }
        let replica = self
            .external_replicas
            .get(&evidence.key)
            .ok_or(ExternalTierError::UnknownObject)?;
        if replica.target != evidence.target {
            return Err(ExternalTierError::DeletionNotConfirmed);
        }
        self.external_replicas
            .remove(&evidence.key)
            .ok_or(ExternalTierError::UnknownObject)
    }

    fn external_export(
        &self,
        transfer_id: ExternalTransferId,
    ) -> Result<&PendingExternalExport, ExternalTierError> {
        if transfer_id.session_epoch != self.session_epoch {
            return Err(ExternalTierError::ForeignTransfer);
        }
        self.external_exports.get(&transfer_id).ok_or({
            if transfer_id.sequence != 0 && transfer_id.sequence < self.next_external_sequence {
                ExternalTierError::StaleTransfer
            } else {
                ExternalTierError::UnknownTransfer
            }
        })
    }

    fn external_restore(
        &self,
        transfer_id: ExternalTransferId,
    ) -> Result<&PendingExternalRestore, ExternalTierError> {
        if transfer_id.session_epoch != self.session_epoch {
            return Err(ExternalTierError::ForeignTransfer);
        }
        self.external_restores.get(&transfer_id).ok_or({
            if transfer_id.sequence != 0 && transfer_id.sequence < self.next_external_sequence {
                ExternalTierError::StaleTransfer
            } else {
                ExternalTierError::UnknownTransfer
            }
        })
    }

    fn preflight_external_request(
        &mut self,
        pending: &PendingExternalExport,
    ) -> Result<(), ExternalTierError> {
        if self
            .requests
            .get(&pending.request_id)
            .is_none_or(|request| {
                request.phase != RequestPhase::ExternalExportPending(pending.plan.transfer_id)
            })
        {
            return Err(self.poison("external export request phase changed").into());
        }
        Ok(())
    }
}

fn external_export_plan(
    transfer_id: ExternalTransferId,
    request_id: EngineRequestId,
    key: ExternalObjectKey,
    target: ExternalReplicaTarget,
    pages: &[PinnedSnapshotPage],
) -> Result<ExternalExportPlan, ExternalTierError> {
    let mut offset = target.base_offset;
    let mut copies = Vec::with_capacity(pages.len());
    for (copy_index, page) in pages.iter().enumerate() {
        copies.push(ExternalExportCopy {
            transfer_id,
            copy_index: u32::try_from(copy_index)
                .map_err(|_| ExternalTierError::ByteGeometryOverflow)?,
            class_id: page.class_id,
            source_backend_domain: page.backend_domain,
            source_backend_index: page.backend_index,
            destination_storage_domain: target.storage_domain,
            destination_object_index: target.object_index,
            destination_offset: offset,
            byte_count: page.payload_bytes,
            logical_ordinal: page.logical_ordinal,
            valid_token_count: page.valid_token_count,
            visible_token_offset: page.visible_token_offset,
            visible_token_count: page.visible_token_count,
        });
        offset = offset
            .checked_add(page.payload_bytes)
            .ok_or(ExternalTierError::ByteGeometryOverflow)?;
    }
    Ok(ExternalExportPlan {
        transfer_id,
        request_id,
        key,
        target,
        total_bytes: offset - target.base_offset,
        copies: copies.into_boxed_slice(),
    })
}

fn validate_export_receipts(
    plan: &ExternalExportPlan,
    receipts: &[ExternalExportReceipt],
) -> Result<ExternalReplica, ExternalTierError> {
    if receipts.len() != plan.copies.len() {
        return Err(ExternalTierError::ReceiptMismatch);
    }
    let mut seen = BTreeSet::new();
    let pages = plan
        .copies
        .iter()
        .zip(receipts)
        .map(|(copy, receipt)| {
            if receipt.copy != *copy
                || !receipt.copied
                || !receipt.durable
                || receipt.checksum == [0; 32]
                || !seen.insert(copy.copy_index)
            {
                return Err(ExternalTierError::ReceiptMismatch);
            }
            Ok(ExternalReplicaPage {
                copy_index: copy.copy_index,
                class_id: copy.class_id,
                logical_ordinal: copy.logical_ordinal,
                storage_offset: copy.destination_offset,
                byte_count: copy.byte_count,
                valid_token_count: copy.valid_token_count,
                visible_token_offset: copy.visible_token_offset,
                visible_token_count: copy.visible_token_count,
                checksum: receipt.checksum,
            })
        })
        .collect::<Result<Vec<_>, ExternalTierError>>()?;
    Ok(ExternalReplica {
        key: plan.key,
        target: plan.target,
        total_bytes: plan.total_bytes,
        pages: pages.into_boxed_slice(),
    })
}

fn build_restore_plan(
    transfer_id: ExternalTransferId,
    request_id: EngineRequestId,
    replica: &ExternalReplica,
    batch: &EngineBatchPlan,
    prepared: &super::EnginePreparedBatchView,
) -> Result<(ExternalRestorePlan, ExecutionEvidence), ExternalTierError> {
    let [step] = batch.steps.as_ref() else {
        return Err(ExternalTierError::RestoreGeometryMismatch);
    };
    let [request] = prepared.requests.as_ref() else {
        return Err(ExternalTierError::RestoreGeometryMismatch);
    };
    if step.request_id != request_id
        || request.request_id != request_id
        || step.previous_boundary != 0
        || step.target_boundary != replica.key.boundary
        || request.pages.len() != replica.pages.len()
        || !step.copy_intents.is_empty()
    {
        return Err(ExternalTierError::RestoreGeometryMismatch);
    }
    let mut destination_pages = std::collections::BTreeMap::new();
    for page in &request.pages {
        destination_pages.insert((page.class_id, page.logical_ordinal), page);
    }
    let mut copies = Vec::with_capacity(replica.pages.len());
    for page in &replica.pages {
        let target = destination_pages
            .get(&(page.class_id, page.logical_ordinal))
            .ok_or(ExternalTierError::RestoreGeometryMismatch)?;
        if page.valid_token_count != target.valid_token_count
            || page.visible_token_offset != target.visible_token_offset
            || page.visible_token_count != target.visible_token_count
            || page.checksum == [0; 32]
        {
            return Err(ExternalTierError::RestoreGeometryMismatch);
        }
        copies.push(ExternalRestoreCopy {
            transfer_id,
            copy_index: page.copy_index,
            class_id: page.class_id,
            source_storage_domain: replica.target.storage_domain,
            source_object_index: replica.target.object_index,
            source_offset: page.storage_offset,
            destination_backend_domain: target.backend_domain,
            destination_backend_index: target.backend_index,
            byte_count: page.byte_count,
            logical_ordinal: page.logical_ordinal,
            valid_token_count: page.valid_token_count,
            visible_token_offset: page.visible_token_offset,
            visible_token_count: page.visible_token_count,
            expected_checksum: page.checksum,
        });
    }
    copies.sort_by_key(|copy| copy.copy_index);
    if copies
        .iter()
        .enumerate()
        .any(|(index, copy)| usize::try_from(copy.copy_index).ok() != Some(index))
    {
        return Err(ExternalTierError::RestoreGeometryMismatch);
    }
    let execution = restore_execution_evidence(batch, &request.pages)?;
    Ok((
        ExternalRestorePlan {
            transfer_id,
            request_id,
            key: replica.key,
            total_bytes: replica.total_bytes,
            copies: copies.into_boxed_slice(),
        },
        execution,
    ))
}

fn restore_execution_evidence(
    batch: &EngineBatchPlan,
    pages: &[SnapshotPage],
) -> Result<ExecutionEvidence, ExternalTierError> {
    let [step] = batch.steps.as_ref() else {
        return Err(ExternalTierError::RestoreGeometryMismatch);
    };
    let mut binds = Vec::new();
    for lowering in &step.class_lowerings {
        let tail_begin = usize::try_from(lowering.tail_offset)
            .map_err(|_| ExternalTierError::RestoreGeometryMismatch)?;
        let tail_end = tail_begin
            .checked_add(usize::try_from(lowering.tail_count).unwrap_or(usize::MAX))
            .ok_or(ExternalTierError::RestoreGeometryMismatch)?;
        for action in step
            .tail_actions
            .get(tail_begin..tail_end)
            .ok_or(ExternalTierError::RestoreGeometryMismatch)?
        {
            if matches!(
                action.kind,
                TailActionKind::Fresh | TailActionKind::CopyOnWrite
            ) {
                binds.push(bind_for_page(pages, action.destination)?);
            }
        }
        let write_begin = usize::try_from(lowering.write_offset)
            .map_err(|_| ExternalTierError::RestoreGeometryMismatch)?;
        let write_end = write_begin
            .checked_add(usize::try_from(lowering.write_count).unwrap_or(usize::MAX))
            .ok_or(ExternalTierError::RestoreGeometryMismatch)?;
        for write in step
            .write_intents
            .get(write_begin..write_end)
            .ok_or(ExternalTierError::RestoreGeometryMismatch)?
        {
            let page = pages
                .iter()
                .find(|page| {
                    page.page.page_id == write.page_id
                        && page.page.generation == write.page_generation
                })
                .ok_or(ExternalTierError::RestoreGeometryMismatch)?;
            binds.push(bind_from_snapshot(page));
        }
    }
    Ok(ExecutionEvidence {
        batch_id: batch.batch_id,
        steps: vec![EngineStepExecutionEvidence {
            request_id: step.request_id,
            bind_receipts: binds.into_boxed_slice(),
            copy_receipts: Box::<[EngineCopyEvidence]>::default(),
            fixed_states: Box::default(),
        }]
        .into_boxed_slice(),
    })
}

fn bind_for_page(
    pages: &[SnapshotPage],
    lease: crate::kv_manager::PageLease,
) -> Result<EngineBindEvidence, ExternalTierError> {
    pages
        .iter()
        .find(|page| page.page == lease)
        .map(bind_from_snapshot)
        .ok_or(ExternalTierError::RestoreGeometryMismatch)
}

fn bind_from_snapshot(page: &SnapshotPage) -> EngineBindEvidence {
    EngineBindEvidence {
        page: page.page,
        backend_domain: page.backend_domain,
        mapped: true,
        writable: true,
        backend_index: page.backend_index,
    }
}

fn validate_restore_receipts(
    plan: &ExternalRestorePlan,
    receipts: &[ExternalRestoreReceipt],
) -> Result<(), ExternalTierError> {
    if receipts.len() != plan.copies.len()
        || plan.copies.iter().zip(receipts).any(|(copy, receipt)| {
            receipt.copy != *copy
                || receipt.checksum != copy.expected_checksum
                || !receipt.copied
                || !receipt.ordered_before_publish
        })
    {
        return Err(ExternalTierError::RestoreReceiptMismatch);
    }
    Ok(())
}

fn validate_external_replica(
    manager: &crate::kv_manager::CanonicalKvManager,
    replica: &ExternalReplica,
) -> Result<(), ExternalTierError> {
    if replica.key.boundary == 0
        || replica.key.digest == [0; 32]
        || replica.key.plan_fingerprint != manager.plan_fingerprint()
        || replica.target.storage_domain == 0
        || replica.target.object_index == 0
        || replica.pages.is_empty()
        || replica.total_bytes == 0
    {
        return Err(ExternalTierError::InvalidDescriptor);
    }
    let mut expected_offset = replica.target.base_offset;
    let mut previous = None;
    for (index, page) in replica.pages.iter().enumerate() {
        if usize::try_from(page.copy_index).ok() != Some(index)
            || page.storage_offset != expected_offset
            || page.checksum == [0; 32]
            || page.visible_token_count == 0
            || page
                .visible_token_offset
                .checked_add(page.visible_token_count)
                .is_none_or(|end| end > page.valid_token_count)
            || previous.is_some_and(|previous| previous >= (page.class_id, page.logical_ordinal))
            || manager.external_payload_bytes(page.class_id, page.valid_token_count)?
                != page.byte_count
        {
            return Err(ExternalTierError::InvalidDescriptor);
        }
        previous = Some((page.class_id, page.logical_ordinal));
        expected_offset = expected_offset
            .checked_add(page.byte_count)
            .ok_or(ExternalTierError::ByteGeometryOverflow)?;
    }
    if expected_offset.checked_sub(replica.target.base_offset) != Some(replica.total_bytes) {
        return Err(ExternalTierError::InvalidDescriptor);
    }
    Ok(())
}

const fn same_external_object(left: ExternalReplicaTarget, right: ExternalReplicaTarget) -> bool {
    left.storage_domain == right.storage_domain && left.object_index == right.object_index
}
