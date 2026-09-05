use serde::Serialize;
use std::collections::BTreeSet;
use thiserror::Error;

use crate::kv_manager::{KvManagerError, PinnedSnapshotPage};

use super::{EngineRequestId, RequestPhase, RuntimeSession, RuntimeSessionError};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ExternalObjectKey {
    pub namespace: [u8; 32],
    pub digest: [u8; 32],
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

#[derive(Clone, Debug)]
pub(super) struct PendingExternalExport {
    pub request_id: EngineRequestId,
    pub plan: ExternalExportPlan,
    pub pinned: Box<[PinnedSnapshotPage]>,
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
    #[error("external transfer completion is not confirmed")]
    CompletionNotConfirmed,
    #[error("external export abort requires proof that the backend was unobserved")]
    AbortObservationUnknown,
    #[error("external replica is unknown")]
    UnknownObject,
    #[error("external replica deletion is not confirmed")]
    DeletionNotConfirmed,
    #[error("external transfer byte geometry overflowed")]
    ByteGeometryOverflow,
}

impl RuntimeSession {
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
            .checked_add(self.external_replicas.len())
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

    #[must_use]
    pub fn external_replica(&self, key: ExternalObjectKey) -> Option<&ExternalReplica> {
        self.external_replicas.get(&key)
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

const fn same_external_object(left: ExternalReplicaTarget, right: ExternalReplicaTarget) -> bool {
    left.storage_domain == right.storage_domain && left.object_index == right.object_index
}
