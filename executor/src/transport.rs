use async_trait::async_trait;
use orbitkv::{
    ExternalExportPlan, ExternalExportReceipt, ExternalReplica, ExternalReplicaDeletionEvidence,
    ExternalRestorePlan, ExternalRestoreReceipt, ExternalTransferCompletion, ExternalTransferId,
};
use thiserror::Error;

use crate::{ExternalRestoreBatch, ExternalTransferBatch};

mod host_memory;
pub use host_memory::{HostFaultPoint, HostMemoryTransport, HostTensorRegion, HostTransportFault};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalTransferOperation {
    Register,
    Export,
    Restore,
    Delete,
    Complete,
}

/// Whether a failed operation is proven invisible or may have mutated its backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransferObservation {
    Unobserved,
    Ambiguous,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("{operation:?} transport failure ({observation:?}): {reason}")]
pub struct ExternalTransportError {
    operation: ExternalTransferOperation,
    observation: TransferObservation,
    reason: String,
}

impl ExternalTransportError {
    fn new(
        operation: ExternalTransferOperation,
        observation: TransferObservation,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            operation,
            observation,
            reason: reason.into(),
        }
    }

    #[must_use]
    pub fn unobserved(operation: ExternalTransferOperation, reason: impl Into<String>) -> Self {
        Self::new(operation, TransferObservation::Unobserved, reason)
    }

    #[must_use]
    pub fn ambiguous(operation: ExternalTransferOperation, reason: impl Into<String>) -> Self {
        Self::new(operation, TransferObservation::Ambiguous, reason)
    }

    #[must_use]
    pub const fn operation(&self) -> ExternalTransferOperation {
        self.operation
    }

    #[must_use]
    pub const fn observation(&self) -> TransferObservation {
        self.observation
    }

    #[must_use]
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExternalCopyChecksum {
    pub copy_index: u32,
    pub checksum: [u8; 32],
}

/// Confirmed transport completion plus checksums computed from moved bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalTransferOutcome {
    operation: ExternalTransferOperation,
    completion: ExternalTransferCompletion,
    checksums: Box<[ExternalCopyChecksum]>,
}

impl ExternalTransferOutcome {
    /// Constructs a confirmed result produced by a transport implementation.
    ///
    /// # Errors
    ///
    /// Rejects non-transfer operations, zero completion coordinates, empty or
    /// non-contiguous checksum reports.
    pub fn confirmed(
        operation: ExternalTransferOperation,
        completion: ExternalTransferCompletion,
        checksums: Box<[ExternalCopyChecksum]>,
    ) -> Result<Self, ExternalTransportError> {
        if !matches!(
            operation,
            ExternalTransferOperation::Export | ExternalTransferOperation::Restore
        ) || !completion.confirmed
            || completion.completion_domain == 0
            || completion.completion_value == 0
            || checksums.is_empty()
            || checksums
                .iter()
                .enumerate()
                .any(|(index, checksum)| usize::try_from(checksum.copy_index).ok() != Some(index))
        {
            return Err(ExternalTransportError::ambiguous(
                operation,
                "invalid confirmed transport outcome",
            ));
        }
        Ok(Self {
            operation,
            completion,
            checksums,
        })
    }

    #[must_use]
    pub const fn operation(&self) -> ExternalTransferOperation {
        self.operation
    }

    #[must_use]
    pub const fn completion(&self) -> ExternalTransferCompletion {
        self.completion
    }

    #[must_use]
    pub fn checksums(&self) -> &[ExternalCopyChecksum] {
        &self.checksums
    }

    /// Binds confirmed transport evidence back to the exact manager export plan.
    ///
    /// # Errors
    ///
    /// Rejects a report from another transfer or any missing/reordered checksum.
    pub fn export_receipts(
        &self,
        plan: &ExternalExportPlan,
    ) -> Result<Box<[ExternalExportReceipt]>, ExternalTransportError> {
        self.validate_plan(
            plan.transfer_id,
            plan.copies.len(),
            ExternalTransferOperation::Export,
        )?;
        Ok(plan
            .copies
            .iter()
            .zip(&self.checksums)
            .map(|(copy, checksum)| ExternalExportReceipt {
                copy: *copy,
                checksum: checksum.checksum,
                copied: true,
                durable: true,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    /// Binds confirmed transport evidence back to the exact manager restore plan.
    ///
    /// # Errors
    ///
    /// Rejects a report from another transfer or any missing/reordered checksum.
    pub fn restore_receipts(
        &self,
        plan: &ExternalRestorePlan,
    ) -> Result<Box<[ExternalRestoreReceipt]>, ExternalTransportError> {
        self.validate_plan(
            plan.transfer_id,
            plan.copies.len(),
            ExternalTransferOperation::Restore,
        )?;
        Ok(plan
            .copies
            .iter()
            .zip(&self.checksums)
            .map(|(copy, checksum)| ExternalRestoreReceipt {
                copy: *copy,
                checksum: checksum.checksum,
                copied: true,
                ordered_before_publish: true,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    fn validate_plan(
        &self,
        transfer_id: ExternalTransferId,
        copy_count: usize,
        operation: ExternalTransferOperation,
    ) -> Result<(), ExternalTransportError> {
        if self.operation != operation
            || self.completion.transfer_id != transfer_id
            || !self.completion.confirmed
            || self.checksums.len() != copy_count
            || self
                .checksums
                .iter()
                .enumerate()
                .any(|(index, checksum)| usize::try_from(checksum.copy_index).ok() != Some(index))
        {
            return Err(ExternalTransportError::ambiguous(
                operation,
                "completion does not match the manager-authored plan",
            ));
        }
        Ok(())
    }
}

/// Backend-neutral asynchronous byte-movement boundary.
///
/// Implementations consume executor-lowered spans and never allocate, retire,
/// or publish `OrbitKV` pages. A backend error must conservatively report whether
/// mutation was proven unobserved or is ambiguous. Dropping an in-flight future
/// is conservatively ambiguous unless the adapter can independently prove that
/// its backend never observed the operation.
#[async_trait]
pub trait ExternalKvTransport: Send + Sync {
    async fn export(
        &self,
        batch: &ExternalTransferBatch,
    ) -> Result<ExternalTransferOutcome, ExternalTransportError>;

    async fn restore(
        &self,
        batch: &ExternalRestoreBatch,
    ) -> Result<ExternalTransferOutcome, ExternalTransportError>;

    async fn delete(
        &self,
        replica: &ExternalReplica,
    ) -> Result<ExternalReplicaDeletionEvidence, ExternalTransportError>;
}
