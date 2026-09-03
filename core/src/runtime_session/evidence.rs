use crate::kv_manager::{
    BackendBindReceipt, BackendCopyReceipt, PreparedStep, ReclamationCertificate,
    ReclamationReceipt, SubmitBatchItem, TailActionKind,
};

use super::{
    EngineBindEvidence, EngineCopyEvidence, EngineRetirement, EngineRetirementEvidence,
    EngineStepAbortEvidence, ExecutionEvidence, PreparedBatch, RuntimeSessionError,
};

pub(super) fn validate_abort_evidence(
    batch: &PreparedBatch,
    evidence: &[EngineStepAbortEvidence],
) -> Result<(), RuntimeSessionError> {
    if evidence.len() != batch.requests.len() {
        return Err(RuntimeSessionError::EvidenceCardinality {
            field: "abort steps",
            expected: batch.requests.len(),
            actual: evidence.len(),
        });
    }
    for (index, (&request_id, evidence)) in batch.requests.iter().zip(evidence.iter()).enumerate() {
        if evidence.request_id != request_id {
            return Err(RuntimeSessionError::EvidenceRequest {
                index,
                expected: request_id,
                actual: evidence.request_id,
            });
        }
    }
    Ok(())
}

type FlattenedExecutionEvidence = (
    Vec<SubmitBatchItem>,
    Vec<BackendBindReceipt>,
    Vec<BackendCopyReceipt>,
);

pub(super) fn flatten_evidence(
    batch: &PreparedBatch,
    evidence: &ExecutionEvidence,
) -> Result<FlattenedExecutionEvidence, RuntimeSessionError> {
    if evidence.steps.len() != batch.steps.len() {
        return Err(RuntimeSessionError::EvidenceCardinality {
            field: "steps",
            expected: batch.steps.len(),
            actual: evidence.steps.len(),
        });
    }
    let mut items = Vec::with_capacity(batch.steps.len());
    let mut binds = Vec::new();
    let mut copies = Vec::new();
    for (index, ((request_id, prepared), step_evidence)) in batch
        .requests
        .iter()
        .copied()
        .zip(batch.steps.iter())
        .zip(evidence.steps.iter())
        .enumerate()
    {
        if step_evidence.request_id != request_id {
            return Err(RuntimeSessionError::EvidenceRequest {
                index,
                expected: request_id,
                actual: step_evidence.request_id,
            });
        }
        let expected_binds = prepared
            .tail_actions
            .iter()
            .filter(|action| {
                matches!(
                    action.kind,
                    TailActionKind::CopyOnWrite | TailActionKind::Fresh
                )
            })
            .count()
            .checked_add(prepared.write_intents.len())
            .ok_or(RuntimeSessionError::EvidenceTooLarge)?;
        if step_evidence.bind_receipts.len() != expected_binds {
            return Err(RuntimeSessionError::EvidenceCardinality {
                field: "bind receipts",
                expected: expected_binds,
                actual: step_evidence.bind_receipts.len(),
            });
        }
        if step_evidence.copy_receipts.len() != prepared.copy_intents.len() {
            return Err(RuntimeSessionError::EvidenceCardinality {
                field: "copy receipts",
                expected: prepared.copy_intents.len(),
                actual: step_evidence.copy_receipts.len(),
            });
        }
        let receipt_offset =
            u32::try_from(binds.len()).map_err(|_| RuntimeSessionError::EvidenceTooLarge)?;
        let receipt_count = u32::try_from(step_evidence.bind_receipts.len())
            .map_err(|_| RuntimeSessionError::EvidenceTooLarge)?;
        let copy_receipt_offset =
            u32::try_from(copies.len()).map_err(|_| RuntimeSessionError::EvidenceTooLarge)?;
        let copy_receipt_count = u32::try_from(step_evidence.copy_receipts.len())
            .map_err(|_| RuntimeSessionError::EvidenceTooLarge)?;
        items.push(SubmitBatchItem {
            step: prepared.step,
            receipt_offset,
            receipt_count,
            copy_receipt_offset,
            copy_receipt_count,
        });
        binds.extend(
            step_evidence
                .bind_receipts
                .iter()
                .map(|evidence| canonical_bind_receipt(prepared, evidence)),
        );
        copies.extend(
            step_evidence
                .copy_receipts
                .iter()
                .map(|evidence| canonical_copy_receipt(prepared, evidence)),
        );
    }
    Ok((items, binds, copies))
}

fn canonical_bind_receipt(
    prepared: &PreparedStep,
    evidence: &EngineBindEvidence,
) -> BackendBindReceipt {
    BackendBindReceipt {
        step: prepared.step,
        page: evidence.page,
        backend_domain: evidence.backend_domain,
        mapped: u8::from(evidence.mapped),
        writable: u8::from(evidence.writable),
        reserved: 0,
        backend_index: evidence.backend_index,
    }
}

fn canonical_copy_receipt(
    prepared: &PreparedStep,
    evidence: &EngineCopyEvidence,
) -> BackendCopyReceipt {
    BackendCopyReceipt {
        step: prepared.step,
        class_id: evidence.class_id,
        backend_domain: evidence.backend_domain,
        token_count: evidence.token_count,
        source_token_offset: evidence.source_token_offset,
        destination_token_offset: evidence.destination_token_offset,
        observed: u8::from(evidence.observed),
        copied: u8::from(evidence.copied),
        ordered_before_writes: u8::from(evidence.ordered_before_writes),
        reserved8: 0,
        reserved32: 0,
        source: evidence.source,
        destination: evidence.destination,
        source_backend_index: evidence.source_backend_index,
        destination_backend_index: evidence.destination_backend_index,
    }
}

pub(super) fn engine_retirements(
    certificates: &[ReclamationCertificate],
) -> Box<[EngineRetirement]> {
    certificates
        .iter()
        .map(|certificate| EngineRetirement {
            page: certificate.page,
            class_id: certificate.class_id,
            backend_domain: certificate.backend_domain,
            logical_ordinal: certificate.logical_ordinal,
            backend_index: certificate.backend_index,
            token_begin: certificate.token_begin,
            token_end_exclusive: certificate.token_end_exclusive,
            completion_domain: certificate.completion_domain,
            completion_value: certificate.completion_value,
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

pub(super) fn validate_reclamation_evidence(
    mirror_cleanup_confirmed: bool,
    certificates: &[ReclamationCertificate],
    evidence: &[EngineRetirementEvidence],
) -> Result<Box<[ReclamationReceipt]>, RuntimeSessionError> {
    if !mirror_cleanup_confirmed {
        return Err(RuntimeSessionError::MirrorCleanupNotConfirmed);
    }
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
