use std::collections::BTreeSet;

use orbitkv::{
    ExternalExportCopy, ExternalExportPlan, ExternalRestoreCopy, ExternalRestorePlan,
    ExternalTransferId,
};

use crate::{AttentionClass, ExecutorArena, ExecutorError, ExecutorPlan, arena_for};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum KvComponent {
    Key,
    Value,
}

/// One contiguous tensor range mapped to an external-object byte range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExternalTransferSpan {
    pub copy_index: u32,
    pub class_id: u16,
    pub layer: u32,
    pub component: KvComponent,
    pub source_backend_domain: u16,
    pub source_tensor_offset: u64,
    pub destination_storage_domain: u64,
    pub destination_object_index: u64,
    pub destination_offset: u64,
    pub bytes: u64,
}

/// Transport-neutral iovecs derived from one manager-authored export plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalTransferBatch {
    pub transfer_id: ExternalTransferId,
    pub total_bytes: u64,
    pub spans: Box<[ExternalTransferSpan]>,
}

/// One contiguous external-object range mapped into a local K/V tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExternalRestoreSpan {
    pub copy_index: u32,
    pub class_id: u16,
    pub layer: u32,
    pub component: KvComponent,
    pub source_storage_domain: u64,
    pub source_object_index: u64,
    pub source_offset: u64,
    pub destination_backend_domain: u16,
    pub destination_tensor_offset: u64,
    pub bytes: u64,
    pub expected_checksum: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalRestoreBatch {
    pub transfer_id: ExternalTransferId,
    pub total_bytes: u64,
    pub spans: Box<[ExternalRestoreSpan]>,
}

impl ExecutorPlan {
    /// Expands logical-page export work into per-layer K/V tensor spans.
    ///
    /// # Errors
    ///
    /// Rejects unknown classes, mismatched arena domains/page ranges, invalid
    /// copy ordering, byte geometry mismatch, or arithmetic overflow.
    pub fn lower_external_export(
        &self,
        source: &ExternalExportPlan,
        arenas: &[ExecutorArena],
    ) -> Result<ExternalTransferBatch, ExecutorError> {
        super::validate_arenas(arenas, self.classes.len())?;
        if source.copies.is_empty() || source.total_bytes == 0 {
            return Err(ExecutorError::ExternalTransferMismatch);
        }
        let mut spans = Vec::new();
        let mut seen = BTreeSet::new();
        let mut expected_destination = source.target.base_offset;
        for (position, copy) in source.copies.iter().enumerate() {
            if copy.transfer_id != source.transfer_id
                || usize::try_from(copy.copy_index).ok() != Some(position)
                || copy.destination_storage_domain != source.target.storage_domain
                || copy.destination_object_index != source.target.object_index
                || copy.destination_offset != expected_destination
                || copy.valid_token_count == 0
                || copy.valid_token_count > self.page_tokens
                || !seen.insert((copy.class_id, copy.source_backend_index))
            {
                return Err(ExecutorError::ExternalTransferMismatch);
            }
            let class = self
                .classes
                .get(usize::from(copy.class_id))
                .filter(|class| class.class_id == copy.class_id)
                .ok_or(ExecutorError::UnknownClass)?;
            let arena = arena_for(arenas, copy.class_id)?;
            expand_page_copy(copy, class, arena, self.page_tokens, &mut spans)?;
            expected_destination = expected_destination
                .checked_add(copy.byte_count)
                .ok_or(ExecutorError::SlotOverflow)?;
        }
        if expected_destination.checked_sub(source.target.base_offset) != Some(source.total_bytes) {
            return Err(ExecutorError::ExternalTransferMismatch);
        }
        Ok(ExternalTransferBatch {
            transfer_id: source.transfer_id,
            total_bytes: source.total_bytes,
            spans: spans.into_boxed_slice(),
        })
    }

    /// Expands external restore work into per-layer K/V tensor spans.
    ///
    /// # Errors
    ///
    /// Rejects unknown classes, mismatched arenas, invalid ordering, checksum
    /// geometry mismatches, or arithmetic overflow.
    pub fn lower_external_restore(
        &self,
        source: &ExternalRestorePlan,
        arenas: &[ExecutorArena],
    ) -> Result<ExternalRestoreBatch, ExecutorError> {
        super::validate_arenas(arenas, self.classes.len())?;
        if source.copies.is_empty() || source.total_bytes == 0 {
            return Err(ExecutorError::ExternalTransferMismatch);
        }
        let mut spans = Vec::new();
        let mut seen = BTreeSet::new();
        let mut total = 0_u64;
        for (position, copy) in source.copies.iter().enumerate() {
            if copy.transfer_id != source.transfer_id
                || usize::try_from(copy.copy_index).ok() != Some(position)
                || copy.expected_checksum == [0; 32]
                || copy.valid_token_count == 0
                || copy.valid_token_count > self.page_tokens
                || !seen.insert((copy.class_id, copy.destination_backend_index))
            {
                return Err(ExecutorError::ExternalTransferMismatch);
            }
            let class = self
                .classes
                .get(usize::from(copy.class_id))
                .filter(|class| class.class_id == copy.class_id)
                .ok_or(ExecutorError::UnknownClass)?;
            let arena = arena_for(arenas, copy.class_id)?;
            expand_restore_copy(copy, class, arena, self.page_tokens, &mut spans)?;
            total = total
                .checked_add(copy.byte_count)
                .ok_or(ExecutorError::SlotOverflow)?;
        }
        if total != source.total_bytes {
            return Err(ExecutorError::ExternalTransferMismatch);
        }
        Ok(ExternalRestoreBatch {
            transfer_id: source.transfer_id,
            total_bytes: source.total_bytes,
            spans: spans.into_boxed_slice(),
        })
    }
}

fn expand_page_copy(
    copy: &ExternalExportCopy,
    class: &AttentionClass,
    arena: ExecutorArena,
    page_tokens: u32,
    spans: &mut Vec<ExternalTransferSpan>,
) -> Result<(), ExecutorError> {
    if class.layers.is_empty()
        || class.key_bytes_per_token_per_layer == 0
        || class.value_bytes_per_token_per_layer == 0
        || copy.source_backend_domain != arena.backend_domain
    {
        return Err(ExecutorError::ExternalTransferMismatch);
    }
    copy.source_backend_index
        .checked_sub(arena.backend_base_index)
        .filter(|page| *page < u64::from(arena.page_count))
        .ok_or(ExecutorError::ExternalTransferMismatch)?;
    let valid_tokens = u64::from(copy.valid_token_count);
    let page_tokens = u64::from(page_tokens);
    let key_page_bytes = page_tokens
        .checked_mul(class.key_bytes_per_token_per_layer)
        .ok_or(ExecutorError::SlotOverflow)?;
    let value_page_bytes = page_tokens
        .checked_mul(class.value_bytes_per_token_per_layer)
        .ok_or(ExecutorError::SlotOverflow)?;
    let key_bytes = valid_tokens
        .checked_mul(class.key_bytes_per_token_per_layer)
        .ok_or(ExecutorError::SlotOverflow)?;
    let value_bytes = valid_tokens
        .checked_mul(class.value_bytes_per_token_per_layer)
        .ok_or(ExecutorError::SlotOverflow)?;
    let expected_page_bytes = (key_bytes + value_bytes)
        .checked_mul(class.layers.len() as u64)
        .ok_or(ExecutorError::SlotOverflow)?;
    if expected_page_bytes != copy.byte_count {
        return Err(ExecutorError::ExternalTransferMismatch);
    }

    let mut destination = copy.destination_offset;
    for &layer in &class.layers {
        for (component, page_bytes, bytes) in [
            (KvComponent::Key, key_page_bytes, key_bytes),
            (KvComponent::Value, value_page_bytes, value_bytes),
        ] {
            spans.push(ExternalTransferSpan {
                copy_index: copy.copy_index,
                class_id: copy.class_id,
                layer,
                component,
                source_backend_domain: copy.source_backend_domain,
                source_tensor_offset: copy
                    .source_backend_index
                    .checked_mul(page_bytes)
                    .ok_or(ExecutorError::SlotOverflow)?,
                destination_storage_domain: copy.destination_storage_domain,
                destination_object_index: copy.destination_object_index,
                destination_offset: destination,
                bytes,
            });
            destination = destination
                .checked_add(bytes)
                .ok_or(ExecutorError::SlotOverflow)?;
        }
    }
    if destination.checked_sub(copy.destination_offset) != Some(copy.byte_count) {
        return Err(ExecutorError::ExternalTransferMismatch);
    }
    Ok(())
}

fn expand_restore_copy(
    copy: &ExternalRestoreCopy,
    class: &AttentionClass,
    arena: ExecutorArena,
    page_tokens: u32,
    spans: &mut Vec<ExternalRestoreSpan>,
) -> Result<(), ExecutorError> {
    if class.layers.is_empty()
        || class.key_bytes_per_token_per_layer == 0
        || class.value_bytes_per_token_per_layer == 0
        || copy.destination_backend_domain != arena.backend_domain
        || copy.source_storage_domain == 0
        || copy.source_object_index == 0
    {
        return Err(ExecutorError::ExternalTransferMismatch);
    }
    copy.destination_backend_index
        .checked_sub(arena.backend_base_index)
        .filter(|page| *page < u64::from(arena.page_count))
        .ok_or(ExecutorError::ExternalTransferMismatch)?;
    let valid_tokens = u64::from(copy.valid_token_count);
    let page_tokens = u64::from(page_tokens);
    let key_page_bytes = page_tokens
        .checked_mul(class.key_bytes_per_token_per_layer)
        .ok_or(ExecutorError::SlotOverflow)?;
    let value_page_bytes = page_tokens
        .checked_mul(class.value_bytes_per_token_per_layer)
        .ok_or(ExecutorError::SlotOverflow)?;
    let key_bytes = valid_tokens
        .checked_mul(class.key_bytes_per_token_per_layer)
        .ok_or(ExecutorError::SlotOverflow)?;
    let value_bytes = valid_tokens
        .checked_mul(class.value_bytes_per_token_per_layer)
        .ok_or(ExecutorError::SlotOverflow)?;
    let expected = (key_bytes + value_bytes)
        .checked_mul(class.layers.len() as u64)
        .ok_or(ExecutorError::SlotOverflow)?;
    if expected != copy.byte_count {
        return Err(ExecutorError::ExternalTransferMismatch);
    }

    let mut source_offset = copy.source_offset;
    for &layer in &class.layers {
        for (component, page_bytes, bytes) in [
            (KvComponent::Key, key_page_bytes, key_bytes),
            (KvComponent::Value, value_page_bytes, value_bytes),
        ] {
            spans.push(ExternalRestoreSpan {
                copy_index: copy.copy_index,
                class_id: copy.class_id,
                layer,
                component,
                source_storage_domain: copy.source_storage_domain,
                source_object_index: copy.source_object_index,
                source_offset,
                destination_backend_domain: copy.destination_backend_domain,
                destination_tensor_offset: copy
                    .destination_backend_index
                    .checked_mul(page_bytes)
                    .ok_or(ExecutorError::SlotOverflow)?,
                bytes,
                expected_checksum: copy.expected_checksum,
            });
            source_offset = source_offset
                .checked_add(bytes)
                .ok_or(ExecutorError::SlotOverflow)?;
        }
    }
    if source_offset.checked_sub(copy.source_offset) != Some(copy.byte_count) {
        return Err(ExecutorError::ExternalTransferMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use orbitkv::{
        EngineRequestId, ExternalObjectKey, ExternalReplicaTarget,
        runtime_session::ExternalTransferId,
    };

    use super::*;
    use crate::AttentionVisibility;

    #[test]
    fn expands_logical_pages_into_layer_component_iovecs() {
        let plan = crate::test_executor_plan(
            "test",
            16,
            vec![AttentionClass {
                class_id: 0,
                name: "attention".into(),
                layers: vec![2, 5].into_boxed_slice(),
                page_tokens: 16,
                key_bytes_per_token_per_layer: 8,
                value_bytes_per_token_per_layer: 12,
                visibility: AttentionVisibility::Full,
            }],
        );
        let transfer_id = ExternalTransferId::from_parts(1, 1);
        let target = ExternalReplicaTarget {
            storage_domain: 7,
            object_index: 9,
            base_offset: 1_000,
        };
        let source = ExternalExportPlan {
            transfer_id,
            request_id: EngineRequestId(3),
            key: ExternalObjectKey {
                namespace: [1; 32],
                digest: [2; 32],
                plan_fingerprint: [3; 32],
                boundary: 16,
            },
            target,
            total_bytes: 640,
            copies: vec![ExternalExportCopy {
                transfer_id,
                copy_index: 0,
                class_id: 0,
                source_backend_domain: 3,
                source_backend_index: 12,
                destination_storage_domain: 7,
                destination_object_index: 9,
                destination_offset: 1_000,
                byte_count: 640,
                logical_ordinal: 0,
                valid_token_count: 16,
                visible_token_offset: 0,
                visible_token_count: 16,
            }]
            .into_boxed_slice(),
        };
        let lowered = plan
            .lower_external_export(
                &source,
                &[ExecutorArena {
                    engine_epoch: 1,
                    pool_epoch: 2,
                    pool_id: 4,
                    class_id: 0,
                    backend_domain: 3,
                    first_page_id: 1,
                    page_count: 8,
                    backend_base_index: 10,
                }],
            )
            .unwrap();
        assert_eq!(lowered.total_bytes, 640);
        assert_eq!(lowered.spans.len(), 4);
        assert_eq!(lowered.spans[0].source_tensor_offset, 1_536);
        assert_eq!(lowered.spans[0].destination_offset, 1_000);
        assert_eq!(lowered.spans[1].source_tensor_offset, 2_304);
        assert_eq!(lowered.spans[1].destination_offset, 1_128);
        assert_eq!(lowered.spans[2].layer, 5);
        assert_eq!(lowered.spans[3].destination_offset, 1_448);
    }

    #[test]
    fn expands_restore_into_external_to_tensor_iovecs() {
        let plan = crate::test_executor_plan(
            "test",
            16,
            vec![AttentionClass {
                class_id: 0,
                name: "attention".into(),
                layers: vec![2, 5].into_boxed_slice(),
                page_tokens: 16,
                key_bytes_per_token_per_layer: 8,
                value_bytes_per_token_per_layer: 12,
                visibility: AttentionVisibility::Full,
            }],
        );
        let transfer_id = ExternalTransferId::from_parts(1, 2);
        let source = ExternalRestorePlan {
            transfer_id,
            request_id: EngineRequestId(4),
            key: ExternalObjectKey {
                namespace: [1; 32],
                digest: [2; 32],
                plan_fingerprint: [3; 32],
                boundary: 16,
            },
            total_bytes: 640,
            copies: vec![ExternalRestoreCopy {
                transfer_id,
                copy_index: 0,
                class_id: 0,
                source_storage_domain: 7,
                source_object_index: 9,
                source_offset: 1_000,
                destination_backend_domain: 3,
                destination_backend_index: 12,
                byte_count: 640,
                logical_ordinal: 0,
                valid_token_count: 16,
                visible_token_offset: 0,
                visible_token_count: 16,
                expected_checksum: [4; 32],
            }]
            .into_boxed_slice(),
        };
        let lowered = plan
            .lower_external_restore(
                &source,
                &[ExecutorArena {
                    engine_epoch: 1,
                    pool_epoch: 2,
                    pool_id: 4,
                    class_id: 0,
                    backend_domain: 3,
                    first_page_id: 1,
                    page_count: 8,
                    backend_base_index: 10,
                }],
            )
            .unwrap();
        assert_eq!(lowered.spans.len(), 4);
        assert_eq!(lowered.spans[0].source_offset, 1_000);
        assert_eq!(lowered.spans[0].destination_tensor_offset, 1_536);
        assert_eq!(lowered.spans[1].source_offset, 1_128);
        assert_eq!(lowered.spans[1].destination_tensor_offset, 2_304);
        assert_eq!(lowered.spans[3].source_offset, 1_448);
    }
}
