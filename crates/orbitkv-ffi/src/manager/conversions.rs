use super::{
    ArenaStats, BackendArenaRegistration, BackendBindReceipt, BackendCopyReceipt,
    BackendUnobservedReceipt, BatchCompletionReceipt, ClassLowering, CopyIntent, DetachedBinding,
    ManagerConfig, ManagerStats, OrbitKvArenaStats, OrbitKvBackendArenaRegistration,
    OrbitKvBackendBindReceipt, OrbitKvBackendCopyReceipt, OrbitKvBackendUnobservedReceipt,
    OrbitKvBatchCompletionReceipt, OrbitKvClassLowering, OrbitKvCopyIntent, OrbitKvDetachedBinding,
    OrbitKvEvictedPrefix, OrbitKvManagerConfig, OrbitKvManagerStats, OrbitKvPageLease,
    OrbitKvPrefixAttachBatchItem, OrbitKvPrefixLease, OrbitKvPrefixLookupHint,
    OrbitKvPrefixPublishBatchItem, OrbitKvPrefixSemanticKey, OrbitKvPublishedPrefix,
    OrbitKvReclamationCertificate, OrbitKvReclamationLease, OrbitKvReclamationReceipt,
    OrbitKvReleaseBatchItem, OrbitKvRequestForkBatchItem, OrbitKvRequestLease, OrbitKvRequestView,
    OrbitKvSnapshotLease, OrbitKvSnapshotPage, OrbitKvStepLease, OrbitKvSubmissionLease,
    OrbitKvSubmitBatchItem, OrbitKvTailAction, OrbitKvWriteIntent, PageLease, PrefixAttachItem,
    PrefixLease, PrefixLookupHint, PrefixPublishItem, PrefixSemanticKey, ReclamationCertificate,
    ReclamationLease, ReclamationReceipt, ReleaseBatchItem, RequestForkItem, RequestLease,
    RequestView, SnapshotLease, SnapshotPage, StepLease, SubmissionLease, SubmitBatchItem,
    TailAction, WriteIntent,
};

macro_rules! lease_conversions {
    ($wire:ident, $core:ident) => {
        impl From<$wire> for $core {
            fn from(value: $wire) -> Self {
                Self {
                    engine_epoch: value.engine_epoch,
                    slot: value.slot,
                    generation: value.generation,
                }
            }
        }
        impl From<$core> for $wire {
            fn from(value: $core) -> Self {
                Self {
                    engine_epoch: value.engine_epoch,
                    slot: value.slot,
                    generation: value.generation,
                }
            }
        }
    };
}

lease_conversions!(OrbitKvRequestLease, RequestLease);
lease_conversions!(OrbitKvSnapshotLease, SnapshotLease);
lease_conversions!(OrbitKvStepLease, StepLease);
lease_conversions!(OrbitKvSubmissionLease, SubmissionLease);
lease_conversions!(OrbitKvReclamationLease, ReclamationLease);
lease_conversions!(OrbitKvPrefixLease, PrefixLease);

impl From<OrbitKvPageLease> for PageLease {
    fn from(value: OrbitKvPageLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            pool_epoch: value.pool_epoch,
            generation: value.generation,
            page_id: value.page_id,
            pool_id: value.pool_id,
        }
    }
}

impl From<PageLease> for OrbitKvPageLease {
    fn from(value: PageLease) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            pool_epoch: value.pool_epoch,
            generation: value.generation,
            page_id: value.page_id,
            pool_id: value.pool_id,
        }
    }
}

impl From<OrbitKvBackendArenaRegistration> for BackendArenaRegistration {
    fn from(value: OrbitKvBackendArenaRegistration) -> Self {
        Self {
            pool_id: value.pool_id,
            class_id: value.class_id,
            backend_domain: value.backend_domain,
            page_count: value.page_count,
            reserved: value.reserved,
            backend_base_index: value.backend_base_index,
        }
    }
}

impl From<OrbitKvManagerConfig> for ManagerConfig {
    fn from(value: OrbitKvManagerConfig) -> Self {
        Self {
            maximum_requests: value.maximum_requests,
            maximum_operations: value.maximum_operations,
            maximum_prefixes: value.maximum_prefixes,
            maximum_reclamations: value.maximum_reclamations,
            maximum_step_tokens: value.maximum_step_tokens,
        }
    }
}

impl From<RequestView> for OrbitKvRequestView {
    fn from(value: RequestView) -> Self {
        Self {
            request: value.request.into(),
            snapshot: value.snapshot.into(),
            view_version: value.view_version.0,
            boundary: value.boundary,
            resident_count: value.resident_count,
            reserved: 0,
        }
    }
}

impl From<SnapshotPage> for OrbitKvSnapshotPage {
    fn from(value: SnapshotPage) -> Self {
        Self {
            page: value.page.into(),
            logical_ordinal: value.logical_ordinal,
            temporal_cell_index: value.temporal_cell_index,
            temporal_cycle: value.temporal_cycle,
            backend_index: value.backend_index,
            class_id: value.class_id,
            backend_domain: value.backend_domain,
            valid_token_count: value.valid_token_count,
            visible_token_offset: value.visible_token_offset,
            visible_token_count: value.visible_token_count,
            reserved: 0,
        }
    }
}

impl From<OrbitKvRequestForkBatchItem> for RequestForkItem {
    fn from(value: OrbitKvRequestForkBatchItem) -> Self {
        Self {
            source_request: value.source_request.into(),
            expected_source_head: value.expected_source_head.into(),
            target_empty_request: value.target_empty_request.into(),
            expected_target_head: value.expected_target_head.into(),
        }
    }
}

impl From<ClassLowering> for OrbitKvClassLowering {
    fn from(value: ClassLowering) -> Self {
        Self {
            class_id: value.class_id,
            flags: value.flags,
            tail_offset: value.tail_offset,
            tail_count: value.tail_count,
            copy_offset: value.copy_offset,
            copy_count: value.copy_count,
            write_offset: value.write_offset,
            write_count: value.write_count,
            reserved: 0,
        }
    }
}

impl From<TailAction> for OrbitKvTailAction {
    fn from(value: TailAction) -> Self {
        Self {
            class_id: value.class_id,
            kind: value.kind as u16,
            valid_token_count: value.valid_token_count,
            logical_ordinal: value.logical_ordinal,
            source: value.source.into(),
            destination: value.destination.into(),
            reserved: value.reserved,
        }
    }
}

impl From<CopyIntent> for OrbitKvCopyIntent {
    fn from(value: CopyIntent) -> Self {
        Self {
            class_id: value.class_id,
            backend_domain: value.backend_domain,
            token_count: value.token_count,
            source_token_offset: value.source_token_offset,
            destination_token_offset: value.destination_token_offset,
            reserved: value.reserved,
            source: value.source.into(),
            destination: value.destination.into(),
            source_backend_index: value.source_backend_index,
            destination_backend_index: value.destination_backend_index,
        }
    }
}

impl From<WriteIntent> for OrbitKvWriteIntent {
    fn from(value: WriteIntent) -> Self {
        Self {
            page_generation: value.page_generation,
            page_id: value.page_id,
            reserved: value.reserved,
        }
    }
}

impl From<OrbitKvBackendBindReceipt> for BackendBindReceipt {
    fn from(value: OrbitKvBackendBindReceipt) -> Self {
        Self {
            step: value.step.into(),
            page: value.page.into(),
            backend_domain: value.backend_domain,
            mapped: value.mapped,
            writable: value.writable,
            reserved: value.reserved,
            backend_index: value.backend_index,
        }
    }
}

impl From<OrbitKvBackendCopyReceipt> for BackendCopyReceipt {
    fn from(value: OrbitKvBackendCopyReceipt) -> Self {
        Self {
            step: value.step.into(),
            class_id: value.class_id,
            backend_domain: value.backend_domain,
            token_count: value.token_count,
            source_token_offset: value.source_token_offset,
            destination_token_offset: value.destination_token_offset,
            observed: value.observed,
            copied: value.copied,
            ordered_before_writes: value.ordered_before_writes,
            reserved8: value.reserved8,
            reserved32: value.reserved32,
            source: value.source.into(),
            destination: value.destination.into(),
            source_backend_index: value.source_backend_index,
            destination_backend_index: value.destination_backend_index,
        }
    }
}

impl From<OrbitKvSubmitBatchItem> for SubmitBatchItem {
    fn from(value: OrbitKvSubmitBatchItem) -> Self {
        Self {
            step: value.step.into(),
            receipt_offset: value.receipt_offset,
            receipt_count: value.receipt_count,
            copy_receipt_offset: value.copy_receipt_offset,
            copy_receipt_count: value.copy_receipt_count,
        }
    }
}

impl From<OrbitKvBatchCompletionReceipt> for BatchCompletionReceipt {
    fn from(value: OrbitKvBatchCompletionReceipt) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            completion_domain: value.completion_domain,
            completion_value: value.completion_value,
            confirmed: value.confirmed,
            reserved: value.reserved,
        }
    }
}

impl From<DetachedBinding> for OrbitKvDetachedBinding {
    fn from(value: DetachedBinding) -> Self {
        Self {
            old: value.old.into(),
            replacement: value.replacement.into(),
            logical_ordinal: value.logical_ordinal,
            old_backend_index: value.old_backend_index,
            replacement_backend_index: value.replacement_backend_index,
            token_begin: value.token_begin,
            token_end_exclusive: value.token_end_exclusive,
            class_id: value.class_id,
            backend_domain: value.backend_domain,
            action: value.action as u16,
            reason: value.reason as u16,
            reserved: value.reserved,
        }
    }
}

impl From<ReclamationCertificate> for OrbitKvReclamationCertificate {
    fn from(value: ReclamationCertificate) -> Self {
        Self {
            reclamation: value.reclamation.into(),
            page: value.page.into(),
            class_id: value.class_id,
            backend_domain: value.backend_domain,
            reserved32: 0,
            logical_ordinal: value.logical_ordinal,
            backend_index: value.backend_index,
            token_begin: value.token_begin,
            token_end_exclusive: value.token_end_exclusive,
            completion_domain: value.completion_domain,
            completion_value: value.completion_value,
        }
    }
}

impl From<OrbitKvBackendUnobservedReceipt> for BackendUnobservedReceipt {
    fn from(value: OrbitKvBackendUnobservedReceipt) -> Self {
        Self {
            step: value.step.into(),
            backend_unobserved: value.backend_unobserved,
            reserved: value.reserved,
        }
    }
}

impl From<OrbitKvReleaseBatchItem> for ReleaseBatchItem {
    fn from(value: OrbitKvReleaseBatchItem) -> Self {
        Self {
            request: value.request.into(),
            expected_head: value.expected_head.into(),
        }
    }
}

impl From<OrbitKvReclamationReceipt> for ReclamationReceipt {
    fn from(value: OrbitKvReclamationReceipt) -> Self {
        Self {
            reclamation: value.reclamation.into(),
            page: value.page.into(),
            backend_domain: value.backend_domain,
            acknowledged: value.acknowledged,
            reserved8: value.reserved8,
            reserved32: value.reserved32,
            backend_index: value.backend_index,
        }
    }
}

impl From<OrbitKvPrefixSemanticKey> for PrefixSemanticKey {
    fn from(value: OrbitKvPrefixSemanticKey) -> Self {
        Self {
            namespace: value.namespace,
            digest: value.digest,
            boundary: value.boundary,
        }
    }
}

impl From<PrefixSemanticKey> for OrbitKvPrefixSemanticKey {
    fn from(value: PrefixSemanticKey) -> Self {
        Self {
            namespace: value.namespace,
            digest: value.digest,
            boundary: value.boundary,
        }
    }
}

impl From<PrefixLookupHint> for OrbitKvPrefixLookupHint {
    fn from(value: PrefixLookupHint) -> Self {
        let candidate = value
            .candidate
            .map_or_else(OrbitKvPrefixLease::default, Into::into);
        Self {
            key: value.key.into(),
            candidate,
            resident_count: value.resident_count,
            candidate_present: u32::from(value.candidate.is_some()),
            reserved: 0,
            reserved_padding: 0,
        }
    }
}

impl From<OrbitKvPrefixLookupHint> for PrefixLookupHint {
    fn from(value: OrbitKvPrefixLookupHint) -> Self {
        Self {
            key: value.key.into(),
            candidate: (value.candidate_present == 1).then(|| value.candidate.into()),
            resident_count: value.resident_count,
        }
    }
}

impl From<OrbitKvPrefixAttachBatchItem> for PrefixAttachItem {
    fn from(value: OrbitKvPrefixAttachBatchItem) -> Self {
        Self {
            request: value.request.into(),
            expected_empty_head: value.expected_empty_head.into(),
            hint: value.hint.into(),
        }
    }
}

impl From<OrbitKvPrefixPublishBatchItem> for PrefixPublishItem {
    fn from(value: OrbitKvPrefixPublishBatchItem) -> Self {
        Self {
            request: value.request.into(),
            expected_head: value.expected_head.into(),
            key: value.key.into(),
        }
    }
}

impl From<orbitkv::kv_manager::PublishedPrefix> for OrbitKvPublishedPrefix {
    fn from(value: orbitkv::kv_manager::PublishedPrefix) -> Self {
        Self {
            prefix: value.prefix.into(),
            key: value.key.into(),
            resident_count: value.resident_count,
            reserved: 0,
        }
    }
}

impl From<orbitkv::kv_manager::EvictedPrefix> for OrbitKvEvictedPrefix {
    fn from(value: orbitkv::kv_manager::EvictedPrefix) -> Self {
        Self {
            prefix: value.prefix.into(),
            key: value.key.into(),
        }
    }
}

impl From<ArenaStats> for OrbitKvArenaStats {
    fn from(value: ArenaStats) -> Self {
        Self {
            engine_epoch: value.engine_epoch,
            pool_epoch: value.pool_epoch,
            class_id: value.class_id,
            backend_domain: value.backend_domain,
            pool_id: value.pool_id,
            page_count: value.page_count,
            first_page_id: value.first_page_id,
            reserved: 0,
            reserved_padding: 0,
            free_pages: value.free_pages,
            reserved_pages: value.reserved_pages,
            writing_pages: value.writing_pages,
            active_pages: value.active_pages,
            retiring_pages: value.retiring_pages,
            quarantined_pages: value.quarantined_pages,
            exhausted_pages: value.exhausted_pages,
            request_page_refs: value.request_page_refs,
            prefix_page_refs: value.prefix_page_refs,
            reader_pins: value.reader_pins,
        }
    }
}

impl From<ManagerStats> for OrbitKvManagerStats {
    fn from(value: ManagerStats) -> Self {
        Self {
            active_requests: value.active_requests,
            active_snapshots: value.active_snapshots,
            active_prefixes: value.active_prefixes,
            evicted_prefixes: value.evicted_prefixes,
            prepared_steps: value.prepared_steps,
            submitted_steps: value.submitted_steps,
            free_pages: value.free_pages,
            reserved_pages: value.reserved_pages,
            writing_pages: value.writing_pages,
            active_pages: value.active_pages,
            retiring_pages: value.retiring_pages,
            quarantined_pages: value.quarantined_pages,
            exhausted_pages: value.exhausted_pages,
            pending_reclamations: value.pending_reclamations,
            total_request_page_refs: value.total_request_page_refs,
            total_prefix_page_refs: value.total_prefix_page_refs,
            total_reader_pins: value.total_reader_pins,
        }
    }
}
