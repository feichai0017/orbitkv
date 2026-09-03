use orbitkv::kv_manager::{
    ArenaStats, BackendArenaRegistration, ClassLowering, CopyIntent, DetachedBinding,
    ManagerConfig, ManagerStats, PageLease, PrefixSemanticKey, SnapshotPage, TailAction,
    TokenDisposition, TokenDispositionKind, TokenLocation, TokenPlacement, WriteIntent,
};

use super::{
    OrbitKvArenaStats, OrbitKvBackendArenaRegistration, OrbitKvClassLowering, OrbitKvCopyIntent,
    OrbitKvDetachedBinding, OrbitKvManagerConfig, OrbitKvManagerStats, OrbitKvPageLease,
    OrbitKvPrefixSemanticKey, OrbitKvSnapshotPage, OrbitKvTailAction, OrbitKvTokenDisposition,
    OrbitKvTokenLocation, OrbitKvTokenPlacement, OrbitKvWriteIntent,
};

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
            previous_layout_boundary: value.previous_layout_boundary,
            target_layout_boundary: value.target_layout_boundary,
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

impl From<OrbitKvTokenDisposition> for TokenDisposition {
    fn from(value: OrbitKvTokenDisposition) -> Self {
        let kind = match value.kind {
            1 => TokenDispositionKind::SemanticallyDead,
            2 => TokenDispositionKind::PolicyEvicted,
            _ => TokenDispositionKind::Retained,
        };
        Self {
            kind,
            policy_or_proof_id: value.policy_or_proof_id,
            version: value.version,
            quality_contract: value.quality_contract,
        }
    }
}

impl From<TokenDisposition> for OrbitKvTokenDisposition {
    fn from(value: TokenDisposition) -> Self {
        Self {
            policy_or_proof_id: value.policy_or_proof_id,
            version: value.version,
            quality_contract: value.quality_contract,
            kind: value.kind as u16,
            reserved16: 0,
            reserved32: 0,
        }
    }
}

impl From<OrbitKvTokenLocation> for TokenLocation {
    fn from(value: OrbitKvTokenLocation) -> Self {
        Self {
            page: value.page.into(),
            backend_index: value.backend_index,
            offset: value.offset,
            reserved: value.reserved,
        }
    }
}

impl From<TokenLocation> for OrbitKvTokenLocation {
    fn from(value: TokenLocation) -> Self {
        Self {
            page: value.page.into(),
            backend_index: value.backend_index,
            offset: value.offset,
            reserved: value.reserved,
        }
    }
}

impl From<TokenPlacement> for OrbitKvTokenPlacement {
    fn from(value: TokenPlacement) -> Self {
        Self {
            token_id: value.token_id,
            disposition: value.disposition.into(),
            location: value
                .location
                .map_or_else(OrbitKvTokenLocation::default, Into::into),
            location_present: u32::from(value.location.is_some()),
            reserved: 0,
        }
    }
}
