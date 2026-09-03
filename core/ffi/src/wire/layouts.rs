pub const ORBITKV_PLAN_FORMAT_KV_PLAN: u32 = 1;
pub const ORBITKV_PLAN_FORMAT_RETENTION_IR: u32 = 2;
pub const ORBITKV_CLASS_LOWERING_PACKED: u16 = orbitkv::kv_manager::CLASS_LOWERING_PACKED;
pub const ORBITKV_CLASS_LOWERING_RESETTABLE: u16 = orbitkv::kv_manager::CLASS_LOWERING_RESETTABLE;
pub const ORBITKV_CLASS_LOWERING_EPOCH_START: u16 = orbitkv::kv_manager::CLASS_LOWERING_EPOCH_START;

pub const ORBITKV_TAIL_NONE: u16 = 0;
pub const ORBITKV_TAIL_IN_PLACE: u16 = 1;
pub const ORBITKV_TAIL_COPY_ON_WRITE: u16 = 2;
pub const ORBITKV_TAIL_FRESH: u16 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct OrbitKvPageLease {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub generation: u64,
    pub page_id: u32,
    pub pool_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvBackendArenaRegistration {
    pub pool_id: u32,
    pub class_id: u16,
    pub backend_domain: u16,
    pub page_count: u32,
    pub reserved: u32,
    pub backend_base_index: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvManagerConfig {
    pub maximum_requests: u32,
    pub maximum_operations: u32,
    pub maximum_prefixes: u32,
    pub maximum_reclamations: u32,
    pub maximum_step_tokens: u32,
    pub plan_format: u32,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvArenaIdentity {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub backend_base_index: u64,
    pub pool_id: u32,
    pub page_count: u32,
    pub page_tokens: u32,
    pub class_id: u16,
    pub backend_domain: u16,
    pub first_page_id: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvArenaStats {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub class_id: u16,
    pub backend_domain: u16,
    pub pool_id: u32,
    pub page_count: u32,
    pub first_page_id: u32,
    pub reserved: u32,
    pub reserved_padding: u32,
    pub free_pages: u64,
    pub reserved_pages: u64,
    pub writing_pages: u64,
    pub active_pages: u64,
    pub retiring_pages: u64,
    pub quarantined_pages: u64,
    pub exhausted_pages: u64,
    pub request_page_refs: u64,
    pub prefix_page_refs: u64,
    pub reader_pins: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvSnapshotPage {
    pub page: OrbitKvPageLease,
    pub logical_ordinal: u64,
    pub temporal_cell_index: u64,
    pub temporal_cycle: u64,
    pub backend_index: u64,
    pub class_id: u16,
    pub backend_domain: u16,
    pub valid_token_count: u32,
    pub visible_token_offset: u32,
    pub visible_token_count: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvClassLowering {
    pub class_id: u16,
    pub flags: u16,
    pub tail_offset: u32,
    pub tail_count: u32,
    pub copy_offset: u32,
    pub copy_count: u32,
    pub write_offset: u32,
    pub write_count: u32,
    pub reserved: u32,
    pub previous_layout_boundary: u64,
    pub target_layout_boundary: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvTailAction {
    pub class_id: u16,
    pub kind: u16,
    pub valid_token_count: u32,
    pub logical_ordinal: u64,
    pub source: OrbitKvPageLease,
    pub destination: OrbitKvPageLease,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvCopyIntent {
    pub class_id: u16,
    pub backend_domain: u16,
    pub token_count: u32,
    pub source_token_offset: u32,
    pub destination_token_offset: u32,
    pub reserved: u32,
    pub source: OrbitKvPageLease,
    pub destination: OrbitKvPageLease,
    pub source_backend_index: u64,
    pub destination_backend_index: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvWriteIntent {
    pub page_generation: u64,
    pub page_id: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvDetachedBinding {
    pub old: OrbitKvPageLease,
    pub replacement: OrbitKvPageLease,
    pub logical_ordinal: u64,
    pub old_backend_index: u64,
    pub replacement_backend_index: u64,
    pub token_begin: u64,
    pub token_end_exclusive: u64,
    pub class_id: u16,
    pub backend_domain: u16,
    pub action: u16,
    pub reason: u16,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvPrefixSemanticKey {
    pub namespace: [u8; 32],
    pub digest: [u8; 32],
    pub boundary: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvManagerStats {
    pub active_requests: u64,
    pub active_snapshots: u64,
    pub active_prefixes: u64,
    pub evicted_prefixes: u64,
    pub prepared_steps: u64,
    pub submitted_steps: u64,
    pub free_pages: u64,
    pub reserved_pages: u64,
    pub writing_pages: u64,
    pub active_pages: u64,
    pub retiring_pages: u64,
    pub quarantined_pages: u64,
    pub exhausted_pages: u64,
    pub pending_reclamations: u64,
    pub total_request_page_refs: u64,
    pub total_prefix_page_refs: u64,
    pub total_reader_pins: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvTokenDisposition {
    pub policy_or_proof_id: u64,
    pub version: u64,
    pub quality_contract: u64,
    pub kind: u16,
    pub reserved16: u16,
    pub reserved32: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvTokenLocation {
    pub page: OrbitKvPageLease,
    pub backend_index: u64,
    pub offset: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvTokenPlacement {
    pub token_id: u64,
    pub disposition: OrbitKvTokenDisposition,
    pub location: OrbitKvTokenLocation,
    pub location_present: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvClassTokenDispositionUpdate {
    pub token_id: u64,
    pub disposition: OrbitKvTokenDisposition,
    pub class_id: u16,
    pub reserved16: u16,
    pub reserved32: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvRelocationPolicy {
    pub maximum_source_pages: u32,
    pub evacuation_headroom_pages: u32,
    pub fragmentation_threshold_milli: u16,
    pub full_evacuation: u8,
    pub reserved8: u8,
    pub reserved32: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitKvTokenMove {
    pub token_id: u64,
    pub source: OrbitKvTokenLocation,
    pub destination: OrbitKvTokenLocation,
}

macro_rules! wire_layout {
    ($ty:ty, $size:expr, $align:expr; $($field:ident = $offset:expr),+ $(,)?) => {
        const _: [(); $size] = [(); std::mem::size_of::<$ty>()];
        const _: [(); $align] = [(); std::mem::align_of::<$ty>()];
        $(const _: [(); $offset] = [(); std::mem::offset_of!($ty, $field)];)+
    };
}

wire_layout!(OrbitKvPageLease, 32, 8; engine_epoch = 0, pool_epoch = 8, generation = 16, page_id = 24, pool_id = 28);
wire_layout!(OrbitKvBackendArenaRegistration, 24, 8; pool_id = 0, class_id = 4, backend_domain = 6, page_count = 8, reserved = 12, backend_base_index = 16);
wire_layout!(OrbitKvManagerConfig, 32, 8; maximum_requests = 0, maximum_operations = 4, maximum_prefixes = 8, maximum_reclamations = 12, maximum_step_tokens = 16, plan_format = 20, reserved = 24);
wire_layout!(OrbitKvArenaIdentity, 48, 8; engine_epoch = 0, pool_epoch = 8, backend_base_index = 16, pool_id = 24, page_count = 28, page_tokens = 32, class_id = 36, backend_domain = 38, first_page_id = 40, reserved = 44);
wire_layout!(OrbitKvArenaStats, 120, 8; engine_epoch = 0, pool_epoch = 8, class_id = 16, backend_domain = 18, pool_id = 20, page_count = 24, first_page_id = 28, reserved = 32, reserved_padding = 36, free_pages = 40, reserved_pages = 48, writing_pages = 56, active_pages = 64, retiring_pages = 72, quarantined_pages = 80, exhausted_pages = 88, request_page_refs = 96, prefix_page_refs = 104, reader_pins = 112);
wire_layout!(OrbitKvSnapshotPage, 88, 8; page = 0, logical_ordinal = 32, temporal_cell_index = 40, temporal_cycle = 48, backend_index = 56, class_id = 64, backend_domain = 66, valid_token_count = 68, visible_token_offset = 72, visible_token_count = 76, reserved = 80);
wire_layout!(OrbitKvClassLowering, 48, 8; class_id = 0, flags = 2, tail_offset = 4, tail_count = 8, copy_offset = 12, copy_count = 16, write_offset = 20, write_count = 24, reserved = 28, previous_layout_boundary = 32, target_layout_boundary = 40);
wire_layout!(OrbitKvTailAction, 88, 8; class_id = 0, kind = 2, valid_token_count = 4, logical_ordinal = 8, source = 16, destination = 48, reserved = 80);
wire_layout!(OrbitKvCopyIntent, 104, 8; class_id = 0, backend_domain = 2, token_count = 4, source_token_offset = 8, destination_token_offset = 12, reserved = 16, source = 24, destination = 56, source_backend_index = 88, destination_backend_index = 96);
wire_layout!(OrbitKvWriteIntent, 16, 8; page_generation = 0, page_id = 8, reserved = 12);
wire_layout!(OrbitKvDetachedBinding, 120, 8; old = 0, replacement = 32, logical_ordinal = 64, old_backend_index = 72, replacement_backend_index = 80, token_begin = 88, token_end_exclusive = 96, class_id = 104, backend_domain = 106, action = 108, reason = 110, reserved = 112);
wire_layout!(OrbitKvPrefixSemanticKey, 72, 8; namespace = 0, digest = 32, boundary = 64);
wire_layout!(OrbitKvManagerStats, 136, 8; active_requests = 0, active_snapshots = 8, active_prefixes = 16, evicted_prefixes = 24, prepared_steps = 32, submitted_steps = 40, free_pages = 48, reserved_pages = 56, writing_pages = 64, active_pages = 72, retiring_pages = 80, quarantined_pages = 88, exhausted_pages = 96, pending_reclamations = 104, total_request_page_refs = 112, total_prefix_page_refs = 120, total_reader_pins = 128);
wire_layout!(OrbitKvTokenDisposition, 32, 8; policy_or_proof_id = 0, version = 8, quality_contract = 16, kind = 24, reserved16 = 26, reserved32 = 28);
wire_layout!(OrbitKvTokenLocation, 48, 8; page = 0, backend_index = 32, offset = 40, reserved = 44);
wire_layout!(OrbitKvTokenPlacement, 96, 8; token_id = 0, disposition = 8, location = 40, location_present = 88, reserved = 92);
wire_layout!(OrbitKvClassTokenDispositionUpdate, 48, 8; token_id = 0, disposition = 8, class_id = 40, reserved16 = 42, reserved32 = 44);
wire_layout!(OrbitKvRelocationPolicy, 16, 4; maximum_source_pages = 0, evacuation_headroom_pages = 4, fragmentation_threshold_milli = 8, full_evacuation = 10, reserved8 = 11, reserved32 = 12);
wire_layout!(OrbitKvTokenMove, 104, 8; token_id = 0, source = 8, destination = 56);
