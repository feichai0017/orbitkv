use serde::Serialize;

pub(in crate::kv_manager) const DEVICE_KV_ACCESS_READ: u32 = 1 << 0;
pub(in crate::kv_manager) const DEVICE_KV_ACCESS_WRITE: u32 = 1 << 1;
pub(in crate::kv_manager) const DEVICE_KV_NEEDS_BINDING: u32 = 1 << 2;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub(in crate::kv_manager) struct DeviceKvEntry {
    pub(in crate::kv_manager) class_id: u16,
    pub(in crate::kv_manager) backend_domain: u16,
    pub(in crate::kv_manager) access_flags: u32,
    pub(in crate::kv_manager) logical_ordinal: u64,
    pub(in crate::kv_manager) token_begin: u64,
    pub(in crate::kv_manager) valid_token_count: u32,
    pub(in crate::kv_manager) visible_token_offset: u32,
    pub(in crate::kv_manager) visible_token_count: u32,
    pub(in crate::kv_manager) pool_id: u32,
    pub(in crate::kv_manager) temporal_cell_index: u64,
    pub(in crate::kv_manager) temporal_cycle: u64,
    pub(in crate::kv_manager) pool_epoch: u64,
    pub(in crate::kv_manager) page_generation: u64,
    pub(in crate::kv_manager) backend_index: u64,
    pub(in crate::kv_manager) page_id: u32,
    pub(in crate::kv_manager) reserved: u32,
}

const _: [(); 88] = [(); std::mem::size_of::<DeviceKvEntry>()];
const _: [(); 8] = [(); std::mem::align_of::<DeviceKvEntry>()];
