use serde::Serialize;

pub(super) const DEVICE_KV_ACCESS_READ: u32 = 1 << 0;
pub(super) const DEVICE_KV_ACCESS_WRITE: u32 = 1 << 1;
pub(super) const DEVICE_KV_NEEDS_BINDING: u32 = 1 << 2;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub(super) struct DeviceKvEntry {
    pub(super) class_id: u16,
    pub(super) backend_domain: u16,
    pub(super) access_flags: u32,
    pub(super) logical_ordinal: u64,
    pub(super) token_begin: u64,
    pub(super) valid_token_count: u32,
    pub(super) visible_token_offset: u32,
    pub(super) visible_token_count: u32,
    pub(super) pool_id: u32,
    pub(super) temporal_cell_index: u64,
    pub(super) temporal_cycle: u64,
    pub(super) pool_epoch: u64,
    pub(super) page_generation: u64,
    pub(super) backend_index: u64,
    pub(super) page_id: u32,
    pub(super) reserved: u32,
}

const _: [(); 88] = [(); std::mem::size_of::<DeviceKvEntry>()];
const _: [(); 8] = [(); std::mem::align_of::<DeviceKvEntry>()];
