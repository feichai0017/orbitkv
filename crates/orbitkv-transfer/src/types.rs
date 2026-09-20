use std::ptr::NonNull;

pub(crate) const INVALID_BATCH: u64 = u64::MAX;
pub(crate) const STATUS_WAITING: i32 = 0;
pub(crate) const STATUS_PENDING: i32 = 1;
pub(crate) const STATUS_COMPLETED: i32 = 4;

pub const P2P_METADATA: &str = "P2PHANDSHAKE";
pub const AUTO_MEMORY_LOCATION: &str = "*";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferOp {
    Read,
    Write,
}

impl TransferOp {
    pub(crate) const fn code(self) -> i32 {
        match self {
            Self::Read => 0,
            Self::Write => 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TransferSlice {
    pub local: NonNull<u8>,
    pub remote_address: u64,
    pub length: usize,
}

unsafe impl Send for TransferSlice {}
unsafe impl Sync for TransferSlice {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Notification {
    pub name: String,
    pub message: String,
}
