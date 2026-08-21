use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct RequestLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

/// Generation-checked identity of one immutable request view.
///
/// ABI6 makes snapshots first-class: request state contains only a head lease,
/// while the snapshot payload lives in a bounded arena.  A recycled slot can
/// therefore never make an old request/prefix hint name a new view.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct SnapshotLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct PrefixLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

/// Token-exact semantic identity for one page-aligned prefix bundle.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct PrefixSemanticKey {
    pub namespace: [u8; 32],
    pub digest: [u8; 32],
    pub boundary: u64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct StepLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct SubmissionLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct ReclamationLease {
    pub engine_epoch: u64,
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct PageLease {
    pub engine_epoch: u64,
    pub pool_epoch: u64,
    pub generation: u64,
    pub page_id: u32,
    pub pool_id: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(transparent)]
pub struct ViewVersion(pub u64);
