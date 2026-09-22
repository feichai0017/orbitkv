#[cfg(feature = "mooncake")]
mod fetch_plan;
#[cfg(feature = "mooncake")]
pub(super) mod mooncake;
#[cfg(feature = "mooncake")]
pub(super) mod mooncake_fetch;
pub(super) mod ssd;
pub(super) mod ssd_cache;
#[cfg(feature = "mooncake")]
mod transfer_lock_guard;
pub(super) mod uring;

use std::sync::Arc;

pub(crate) use ssd_cache::SSD_ALIGNMENT;
#[allow(
    unreachable_pub,
    reason = "SSD config types are re-exported through the public crate API"
)]
pub use ssd_cache::{
    DEFAULT_SSD_PREFETCH_INFLIGHT, DEFAULT_SSD_PREFETCH_QUEUE_DEPTH, DEFAULT_SSD_WRITE_INFLIGHT,
    DEFAULT_SSD_WRITE_QUEUE_DEPTH, SsdCacheConfig,
};

use crate::block::{SealedBlock, StateKey};
use crate::numa::NumaNode;
use crate::pinned_pool::PinnedAllocation;

#[cfg(feature = "mooncake")]
pub(crate) use mooncake::{MooncakeTransport, new_mooncake};
#[cfg(feature = "mooncake")]
pub(crate) use mooncake_fetch::MooncakeFetchStore;
pub(crate) use ssd::SsdBackingStore;
pub(crate) use ssd::new_ssd;

pub(crate) type PrefetchResult = Vec<(StateKey, Arc<SealedBlock>)>;

/// Allocator closure for pinned memory, passed to the SSD backing store.
pub(crate) type AllocateFn =
    Arc<dyn Fn(u64, Option<NumaNode>) -> Option<Arc<PinnedAllocation>> + Send + Sync>;
