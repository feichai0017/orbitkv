//! Pinned allocation, memory accounting, NUMA placement.
pub(crate) mod allocator;
pub(crate) mod numa;
pub(crate) mod pinned;
pub(crate) mod pool;

use std::sync::Arc;

/// Pinned allocation with local reclamation, used by physical read workers.
pub(crate) type AllocateFn =
    Arc<dyn Fn(u64, Option<numa::NumaNode>) -> Option<Arc<pool::PinnedAllocation>> + Send + Sync>;
