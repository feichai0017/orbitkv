//! Framework-neutral cache engine with leased state, tiered storage and GPU transfer.

#[cfg(feature = "test-hooks")]
#[path = "../tests/support/faults.rs"]
pub mod test_faults;

#[macro_use]
mod trace;
mod backing;
mod block;
mod engine;
mod internode;
mod memory;
mod metrics;
mod query;
mod storage;
pub mod transfer;

pub use backing::ssd::SsdReadLease;
pub use backing::{
    DEFAULT_SSD_PREFETCH_INFLIGHT, DEFAULT_SSD_PREFETCH_QUEUE_DEPTH, DEFAULT_SSD_WRITE_INFLIGHT,
    DEFAULT_SSD_WRITE_QUEUE_DEPTH, SsdBackend, SsdCacheConfig, SsdCompression, SsdWritePolicy,
};
pub use block::{
    BlockHash, LayerBlock, LayerSave, QueryResult, RawBlock, RestoreSource, SealedBlock, StateKey,
};
pub use engine::instance::{GpuContext, InstanceContext};
pub use engine::{EngineError, OrbitKVEngine};
pub use internode::P2pTransferService;
pub use memory::numa::NumaNode;
pub use memory::pool::PinnedAllocation;
pub use orbitkv_state::{
    BundleComponent, LocalPageRef, RecoveryContract, StateBundle, StateComponent, StateDescriptor,
    StateFormat, TokenRange,
};
pub use query::lease::QueryLeaseId;
pub use query::{QueryAdmission, QueryMode, QueryOwner, QueryReservation};
pub use storage::inventory::DEFAULT_INVENTORY_JOURNAL_BYTES;
pub use storage::metadata::SlotMeta;
pub use storage::{MemoryCacheCleanupStats, StorageConfig};
pub use trace::{set_trace_sample_rate, should_sample};
pub use transfer::TransferMode;
pub use transfer::worker::LoadOutcome;
