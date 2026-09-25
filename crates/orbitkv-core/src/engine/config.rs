use std::sync::Arc;
use std::time::Duration;

use crate::SsdCacheConfig;

#[derive(Clone)]
pub struct EngineConfig {
    /// Query-owned bytes across preparation, ready leases, and GPU loads.
    /// Defaults to three quarters of the pinned pool; the allocator remains
    /// the physical-memory limit, including publish and cache residency.
    pub query_budget_bytes: Option<usize>,
    /// Per-instance query limit, defaulting to the global query limit.
    pub query_instance_budget_bytes: Option<usize>,
    pub enable_lfu_admission: bool,
    /// Maximum percent of host capacity in demand-promoted cache entries.
    /// Zero keeps the ordinary LRU classes; positive values enable segmented LRU.
    pub cache_protected_percent: u8,
    /// Optional hint for expected value size in bytes (tunes cache + allocator granularity).
    pub hint_value_size_bytes: Option<usize>,
    /// Optional SSD cache for sealed blocks (single-node, FIFO).
    pub ssd_cache_config: Option<SsdCacheConfig>,
    pub codec: crate::StorageCodec,
    /// GPU scratch per transfer worker.
    pub codec_budget: usize,
    /// Optional Mooncake RDMA rail filter. Empty means that Mooncake selects
    /// the available transport, including TCP fallback.
    pub mooncake_nic_names: Vec<String>,
    /// Enable NUMA-aware memory allocation.
    pub enable_numa_affinity: bool,
    /// Allocate each block separately instead of contiguous batch allocation.
    /// Reduces fragmentation when blocks are freed in different order.
    /// SSD-backed storage always uses independent allocations for reads and saves.
    pub blockwise_alloc: bool,
    /// Overdue threshold for cross-node transfers; expiry retains source allocations.
    pub transfer_lock_timeout: Duration,
    /// Source allocation reservations, including overdue transfers. Defaults to half the pool.
    pub transfer_budget_bytes: Option<usize>,
    /// Optional leased membership. Its incarnation also identifies this inventory.
    pub membership: Option<Arc<orbitkv_catalog::MembershipView>>,
    /// Byte limit for retained residency changes used by directory synchronization.
    pub inventory_journal_bytes: usize,
    /// Number of shards for the pinned memory pool (reduces allocator lock contention).
    pub pool_shards: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            query_budget_bytes: None,
            query_instance_budget_bytes: None,
            enable_lfu_admission: false,
            cache_protected_percent: 0,
            hint_value_size_bytes: None,
            ssd_cache_config: None,
            codec: crate::StorageCodec::None,
            codec_budget: 64 * 1024 * 1024,
            mooncake_nic_names: Vec::new(),
            enable_numa_affinity: true,
            blockwise_alloc: false,
            transfer_lock_timeout: Duration::from_secs(120),
            transfer_budget_bytes: None,
            membership: None,
            inventory_journal_bytes:
                crate::storage::dram::inventory::DEFAULT_INVENTORY_JOURNAL_BYTES,
            pool_shards: 1,
        }
    }
}
