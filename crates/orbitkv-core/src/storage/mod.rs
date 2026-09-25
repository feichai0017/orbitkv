pub(crate) mod inventory;
pub(crate) mod metadata;
pub(crate) mod read_cache;
pub(crate) mod transfer_lock;
pub(crate) mod write_path;

use bytesize::ByteSize;
use futures::{StreamExt, stream};
use log::{debug, info};
use std::collections::HashSet;
use std::num::NonZeroU64;
use std::sync::{Arc, Weak};
use std::time::Duration;

use crate::backing::{AllocateFn, SsdBackingStore, SsdCacheConfig};
#[cfg(feature = "mooncake")]
use crate::backing::{MooncakeFetchStore, MooncakeTransport};
use crate::block::{QueryResult, SealedBlock, StateKey};
use crate::internode::CatalogClient;
use crate::memory::numa::NumaNode;
use crate::memory::pool::{PinnedAllocation, PinnedAllocator};
use crate::metrics::core_metrics;

use crate::planning::replica::ReplicaSet;
use crate::query::read::ReadCoordinator;
pub(crate) use read_cache::ReadCache;
use write_path::{InsertDeps, WritePipeline};

const RECLAIM_BATCH_SIZE: usize = 512;
// One catalog budget across every bounded batch in a candidate discovery.
pub(crate) const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MemoryCacheCleanupStats {
    pub evicted_blocks: usize,
    pub evicted_bytes: u64,
    pub reclaimed_bytes: u64,
    pub still_referenced_blocks: u64,
}

#[derive(Clone)]
pub struct StorageConfig {
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

impl Default for StorageConfig {
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
            inventory_journal_bytes: inventory::DEFAULT_INVENTORY_JOURNAL_BYTES,
            pool_shards: 1,
        }
    }
}

pub(crate) enum TransferAuthorizationError {
    StaleReplica,
    Lock(transfer_lock::TransferLockError),
}

pub(crate) struct StorageEngine {
    allocator: Arc<PinnedAllocator>,
    pub(crate) codec: crate::StorageCodec,
    pub(crate) codec_budget: usize,
    read_cache: Arc<ReadCache>,
    reads: ReadCoordinator,
    write_pipeline: Arc<WritePipeline>,
    pub(crate) ssd_store: Option<Arc<SsdBackingStore>>,
    #[cfg(feature = "mooncake")]
    mooncake_transport: Option<Arc<MooncakeTransport>>,
    blockwise_alloc: bool,
    catalog_client: Option<Arc<CatalogClient>>,
    membership: Option<Arc<orbitkv_catalog::MembershipView>>,
    pub(crate) transfer_lock: Arc<transfer_lock::TransferLockManager>,
}

impl StorageEngine {
    pub(crate) fn new_with_config(
        capacity_bytes: usize,
        use_hugepages: bool,
        config: StorageConfig,
        numa_nodes: &[NumaNode],
    ) -> Result<Arc<Self>, String> {
        if config.codec_budget < 4096 || config.codec_budget > u32::MAX as usize {
            return Err("codec budget must be between 4 KiB and 4 GiB - 1".into());
        }
        if config.codec == crate::StorageCodec::Ans {
            crate::codec::ans::Library::load()?;
        }
        if config.cache_protected_percent > 100 {
            return Err("cache protected percent must be between 0 and 100".into());
        }
        let value_size_hint = config.hint_value_size_bytes.filter(|size| *size > 0);
        let unit_hint = value_size_hint.and_then(|size| NonZeroU64::new(size as u64));
        let ssd_cache_config = config.ssd_cache_config;
        #[cfg(feature = "mooncake")]
        let mooncake_nic_names = config.mooncake_nic_names;
        let blockwise_alloc = config.blockwise_alloc || ssd_cache_config.is_some();
        let transfer_lock_timeout = config.transfer_lock_timeout;
        let transfer_budget = config.transfer_budget_bytes.unwrap_or(capacity_bytes / 2);
        if transfer_budget == 0
            || transfer_budget > capacity_bytes
            || transfer_budget > i64::MAX as usize
        {
            return Err(
                "transfer budget must be positive and no larger than the pinned pool or i64::MAX"
                    .into(),
            );
        }

        if blockwise_alloc {
            info!("Blockwise allocation enabled for batch_save");
        }

        let cpu_readable = ssd_cache_config.is_some() || config.codec != crate::StorageCodec::None;
        let pool_shards = config.pool_shards;

        // Create unified allocator based on NUMA configuration
        let allocator = if !numa_nodes.is_empty() {
            info!(
                "Creating NUMA-aware pinned pools for {} nodes",
                numa_nodes.len()
            );
            Arc::new(PinnedAllocator::new_numa(
                capacity_bytes,
                numa_nodes,
                pool_shards,
                use_hugepages,
                cpu_readable,
                unit_hint,
            ))
        } else {
            info!("Creating global pinned pool (NUMA affinity disabled)");
            Arc::new(PinnedAllocator::new_global(
                capacity_bytes,
                pool_shards,
                use_hugepages,
                cpu_readable,
                unit_hint,
            ))
        };

        // Sub-components
        let read_cache = Arc::new(ReadCache::new(
            capacity_bytes,
            config.enable_lfu_admission,
            value_size_hint,
            config
                .membership
                .as_ref()
                .map(|_| config.inventory_journal_bytes),
            (capacity_bytes as u128 * config.cache_protected_percent as u128 / 100) as u64,
        ));

        let catalog_client = config
            .membership
            .as_ref()
            .map(|view| CatalogClient::new(view.clone(), Arc::downgrade(&read_cache)).map(Arc::new))
            .transpose()?;

        let (write_pipeline, insert_rx) = WritePipeline::new();
        let write_pipeline = Arc::new(write_pipeline);

        // Mooncake must be created after the allocator so it can register the
        // pinned pool. An empty rail filter lets Mooncake choose TCP fallback.
        #[cfg(feature = "mooncake")]
        let mooncake_transport = if catalog_client.is_some() {
            let advertise = &config
                .membership
                .as_ref()
                .expect("distributed configuration")
                .owner()
                .endpoint;
            let transfer =
                crate::backing::new_mooncake(&mooncake_nic_names, &allocator, advertise)?;
            Some(transfer)
        } else {
            None
        };

        #[cfg(not(feature = "mooncake"))]
        if catalog_client.is_some() {
            log::warn!(
                "Catalog was configured, but this binary was built without the `mooncake` feature; remote transfer is disabled"
            );
        }

        let is_numa = allocator.is_numa();
        let engine = Arc::new_cyclic(move |weak_engine: &Weak<Self>| {
            // Build shared allocate_fn for backing stores.
            let alloc_weak = weak_engine.clone();
            let allocate_fn: AllocateFn = Arc::new(move |size, numa_node| {
                alloc_weak
                    .upgrade()
                    .and_then(|engine| engine.allocate(NonZeroU64::new(size)?, numa_node))
            });

            let ssd_store = ssd_cache_config
                .map(|cfg| crate::backing::new_ssd(cfg, allocate_fn.clone(), is_numa));

            #[cfg(feature = "mooncake")]
            let remote_fetch = mooncake_transport.as_ref().and_then(|transfer| {
                let ms = catalog_client.as_ref()?;
                Some(Arc::new(MooncakeFetchStore::new(
                    Arc::clone(ms),
                    Arc::clone(transfer),
                    allocate_fn.clone(),
                    config
                        .membership
                        .clone()
                        .expect("distributed configuration"),
                )))
            });

            let reads = ReadCoordinator::new(
                ssd_store.clone(),
                #[cfg(feature = "mooncake")]
                remote_fetch,
                config.codec_budget,
            );

            let transfer_lock = Arc::new(transfer_lock::TransferLockManager::new(
                transfer_lock_timeout,
                transfer_budget as u64,
            ));

            Self {
                allocator,
                codec: config.codec,
                codec_budget: config.codec_budget,
                read_cache: read_cache.clone(),
                reads,
                write_pipeline: write_pipeline.clone(),
                ssd_store,
                #[cfg(feature = "mooncake")]
                mooncake_transport,
                blockwise_alloc,
                catalog_client,
                membership: config.membership.clone(),
                transfer_lock,
            }
        });

        // Spawn insert worker on a dedicated OS thread (CPU-bound work)
        {
            let deps = Arc::new(InsertDeps {
                read_cache: engine.read_cache.clone(),
                ssd_store: engine.ssd_store.clone(),
            });
            let weak_deps = Arc::downgrade(&deps);
            // Keep deps alive by leaking it into the thread. The worker holds
            // a Weak, so it won't prevent engine drop. The Arc is dropped when
            // the thread exits (channel closed).
            std::thread::Builder::new()
                .name("orbitkv-insert".into())
                .spawn(move || {
                    let _keep_alive = deps;
                    write_path::insert_worker_loop(insert_rx, weak_deps);
                })
                .expect("failed to spawn insert worker thread");
        }

        Ok(engine)
    }

    pub(crate) fn is_ssd_enabled(&self) -> bool {
        self.ssd_store.is_some()
    }

    pub(crate) fn is_numa_enabled(&self) -> bool {
        self.allocator.is_numa()
    }

    /// Returns true if blockwise allocation is enabled.
    pub(crate) fn blockwise_alloc(&self) -> bool {
        self.blockwise_alloc
    }

    /// Allocate pinned memory, returning `None` if the pool is exhausted after eviction.
    pub(crate) fn allocate(
        &self,
        size: NonZeroU64,
        numa_node: Option<NumaNode>,
    ) -> Option<Arc<PinnedAllocation>> {
        let requested_bytes = size.get();
        let node = numa_node.unwrap_or(NumaNode::UNKNOWN);

        loop {
            if let Some(alloc) = self.allocator.allocate(size, node) {
                return Some(alloc);
            }

            let (freed_blocks, _freed_bytes, largest_free) =
                self.reclaim_until_allocator_can_allocate(requested_bytes, node);

            if freed_blocks == 0 && largest_free < requested_bytes {
                // Final retry: absorb concurrent frees that may have raced with reclaim probing.
                if let Some(alloc) = self.allocator.allocate(size, node) {
                    return Some(alloc);
                }
                break;
            }
        }

        let (used, total) = self.allocator.usage();
        log::error!(
            "Pinned memory pool exhausted; cannot satisfy allocation: \
             requested={} used={} total={} numa={:?}",
            ByteSize(requested_bytes),
            ByteSize(used),
            ByteSize(total),
            numa_node
        );
        core_metrics().pool_alloc_failures.add(1, &[]);
        None
    }

    pub(crate) fn send_raw_insert(&self, batch: write_path::RawSaveBatch) {
        self.write_pipeline.send_raw_insert(batch);
    }

    /// Flush the write pipeline.
    ///
    /// Returns a receiver that resolves once all batches enqueued before this
    /// call have been processed by the insert worker.
    pub(crate) async fn flush_write_pipeline(&self) {
        if let Some(rx) = self.write_pipeline.flush() {
            let _ = rx.await;
        }
    }

    /// Flush the SSD writer: waits until all enqueued writes are committed.
    pub(crate) async fn flush_ssd(&self) {
        if let Some(ssd) = &self.ssd_store {
            ssd.flush().await;
        }
    }

    pub(crate) async fn flush_inventory(&self) -> Result<(), String> {
        if let Some(client) = &self.catalog_client {
            client.flush().await?;
        }
        Ok(())
    }

    pub(crate) fn filter_hashes_not_in_cache_inplace(
        &self,
        namespace: &str,
        hashes: &mut HashSet<Vec<u8>>,
    ) {
        let namespace = namespace.to_string();
        let hash_vec: Vec<Vec<u8>> = hashes.iter().cloned().collect();
        let keys: Vec<StateKey> = hash_vec
            .iter()
            .map(|hash| StateKey::new(namespace.clone(), hash.clone()))
            .collect();
        let present = self.read_cache.contains_keys(&keys);
        for (hash, is_present) in hash_vec.into_iter().zip(present) {
            if is_present {
                hashes.remove(&hash);
            }
        }
    }

    /// Availability hints; concurrent eviction can invalidate them immediately.
    /// Only a subsequent payload read and lease establishes recoverability.
    pub(crate) async fn discover(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
        deadline: tokio::time::Instant,
    ) -> Vec<ReplicaSet> {
        #[cfg(not(feature = "mooncake"))]
        let _ = deadline;
        let keys: Vec<_> = hashes
            .iter()
            .map(|hash| StateKey::new(namespace.to_owned(), hash.clone()))
            .collect();
        let mut candidates: Vec<_> = keys
            .iter()
            .cloned()
            .zip(self.read_cache.discover(&keys))
            .map(|(key, dram)| {
                let mut candidates = ReplicaSet::new(key);
                if let Some(dram) = dram {
                    candidates.set_memory(dram);
                }
                candidates
            })
            .collect();
        if let Some(ssd) = &self.ssd_store {
            for (candidate, backing) in candidates.iter_mut().zip(ssd.discover(&keys)) {
                if let Some(backing) = backing {
                    candidate.set_ssd(backing);
                }
            }
        }
        #[cfg(feature = "mooncake")]
        if let Some(catalog) = &self.catalog_client {
            for (candidate, cached) in candidates.iter_mut().zip(catalog.cached_blocks(&keys)) {
                if let Some(cached) = cached {
                    candidate.set_peer_dram(cached.replicas);
                }
            }
            let missing: Vec<_> = candidates
                .iter()
                .enumerate()
                .filter_map(|(i, candidate)| (!candidate.is_available()).then_some(i))
                .collect();
            let hashes: Vec<_> = missing.iter().map(|&i| hashes[i].clone()).collect();
            if !hashes.is_empty() && tokio::time::Instant::now() < deadline {
                let remote = match tokio::time::timeout_at(
                    deadline,
                    catalog.locate_blocks(namespace, &hashes),
                )
                .await
                {
                    Ok(Ok(remote)) => remote.into_iter().map(Some).collect(),
                    Ok(Err(error)) => {
                        log::warn!("candidate discovery failed: {error}");
                        Vec::new()
                    }
                    Err(_) => {
                        // Healthy peers may have published evidence before a
                        // different peer exhausted this discovery's deadline.
                        let keys: Vec<_> = missing.iter().map(|&i| keys[i].clone()).collect();
                        catalog.cached_blocks(&keys)
                    }
                };
                for (i, candidate) in missing.into_iter().zip(remote) {
                    if let Some(candidate) = candidate
                        && candidates[i].key == candidate.key
                    {
                        candidates[i].set_peer_dram(candidate.replicas);
                    }
                }
            }
        }
        candidates
    }

    /// Position-aligned membership across resident and backing tiers: entry
    /// `i` is the sealed block for `hashes[i]`, or `None` on miss. Hashes must
    /// already carry any group encoding (see `group_hash`).
    pub(crate) async fn get_membership(
        &self,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        mode: crate::QueryMode,
    ) -> Vec<Option<crate::RestoreSource>> {
        let keys: Vec<StateKey> = hashes
            .iter()
            .map(|hash| StateKey::new(namespace.to_string(), hash.clone()))
            .collect();
        let resident = self.read_cache.get_blocks_aligned(&keys);
        // Auxiliary state can have holes (checkpoints or evicted windows).
        // Bound independent reads and reuse prefix fetch coalescing/cancellation
        // without waiting for absent checkpoints to be published.
        stream::iter(
            hashes
                .iter()
                .cloned()
                .zip(resident)
                .map(|(hash, block)| async move {
                    if block.is_some() {
                        return block.map(crate::RestoreSource::Memory);
                    }
                    self.reads
                        .read_prefix(
                            &self.read_cache,
                            req_id,
                            namespace,
                            std::slice::from_ref(&hash),
                            mode,
                        )
                        .await
                        .blocks
                        .pop()
                }),
        )
        .buffered(8)
        .collect()
        .await
    }

    /// Evict all blocks from the resident in-memory read cache.
    ///
    /// This preserves backing-store copies. Blocks with
    /// outstanding references may remain allocated until those holders release
    /// the last `Arc`.
    pub(crate) fn cleanup_memory_cache(&self) -> MemoryCacheCleanupStats {
        let used_before = self.allocator.usage().0;
        let evicted = self.read_cache.remove_all();
        if evicted.is_empty() {
            return MemoryCacheCleanupStats::default();
        }

        let mut evicted_bytes = 0u64;
        let mut still_referenced_blocks = 0u64;
        for (_key, block) in &evicted {
            let bytes = block.memory_footprint();
            evicted_bytes = evicted_bytes.saturating_add(bytes);
            if Arc::strong_count(block) > 1 {
                still_referenced_blocks += 1;
            }
            core_metrics()
                .cache_resident_bytes
                .add(-(bytes as i64), &[]);
        }

        let evicted_blocks = evicted.len();
        if still_referenced_blocks > 0 {
            core_metrics()
                .cache_block_evictions_still_referenced
                .add(still_referenced_blocks, &[]);
        }
        core_metrics()
            .cache_block_evictions
            .add(evicted_blocks as u64, &[]);

        drop(evicted);
        let reclaimed_bytes = used_before.saturating_sub(self.allocator.usage().0);
        if reclaimed_bytes > 0 {
            core_metrics()
                .cache_eviction_reclaimed_bytes
                .add(reclaimed_bytes, &[]);
        }

        info!(
            "Cleaned resident memory cache: evicted_blocks={} evicted_bytes={} reclaimed_bytes={} still_referenced_blocks={}",
            evicted_blocks,
            ByteSize(evicted_bytes),
            ByteSize(reclaimed_bytes),
            still_referenced_blocks
        );

        MemoryCacheCleanupStats {
            evicted_blocks,
            evicted_bytes,
            reclaimed_bytes,
            still_referenced_blocks,
        }
    }

    /// Check prefix blocks and schedule backing-store reads if needed.
    pub(crate) async fn check_prefix_and_prefetch(
        &self,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        mode: crate::QueryMode,
    ) -> QueryResult {
        self.reads
            .read_prefix(&self.read_cache, req_id, namespace, hashes, mode)
            .await
    }

    fn reclaim_until_allocator_can_allocate(
        &self,
        required_bytes: u64,
        target_node: NumaNode,
    ) -> (usize, u64, u64) {
        if required_bytes == 0 {
            return (
                0,
                0,
                self.allocator.largest_free_allocation_for_node(target_node),
            );
        }

        let mut freed_blocks = 0usize;
        let mut freed_bytes = 0u64;
        let mut largest_free = self.allocator.largest_free_allocation_for_node(target_node);

        while largest_free < required_bytes {
            let used_before = self.allocator.usage().0;

            let evicted = self
                .read_cache
                .remove_lru_batch(RECLAIM_BATCH_SIZE, required_bytes);

            if evicted.is_empty() {
                break;
            }

            let mut batch_bytes = 0u64;
            for (_key, block) in &evicted {
                let b = block.memory_footprint();
                batch_bytes = batch_bytes.saturating_add(b);
                core_metrics().cache_resident_bytes.add(-(b as i64), &[]);
            }

            freed_bytes = freed_bytes.saturating_add(batch_bytes);
            freed_blocks += evicted.len();

            drop(evicted);
            let used_after = self.allocator.usage().0;
            let reclaimed = used_before.saturating_sub(used_after);
            if reclaimed > 0 {
                core_metrics()
                    .cache_eviction_reclaimed_bytes
                    .add(reclaimed, &[]);
            }

            largest_free = self.allocator.largest_free_allocation_for_node(target_node);
        }

        if freed_blocks > 0 {
            debug!(
                "Reclaimed cache blocks toward allocator request: \
                 freed_blocks={} freed_bytes={} largest_free={} required={}",
                freed_blocks,
                ByteSize(freed_bytes),
                ByteSize(largest_free),
                ByteSize(required_bytes)
            );
            core_metrics()
                .cache_block_evictions
                .add(freed_blocks as u64, &[]);
        }

        (freed_blocks, freed_bytes, largest_free)
    }

    pub(crate) async fn gc_stale_inflight(&self, max_age: std::time::Duration) -> usize {
        self.write_pipeline.gc_stale_inflight(max_age).await
    }

    // ---- Cross-node transfer: serving side ----

    pub(crate) fn validate_transfer_owner(
        &self,
        owner: uuid::Uuid,
    ) -> Result<(), TransferAuthorizationError> {
        if self
            .catalog_client
            .as_ref()
            .is_none_or(|client| client.node_id != owner)
            || self
                .membership
                .as_ref()
                .is_some_and(|view| !view.permits(view.owner()))
        {
            return Err(TransferAuthorizationError::StaleReplica);
        }
        Ok(())
    }

    pub(crate) fn authorize_transfer(
        &self,
        owner: uuid::Uuid,
        ticket: transfer_lock::TransferTicket,
        records: &[orbitkv_state::InventoryRecord],
    ) -> Result<Vec<(StateKey, Arc<SealedBlock>)>, TransferAuthorizationError> {
        self.validate_transfer_owner(owner)?;
        let found = self
            .read_cache
            .pin_residencies(records)
            .ok_or(TransferAuthorizationError::StaleReplica)?;
        self.transfer_lock
            .lock_blocks(ticket, found.clone())
            .map_err(TransferAuthorizationError::Lock)?;
        Ok(found)
    }

    /// Return `(base_ptr, size)` for each contiguous pinned memory region.
    /// Used for Mooncake memory registration.
    pub(crate) fn pinned_memory_regions(&self) -> Vec<(u64, usize)> {
        self.allocator
            .memory_regions()
            .into_iter()
            .map(|(ptr, len)| (ptr.as_ptr() as u64, len))
            .collect()
    }

    #[cfg(feature = "mooncake")]
    pub(crate) fn mooncake_transport(&self) -> Option<&Arc<MooncakeTransport>> {
        self.mooncake_transport.as_ref()
    }

    #[cfg(feature = "mooncake")]
    pub(crate) fn transfer_endpoint(&self) -> Option<&str> {
        self.mooncake_transport
            .as_ref()
            .map(|transport| transport.transfer_endpoint())
    }

    pub(crate) async fn shutdown_catalog_client(&self) {
        if let Some(client) = &self.catalog_client {
            client.shutdown().await;
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/storage/mod.rs"]
mod tests;
