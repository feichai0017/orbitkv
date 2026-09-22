//! OrbitKV Core Engine
//!
//! A GPU-aware KV cache offloading engine with support for:
//! - Multi-tenant instance isolation
//! - Tensor parallelism (TP) across multiple GPUs
//! - Split-storage layout for efficient K/V batch transfers
//! - SSD caching tier
//! - Catalog-backed block discovery and Mooncake remote fetch

#[cfg(feature = "test-hooks")]
#[path = "../tests/support/faults.rs"]
pub mod test_faults;

#[macro_use]
mod trace;

mod allocator;
mod backing;
mod block;
mod cache;
mod gpu_worker;
mod instance;
mod internode;
mod layout;
mod lease;
mod metrics;
mod numa;
mod offload;
mod pinned_mem;
mod pinned_pool;
mod query;
mod seal_offload;
mod storage;
pub mod sync_state;
pub mod transfer;

pub use crate::numa::NumaNode;
pub use backing::{
    DEFAULT_SSD_PREFETCH_INFLIGHT, DEFAULT_SSD_PREFETCH_QUEUE_DEPTH, DEFAULT_SSD_WRITE_INFLIGHT,
    DEFAULT_SSD_WRITE_QUEUE_DEPTH, SsdCacheConfig,
};
pub use block::{BlockHash, LayerBlock, LayerSave, QueryResult, RawBlock, SealedBlock, StateKey};
use instance::GpuRegistration;
pub use instance::{GpuContext, InstanceContext};
pub use internode::P2pTransferService;
use layout::KVCacheLayout;
pub use lease::QueryLeaseId;
use numa::NumaTopology;
use orbitkv_state::group_hash;
pub use orbitkv_state::{
    BundleComponent, LocalPageRef, RecoveryContract, StateBundle, StateComponent, StateDescriptor,
    StateFormat, TokenRange,
};
pub use pinned_pool::PinnedAllocation;
pub use query::{QueryAdmission, QueryMode, QueryOwner, QueryReservation};
pub use seal_offload::SlotMeta;
pub use storage::inventory::DEFAULT_INVENTORY_JOURNAL_BYTES;
pub use storage::{MemoryCacheCleanupStats, StorageConfig};
pub use sync_state::{LoadState, LoadStateError};
pub use trace::{set_trace_sample_rate, should_sample};
pub use transfer::TransferMode;

use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, RwLock},
};

use log::{debug, info};

use crate::backing::SSD_ALIGNMENT;
use crate::gpu_worker::{HostBlock, LayerTransferData, LoadCompletion, LoadTask, TransferBlock};
use crate::lease::QueryLeaseManager;
use crate::metrics::core_metrics;
use crate::storage::StorageEngine;
use tokio::sync::oneshot;

/// Errors that can occur during engine operations.
#[derive(Debug)]
pub enum EngineError {
    /// Instance not found in the registry.
    InstanceMissing(String),
    /// GPU worker not found for the specified device.
    WorkerMissing(String, i32),
    /// Invalid argument provided.
    InvalidArgument(String),
    /// CUDA initialization or runtime error.
    CudaInit(String),
    /// Storage engine error.
    Storage(String),
    /// Internal lock poisoned.
    Poisoned(&'static str),
    /// Topology mismatch between registration and existing instance.
    TopologyMismatch(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::InstanceMissing(ctx) => write!(f, "instance {ctx} not found"),
            EngineError::WorkerMissing(ctx, device) => {
                write!(f, "device {device} not found in instance {ctx}")
            }
            EngineError::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            EngineError::CudaInit(msg) => write!(f, "failed to initialize CUDA: {msg}"),
            EngineError::Storage(msg) => write!(f, "storage error: {msg}"),
            EngineError::Poisoned(what) => write!(f, "internal lock poisoned: {what}"),
            EngineError::TopologyMismatch(msg) => write!(f, "topology mismatch: {msg}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<LoadStateError> for EngineError {
    fn from(err: LoadStateError) -> Self {
        EngineError::Storage(format!("LoadState: {err}"))
    }
}

/// Main engine for managing KV cache offloading.
///
/// `OrbitKVEngine` is the top-level orchestrator that:
/// - Manages multiple inference instances
/// - Coordinates GPU worker pools for async transfers
/// - Interfaces with the storage engine for block caching
/// - Tracks GPU-NUMA topology for optimal memory locality
///
/// The engine is thread-safe and can be shared across async tasks.
pub struct OrbitKVEngine {
    /// Active inference instances indexed by instance ID.
    instances: Arc<RwLock<HashMap<String, Arc<InstanceContext>>>>,
    /// Storage engine for pinned memory, block cache, and SSD tier.
    storage: Arc<StorageEngine>,
    /// GPU-NUMA topology for memory allocation decisions.
    topology: Arc<NumaTopology>,
    /// Query-ready blocks owned by opaque scheduler leases.
    query_leases: QueryLeaseManager,
    query_budget: Arc<query::QueryBudget>,
}

impl OrbitKVEngine {
    /// Create an engine with full custom configuration.
    ///
    /// If `storage_config.enable_numa_affinity` is true and the system has multiple
    /// NUMA nodes, per-node pinned memory pools are created for optimal bandwidth.
    pub fn new_with_config(
        pool_size: usize,
        use_hugepages: bool,
        storage_config: storage::StorageConfig,
    ) -> Result<Self, EngineError> {
        let topology = Arc::new(NumaTopology::detect());
        topology.log_summary();

        let config = storage_config;
        let query_limit = config
            .query_budget_bytes
            .unwrap_or_else(|| (pool_size / 4 * 3).max(1));
        if query_limit > pool_size {
            return Err(EngineError::InvalidArgument(
                "query budget exceeds the pinned pool".into(),
            ));
        }
        let query_budget = query::QueryBudget::new(
            query_limit as u64,
            config.query_instance_budget_bytes.unwrap_or(query_limit) as u64,
        )
        .map_err(EngineError::InvalidArgument)?;
        let numa_nodes: Vec<NumaNode> = if config.enable_numa_affinity && topology.is_multi_numa() {
            let gpu_numa_nodes = topology.gpu_numa_nodes();
            if gpu_numa_nodes.is_empty() {
                info!(
                    "Auto-enabling NUMA-aware memory allocation for {} CPU NUMA nodes; no GPU NUMA affinity detected",
                    topology.num_nodes()
                );
                topology.numa_nodes().to_vec()
            } else {
                info!(
                    "Auto-enabling NUMA-aware memory allocation for {} GPU-local NUMA nodes ({} CPU NUMA nodes detected)",
                    gpu_numa_nodes.len(),
                    topology.num_nodes()
                );
                gpu_numa_nodes
            }
        } else {
            vec![]
        };

        let storage = StorageEngine::new_with_config(pool_size, use_hugepages, config, &numa_nodes)
            .map_err(EngineError::Storage)?;

        Ok(OrbitKVEngine {
            instances: Arc::new(RwLock::new(HashMap::new())),
            storage,
            topology,
            query_leases: QueryLeaseManager::default(),
            query_budget,
        })
    }

    /// Get or create an instance with the specified topology.
    ///
    /// If an instance with the same ID exists but different topology,
    /// returns a `TopologyMismatch` error.
    fn get_or_create_instance(
        &self,
        instance_id: &str,
        namespace: &str,
        tp_size: usize,
        world_size: usize,
        page_first: bool,
    ) -> Result<Arc<InstanceContext>, EngineError> {
        let mut instances = self
            .instances
            .write()
            .expect("instances write lock poisoned");

        if let Some(instance) = instances.get(instance_id) {
            // Already exists, verify topology
            instance
                .verify_identity(namespace, tp_size, world_size, Some(page_first))
                .map_err(|e| {
                    EngineError::TopologyMismatch(format!("instance {instance_id} {e}"))
                })?;
            return Ok(Arc::clone(instance));
        }

        // Create new instance
        let instance = InstanceContext::new(
            instance_id.to_string(),
            namespace.to_string(),
            tp_size,
            world_size,
            page_first,
        )
        .map_err(EngineError::InvalidArgument)?;

        let instance = Arc::new(instance);
        instances.insert(instance_id.to_string(), Arc::clone(&instance));
        Ok(instance)
    }

    /// Look up an instance by ID.
    fn get_instance(&self, instance_id: &str) -> Result<Arc<InstanceContext>, EngineError> {
        let instances = self.instances.read().expect("instances read lock poisoned");
        instances
            .get(instance_id)
            .cloned()
            .ok_or_else(|| EngineError::InstanceMissing(instance_id.to_string()))
    }

    /// Batch register multiple KV cache layers for a single GPU.
    ///
    /// This reduces gRPC round-trips from N to 1 and allows the engine to
    /// construct the GPU context with all layer registrations at once.
    ///
    /// Argument contract:
    /// - `device_id` must be non-negative; RPC callers validate this in service.rs.
    /// - `tp_size` and `world_size` must be non-zero.
    /// - `tp_rank` must be less than `tp_size`.
    /// - Layer metadata arrays must all have the same length as `layer_names`.
    /// - Registration sizes and pointers must describe a valid KV cache layout.
    ///
    /// The instance's layer-id space is sealed by the engine once `world_size`
    /// devices have registered; callers declare only the layers that actually
    /// exist on each device.
    #[allow(
        clippy::too_many_arguments,
        reason = "public API mirrors one batched registration RPC payload"
    )]
    pub fn register_context_layer_batch(
        &self,
        instance_id: &str,
        namespace: &str,
        device_id: i32,
        tp_rank: usize,
        pp_rank: usize,
        tp_size: usize,
        world_size: usize,
        layer_names: &[String],
        data_ptrs: &[u64],
        size_bytes_list: &[usize],
        num_blocks_list: &[usize],
        bytes_per_block_list: &[usize],
        kv_stride_bytes_list: &[usize],
        segments_list: &[usize],
        transfer_mode: TransferMode,
        page_first: bool,
    ) -> Result<(), EngineError> {
        // Dense default: block stride == bytes_per_block (layer-first layout).
        self.register_context_layer_batch_strided(
            instance_id,
            namespace,
            device_id,
            tp_rank,
            pp_rank,
            tp_size,
            world_size,
            layer_names,
            data_ptrs,
            size_bytes_list,
            num_blocks_list,
            bytes_per_block_list,
            kv_stride_bytes_list,
            segments_list,
            None,
            None,
            transfer_mode,
            page_first,
        )
    }

    /// Like [`Self::register_context_layer_batch`] but with an explicit per-layer
    /// block stride (see [`KVCacheRegistration::with_block_stride`]). When
    /// `block_stride_bytes_list` is `Some` it must match `layer_names` in length;
    /// each entry overrides that layer's stride.
    ///
    /// `transfer_mode` selects the GPU worker pools' H2D/D2H backend for this
    /// instance. The pools are spawned per (instance, GPU) at registration, so
    /// each instance can run a different backend.
    #[allow(
        clippy::too_many_arguments,
        reason = "public API mirrors one batched registration RPC payload"
    )]
    pub fn register_context_layer_batch_strided(
        &self,
        instance_id: &str,
        namespace: &str,
        device_id: i32,
        tp_rank: usize,
        pp_rank: usize,
        tp_size: usize,
        world_size: usize,
        layer_names: &[String],
        data_ptrs: &[u64],
        size_bytes_list: &[usize],
        num_blocks_list: &[usize],
        bytes_per_block_list: &[usize],
        kv_stride_bytes_list: &[usize],
        segments_list: &[usize],
        block_stride_bytes_list: Option<&[usize]>,
        layer_group_ids: Option<&[u32]>,
        transfer_mode: TransferMode,
        page_first: bool,
    ) -> Result<(), EngineError> {
        // Build all registrations
        let ssd_enabled = self.storage.is_ssd_enabled();
        let batch_size = layer_names.len();
        if data_ptrs.len() != batch_size
            || size_bytes_list.len() != batch_size
            || num_blocks_list.len() != batch_size
            || bytes_per_block_list.len() != batch_size
            || kv_stride_bytes_list.len() != batch_size
            || segments_list.len() != batch_size
        {
            return Err(EngineError::InvalidArgument(format!(
                "registration metadata length mismatch: layer_names={batch_size}, data_ptrs={}, size_bytes={}, num_blocks={}, bytes_per_block={}, kv_stride_bytes={}, segments={}",
                data_ptrs.len(),
                size_bytes_list.len(),
                num_blocks_list.len(),
                bytes_per_block_list.len(),
                kv_stride_bytes_list.len(),
                segments_list.len()
            )));
        }
        if let Some(strides) = block_stride_bytes_list
            && strides.len() != batch_size
        {
            return Err(EngineError::InvalidArgument(format!(
                "registration metadata length mismatch: layer_names={batch_size}, block_stride_bytes={}",
                strides.len()
            )));
        }
        if let Some(group_ids) = layer_group_ids
            && group_ids.len() != batch_size
        {
            return Err(EngineError::InvalidArgument(format!(
                "registration metadata length mismatch: layer_names={batch_size}, layer_group_ids={}",
                group_ids.len()
            )));
        }
        let mut kv_caches = HashMap::with_capacity(batch_size);

        for i in 0..batch_size {
            let layer_name = &layer_names[i];
            let mut layout = KVCacheLayout::new(
                data_ptrs[i],
                size_bytes_list[i],
                num_blocks_list[i],
                bytes_per_block_list[i],
                kv_stride_bytes_list[i],
                segments_list[i],
            )
            .map_err(|e| EngineError::InvalidArgument(format!("layer {layer_name}: {e}")))?;

            if let Some(strides) = block_stride_bytes_list {
                layout = layout.with_block_stride(strides[i]).map_err(|e| {
                    EngineError::InvalidArgument(format!("layer {layer_name}: {e}"))
                })?;
            }

            if ssd_enabled {
                layout = layout.with_ssd_padding(SSD_ALIGNMENT);
                if layout.padded_segment_bytes() != layout.segment_bytes() {
                    info!(
                        "SSD alignment padding: layer={layer_name}, bytes_per_block={} -> padded={}",
                        layout.segment_bytes(),
                        layout.padded_segment_bytes()
                    );
                }
            }

            if kv_caches.insert(layer_name.clone(), layout).is_some() {
                return Err(EngineError::InvalidArgument(format!(
                    "duplicate layer name in registration batch: {layer_name}"
                )));
            }
        }

        let layer_groups: HashMap<String, u32> = match layer_group_ids {
            Some(group_ids) => layer_names
                .iter()
                .cloned()
                .zip(group_ids.iter().copied())
                .collect(),
            None => HashMap::new(),
        };

        // Get or create instance
        let instance =
            self.get_or_create_instance(instance_id, namespace, tp_size, world_size, page_first)?;

        // Get NUMA affinity for this GPU
        let numa_node = self.topology.numa_for_gpu(device_id);

        // Validate NUMA topology if NUMA-aware allocation is enabled
        if self.storage.is_numa_enabled() && numa_node.is_unknown() {
            return Err(EngineError::InvalidArgument(format!(
                "NUMA-aware allocation is enabled, but GPU {} NUMA affinity is unknown. \
                 Please ensure nvidia-smi is available and GPU NUMA topology is detectable, \
                 or disable NUMA-aware allocation.",
                device_id
            )));
        }

        // Register GPU with all layers. The connector picks the backend per
        // model and sends it with the registration.
        instance.register_new_gpu(GpuRegistration {
            device_id,
            tp_rank,
            pp_rank,
            numa_node,
            transfer_mode,
            kv_caches,
            layer_groups,
        })?;

        info!(
            "Registered context batch: instance={instance_id}, namespace={namespace}, \
             device={device_id}, layers={batch_size}, tp_rank={tp_rank}/{tp_size}, pp_rank={pp_rank}"
        );
        Ok(())
    }

    /// Drain all GPU transfers before callers release imported CUDA mappings.
    /// Callers must serialize registration and cleanup for this instance.
    pub async fn unregister_instance_and_wait(&self, instance_id: &str) -> Result<(), EngineError> {
        self.get_instance(instance_id)?.drain_workers().await?;
        self.unregister_instance(instance_id)
    }

    /// Unregister an instance and release all associated resources.
    pub fn unregister_instance(&self, instance_id: &str) -> Result<(), EngineError> {
        let removed = self
            .instances
            .write()
            .expect("instances write lock poisoned")
            .remove(instance_id);

        if removed.is_none() {
            return Err(EngineError::InstanceMissing(instance_id.to_string()));
        }
        self.query_leases.release_instance(instance_id);
        info!("Unregistered instance: {}", instance_id);
        Ok(())
    }

    /// Unregister all instances, returning the IDs that were removed.
    pub fn unregister_all_instances(&self) -> Vec<String> {
        let mut instances = self
            .instances
            .write()
            .expect("instances write lock poisoned");
        let ids: Vec<String> = instances.keys().cloned().collect();
        instances.clear();
        drop(instances);
        for id in &ids {
            self.query_leases.release_instance(id);
        }
        if !ids.is_empty() {
            info!("Unregistered all instances: {:?}", ids);
        }
        ids
    }

    /// List all registered instance IDs.
    pub fn list_instance_ids(&self) -> Vec<String> {
        self.instances
            .read()
            .expect("instances read lock poisoned")
            .keys()
            .cloned()
            .collect()
    }

    /// A scheduler session may arrive before or after GPU registration. Once an
    /// instance exists it must describe the same computation and worker set.
    pub fn validate_session_identity(
        &self,
        instance_id: &str,
        namespace: &str,
        tp_size: usize,
        world_size: usize,
    ) -> Result<(), EngineError> {
        if let Some(instance) = self
            .instances
            .read()
            .expect("instances read lock poisoned")
            .get(instance_id)
        {
            instance
                .verify_identity(namespace, tp_size, world_size, None)
                .map_err(EngineError::TopologyMismatch)?;
        }
        Ok(())
    }

    /// Return the namespace associated with a registered instance.
    pub fn instance_namespace(&self, instance_id: &str) -> Result<String, EngineError> {
        Ok(self
            .get_instance(instance_id)?
            .sealed_topology()?
            .cache_namespace
            .clone())
    }

    /// Count prefix hit blocks with SSD prefetch support.
    ///
    /// Argument contract:
    /// - `instance_id` must identify a registered instance.
    /// - `req_id` must be non-empty; the Cache Manager validates it.
    /// - `block_hashes` may be empty.
    ///
    /// Returns:
    /// A terminal `QueryResult` with the ready prefix and missing suffix.
    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "query_prefetch.count_prefix_hit")
    )]
    pub async fn count_prefix_hit_blocks_with_prefetch(
        &self,
        instance_id: &str,
        req_id: &str,
        block_hashes: &[Vec<u8>],
        mode: QueryMode,
    ) -> Result<QueryResult, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let topology = instance.sealed_topology()?;
        let namespace = &topology.cache_namespace;
        let encoded: Vec<Vec<u8>> = block_hashes
            .iter()
            .map(|hash| group_hash(hash, 0))
            .collect();

        let status = self
            .storage
            .check_prefix_and_prefetch(req_id, namespace, &encoded, mode)
            .await;

        {
            let QueryResult { blocks, missing } = &status;
            let metrics = core_metrics();
            metrics.cache_block_hits.add(blocks.len() as u64, &[]);
            if *missing > 0 {
                metrics.cache_block_misses.add(*missing as u64, &[]);
            }
        }

        Ok(status)
    }

    /// Find candidate positions without loading or pinning their payloads.
    pub async fn discover_candidates(
        &self,
        instance_id: &str,
        group_id: u32,
        hashes: &[Vec<u8>],
    ) -> Result<Vec<u32>, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let topology = instance.sealed_topology()?;
        topology.group_total_slots(group_id)?;
        let encoded: Vec<_> = hashes
            .iter()
            .map(|hash| group_hash(hash, group_id))
            .collect();
        let hits = self
            .storage
            .discover(&topology.cache_namespace, &encoded)
            .await;
        let limit = if group_id == 0 {
            hits.iter().take_while(|&&hit| hit).count()
        } else {
            hits.len()
        };
        Ok(hits
            .into_iter()
            .take(limit)
            .enumerate()
            .filter_map(|(i, hit)| hit.then_some(i as u32))
            .collect())
    }

    /// All-or-nothing membership fetch over one hybrid-cache storage group,
    /// eligible for the same SSD prefetch and Catalog + Mooncake remote fetch
    /// as prefix queries.
    ///
    /// Where [`Self::query_group_membership`] allows sparse hits, this treats
    /// `block_hashes` as an exact want-set: misses are
    /// pulled from remote tiers, and the future waits until the whole
    /// set is fetchable (the prefix machinery over an explicit key list *is*
    /// a set fetch once the full length is required). Use it when partial
    /// state is useless — e.g. restoring a recurrent-state checkpoint on a
    /// prefill/decode handoff, where the peer that saved the set holds every
    /// member. A result with fewer blocks than requested means the
    /// set could not be completed anywhere; callers treat that as a miss.
    pub async fn query_group_membership_with_fetch(
        &self,
        instance_id: &str,
        req_id: &str,
        group_id: u32,
        block_hashes: &[Vec<u8>],
    ) -> Result<QueryResult, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let topology = instance.sealed_topology()?;
        // Same contract as the local membership query: an unknown group is a
        // bug in the caller, not an all-miss answer.
        topology.group_total_slots(group_id)?;

        let namespace = &topology.cache_namespace;
        let encoded: Vec<Vec<u8>> = block_hashes
            .iter()
            .map(|hash| group_hash(hash, group_id))
            .collect();

        let status = self
            .storage
            .check_prefix_and_prefetch(req_id, namespace, &encoded, QueryMode::WaitForFullPrefix)
            .await;

        {
            let QueryResult { blocks, missing } = &status;
            let metrics = core_metrics();
            metrics.cache_block_hits.add(blocks.len() as u64, &[]);
            if *missing > 0 {
                metrics.cache_block_misses.add(*missing as u64, &[]);
            }
        }

        Ok(status)
    }

    /// Position-aligned membership query over one hybrid-cache storage group.
    ///
    /// Unlike prefix queries, every position reports independently: entry `i`
    /// is the sealed block for `block_hashes[i]` in `group_id`, or `None` on
    /// miss. Sparse hit patterns are the point — callers (e.g. the vLLM
    /// connector's hybrid reconcile) pick the rightmost hit themselves.
    /// Every group uses the same versioned content-hash encoding.
    ///
    /// The returned blocks hold plain `Arc` refs, not leases: pin what you
    /// need via [`Self::create_query_lease`].
    pub async fn query_group_membership(
        &self,
        instance_id: &str,
        req_id: &str,
        group_id: u32,
        block_hashes: &[Vec<u8>],
    ) -> Result<Vec<Option<Arc<SealedBlock>>>, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let topology = instance.sealed_topology()?;
        // Validate the group against the sealed topology; an unknown group is
        // a contract bug, not an answer of "all miss".
        topology.group_total_slots(group_id)?;

        let namespace = &topology.cache_namespace;
        let encoded: Vec<Vec<u8>> = block_hashes
            .iter()
            .map(|hash| group_hash(hash, group_id))
            .collect();
        Ok(self
            .storage
            .get_membership(req_id, namespace, &encoded)
            .await)
    }

    /// Create an opaque lease that owns query-ready blocks.
    pub fn create_query_lease(
        &self,
        instance_id: &str,
        blocks: Vec<Arc<SealedBlock>>,
    ) -> Result<QueryLeaseId, EngineError> {
        let instance = self.get_instance(instance_id)?;
        if blocks.is_empty() {
            return Err(EngineError::InvalidArgument(
                "query lease requires at least one block".to_string(),
            ));
        }
        Ok(self
            .query_leases
            .create(instance_id, blocks, instance.world_size(), None))
    }

    /// Reserve registered group bytes before a process query retains any pages.
    pub fn reserve_query(
        &self,
        instance_id: &str,
        group_id: u32,
        blocks: usize,
        warming: bool,
    ) -> Result<QueryAdmission, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let topology = instance.sealed_topology()?;
        let bytes = topology
            .group_block_bytes(group_id)?
            .checked_mul(blocks as u64)
            .ok_or_else(|| EngineError::InvalidArgument("query bytes overflow".into()))?;
        Ok(self
            .query_budget
            .reserve(instance_id, &topology.cache_namespace, bytes, warming))
    }

    /// Move preparation ownership into the result lease and then GPU consumers.
    pub fn finish_query(
        &self,
        reservation: QueryReservation,
        owner: QueryOwner,
        blocks: Vec<Arc<SealedBlock>>,
    ) -> Result<QueryLeaseId, EngineError> {
        let instance_id = reservation.instance();
        let instance = self.get_instance(instance_id)?;
        if instance.sealed_topology()?.cache_namespace != reservation.namespace() {
            return Err(EngineError::InvalidArgument(
                "query instance registration changed".into(),
            ));
        }
        let bytes = blocks
            .iter()
            .try_fold(0u64, |sum, block| sum.checked_add(block.memory_footprint()))
            .ok_or_else(|| EngineError::InvalidArgument("query result bytes overflow".into()))?;
        reservation
            .ready(bytes)
            .map_err(EngineError::InvalidArgument)?;
        Ok(self.query_leases.create(
            instance_id,
            blocks,
            instance.world_size(),
            Some((owner, reservation.clone())),
        ))
    }

    pub fn release_query_session(&self, session: u64) {
        self.query_leases
            .release_owner(|owner| owner.session == session);
    }

    pub fn cancel_query(&self, owner: QueryOwner) {
        self.query_leases
            .release_owner(|candidate| candidate == owner);
    }

    /// Release a query lease. Returns false when the lease is unknown or expired.
    pub fn release_query_lease(&self, lease: &QueryLeaseId) -> bool {
        self.query_leases.release(lease)
    }

    /// Evict all resident in-memory cache blocks while preserving backing-store data.
    pub fn cleanup_memory_cache(&self) -> MemoryCacheCleanupStats {
        self.query_leases.sweep_expired();
        self.storage.cleanup_memory_cache()
    }

    /// Best-effort graceful unregister from Catalog, if configured.
    pub async fn shutdown_catalog_client(&self) {
        self.storage.shutdown_catalog_client().await;
    }

    /// Batch load KV blocks for multiple layers asynchronously.
    ///
    /// Returns immediately after submitting the task to the GPU worker pool.
    /// The connector spin-waits on the `LoadState` until completion.
    #[allow(
        clippy::too_many_arguments,
        reason = "public API mirrors one batched load RPC payload"
    )]
    pub fn batch_load_kv_blocks_multi_layer(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        load_state_shm: &str,
        layer_groups: &[Vec<&str>],
        loads: &[(QueryLeaseId, Vec<Vec<Option<usize>>>)],
    ) -> Result<(), EngineError> {
        let load_state = LoadState::attach(load_state_shm)?;

        let result = self.batch_load_kv_blocks_multi_layer_inner(
            instance_id,
            tp_rank,
            device_id,
            layer_groups,
            loads,
            LoadCompletion::Shm(load_state_shm.to_string()),
        );

        if let Err(ref e) = result {
            log::error!("batch_load_kv_blocks_multi_layer pre-submit error: {e:?}");
            load_state.set_error();
        }

        result
    }

    /// In-process variant of [`Self::batch_load_kv_blocks_multi_layer`]: instead
    /// of a caller-managed shared-memory `LoadState`, it returns a oneshot
    /// receiver that resolves when the GPU worker finishes the load (`Ok`) or it
    /// fails (`Err`). Poll it with `try_recv` to keep admission non-blocking, or
    /// await it. For in-process Rust embedders that register raw device pointers
    /// and have no second process to coordinate a `LoadState` with.
    ///
    /// On a pre-submit error the receiver is dropped (yields `RecvError`); the
    /// same error is returned synchronously here.
    pub fn batch_load_kv_blocks_multi_layer_inproc(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        layer_groups: &[Vec<&str>],
        loads: &[(QueryLeaseId, Vec<Vec<Option<usize>>>)],
    ) -> Result<oneshot::Receiver<Result<(), EngineError>>, EngineError> {
        let (reply, rx) = oneshot::channel();
        self.batch_load_kv_blocks_multi_layer_inner(
            instance_id,
            tp_rank,
            device_id,
            layer_groups,
            loads,
            LoadCompletion::Channel(reply),
        )?;
        Ok(rx)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "internal helper keeps the public load API validation path explicit"
    )]
    fn batch_load_kv_blocks_multi_layer_inner(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        layer_groups: &[Vec<&str>],
        loads: &[(QueryLeaseId, Vec<Vec<Option<usize>>>)],
        completion: LoadCompletion,
    ) -> Result<(), EngineError> {
        let instance = self.get_instance(instance_id)?;
        let topology = instance.sealed_topology()?;
        let gpu = instance
            .get_gpu(device_id)
            .ok_or_else(|| EngineError::WorkerMissing(instance_id.to_string(), device_id))?;

        if layer_groups.is_empty() {
            return Err(EngineError::InvalidArgument(
                "load requires at least one layer group".to_string(),
            ));
        }
        let layer_count: usize = layer_groups.iter().map(Vec::len).sum();
        let unique_layer_count = layer_groups
            .iter()
            .flatten()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len();
        if unique_layer_count != layer_count {
            return Err(EngineError::InvalidArgument(
                "load layer names must be unique across groups".to_string(),
            ));
        }

        // Resolve each load group's storage group and require homogeneity: a
        // group's blocks seal against exactly one slot space. An empty load
        // group owns no storage group; targets pointing at it load into
        // nothing (it exists only to keep group indices aligned with the
        // connector's cache-group layout).
        let mut storage_group_of_load_group: Vec<Option<u32>> =
            Vec::with_capacity(layer_groups.len());
        for layer_names in layer_groups {
            let mut group: Option<u32> = None;
            for layer_name in layer_names {
                let layer_id = topology.layer_id(layer_name)?;
                let layer_group = topology.group_of_layer(layer_id);
                match group {
                    None => group = Some(layer_group),
                    Some(existing) if existing == layer_group => {}
                    Some(existing) => {
                        return Err(EngineError::InvalidArgument(format!(
                            "load group mixes storage groups {existing} and {layer_group} \
                             (layer {layer_name})",
                        )));
                    }
                }
            }
            storage_group_of_load_group.push(group);
        }

        // Consume query leases reserved for this load.
        trace_scope!("load.cache_lookup", _s);
        let mut block_targets_by_group = vec![Vec::new(); layer_groups.len()];
        let mut block_cache = Vec::new();
        let mut reservations = Vec::new();
        for (lease, lease_block_ids_by_group) in loads {
            let (blocks, reservation) = self
                .query_leases
                .consume(instance_id, lease)
                .map_err(EngineError::Storage)?;
            if let Some(reservation) = reservation {
                reservation.restoring();
                reservations.push(reservation);
            }
            if lease_block_ids_by_group.len() != layer_groups.len() {
                return Err(EngineError::InvalidArgument(format!(
                    "load group count {} does not match layer group count {}",
                    lease_block_ids_by_group.len(),
                    layer_groups.len()
                )));
            }
            let block_cache_start = block_cache.len();
            for (group_index, lease_block_targets) in lease_block_ids_by_group.iter().enumerate() {
                if blocks.len() != lease_block_targets.len() {
                    return Err(EngineError::InvalidArgument(format!(
                        "query lease block count {} does not match destination block count {} for group {}",
                        blocks.len(),
                        lease_block_targets.len(),
                        group_index
                    )));
                }
                // A stored block must carry exactly the slot layout of the
                // storage group this target group loads into. A mismatch
                // means the namespace is shared by instances with different
                // layer sets (e.g. MTP enabled vs disabled) — loading would
                // silently leave layers uninitialized, so fail loudly and
                // let vLLM recompute. Empty load groups own no storage
                // group and skip the check entirely.
                if let Some(storage_group) = storage_group_of_load_group[group_index] {
                    let expected_slots = topology.group_total_slots(storage_group)?;
                    for (source_index, destination) in lease_block_targets.iter().enumerate() {
                        if destination.is_some()
                            && blocks[source_index].slots().len() != expected_slots
                        {
                            return Err(EngineError::InvalidArgument(format!(
                                "stored block has {} slots but storage group {storage_group} of \
                                 instance {instance_id} expects {expected_slots}: \
                                 namespace is shared by incompatible KV layouts",
                                blocks[source_index].slots().len(),
                            )));
                        }
                    }
                }
                block_targets_by_group[group_index].extend(
                    lease_block_targets.iter().enumerate().filter_map(
                        |(source_index, destination)| {
                            destination.map(|block_id| (block_id, block_cache_start + source_index))
                        },
                    ),
                );
            }
            block_cache.extend(blocks);
        }
        trace_drop!(_s);

        // Build load tasks for each layer
        trace_scope!("load.build_tasks");
        let mut layers = Vec::with_capacity(layer_count);

        for (group_index, layer_names) in layer_groups.iter().enumerate() {
            let group_block_targets = &block_targets_by_group[group_index];
            for layer_name in layer_names {
                let layer_id = topology.layer_id(layer_name)?;

                let layout = gpu.get_layout(layer_name).ok_or_else(|| {
                    EngineError::InvalidArgument(format!(
                        "layer {layer_name} not registered on device {device_id}"
                    ))
                })?;

                let slot_id = topology.slot_index(layer_id, tp_rank)?;
                // Page-first: every layer reads from the one page slot (tp_rank) at
                // its sealed byte offset. Layer-first: offset 0 (the whole slot
                // RawBlock is the layer).
                let host_offset = topology
                    .page_placement(layer_id)
                    .map_or(0, |(offset, _)| offset);

                let mut blocks = Vec::with_capacity(group_block_targets.len());
                for &(block_idx, source_index) in group_block_targets {
                    let block_entry = &block_cache[source_index];
                    if block_entry.get_slot(slot_id).is_none() {
                        return Err(EngineError::InvalidArgument(format!(
                            "stored block is missing slot {slot_id} for layer {layer_name}"
                        )));
                    }
                    blocks.push(TransferBlock {
                        block_idx,
                        block: HostBlock::Cached {
                            sealed: Arc::clone(block_entry),
                            slot_id,
                            offset: host_offset,
                        },
                    });
                }

                if !blocks.is_empty() {
                    layers.push(LayerTransferData {
                        layer_name: (*layer_name).to_string(),
                        layout,
                        blocks,
                    });
                }
            }
        }

        // Complete immediately if no blocks to load
        if layers.is_empty() {
            debug!("No blocks to load, completing immediately");
            completion.signal(Ok(()));
            return Ok(());
        }

        // Submit to worker pool (fire and forget)
        gpu.worker_pool().submit_load(LoadTask {
            layers,
            completion,
            reservations,
        })
    }

    /// Wait until all previously submitted save batches have been processed
    /// by the insert worker.
    ///
    /// This is a flush barrier: it guarantees that every `batch_save_kv_blocks_from_ipc`
    /// call that returned before this call will have its blocks inserted into the
    /// read cache (or inflight map) by the time this future resolves.
    pub async fn flush_saves(&self) {
        self.storage.flush_write_pipeline().await;
    }

    /// Flush saves and wait for directory acknowledgement of current residency.
    /// Returns an error if synchronization cannot complete within its deadline.
    pub async fn flush_saves_and_inventory(&self) -> Result<(), EngineError> {
        self.storage.flush_write_pipeline().await;
        self.storage
            .flush_inventory()
            .await
            .map_err(EngineError::Storage)
    }

    /// Flush write pipeline and SSD writer.
    ///
    /// Guarantees that all saves submitted before this call are both
    /// cache-visible and persisted to SSD (if SSD is enabled).
    pub async fn flush_all(&self) {
        self.storage.flush_write_pipeline().await;
        self.storage.flush_ssd().await;
    }

    /// Remove abandoned writes. Query futures drain independently of polling.
    pub async fn gc_stale_inflight(&self, max_age: std::time::Duration) -> usize {
        self.storage.gc_stale_inflight(max_age).await
    }

    // =========================================================================
    // Cross-node transfer: serving side
    // =========================================================================

    pub fn transfer_lock_timeout(&self) -> std::time::Duration {
        self.storage.transfer_lock_timeout()
    }

    /// Release a transfer lock session. Returns the number of blocks released.
    pub fn release_transfer_lock(&self, session_id: &str) -> usize {
        self.storage.release_transfer_lock(session_id)
    }

    /// GC expired transfer lock sessions.
    pub fn gc_expired_transfer_locks(&self) -> usize {
        self.storage.gc_expired_transfer_locks()
    }

    /// Return `(base_ptr, size)` for each contiguous pinned memory region.
    /// Used for Mooncake memory registration.
    pub fn pinned_memory_regions(&self) -> Vec<(u64, usize)> {
        self.storage.pinned_memory_regions()
    }

    /// Returns true if the Mooncake remote transfer engine is available.
    #[cfg(feature = "mooncake")]
    pub fn has_remote_transport(&self) -> bool {
        self.storage.mooncake_transport().is_some()
    }

    /// Returns true if the Mooncake remote transfer engine is available.
    #[cfg(not(feature = "mooncake"))]
    pub fn has_remote_transport(&self) -> bool {
        false
    }

    #[cfg(feature = "mooncake")]
    pub fn transfer_endpoint(&self) -> Option<&str> {
        self.storage.transfer_endpoint()
    }

    #[cfg(not(feature = "mooncake"))]
    pub fn transfer_endpoint(&self) -> Option<&str> {
        None
    }
}

#[cfg(test)]
#[path = "../tests/unit/lib.rs"]
mod tests;
