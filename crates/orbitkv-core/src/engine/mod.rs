//! Cache orchestration and registration; storage and transfer owners retain resources.

pub(crate) mod instance;
mod publish;
mod query;
mod restore;

use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, RwLock},
};

use log::info;

use crate::backing::SSD_ALIGNMENT;
use crate::memory::numa::{NumaNode, NumaTopology};
use crate::query::QueryBudget;
use crate::query::lease::QueryLeaseManager;
use crate::storage::{self, MemoryCacheCleanupStats, StorageEngine};
use crate::transfer::TransferMode;
use crate::transfer::layout::KVCacheLayout;
use instance::{GpuRegistration, InstanceContext};

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
    pub(crate) storage: Arc<StorageEngine>,
    /// GPU-NUMA topology for memory allocation decisions.
    topology: Arc<NumaTopology>,
    /// Query-ready blocks owned by opaque scheduler leases.
    query_leases: QueryLeaseManager,
    query_budget: Arc<QueryBudget>,
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
        let query_budget = QueryBudget::new(
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
    /// One registration batch allows the engine to
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
        storage_formats: Option<&[orbitkv_state::StorageFormat]>,
        transfer_mode: TransferMode,
        page_first: bool,
    ) -> Result<(), EngineError> {
        // Build all registrations
        let ssd_enabled = self.storage.is_ssd_enabled();
        let batch_size = layer_names.len();
        if storage_formats.is_some_and(|formats| formats.len() != batch_size) {
            return Err(EngineError::InvalidArgument(
                "storage format count differs from layers".into(),
            ));
        }
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

            layout.storage_format = self
                .storage
                .codec
                .format(storage_formats.map_or(Default::default(), |formats| formats[i]));

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
        let page_first = page_first && self.storage.codec == crate::StorageCodec::None;
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

    /// Evict all resident in-memory cache blocks while preserving backing-store data.
    pub fn cleanup_memory_cache(&self) -> MemoryCacheCleanupStats {
        self.query_leases.sweep_expired();
        self.storage.cleanup_memory_cache()
    }

    /// Best-effort graceful unregister from Catalog, if configured.
    pub async fn shutdown_catalog_client(&self) {
        self.storage.shutdown_catalog_client().await;
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

    /// Mark overdue source transfers without releasing their memory.
    pub fn expire_transfer_locks(&self) -> usize {
        self.storage.transfer_lock.expire()
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
#[path = "../../tests/unit/engine/mod.rs"]
mod tests;
