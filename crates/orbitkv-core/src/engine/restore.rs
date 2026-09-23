use std::sync::Arc;

use log::debug;
use tokio::sync::oneshot;

use super::{EngineError, OrbitKVEngine};
use crate::block::RestoreSource;
use crate::query::lease::QueryLeaseId;
use crate::transfer::worker::{
    LayerTransferData, LoadOutcome, LoadTask, TransferBlock, TransferPayload,
};

impl OrbitKVEngine {
    /// Restore leased state into registered GPU pages. The receiver resolves only
    /// after all submitted transfers complete; dropping it does not revoke DMA.
    pub fn restore(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        layer_groups: &[Vec<&str>],
        loads: &[(QueryLeaseId, Vec<Vec<Option<usize>>>)],
    ) -> Result<oneshot::Receiver<LoadOutcome>, EngineError> {
        let (completion, receiver) = oneshot::channel();
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
                            && blocks[source_index].slot_count() != expected_slots
                        {
                            return Err(EngineError::InvalidArgument(format!(
                                "stored block has {} slots but storage group {storage_group} of \
                                 instance {instance_id} expects {expected_slots}: \
                                 namespace is shared by incompatible KV layouts",
                                blocks[source_index].slot_count(),
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
                    if slot_id >= block_entry.slot_count() {
                        return Err(EngineError::InvalidArgument(format!(
                            "stored block is missing slot {slot_id} for layer {layer_name}"
                        )));
                    }
                    blocks.push(TransferBlock {
                        block_idx,
                        block: match block_entry {
                            RestoreSource::Memory(sealed) => TransferPayload::Cached {
                                sealed: Arc::clone(sealed),
                                slot_id,
                                offset: host_offset,
                            },
                            RestoreSource::Ssd(source) => TransferPayload::Ssd {
                                source: Arc::clone(source),
                                slot_id,
                                offset: host_offset,
                            },
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
            let _ = completion.send(LoadOutcome {
                result: Ok(()),
                completed_at: std::time::Instant::now(),
            });
            return Ok(receiver);
        }

        // Submit to worker pool (fire and forget)
        gpu.worker_pool().submit_load(LoadTask {
            layers,
            completion,
            reservations,
        })?;
        Ok(receiver)
    }
}
