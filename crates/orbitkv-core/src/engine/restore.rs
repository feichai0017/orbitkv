use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::oneshot;

use super::instance::LayerTopology;
use super::{EngineError, OrbitKVEngine};
use crate::QueryReservation;
use crate::block::RestoreSource;
use crate::planning::restore::RestorePlan;
use crate::query::lease::{QueryLeaseId, QueryLeaseManager};
use crate::transfer::layout::KVCacheLayout;
use crate::transfer::worker::{
    LayerTransferData, LoadOutcome, LoadPayload, LoadTask, TransferBlock, TransferPayload,
    append_copy_descs,
};

struct RestoreLayer {
    name: String,
    slot_id: usize,
    host_offset: usize,
}

struct RestoreGroup {
    layers: Vec<RestoreLayer>,
    storage_slots: Option<(u32, usize)>,
    targets: Vec<(usize, usize)>,
}

/// Owns leased sources and their byte reservations before GPU addresses are bound.
/// Dropping this owner is safe: no transfer has been submitted yet.
struct PreparedRestore {
    plan: RestorePlan,
    groups: Vec<RestoreGroup>,
    sources: Vec<RestoreSource>,
    reservations: Vec<QueryReservation>,
}

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
        let instance = self.get_instance(instance_id)?;
        let topology = instance.sealed_topology()?;
        let gpu = instance
            .get_gpu(device_id)
            .ok_or_else(|| EngineError::WorkerMissing(instance_id.to_string(), device_id))?;
        if gpu.tp_rank() != tp_rank {
            return Err(EngineError::InvalidArgument(format!(
                "device_id {device_id} represents tp_rank {}, got {tp_rank}",
                gpu.tp_rank()
            )));
        }
        let groups = RestoreGroup::resolve(&topology, tp_rank, layer_groups)?;
        for (_, targets) in loads {
            if targets.len() != groups.len() {
                return Err(EngineError::InvalidArgument(format!(
                    "load group count {} does not match layer group count {}",
                    targets.len(),
                    groups.len()
                )));
            }
        }

        // Check registrations and destination ranges before consuming any lease.
        // Keep the resolved bindings so submission does not look up layers again.
        let mut layouts = Vec::with_capacity(groups.iter().map(|g| g.layers.len()).sum());
        for (group_index, group) in groups.iter().enumerate() {
            let mut capacity = None;
            for layer in &group.layers {
                let layout = gpu.get_layout(&layer.name).ok_or_else(|| {
                    EngineError::InvalidArgument(format!(
                        "layer {} not registered on device {device_id}",
                        layer.name
                    ))
                })?;
                let blocks = layout.geometry().num_blocks();
                capacity = Some(capacity.map_or(blocks, |limit: usize| limit.min(blocks)));
                layouts.push(layout);
            }
            if let Some(capacity) = capacity {
                for (_, targets) in loads {
                    for &block in targets[group_index].iter().flatten() {
                        if block >= capacity {
                            return Err(EngineError::InvalidArgument(format!(
                                "block {block} out of range ({capacity} blocks) for load group {group_index}"
                            )));
                        }
                    }
                }
            }
        }

        trace_scope!("load.cache_lookup", lookup);
        let prepared =
            PreparedRestore::prepare(&self.query_leases, instance_id, device_id, groups, loads)?;
        trace_drop!(lookup);
        trace_scope!("load.build_tasks");
        let (completion, receiver) = oneshot::channel();
        let task = prepared.bind(layouts, completion, self.storage.codec_budget)?;
        let empty = match &task.payload {
            LoadPayload::Raw { copies, .. } => copies.is_empty(),
            LoadPayload::Layers(layers) => layers.is_empty(),
        };
        if empty {
            drop(task.payload);
            drop(task.reservations);
            let _ = task.completion.send(LoadOutcome {
                result: Ok(()),
                completed_at: std::time::Instant::now(),
            });
        } else {
            gpu.worker_pool().submit_load(task)?;
        }
        Ok(receiver)
    }
}

impl RestoreGroup {
    fn resolve(
        topology: &LayerTopology,
        tp_rank: usize,
        layer_groups: &[Vec<&str>],
    ) -> Result<Vec<Self>, EngineError> {
        if layer_groups.is_empty() {
            return Err(EngineError::InvalidArgument(
                "load requires at least one layer group".into(),
            ));
        }
        let mut seen = HashSet::new();
        let mut groups = Vec::with_capacity(layer_groups.len());
        for names in layer_groups {
            let mut group = Self {
                layers: Vec::with_capacity(names.len()),
                storage_slots: None,
                targets: Vec::new(),
            };
            for &name in names {
                if !seen.insert(name) {
                    return Err(EngineError::InvalidArgument(
                        "load layer names must be unique across groups".into(),
                    ));
                }
                let layer_id = topology.layer_id(name)?;
                let storage_group = topology.group_of_layer(layer_id);
                match group.storage_slots {
                    None => {
                        group.storage_slots =
                            Some((storage_group, topology.group_total_slots(storage_group)?));
                    }
                    Some((existing, _)) if existing == storage_group => {}
                    Some((existing, _)) => {
                        return Err(EngineError::InvalidArgument(format!(
                            "load group mixes storage groups {existing} and {storage_group} (layer {name})"
                        )));
                    }
                }
                group.layers.push(RestoreLayer {
                    name: name.into(),
                    slot_id: topology.slot_index(layer_id, tp_rank)?,
                    host_offset: topology
                        .page_placement(layer_id)
                        .map_or(0, |(offset, _)| offset),
                });
            }
            groups.push(group);
        }
        Ok(groups)
    }
}

impl PreparedRestore {
    fn prepare(
        leases: &QueryLeaseManager,
        instance_id: &str,
        device_id: i32,
        mut groups: Vec<RestoreGroup>,
        loads: &[(QueryLeaseId, Vec<Vec<Option<usize>>>)],
    ) -> Result<Self, EngineError> {
        let tokens: Vec<_> = loads.iter().map(|(token, _)| *token).collect();
        let (plan, sources, reservations) = leases
            .consume_batch(instance_id, &tokens, |leased| {
                let mut sources = Vec::with_capacity(leased.iter().map(|blocks| blocks.len()).sum());
                for (blocks, (_, targets)) in leased.iter().zip(loads) {
                    let source_start = sources.len();
                    for (group_index, (group, targets)) in
                        groups.iter_mut().zip(targets).enumerate()
                    {
                        if blocks.len() != targets.len() {
                            return Err(EngineError::InvalidArgument(format!(
                                "query lease block count {} does not match destination block count {} for group {group_index}",
                                blocks.len(), targets.len()
                            )));
                        }
                        // Empty groups preserve connector indices but have no source route.
                        let Some((storage_group, expected_slots)) = group.storage_slots else {
                            continue;
                        };
                        for (source_index, destination) in targets.iter().enumerate() {
                            let Some(block_id) = destination else {
                                continue;
                            };
                            if blocks[source_index].slot_count() != expected_slots {
                                return Err(EngineError::InvalidArgument(format!(
                                    "stored block has {} slots but storage group {storage_group} of instance {instance_id} expects {expected_slots}: namespace is shared by incompatible KV layouts",
                                    blocks[source_index].slot_count()
                                )));
                            }
                            group.targets.push((*block_id, source_start + source_index));
                        }
                    }
                    sources.extend(blocks.iter());
                }
                RestorePlan::new(
                    device_id,
                    groups.iter().flat_map(|group| &group.targets)
                        .map(|&(_, index)| (index, sources[index])),
                ).map_err(EngineError::InvalidArgument)
            })?;
        for reservation in &reservations {
            reservation.restoring();
        }
        Ok(Self {
            plan,
            groups,
            sources,
            reservations,
        })
    }

    fn bind(
        self,
        layouts: Vec<KVCacheLayout>,
        completion: oneshot::Sender<LoadOutcome>,
        codec_budget: usize,
    ) -> Result<LoadTask, EngineError> {
        debug_assert_eq!(
            layouts.len(),
            self.groups.iter().map(|g| g.layers.len()).sum::<usize>()
        );
        let raw = self.groups.iter().all(|group| {
            group.targets.iter().all(|&(_, source_index)| {
                matches!(&self.sources[source_index], RestoreSource::Memory(sealed)
                    if group.layers.iter().all(|layer| sealed.get_slot(layer.slot_id)
                        .is_some_and(|slot| slot.encoding.is_none())))
            })
        });
        let payload = if raw {
            let count = self
                .groups
                .iter()
                .flat_map(|group| group.layers.iter().map(move |_| group.targets.len()))
                .zip(&layouts)
                .map(|(count, layout)| count * if layout.geometry().is_split() { 2 } else { 1 })
                .sum();
            let mut copies = Vec::with_capacity(count);
            let mut used = vec![false; self.sources.len()];
            let mut layouts = layouts.into_iter().enumerate();
            for group in self.groups {
                for &(_, source_index) in &group.targets {
                    used[source_index] = true;
                }
                for (layer, (allocation, layout)) in group.layers.into_iter().zip(&mut layouts) {
                    for &(block_index, source_index) in &group.targets {
                        let RestoreSource::Memory(sealed) = &self.sources[source_index] else {
                            unreachable!("raw source selection checked above")
                        };
                        let slot = sealed.get_slot(layer.slot_id).ok_or_else(|| {
                            EngineError::Storage(format!("missing source slot for {}", layer.name))
                        })?;
                        append_copy_descs(
                            &mut copies,
                            allocation,
                            layout
                                .block_copies(block_index)
                                .map_err(EngineError::Storage)?,
                            slot,
                            layer.host_offset,
                        )?;
                    }
                }
            }
            let sources = self
                .sources
                .into_iter()
                .zip(used)
                .filter_map(|(source, used)| {
                    if !used {
                        return None;
                    }
                    let RestoreSource::Memory(sealed) = source else {
                        unreachable!("raw source selection checked above")
                    };
                    Some(sealed)
                })
                .collect();
            LoadPayload::Raw { copies, sources }
        } else {
            let mut layers = Vec::with_capacity(layouts.len());
            let mut layouts = layouts.into_iter();
            for group in self.groups {
                for (layer, layout) in group.layers.into_iter().zip(&mut layouts) {
                    if group.targets.is_empty() {
                        continue;
                    }
                    let blocks = group
                        .targets
                        .iter()
                        .map(|&(block_idx, source_index)| TransferBlock {
                            block_idx,
                            block: match &self.sources[source_index] {
                                RestoreSource::Memory(sealed) => TransferPayload::Cached {
                                    sealed: Arc::clone(sealed),
                                    slot_id: layer.slot_id,
                                    offset: layer.host_offset,
                                },
                                RestoreSource::Ssd { lease, path, .. } => TransferPayload::Ssd {
                                    source: Arc::clone(lease),
                                    path: *path,
                                    slot_id: layer.slot_id,
                                    offset: layer.host_offset,
                                },
                            },
                        })
                        .collect();
                    layers.push(LayerTransferData {
                        layer_name: layer.name,
                        layout,
                        blocks,
                    });
                }
            }
            LoadPayload::Layers(layers)
        };
        Ok(LoadTask {
            plan: self.plan,
            payload,
            completion,
            reservations: self.reservations,
            codec_budget,
        })
    }
}
