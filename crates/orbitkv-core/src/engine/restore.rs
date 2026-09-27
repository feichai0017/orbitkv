use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::oneshot;

use super::instance::LayerTopology;
use super::{EngineError, OrbitKVEngine};
use crate::QueryReservation;
use crate::block::{RestoreSource, SealedBlock};
use crate::metrics::core_metrics;
use crate::planning::restore::RestorePlan;
use crate::query::lease::{QueryLeaseId, QueryLeaseManager};
use crate::transfer::layout::{BlockRanges, KVCacheLayout};
use crate::transfer::local::{MAX_PLAN_BYTES, RawCopy, RawRestorePlan};
use crate::transfer::worker::{
    LayerTransferData, LoadOutcome, LoadTask, TransferBlock, TransferPayload,
};

/// Source ownership for an engine-local transfer. The grant owner may release it
/// only after a never-claimed revocation or an authoritative local DMA drain.
pub struct RawRestoreGrant {
    plan: Vec<u8>,
    sources: Vec<Arc<SealedBlock>>,
    reservations: Vec<QueryReservation>,
    bytes: u64,
    started: std::time::Instant,
}

impl RawRestoreGrant {
    pub fn encoded_plan(&self) -> &[u8] {
        &self.plan
    }

    pub fn finish(self, success: bool) {
        if success {
            for source in &self.sources {
                source.mark_warmup_restored();
            }
            if self.bytes != 0 {
                core_metrics().load_bytes.add(self.bytes, &[]);
                core_metrics()
                    .load_duration_seconds
                    .record(self.started.elapsed().as_secs_f64(), &[]);
            }
        } else {
            core_metrics().load_failures.add(1, &[]);
        }
        drop(self.sources);
        drop(self.reservations);
    }
}

impl std::fmt::Debug for RawRestoreGrant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawRestoreGrant")
            .field("plan_bytes", &self.plan.len())
            .field("sources", &self.sources.len())
            .field("bytes", &self.bytes)
            .finish()
    }
}

#[derive(Debug)]
pub enum RestoreExecution {
    Local(RawRestoreGrant),
    Managed(oneshot::Receiver<LoadOutcome>),
}

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
    raw: Option<(Vec<u8>, u64)>,
    plan: RestorePlan,
    groups: Vec<RestoreGroup>,
    sources: Vec<RestoreSource>,
    reservations: Vec<QueryReservation>,
}

impl OrbitKVEngine {
    pub fn payload_arenas(&self) -> Result<Vec<crate::PayloadArena>, EngineError> {
        self.storage
            .allocator
            .payload_arenas()
            .map_err(|error| EngineError::Storage(error.to_string()))
    }

    /// Prepare leased sources for one GPU. Raw DRAM grants execute in the engine
    /// process; SSD and codec routes retain the Manager's worker completion fence.
    pub fn restore(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        layer_groups: &[Vec<&str>],
        loads: &[(QueryLeaseId, Vec<Vec<Option<usize>>>)],
    ) -> Result<RestoreExecution, EngineError> {
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
        let prepared = PreparedRestore::prepare(
            &self.query_leases,
            instance_id,
            device_id,
            groups,
            loads,
            &layouts,
        )?;
        trace_drop!(lookup);
        trace_scope!("load.build_tasks");
        if let Some((plan, bytes)) = prepared.raw {
            let mut used = vec![false; prepared.sources.len()];
            for group in &prepared.groups {
                for &(_, source) in &group.targets {
                    used[source] = true;
                }
            }
            let sources = prepared
                .sources
                .into_iter()
                .zip(used)
                .filter_map(|(source, used)| {
                    if !used {
                        return None;
                    }
                    let RestoreSource::Memory(source) = source else {
                        unreachable!("raw plan contains only checked DRAM sources")
                    };
                    Some(source)
                })
                .collect();
            return Ok(RestoreExecution::Local(RawRestoreGrant {
                plan,
                sources,
                reservations: prepared.reservations,
                bytes,
                started: std::time::Instant::now(),
            }));
        }
        let (completion, receiver) = oneshot::channel();
        let task = prepared.bind(layouts, completion, self.storage.codec_budget);
        if task.layers.is_empty() {
            drop(task.layers);
            drop(task.reservations);
            let _ = task.completion.send(LoadOutcome {
                result: Ok(()),
                completed_at: std::time::Instant::now(),
            });
        } else {
            gpu.worker_pool().submit_load(task)?;
        }
        Ok(RestoreExecution::Managed(receiver))
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
        layouts: &[KVCacheLayout],
    ) -> Result<Self, EngineError> {
        let tokens: Vec<_> = loads.iter().map(|(token, _)| *token).collect();
        let ((plan, raw), sources, reservations) = leases
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
                let plan = RestorePlan::new(
                    device_id,
                    groups.iter().flat_map(|group| &group.targets)
                        .map(|&(_, index)| (index, sources[index])),
                ).map_err(EngineError::InvalidArgument)?;
                let raw = Self::raw_plan(&groups, &sources, layouts)?;
                Ok((plan, raw))
            })?;
        for reservation in &reservations {
            reservation.restoring();
        }
        Ok(Self {
            raw,
            plan,
            groups,
            sources,
            reservations,
        })
    }

    fn raw_plan(
        groups: &[RestoreGroup],
        sources: &[&RestoreSource],
        layouts: &[KVCacheLayout],
    ) -> Result<Option<(Vec<u8>, u64)>, EngineError> {
        let raw = groups.iter().all(|group| {
            group.targets.iter().all(|&(_, source)| {
                matches!(sources[source], RestoreSource::Memory(sealed)
                    if group.layers.iter().all(|layer| sealed.get_slot(layer.slot_id)
                        .is_some_and(|slot| slot.encoding.is_none())))
            })
        });
        if !raw {
            return Ok(None);
        }
        let mut plan_size = 8usize;
        let mut layouts_iter = layouts.iter();
        for group in groups {
            for layer in &group.layers {
                let layout = layouts_iter
                    .next()
                    .ok_or_else(|| EngineError::Storage("missing restore layout".into()))?;
                let copies = group
                    .targets
                    .len()
                    .checked_mul(if layout.geometry().is_split() { 2 } else { 1 });
                plan_size = copies.and_then(|count| (58usize.checked_add(layer.name.len())?).checked_mul(count))
                    .and_then(|bytes| plan_size.checked_add(bytes))
                    .filter(|size| *size <= MAX_PLAN_BYTES)
                    .ok_or_else(|| EngineError::InvalidArgument("raw restore plan exceeds the bounded plan bank; split the restore batch".into()))?;
            }
        }
        let mut copies = Vec::new();
        let mut bytes = 0u64;
        let mut layouts = layouts.iter();
        let mut targets = Vec::new();
        for group in groups {
            for layer in &group.layers {
                let layout = layouts
                    .next()
                    .ok_or_else(|| EngineError::Storage("missing restore layout".into()))?;
                for &(block, source) in &group.targets {
                    let RestoreSource::Memory(sealed) = sources[source] else {
                        unreachable!("raw sources checked above")
                    };
                    let slot = sealed
                        .get_slot(layer.slot_id)
                        .ok_or_else(|| EngineError::Storage("missing raw source slot".into()))?;
                    let mut append = |segment,
                                      host_offset,
                                      destination: std::ops::Range<usize>|
                     -> Result<(), EngineError> {
                        let source = slot
                            .source_range(segment, host_offset, destination.len())
                            .map_err(EngineError::Storage)?;
                        bytes = bytes.checked_add(source.size).ok_or_else(|| {
                            EngineError::Storage("restore byte count overflow".into())
                        })?;
                        copies.push(RawCopy {
                            source,
                            layer: layer.name.clone(),
                            destination_offset: destination.start as u64,
                        });
                        Ok(())
                    };
                    match layout
                        .geometry()
                        .block_ranges(block)
                        .map_err(EngineError::Storage)?
                    {
                        BlockRanges::Contiguous(range) => append(0, layer.host_offset, range)?,
                        BlockRanges::Split { k, v } => {
                            let (segment, offset) = if slot.num_segments() > 1 {
                                (1, layer.host_offset)
                            } else {
                                (
                                    0,
                                    layer.host_offset.checked_add(k.len()).ok_or_else(|| {
                                        EngineError::Storage("raw source offset overflow".into())
                                    })?,
                                )
                            };
                            append(0, layer.host_offset, k)?;
                            append(segment, offset, v)?;
                        }
                    }
                    match layout.block_copies(block).map_err(EngineError::Storage)? {
                        crate::transfer::layout::BlockCopies::Contiguous(copy) => {
                            targets.push((copy.addr, copy.bytes));
                        }
                        crate::transfer::layout::BlockCopies::Split { k, v } => {
                            targets.extend([(k.addr, k.bytes), (v.addr, v.bytes)]);
                        }
                    }
                }
            }
        }
        crate::codec::gpu::validate_targets(targets).map_err(EngineError::Storage)?;
        let encoded = RawRestorePlan { copies }
            .encode()
            .map_err(EngineError::Storage)?;
        Ok(Some((encoded, bytes)))
    }

    fn bind(
        self,
        layouts: Vec<KVCacheLayout>,
        completion: oneshot::Sender<LoadOutcome>,
        codec_budget: usize,
    ) -> LoadTask {
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
        LoadTask {
            plan: self.plan,
            layers,
            completion,
            reservations: self.reservations,
            codec_budget,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/engine/restore.rs"]
mod tests;
