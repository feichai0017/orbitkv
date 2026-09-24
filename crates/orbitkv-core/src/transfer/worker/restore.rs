use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::cost::{CostKey, CostPath};
use crate::{EngineError, SsdReadPath, TransferMode};

use super::{LayerTransferData, LoadTask, TransferPayload};

pub(super) fn ssd_path(layers: &[LayerTransferData]) -> Result<Option<SsdReadPath>, String> {
    let mut selected = None;
    for block in layers.iter().flat_map(|layer| &layer.blocks) {
        if let TransferPayload::Ssd { path, .. } = block.block {
            if selected.is_some_and(|selected| selected != path) {
                return Err("one restore task cannot mix SSD read routes".into());
            }
            selected = Some(path);
        }
    }
    Ok(selected)
}

pub(super) fn cost_key(
    layers: &[LayerTransferData],
    device: i32,
    mode: TransferMode,
    path: SsdReadPath,
    shape: CostKey,
) -> CostKey {
    let mut resources: Vec<_> = layers
        .iter()
        .flat_map(|layer| &layer.blocks)
        .filter_map(|block| match &block.block {
            TransferPayload::Ssd { source, .. } => Some(source.cost_resource()),
            _ => None,
        })
        .collect();
    resources.sort_unstable();
    resources.dedup();
    let mut sources = HashSet::new();
    let mut source_bytes = 0u64;
    let mut source_fragments = 0usize;
    let mut target_bytes = 0u64;
    let mut target_fragments = 0usize;
    for layer in layers {
        for block in &layer.blocks {
            if let TransferPayload::Ssd { source, .. } = &block.block {
                if sources.insert(Arc::as_ptr(&source.entry.readers)) {
                    for slot in &source.entry.slots {
                        source_bytes = source_bytes.saturating_add(slot.total_size());
                        source_fragments = source_fragments.saturating_add(slot.num_segments());
                    }
                }
                if let Ok(copies) = layer.layout.block_copies(block.block_idx) {
                    use crate::transfer::layout::BlockCopies;
                    match copies {
                        BlockCopies::Contiguous(copy) => {
                            target_bytes = target_bytes.saturating_add(copy.bytes as u64);
                            target_fragments += 1;
                        }
                        BlockCopies::Split { k, v } => {
                            target_bytes = target_bytes
                                .saturating_add(k.bytes as u64)
                                .saturating_add(v.bytes as u64);
                            target_fragments += 2;
                        }
                    }
                }
            }
        }
    }
    let has_memory = layers
        .iter()
        .flat_map(|layer| &layer.blocks)
        .any(|block| !matches!(block.block, TransferPayload::Ssd { .. }));
    let resource = crate::cost::resource_id(&(device, mode as u8, resources, has_memory));
    shape
        .with_ssd_shape(
            source_bytes,
            source_fragments,
            target_bytes,
            target_fragments,
        )
        .with_path_resource(
            match path {
                SsdReadPath::Uring => CostPath::SsdUringRestore,
                SsdReadPath::Cufile => CostPath::SsdCufileRestore,
            },
            resource,
        )
}

pub(super) fn shadow(task: &LoadTask, selected: SsdReadPath, key: CostKey) {
    let uring = key.with_path(CostPath::SsdUringRestore);
    let cufile = key.with_path(CostPath::SsdCufileRestore);
    let cufile_eligible = task
        .layers
        .iter()
        .flat_map(|layer| &layer.blocks)
        .all(|block| match &block.block {
            TransferPayload::Ssd { source, .. } => source.cufile_eligible(task.codec_budget),
            _ => true,
        });
    let candidates = [uring, cufile];
    crate::cost::shadow(
        if cufile_eligible {
            &candidates
        } else {
            &candidates[..1]
        },
        usize::from(selected == SsdReadPath::Cufile),
    );
}

/// Runs only on the independent SSD host lane. The existing bounded reader
/// owns submitted I/O; the task keeps engine mappings through final GPU completion.
pub(super) fn materialize_host(task: &mut LoadTask) -> Result<(), EngineError> {
    super::ssd::validate_host_sources(&task.layers)?;
    let mut sources = HashMap::new();
    for block in task.layers.iter().flat_map(|layer| &layer.blocks) {
        if let TransferPayload::Ssd { source, path, .. } = &block.block {
            if *path != SsdReadPath::Uring {
                return Err(EngineError::Storage(
                    "cuFile source reached the SSD host lane".into(),
                ));
            }
            sources
                .entry(Arc::as_ptr(&source.entry.readers) as usize)
                .or_insert_with(|| Arc::clone(source));
        }
    }
    let reads = sources.into_iter().map(|(identity, source)| async move {
        source.read_host().await.map(|block| (identity, block))
    });
    // Poll every submitted read to terminal completion, including after one fails.
    let results = futures::executor::block_on(futures::future::join_all(reads));
    let blocks = results.into_iter().collect::<Result<HashMap<_, _>, _>>()?;
    for block in task.layers.iter_mut().flat_map(|layer| &mut layer.blocks) {
        if let TransferPayload::Ssd {
            source,
            slot_id,
            offset,
            ..
        } = &block.block
        {
            let sealed = blocks
                .get(&(Arc::as_ptr(&source.entry.readers) as usize))
                .ok_or_else(|| EngineError::Storage("SSD host result is missing".into()))?;
            block.block = TransferPayload::Cached {
                sealed: Arc::clone(sealed),
                slot_id: *slot_id,
                offset: *offset,
            };
        }
    }
    Ok(())
}
