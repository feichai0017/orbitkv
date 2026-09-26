use std::collections::HashMap;
use std::sync::Arc;

use crate::cost::{CostKey, CostPath, Resource};
use crate::{EngineError, SsdReadPath, TransferMode};

use super::{LayerTransferData, LoadTask, TransferPayload};

fn ssd_path(layers: &[LayerTransferData]) -> Result<Option<SsdReadPath>, String> {
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

pub(super) fn validate_plan(task: &LoadTask) -> Result<(), EngineError> {
    let path = ssd_path(&task.layers).map_err(EngineError::Storage)?;
    if path != task.plan.ssd_path() {
        return Err(EngineError::InvalidArgument(
            "restore plan source path differs from transfer payloads".into(),
        ));
    }
    Ok(())
}

pub(super) fn cost_key(
    task: &LoadTask,
    mode: TransferMode,
    path: SsdReadPath,
    shape: CostKey,
) -> CostKey {
    let mut resources: Vec<_> = task
        .layers
        .iter()
        .flat_map(|layer| &layer.blocks)
        .filter_map(|block| match &block.block {
            TransferPayload::Ssd { source, .. } => Some(source.cost_resource()),
            _ => None,
        })
        .collect();
    resources.sort_unstable();
    resources.dedup();
    let mut target_bytes = 0u64;
    let mut target_fragments = 0usize;
    for layer in &task.layers {
        for block in &layer.blocks {
            if let TransferPayload::Ssd { .. } = &block.block
                && let Ok(copies) = layer.layout.block_copies(block.block_idx)
            {
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
    let resource = Resource::SsdRestore {
        device: task.plan.device_id() as u64,
        copy_backend: mode as u8,
        stores: crate::cost::resource_id(&resources),
        has_memory: task.plan.has_memory(),
    };
    shape
        .with_ssd_shape(
            task.plan.ssd_source_bytes(),
            task.plan.ssd_source_fragments(),
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
