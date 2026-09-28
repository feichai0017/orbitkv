use std::collections::HashMap;
use std::sync::Arc;

use crate::cost::{CostEstimateKey, CostObservationKind, ExecutionResource};
use crate::planning::restore::RestorePlan;
use crate::{EngineError, SsdReadPath, TransferMode};

use super::{LayerTransferData, TransferPayload};

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

pub(super) fn validate_plan(
    plan: &RestorePlan,
    layers: &[LayerTransferData],
) -> Result<(), EngineError> {
    let path = ssd_path(layers).map_err(EngineError::Storage)?;
    if path != plan.ssd_path() {
        return Err(EngineError::InvalidArgument(
            "restore plan source path differs from transfer payloads".into(),
        ));
    }
    Ok(())
}

pub(super) fn set_ssd_path(layers: &mut [LayerTransferData], path: SsdReadPath) {
    for block in layers.iter_mut().flat_map(|layer| &mut layer.blocks) {
        if let TransferPayload::Ssd { path: selected, .. } = &mut block.block {
            *selected = path;
        }
    }
}

pub(super) fn cost_estimate_key(
    plan: &RestorePlan,
    layers: &[LayerTransferData],
    mode: TransferMode,
    path: SsdReadPath,
    shape: CostEstimateKey,
) -> CostEstimateKey {
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
    let mut target_bytes = 0u64;
    let mut target_fragments = 0usize;
    for layer in layers {
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
    let resource = ExecutionResource::SsdRestore {
        device: plan.device_id() as u64,
        copy_backend: mode as u8,
        stores: crate::cost::resource_id(&resources),
        has_memory: plan.has_memory(),
    };
    shape
        .with_ssd_shape(
            plan.ssd_source_bytes(),
            plan.ssd_source_fragments(),
            target_bytes,
            target_fragments,
        )
        .with_observation_kind_and_resource(
            match path {
                SsdReadPath::Uring => CostObservationKind::SsdUringRestore,
                SsdReadPath::Cufile => CostObservationKind::SsdCufileRestore,
            },
            resource,
        )
}

pub(super) fn shadow(
    layers: &[LayerTransferData],
    codec_budget: usize,
    selected: SsdReadPath,
    key: CostEstimateKey,
) {
    let uring = key.with_observation_kind(CostObservationKind::SsdUringRestore);
    let cufile = key.with_observation_kind(CostObservationKind::SsdCufileRestore);
    let cufile_eligible =
        layers
            .iter()
            .flat_map(|layer| &layer.blocks)
            .all(|block| match &block.block {
                TransferPayload::Ssd { source, .. } => source.cufile_eligible(codec_budget),
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
pub(super) fn materialize_host(layers: &mut [LayerTransferData]) -> Result<(), EngineError> {
    super::ssd::validate_host_sources(layers)?;
    let mut sources = HashMap::new();
    for block in layers.iter().flat_map(|layer| &layer.blocks) {
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
    for block in layers.iter_mut().flat_map(|layer| &mut layer.blocks) {
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
