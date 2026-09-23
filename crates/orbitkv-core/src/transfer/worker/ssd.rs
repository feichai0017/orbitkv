use std::collections::HashMap;
use std::sync::Arc;

use crate::backing::ssd::GpuWriteLease;
use crate::backing::ssd::cufile::{CopyRange, IoBatch, plan_reads};
use crate::transfer::layout::BlockCopies;
use crate::{EngineError, SlotMeta, SsdReadLease};

use super::{LayerTransferData, TransferPayload};

type SourceReads = (Arc<SsdReadLease>, Vec<IoBatch>);

pub(crate) struct GpuWrite {
    pub lease: GpuWriteLease,
    pub batches: Vec<IoBatch>,
}

/// Validate all source/destination ranges before issuing any storage I/O.
pub(super) fn plan(layers: &[LayerTransferData]) -> Result<Vec<SourceReads>, EngineError> {
    let mut sources: HashMap<usize, (Arc<SsdReadLease>, Vec<CopyRange>)> = HashMap::new();
    for layer in layers {
        for block in &layer.blocks {
            let TransferPayload::Ssd {
                source,
                slot_id,
                offset,
            } = &block.block
            else {
                continue;
            };
            let slot = source
                .entry
                .slots
                .get(*slot_id)
                .ok_or_else(|| EngineError::Storage("SSD slot is missing".into()))?;
            let base = source.entry.file_offset
                + source.entry.slots[..*slot_id]
                    .iter()
                    .map(|slot| slot.total_size())
                    .sum::<u64>();
            let copies = layer
                .layout
                .block_copies(block.block_idx)
                .map_err(EngineError::Storage)?;
            let reads = &mut sources
                .entry(Arc::as_ptr(source) as usize)
                .or_insert_with(|| (Arc::clone(source), Vec::new()))
                .1;
            let mut push =
                |segment: usize, relative: usize, device, bytes| -> Result<(), EngineError> {
                    let file_offset = segment_offset(slot, base, segment, relative, bytes)?;
                    reads.push(CopyRange {
                        file_offset,
                        device,
                        bytes,
                    });
                    Ok(())
                };
            match copies {
                BlockCopies::Contiguous(copy) => push(0, *offset, copy.addr, copy.bytes)?,
                BlockCopies::Split { k, v } => {
                    push(0, *offset, k.addr, k.bytes)?;
                    if slot.num_segments() > 1 {
                        push(1, *offset, v.addr, v.bytes)?;
                    } else {
                        push(
                            0,
                            offset.checked_add(k.bytes).ok_or_else(|| {
                                EngineError::Storage("SSD slot offset overflow".into())
                            })?,
                            v.addr,
                            v.bytes,
                        )?;
                    }
                }
            }
        }
    }
    sources
        .into_values()
        .map(|(source, copies)| {
            plan_reads(copies)
                .map(|batches| (source, batches))
                .map_err(EngineError::Storage)
        })
        .collect()
}

fn segment_offset(
    slot: &SlotMeta,
    base: u64,
    segment: usize,
    offset: usize,
    bytes: usize,
) -> Result<u64, EngineError> {
    let size = slot
        .segment_sizes
        .get(segment)
        .copied()
        .ok_or_else(|| EngineError::Storage("SSD segment missing".into()))?;
    if offset
        .checked_add(bytes)
        .is_none_or(|end| end as u64 > size)
    {
        return Err(EngineError::Storage(
            "SSD segment is smaller than the registered GPU layout".into(),
        ));
    }
    base.checked_add(slot.segment_sizes[..segment].iter().sum::<u64>())
        .and_then(|start| start.checked_add(offset as u64))
        .ok_or_else(|| EngineError::Storage("SSD segment offset overflow".into()))
}

#[cfg(test)]
#[path = "../../../tests/unit/transfer/worker/ssd.rs"]
mod tests;
