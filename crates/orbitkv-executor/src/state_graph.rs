//! Shared graph addressing for fixed-size state arenas.

use std::collections::BTreeMap;

use luminal::{
    dtype::DType,
    prelude::{Expression, Graph, GraphTensor},
};
use luminal_cuda_lite::{cudarc::driver::CudaSlice, runtime::CudaRuntime};
use thiserror::Error;

use crate::{FixedStateArenaRegistration, FixedStateDeviceBatch};

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum FixedStateGraphError {
    #[error("fixed-state graph geometry is invalid")]
    InvalidGeometry,
    #[error("fixed-state batch does not match the compiled state class")]
    InvalidBatch,
}

/// Final graph handles for one fixed-state arena.
#[derive(Clone, Copy)]
pub struct FixedStateGraphBinding {
    pub state_id: u16,
    pub arena_input: GraphTensor,
    pub arena_output: GraphTensor,
    pub destination_slots: GraphTensor,
    arena_bytes: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct FixedStateGraphLayout<'a> {
    pub(crate) state_id: u16,
    pub(crate) layers: &'a [u32],
    pub(crate) bytes_per_layer: u64,
    pub(crate) slots_per_request: u32,
    pub(crate) bytes_per_request: u64,
    pub(crate) dtype: DType,
}

pub(crate) struct FixedStateGraphArena {
    state_id: u16,
    batch_size: Expression,
    dtype: DType,
    arena_input: GraphTensor,
    arena_version: GraphTensor,
    destination_slots: GraphTensor,
    arena_bytes: usize,
    slot_elements: usize,
    layer_elements: usize,
    layer_positions: BTreeMap<u32, usize>,
}

impl FixedStateGraphArena {
    pub(crate) fn new(
        graph: &mut Graph,
        layout: FixedStateGraphLayout<'_>,
        registration: FixedStateArenaRegistration,
        batch_size: Expression,
    ) -> Result<Self, FixedStateGraphError> {
        let element_bytes = layout
            .dtype
            .bits()
            .checked_div(8)
            .filter(|bytes| *bytes > 0 && layout.dtype.bits().is_multiple_of(8))
            .ok_or(FixedStateGraphError::InvalidGeometry)?;
        let layer_elements = elements(layout.bytes_per_layer, element_bytes)?;
        let slot_elements = elements(registration.slot_bytes, element_bytes)?;
        if registration.state_id != layout.state_id
            || layout.layers.is_empty()
            || batch_size.to_usize() == Some(0)
            || registration
                .slot_bytes
                .checked_mul(u64::from(layout.slots_per_request))
                != Some(layout.bytes_per_request)
            || layer_elements.checked_mul(layout.layers.len()) != Some(slot_elements)
        {
            return Err(FixedStateGraphError::InvalidGeometry);
        }
        let arena_elements = slot_elements
            .checked_mul(
                usize::try_from(registration.slot_count)
                    .map_err(|_| FixedStateGraphError::InvalidGeometry)?,
            )
            .ok_or(FixedStateGraphError::InvalidGeometry)?;
        if i32::try_from(arena_elements).is_err() {
            return Err(FixedStateGraphError::InvalidGeometry);
        }
        let arena_bytes = arena_elements
            .checked_mul(element_bytes)
            .ok_or(FixedStateGraphError::InvalidGeometry)?;
        let prefix = format!("state.{}", layout.state_id);
        let arena_input = graph
            .named_tensor(format!("{prefix}.arena"), arena_elements)
            .persist()
            .as_dtype(layout.dtype);
        let destination_slots = graph
            .named_tensor(format!("{prefix}.destination_slots"), batch_size)
            .as_dtype(DType::Int);
        let layer_positions = layout
            .layers
            .iter()
            .copied()
            .enumerate()
            .map(|(position, layer)| (layer, position))
            .collect::<BTreeMap<_, _>>();
        if layer_positions.len() != layout.layers.len() {
            return Err(FixedStateGraphError::InvalidGeometry);
        }
        Ok(Self {
            state_id: layout.state_id,
            batch_size,
            dtype: layout.dtype,
            arena_input,
            arena_version: arena_input,
            destination_slots,
            arena_bytes,
            slot_elements,
            layer_elements,
            layer_positions,
        })
    }

    pub(crate) fn layer_state(
        &self,
        layer: u32,
        dimensions: &[usize],
    ) -> Result<GraphTensor, FixedStateGraphError> {
        let indices = self.layer_indices(layer, dimensions)?;
        Ok(self.arena_version.gather(indices))
    }

    pub(crate) fn commit_layer(
        &mut self,
        layer: u32,
        dimensions: &[usize],
        next_state: &GraphTensor,
    ) -> Result<(), FixedStateGraphError> {
        let expected = std::iter::once(self.batch_size)
            .chain(dimensions.iter().copied().map(Expression::from))
            .collect::<Vec<_>>();
        if next_state.graph_ref != self.arena_input.graph_ref
            || next_state.dtype != self.dtype
            || next_state.dims() != expected
        {
            return Err(FixedStateGraphError::InvalidGeometry);
        }
        let indices = self.layer_indices(layer, dimensions)?;
        self.arena_version = (*next_state).scatter(indices, self.arena_version);
        Ok(())
    }

    pub(crate) fn finish(self) -> FixedStateGraphBinding {
        FixedStateGraphBinding {
            state_id: self.state_id,
            arena_input: self.arena_input,
            arena_output: self.arena_version.output(),
            destination_slots: self.destination_slots,
            arena_bytes: self.arena_bytes,
        }
    }

    fn layer_indices(
        &self,
        layer: u32,
        dimensions: &[usize],
    ) -> Result<GraphTensor, FixedStateGraphError> {
        let shape_elements = dimensions
            .iter()
            .copied()
            .try_fold(1_usize, usize::checked_mul)
            .filter(|elements| *elements == self.layer_elements)
            .ok_or(FixedStateGraphError::InvalidGeometry)?;
        let layer_position = *self
            .layer_positions
            .get(&layer)
            .ok_or(FixedStateGraphError::InvalidGeometry)?;
        let layer_offset = layer_position
            .checked_mul(shape_elements)
            .ok_or(FixedStateGraphError::InvalidGeometry)?;
        let graph = self.arena_input.graph();
        let mut local = graph
            .iota('z', dimensions.to_vec())
            .expand_dim(0, self.batch_size);
        let mut slot_base = self.destination_slots * self.slot_elements;
        for (axis, dimension) in dimensions.iter().copied().enumerate() {
            slot_base = slot_base.expand_dim(axis + 1, dimension);
        }
        local = local + slot_base + layer_offset;
        Ok(local)
    }
}

impl FixedStateGraphBinding {
    #[must_use]
    pub fn allocate_compile_scratch(self, runtime: &mut CudaRuntime) -> CudaSlice<u8> {
        runtime.alias_state_required(self.arena_input, self.arena_output, self.arena_bytes)
    }

    /// Seeds a stable-capacity slot-id input before graph compilation.
    ///
    /// # Errors
    ///
    /// Rejects zero or inconsistent representative and maximum batch sizes.
    pub fn seed_destination_slots(
        self,
        runtime: &mut CudaRuntime,
        representative_batch: usize,
        maximum_batch: usize,
    ) -> Result<(), FixedStateGraphError> {
        if representative_batch == 0
            || representative_batch > maximum_batch
            || maximum_batch
                .checked_mul(std::mem::size_of::<i32>())
                .is_none()
        {
            return Err(FixedStateGraphError::InvalidGeometry);
        }
        runtime.set_data_with_capacity(
            self.destination_slots,
            vec![0_i32; representative_batch],
            maximum_batch * std::mem::size_of::<i32>(),
        );
        Ok(())
    }

    pub(crate) fn upload_destination_slots(
        self,
        runtime: &mut CudaRuntime,
        batches: &[FixedStateDeviceBatch],
    ) -> Result<(), FixedStateGraphError> {
        runtime.set_data(
            self.destination_slots,
            destination_slot_ids(self.state_id, batches)?,
        );
        Ok(())
    }
}

pub(crate) fn destination_slot_ids(
    state_id: u16,
    batches: &[FixedStateDeviceBatch],
) -> Result<Vec<i32>, FixedStateGraphError> {
    if batches.is_empty() {
        return Err(FixedStateGraphError::InvalidBatch);
    }
    batches
        .iter()
        .map(|batch| {
            let mut matches = batch
                .destinations
                .iter()
                .filter(|destination| destination.state_id == state_id);
            let destination = matches.next().ok_or(FixedStateGraphError::InvalidBatch)?;
            if matches.next().is_some() {
                return Err(FixedStateGraphError::InvalidBatch);
            }
            i32::try_from(destination.lease.slot_id).map_err(|_| FixedStateGraphError::InvalidBatch)
        })
        .collect()
}

fn elements(bytes: u64, element_bytes: usize) -> Result<usize, FixedStateGraphError> {
    let element_bytes =
        u64::try_from(element_bytes).map_err(|_| FixedStateGraphError::InvalidGeometry)?;
    if !bytes.is_multiple_of(element_bytes) {
        return Err(FixedStateGraphError::InvalidGeometry);
    }
    usize::try_from(bytes / element_bytes).map_err(|_| FixedStateGraphError::InvalidGeometry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FixedStateDeviceRange;

    #[test]
    fn destination_slots_preserve_request_order_and_reject_bad_classes() {
        let range = |request_id: u64, state_id: u16, slot_id: u32| FixedStateDeviceBatch {
            request_id,
            sources: Box::default(),
            destinations: vec![FixedStateDeviceRange {
                state_id,
                lease: orbitkv::StateSlotLease {
                    engine_epoch: 1,
                    pool_epoch: 2,
                    generation: 1,
                    slot_id,
                    pool_id: 3,
                },
                device_ptr: 64,
                byte_offset: 0,
                byte_count: 16,
            }]
            .into_boxed_slice(),
        };
        let ordered = [range(9, 4, 2), range(3, 4, 0)];
        assert_eq!(destination_slot_ids(4, &ordered).unwrap(), vec![2, 0]);
        assert!(matches!(
            destination_slot_ids(5, &ordered),
            Err(FixedStateGraphError::InvalidBatch)
        ));
        let duplicate = FixedStateDeviceBatch {
            destinations: vec![ordered[0].destinations[0], ordered[0].destinations[0]]
                .into_boxed_slice(),
            ..ordered[0].clone()
        };
        assert!(matches!(
            destination_slot_ids(4, &[duplicate]),
            Err(FixedStateGraphError::InvalidBatch)
        ));
    }
}
