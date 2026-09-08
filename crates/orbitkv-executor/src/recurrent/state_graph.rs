//! Luminal graph views over an OrbitKV-owned recurrent-state arena.

use std::collections::BTreeMap;

use luminal::{
    dtype::DType,
    prelude::{Expression, Graph, GraphTensor},
};
use luminal_cuda_lite::{cudarc::driver::CudaSlice, runtime::CudaRuntime};

use crate::{
    FixedStateArenaRegistration, FixedStateClass, FixedStateDeviceBatch, FixedStateStorage,
    GatedDeltaGeometry, RecurrentError,
};

/// Mutable graph builder for one fixed-address recurrent-state arena.
pub struct RecurrentStateGraphArena {
    state_id: u16,
    batch_size: Expression,
    arena_input: GraphTensor,
    arena_version: GraphTensor,
    destination_slots: GraphTensor,
    arena_bytes: usize,
    slot_elements: usize,
    layer_elements: usize,
    layer_positions: BTreeMap<u32, usize>,
}

/// Final graph handles used to bind the physical arena and upload per-step
/// slot metadata.
#[derive(Clone, Copy)]
pub struct RecurrentStateGraphBinding {
    pub state_id: u16,
    pub arena_input: GraphTensor,
    pub arena_output: GraphTensor,
    pub destination_slots: GraphTensor,
    arena_bytes: usize,
}

impl RecurrentStateGraphArena {
    /// Creates graph inputs for the complete fixed-size arena and the dynamic
    /// destination slot selected for each request.
    ///
    /// # Errors
    ///
    /// Rejects non-recurrent storage, mismatched registration, non-f32 byte
    /// geometry, or an empty request batch.
    pub fn new(
        graph: &mut Graph,
        class: &FixedStateClass,
        registration: FixedStateArenaRegistration,
        batch_size: Expression,
    ) -> Result<Self, RecurrentError> {
        let FixedStateStorage::Recurrent {
            bytes_per_layer,
            slots_per_request,
            bytes_per_request,
            ..
        } = class.storage
        else {
            return Err(RecurrentError::InvalidGeometry);
        };
        let layer_count = class.layers.len();
        let layer_elements = elements(bytes_per_layer)?;
        let slot_elements = elements(registration.slot_bytes)?;
        if registration.state_id != class.state_id
            || layer_count == 0
            || batch_size.to_usize() == Some(0)
            || registration
                .slot_bytes
                .checked_mul(u64::from(slots_per_request))
                != Some(bytes_per_request)
            || layer_elements.checked_mul(layer_count) != Some(slot_elements)
        {
            return Err(RecurrentError::InvalidGeometry);
        }
        let arena_elements = slot_elements
            .checked_mul(
                usize::try_from(registration.slot_count)
                    .map_err(|_| RecurrentError::InvalidGeometry)?,
            )
            .ok_or(RecurrentError::InvalidGeometry)?;
        if i32::try_from(arena_elements).is_err() {
            return Err(RecurrentError::InvalidGeometry);
        }
        let arena_bytes = arena_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(RecurrentError::InvalidGeometry)?;
        let prefix = format!("state.{}", class.state_id);
        let arena_input = graph
            .named_tensor(format!("{prefix}.arena"), arena_elements)
            .persist()
            .as_dtype(DType::F32);
        let destination_slots = graph
            .named_tensor(format!("{prefix}.destination_slots"), batch_size)
            .as_dtype(DType::Int);
        let layer_positions = class
            .layers
            .iter()
            .copied()
            .enumerate()
            .map(|(position, layer)| (layer, position))
            .collect::<BTreeMap<_, _>>();
        if layer_positions.len() != layer_count {
            return Err(RecurrentError::InvalidGeometry);
        }
        Ok(Self {
            state_id: class.state_id,
            batch_size,
            arena_input,
            arena_version: arena_input,
            destination_slots,
            arena_bytes,
            slot_elements,
            layer_elements,
            layer_positions,
        })
    }

    /// Gathers one layer's request states from manager-selected destination
    /// slots and views them as `[batch, heads, key, value]`.
    ///
    /// # Errors
    ///
    /// Rejects a layer outside this class or incompatible recurrent geometry.
    pub fn layer_state(
        &self,
        layer: u32,
        geometry: GatedDeltaGeometry,
    ) -> Result<GraphTensor, RecurrentError> {
        self.validate_layer(layer, geometry)?;
        Ok(self
            .arena_version
            .gather(self.layer_indices(layer, geometry)?))
    }

    /// Commits one layer's next state into the current logical arena version.
    ///
    /// # Errors
    ///
    /// Rejects a foreign graph tensor, layer, dtype, or shape.
    pub fn commit_layer(
        &mut self,
        layer: u32,
        geometry: GatedDeltaGeometry,
        next_state: GraphTensor,
    ) -> Result<(), RecurrentError> {
        self.validate_layer(layer, geometry)?;
        if next_state.graph_ref != self.arena_input.graph_ref
            || next_state.dtype != DType::F32
            || next_state.dims()
                != [
                    self.batch_size,
                    geometry.heads.into(),
                    geometry.key_width.into(),
                    geometry.value_width.into(),
                ]
        {
            return Err(RecurrentError::InvalidGeometry);
        }
        let indices = self.layer_indices(layer, geometry)?;
        self.arena_version = next_state.scatter(indices, self.arena_version);
        Ok(())
    }

    /// Marks the final arena version as an output required to alias the input.
    #[must_use]
    pub fn finish(self) -> RecurrentStateGraphBinding {
        RecurrentStateGraphBinding {
            state_id: self.state_id,
            arena_input: self.arena_input,
            arena_output: self.arena_version.output(),
            destination_slots: self.destination_slots,
            arena_bytes: self.arena_bytes,
        }
    }

    fn validate_layer(
        &self,
        layer: u32,
        geometry: GatedDeltaGeometry,
    ) -> Result<(), RecurrentError> {
        geometry.validate()?;
        if !self.layer_positions.contains_key(&layer)
            || geometry
                .heads
                .checked_mul(geometry.key_width)
                .and_then(|elements| elements.checked_mul(geometry.value_width))
                != Some(self.layer_elements)
        {
            return Err(RecurrentError::InvalidGeometry);
        }
        Ok(())
    }

    fn layer_indices(
        &self,
        layer: u32,
        geometry: GatedDeltaGeometry,
    ) -> Result<GraphTensor, RecurrentError> {
        let layer_position = *self
            .layer_positions
            .get(&layer)
            .ok_or(RecurrentError::InvalidGeometry)?;
        let layer_offset = layer_position
            .checked_mul(self.layer_elements)
            .ok_or(RecurrentError::InvalidGeometry)?;
        let graph = self.arena_input.graph();
        let local = graph
            .iota(
                'z',
                (geometry.heads, geometry.key_width, geometry.value_width),
            )
            .expand_dim(0, self.batch_size);
        let slot_base = (self.destination_slots * self.slot_elements)
            .expand_dim(1, geometry.heads)
            .expand_dim(2, geometry.key_width)
            .expand_dim(3, geometry.value_width);
        Ok(local + slot_base + layer_offset)
    }
}

impl RecurrentStateGraphBinding {
    /// Allocates a temporary runtime-owned arena used only while Luminal
    /// searches candidates. Bind the `OrbitKV` arena after compilation.
    #[must_use]
    pub fn allocate_compile_scratch(self, runtime: &mut CudaRuntime) -> CudaSlice<u8> {
        runtime.alias_state_required(self.arena_input, self.arena_output, self.arena_bytes)
    }
    /// Seeds a stable-capacity slot-id input before graph compilation.
    ///
    /// # Errors
    ///
    /// Rejects zero or inconsistent representative/maximum batch sizes.
    pub fn seed_destination_slots(
        self,
        runtime: &mut CudaRuntime,
        representative_batch: usize,
        maximum_batch: usize,
    ) -> Result<(), RecurrentError> {
        if representative_batch == 0
            || representative_batch > maximum_batch
            || maximum_batch
                .checked_mul(std::mem::size_of::<i32>())
                .is_none()
        {
            return Err(RecurrentError::InvalidGeometry);
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
    ) -> Result<(), RecurrentError> {
        runtime.set_data(
            self.destination_slots,
            destination_slot_ids(self.state_id, batches)?,
        );
        Ok(())
    }
}

fn destination_slot_ids(
    state_id: u16,
    batches: &[FixedStateDeviceBatch],
) -> Result<Vec<i32>, RecurrentError> {
    if batches.is_empty() {
        return Err(RecurrentError::InvalidStateBatch);
    }
    batches
        .iter()
        .map(|batch| {
            let mut matches = batch
                .destinations
                .iter()
                .filter(|destination| destination.state_id == state_id);
            let destination = matches.next().ok_or(RecurrentError::InvalidStateBatch)?;
            if matches.next().is_some() {
                return Err(RecurrentError::InvalidStateBatch);
            }
            i32::try_from(destination.lease.slot_id).map_err(|_| RecurrentError::InvalidStateBatch)
        })
        .collect()
}

fn elements(bytes: u64) -> Result<usize, RecurrentError> {
    if !bytes.is_multiple_of(4) {
        return Err(RecurrentError::InvalidGeometry);
    }
    usize::try_from(bytes / 4).map_err(|_| RecurrentError::InvalidGeometry)
}

#[cfg(test)]
mod tests {
    use luminal::prelude::{CompileOptions, ReferenceRuntime, Runtime};
    use orbitkv::RecurrentFamily;

    use super::*;

    fn class() -> FixedStateClass {
        FixedStateClass {
            state_id: 4,
            name: "recurrent".into(),
            layers: vec![3, 7].into_boxed_slice(),
            storage: FixedStateStorage::Recurrent {
                family: RecurrentFamily::Gdn,
                bytes_per_layer: 16,
                slots_per_request: 2,
                bytes_per_request: 64,
            },
        }
    }

    fn registration() -> FixedStateArenaRegistration {
        FixedStateArenaRegistration {
            state_id: 4,
            engine_epoch: 1,
            pool_epoch: 2,
            pool_id: 3,
            slot_count: 4,
            slot_bytes: 32,
        }
    }

    #[test]
    fn dynamic_slots_and_manifest_layer_order_select_exact_arena_ranges() {
        let mut graph = Graph::new();
        let geometry = GatedDeltaGeometry {
            heads: 1,
            key_width: 2,
            value_width: 2,
            normalization_epsilon: 1e-6,
        };
        let mut arena =
            RecurrentStateGraphArena::new(&mut graph, &class(), registration(), 2.into()).unwrap();
        let selected = arena.layer_state(7, geometry).unwrap().output();
        arena.commit_layer(7, geometry, selected + 100.0).unwrap();
        let binding = arena.finish();
        let mut runtime = graph.compile(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        let initial = (0..32)
            .map(|value| f32::from(u16::try_from(value).unwrap()))
            .collect::<Vec<_>>();
        runtime.set_data(binding.arena_input, initial.clone());
        runtime.set_data(binding.destination_slots, vec![2_i32, 0]);
        runtime.execute(&graph.dyn_map);

        assert_eq!(
            runtime.get_f32(selected),
            &vec![20.0, 21.0, 22.0, 23.0, 4.0, 5.0, 6.0, 7.0]
        );
        let mut expected = initial;
        for index in [4, 5, 6, 7, 20, 21, 22, 23] {
            expected[index] += 100.0;
        }
        assert_eq!(runtime.get_f32(binding.arena_output), &expected);
    }

    #[test]
    fn recurrent_arena_exposes_update_and_in_place_commit_candidates() {
        let mut graph = Graph::new();
        let geometry = GatedDeltaGeometry {
            heads: 1,
            key_width: 2,
            value_width: 2,
            normalization_epsilon: 1e-6,
        };
        let mut arena =
            RecurrentStateGraphArena::new(&mut graph, &class(), registration(), 2.into()).unwrap();
        let previous_state = arena.layer_state(3, geometry).unwrap();
        let recurrent = super::super::gated_delta_step(
            super::super::GatedDeltaStepInputs {
                query: graph.tensor((2, 1, 2)),
                key: graph.tensor((2, 1, 2)),
                value: graph.tensor((2, 1, 2)),
                log_decay: graph.tensor((2, 1)),
                update_gate: graph.tensor((2, 1)),
                previous_state,
                batch_size: 2.into(),
            },
            geometry,
        )
        .unwrap();
        arena
            .commit_layer(3, geometry, recurrent.next_state)
            .unwrap();
        let binding = arena.finish();
        graph.build_search_space::<CudaRuntime>(CompileOptions::default());
        assert!(egraph_has_kernel(&graph, "KernelDeltaStateUpdate"));
        assert!(egraph_has_kernel(&graph, "KernelScatterNoCopy"));
        assert_ne!(binding.arena_input.id, binding.arena_output.id);
    }

    #[test]
    fn graph_arena_rejects_non_manifest_layer_and_wrong_geometry() {
        let mut graph = Graph::new();
        let arena =
            RecurrentStateGraphArena::new(&mut graph, &class(), registration(), 1.into()).unwrap();
        let valid = GatedDeltaGeometry {
            heads: 1,
            key_width: 2,
            value_width: 2,
            normalization_epsilon: 1e-6,
        };
        assert!(matches!(
            arena.layer_state(8, valid),
            Err(RecurrentError::InvalidGeometry)
        ));
        assert!(matches!(
            arena.layer_state(
                3,
                GatedDeltaGeometry {
                    value_width: 3,
                    ..valid
                }
            ),
            Err(RecurrentError::InvalidGeometry)
        ));
    }

    #[test]
    fn destination_slots_follow_request_order_and_reject_bad_classes() {
        let range = |request_id: u64, state_id: u16, slot_id: u32| FixedStateDeviceBatch {
            request_id,
            sources: Box::default(),
            destinations: vec![crate::FixedStateDeviceRange {
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
            Err(RecurrentError::InvalidStateBatch)
        ));
        let duplicate = FixedStateDeviceBatch {
            destinations: vec![ordered[0].destinations[0], ordered[0].destinations[0]]
                .into_boxed_slice(),
            ..ordered[0].clone()
        };
        assert!(matches!(
            destination_slot_ids(4, &[duplicate]),
            Err(RecurrentError::InvalidStateBatch)
        ));
    }

    fn egraph_has_kernel(graph: &Graph, kind: &str) -> bool {
        let egraph = graph.egraph().expect("CUDA search space");
        egraph.eclasses.values().any(|(sort, nodes)| {
            sort == "IR"
                && nodes.iter().any(|node| {
                    let Some(("Op", children)) = egraph
                        .enodes
                        .get(node)
                        .map(|(label, children)| (label.as_str(), children))
                    else {
                        return false;
                    };
                    children.first().is_some_and(|kind_class| {
                        egraph.eclasses[kind_class]
                            .1
                            .iter()
                            .any(|kind_node| egraph.enodes[kind_node].0 == kind)
                    })
                })
        })
    }
}
