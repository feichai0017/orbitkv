//! Luminal graph views over an OrbitKV-owned recurrent-state arena.

use luminal::prelude::{Expression, Graph, GraphTensor};

use crate::{
    FixedStateArenaRegistration, FixedStateClass, FixedStateGraphBinding, FixedStateStorage,
    GatedDeltaGeometry, RecurrentError,
    state_graph::{FixedStateGraphArena, FixedStateGraphLayout},
};

/// Mutable graph builder for one fixed-address recurrent-state arena.
pub struct RecurrentStateGraphArena {
    inner: FixedStateGraphArena,
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
        Ok(Self {
            inner: FixedStateGraphArena::new(
                graph,
                FixedStateGraphLayout {
                    state_id: class.state_id,
                    layers: &class.layers,
                    bytes_per_layer,
                    slots_per_request,
                    bytes_per_request,
                    dtype: luminal::dtype::DType::F32,
                },
                registration,
                batch_size,
            )
            .map_err(|_| RecurrentError::InvalidGeometry)?,
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
        geometry.validate()?;
        self.inner
            .layer_state(
                layer,
                &[
                    geometry.value_heads,
                    geometry.key_width,
                    geometry.value_width,
                ],
            )
            .map_err(|_| RecurrentError::InvalidGeometry)
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
        geometry.validate()?;
        self.inner
            .commit_layer(
                layer,
                &[
                    geometry.value_heads,
                    geometry.key_width,
                    geometry.value_width,
                ],
                &next_state,
            )
            .map_err(|_| RecurrentError::InvalidGeometry)
    }

    /// Marks the final arena version as an output required to alias the input.
    #[must_use]
    pub fn finish(self) -> FixedStateGraphBinding {
        self.inner.finish()
    }
}

#[cfg(test)]
mod tests {
    use luminal::prelude::{CompileOptions, ReferenceRuntime, Runtime};
    use luminal_cuda_lite::runtime::CudaRuntime;
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
            key_heads: 1,
            value_heads: 1,
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
            key_heads: 1,
            value_heads: 1,
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
            key_heads: 1,
            value_heads: 1,
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
