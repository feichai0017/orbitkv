//! `OrbitKV` graph views over an `OrbitKV`-owned recurrent-state arena.

use orbitkv_compiler::prelude::{Expression, Graph, GraphTensor};

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
                    dtype: orbitkv_compiler::dtype::DType::F32,
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
#[path = "../../tests/unit/recurrent/state_graph/mod.rs"]
mod tests;
