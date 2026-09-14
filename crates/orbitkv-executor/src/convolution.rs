//! Causal depthwise convolution with minimal persistent history.

use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CausalConvolutionGeometry {
    pub channels: usize,
    pub kernel_width: usize,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ConvolutionError {
    #[error("causal-convolution geometry is invalid")]
    InvalidGeometry,
    #[error("causal-convolution {field} length is {actual}, expected {expected}")]
    InputLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
}

/// Values emitted for the current token and the minimal future-visible history.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalConvolutionReferenceOutput {
    pub values: Box<[f32]>,
    pub history: Box<[f32]>,
}

impl CausalConvolutionGeometry {
    fn validate(self) -> Result<(), ConvolutionError> {
        if self.channels == 0
            || self.kernel_width < 2
            || self.channels.checked_mul(self.kernel_width).is_none()
        {
            return Err(ConvolutionError::InvalidGeometry);
        }
        Ok(())
    }

    fn history_elements(self) -> Result<usize, ConvolutionError> {
        self.channels
            .checked_mul(self.kernel_width - 1)
            .ok_or(ConvolutionError::InvalidGeometry)
    }
}

/// Reference one-token depthwise convolution. Persistent state contains only
/// the `kernel_width - 1` values that can affect a future token.
///
/// # Errors
///
/// Rejects zero or overflowing geometry and input slices whose lengths do not
/// match that geometry.
pub fn causal_convolution_reference(
    geometry: CausalConvolutionGeometry,
    input: &[f32],
    weights: &[f32],
    history: &[f32],
) -> Result<CausalConvolutionReferenceOutput, ConvolutionError> {
    geometry.validate()?;
    require_len("input", input, geometry.channels)?;
    require_len(
        "weights",
        weights,
        geometry.channels * geometry.kernel_width,
    )?;
    require_len("history", history, geometry.history_elements()?)?;
    let history_width = geometry.kernel_width - 1;
    let mut output = vec![0.0_f32; geometry.channels];
    let mut next = vec![0.0_f32; geometry.history_elements()?];
    for channel in 0..geometry.channels {
        let history_base = channel * history_width;
        let weight_base = channel * geometry.kernel_width;
        let mut value = input[channel] * weights[weight_base + history_width];
        for offset in 0..history_width {
            value += history[history_base + offset] * weights[weight_base + offset];
        }
        output[channel] = silu(value);
        if history_width > 1 {
            next[history_base..history_base + history_width - 1]
                .copy_from_slice(&history[history_base + 1..history_base + history_width]);
        }
        next[history_base + history_width - 1] = input[channel];
    }
    Ok(CausalConvolutionReferenceOutput {
        values: output.into_boxed_slice(),
        history: next.into_boxed_slice(),
    })
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn require_len(
    field: &'static str,
    values: &[f32],
    expected: usize,
) -> Result<(), ConvolutionError> {
    if values.len() != expected {
        return Err(ConvolutionError::InputLength {
            field,
            expected,
            actual: values.len(),
        });
    }
    Ok(())
}

#[cfg(feature = "cuda")]
mod graph {
    use luminal::{
        dtype::DType,
        prelude::{Expression, GraphTensor},
    };

    use super::{CausalConvolutionGeometry, ConvolutionError};

    #[derive(Clone, Copy)]
    pub struct CausalConvolutionStepInputs {
        pub input: GraphTensor,
        pub weights: GraphTensor,
        pub previous_history: GraphTensor,
        pub batch_size: Expression,
    }

    #[derive(Clone, Copy)]
    pub struct CausalConvolutionStepOutputs {
        pub values: GraphTensor,
        pub next_history: GraphTensor,
    }

    /// Builds a one-token K-tap depthwise convolution over a `K-1` history.
    ///
    /// # Errors
    ///
    /// Rejects incompatible graph ownership, dtype, shape, or geometry.
    pub fn causal_convolution_step(
        inputs: CausalConvolutionStepInputs,
        geometry: CausalConvolutionGeometry,
    ) -> Result<CausalConvolutionStepOutputs, ConvolutionError> {
        geometry.validate()?;
        let CausalConvolutionStepInputs {
            input,
            weights,
            previous_history,
            batch_size,
        } = inputs;
        let history_width = geometry.kernel_width - 1;
        if !matches!(input.dtype, DType::F32 | DType::Bf16)
            || weights.dtype != input.dtype
            || previous_history.dtype != input.dtype
            || weights.graph_ref != input.graph_ref
            || previous_history.graph_ref != input.graph_ref
            || input.dims() != [batch_size, geometry.channels.into()]
            || weights.dims()
                != [
                    Expression::from(geometry.channels),
                    Expression::from(geometry.kernel_width),
                ]
            || previous_history.dims()
                != [batch_size, geometry.channels.into(), history_width.into()]
        {
            return Err(ConvolutionError::InvalidGeometry);
        }
        let window = previous_history.concat_along(input.expand_dim(2, 1), 2);
        let values = (window * weights.expand_dim(0, batch_size)).sum(2).silu();
        let next_history = window.slice((.., .., 1..));
        Ok(CausalConvolutionStepOutputs {
            values,
            next_history,
        })
    }
}

#[cfg(feature = "cuda")]
mod state_graph {
    use luminal::{
        dtype::DType,
        prelude::{Expression, Graph, GraphTensor},
    };

    use super::{CausalConvolutionGeometry, ConvolutionError};
    use crate::{
        FixedStateArenaRegistration, FixedStateClass, FixedStateGraphBinding, FixedStateStorage,
        state_graph::{FixedStateGraphArena, FixedStateGraphLayout},
    };

    pub struct ConvolutionStateGraphArena {
        inner: FixedStateGraphArena,
    }

    impl ConvolutionStateGraphArena {
        /// Creates a graph view over one manager-owned convolution arena.
        ///
        /// # Errors
        ///
        /// Rejects a non-convolution class or incompatible registration.
        pub fn new(
            graph: &mut Graph,
            class: &FixedStateClass,
            registration: FixedStateArenaRegistration,
            batch_size: Expression,
        ) -> Result<Self, ConvolutionError> {
            let FixedStateStorage::Convolution {
                bytes_per_layer,
                slots_per_request,
                bytes_per_request,
                ..
            } = class.storage
            else {
                return Err(ConvolutionError::InvalidGeometry);
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
                        dtype: DType::Bf16,
                    },
                    registration,
                    batch_size,
                )
                .map_err(|_| ConvolutionError::InvalidGeometry)?,
            })
        }

        /// Gathers one layer's history for the current request slots.
        ///
        /// # Errors
        ///
        /// Rejects a foreign layer or mismatched convolution geometry.
        pub fn layer_state(
            &self,
            layer: u32,
            geometry: CausalConvolutionGeometry,
        ) -> Result<GraphTensor, ConvolutionError> {
            geometry.validate()?;
            self.inner
                .layer_state(layer, &[geometry.channels, geometry.kernel_width - 1])
                .map_err(|_| ConvolutionError::InvalidGeometry)
        }

        /// Commits one layer's updated minimal history.
        ///
        /// # Errors
        ///
        /// Rejects a foreign layer or mismatched tensor shape and dtype.
        pub fn commit_layer(
            &mut self,
            layer: u32,
            geometry: CausalConvolutionGeometry,
            next_history: GraphTensor,
        ) -> Result<(), ConvolutionError> {
            geometry.validate()?;
            self.inner
                .commit_layer(
                    layer,
                    &[geometry.channels, geometry.kernel_width - 1],
                    &next_history,
                )
                .map_err(|_| ConvolutionError::InvalidGeometry)
        }

        #[must_use]
        pub fn finish(self) -> FixedStateGraphBinding {
            self.inner.finish()
        }
    }
}

#[cfg(feature = "cuda")]
pub use graph::{
    CausalConvolutionStepInputs, CausalConvolutionStepOutputs, causal_convolution_step,
};
#[cfg(feature = "cuda")]
pub use state_graph::ConvolutionStateGraphArena;

#[cfg(test)]
#[path = "../tests/unit/convolution/mod.rs"]
mod tests;
