//! Checkpoint-backed gated-delta graph components.

use orbitkv_compiler::{
    dtype::DType,
    prelude::{Expression, Graph, GraphTensor},
};

use super::{
    DecoderConfig, DecoderError, DecoderLinearWeight, GatedDeltaConfig, linear_weight,
    topology::DecoderTopology, weight,
};
use crate::recurrent::gated_delta::softplus;
use crate::{
    CausalConvolutionGeometry, CausalConvolutionStepInputs, ConvolutionStateGraphArena,
    ExecutorPlan, FixedStateArenaRegistration, FixedStateClass, FixedStateGraphBinding,
    GatedDeltaGeometry, GatedDeltaProjectedInputs, RecurrentStateGraphArena,
    causal_convolution_step, gated_delta_projected_step,
};
use orbitkv_cuda::kernel::sequence_state::{
    PackedConvolutionPlan, PackedConvolutionSpec, PackedDeltaScanPlan, PackedDeltaScanSpec,
    packed_causal_convolution, packed_delta_scan,
};

/// Projection outputs whose q/k/v channels still require causal convolution.
#[derive(Clone, Copy)]
pub struct GatedDeltaProjection {
    pub convolution_input: GraphTensor,
    output_gate: GraphTensor,
    update_gate_logits: GraphTensor,
    decay_gate_logits: GraphTensor,
}

/// Output of the recurrent core before the decoder residual/MLP envelope.
#[derive(Clone, Copy)]
pub struct GatedDeltaCoreOutput {
    pub hidden: GraphTensor,
    pub next_recurrent_state: GraphTensor,
}

/// Complete GDN result including both persistent state updates.
#[derive(Clone, Copy)]
pub struct GatedDeltaDecodeOutput {
    pub hidden: GraphTensor,
    pub next_recurrent_state: GraphTensor,
    pub next_convolution_history: GraphTensor,
}

/// Weight-backed GDN core shared by decode and packed-prefill builders.
pub struct GatedDeltaCore {
    geometry: GatedDeltaConfig,
    normalization_epsilon: f32,
    input_qkv: DecoderLinearWeight,
    input_z: DecoderLinearWeight,
    input_b: GraphTensor,
    input_a: GraphTensor,
    convolution_weight: GraphTensor,
    decay_log_rates: GraphTensor,
    decay_bias: GraphTensor,
    output_norm: GraphTensor,
    output: DecoderLinearWeight,
}

/// Joint graph builder for the recurrent and convolution arenas used by a
/// gated-delta decoder stack.
pub struct GatedDeltaStateGraph {
    geometry: GatedDeltaConfig,
    recurrent: RecurrentStateGraphArena,
    convolution: ConvolutionStateGraphArena,
}

#[derive(Clone, Copy)]
pub struct GatedDeltaStateBindings {
    pub recurrent: FixedStateGraphBinding,
    pub convolution: FixedStateGraphBinding,
}

impl GatedDeltaCore {
    /// Declares checkpoint-backed GDN weights for one decoder layer.
    ///
    /// # Errors
    ///
    /// Rejects a missing structural GDN config or unsupported activation dtype.
    pub fn new(
        graph: &mut Graph,
        config: &DecoderConfig,
        layer: usize,
        activation_dtype: DType,
    ) -> Result<Self, DecoderError> {
        if !matches!(activation_dtype, DType::F32 | DType::Bf16) {
            return Err(DecoderError::InvalidGeometry(
                "gated-delta activation dtype",
            ));
        }
        let geometry = config.gated_delta.ok_or(DecoderError::UnsupportedPlan)?;
        let value_elements = geometry
            .value_elements()
            .ok_or(DecoderError::InvalidGeometry("gated-delta value width"))?;
        let convolution_channels =
            geometry
                .convolution_channels()
                .ok_or(DecoderError::InvalidGeometry(
                    "gated-delta convolution width",
                ))?;
        let prefix = format!("{}.layers.{layer}.linear_attn", config.tensor_prefix);
        Ok(Self {
            geometry,
            normalization_epsilon: config.rms_epsilon,
            input_qkv: linear_weight(
                graph,
                config,
                &format!("{prefix}.in_proj_qkv.weight"),
                convolution_channels,
                config.hidden_size,
                activation_dtype,
            ),
            input_z: linear_weight(
                graph,
                config,
                &format!("{prefix}.in_proj_z.weight"),
                value_elements,
                config.hidden_size,
                activation_dtype,
            ),
            input_b: weight(
                graph,
                format!("{prefix}.in_proj_b.weight"),
                (geometry.value_heads, config.hidden_size),
                activation_dtype,
            ),
            input_a: weight(
                graph,
                format!("{prefix}.in_proj_a.weight"),
                (geometry.value_heads, config.hidden_size),
                activation_dtype,
            ),
            convolution_weight: weight(
                graph,
                format!("{prefix}.conv1d.weight"),
                (convolution_channels, 1, geometry.convolution_kernel_width),
                activation_dtype,
            ),
            decay_log_rates: weight(
                graph,
                format!("{prefix}.A_log"),
                geometry.value_heads,
                DType::F32,
            ),
            decay_bias: weight(
                graph,
                format!("{prefix}.dt_bias"),
                geometry.value_heads,
                DType::F32,
            ),
            output_norm: weight(
                graph,
                format!("{prefix}.norm.weight"),
                geometry.value_width,
                DType::F32,
            ),
            output: linear_weight(
                graph,
                config,
                &format!("{prefix}.out_proj.weight"),
                config.hidden_size,
                value_elements,
                activation_dtype,
            ),
        })
    }

    /// Projects normalized hidden states into q/k/v, output-gate, and recurrence gates.
    ///
    /// # Errors
    ///
    /// Rejects a tensor that is not rank two.
    pub fn project(&self, hidden: &GraphTensor) -> Result<GatedDeltaProjection, DecoderError> {
        if hidden.dims().len() != 2 {
            return Err(DecoderError::InvalidGeometry("gated-delta hidden shape"));
        }
        Ok(GatedDeltaProjection {
            convolution_input: self.input_qkv.forward(hidden),
            output_gate: self.input_z.forward(hidden).cast(DType::F32),
            update_gate_logits: (*hidden).matmul(self.input_b.t()).cast(DType::F32),
            decay_gate_logits: (*hidden).matmul(self.input_a.t()).cast(DType::F32),
        })
    }

    /// Applies gated-delta recurrence and checkpoint-defined gated output normalization.
    ///
    /// # Errors
    ///
    /// Rejects incompatible convolution output or recurrent-state geometry.
    pub fn finish_after_convolution(
        &self,
        projection: &GatedDeltaProjection,
        convolved_qkv: &GraphTensor,
        previous_state: &GraphTensor,
        batch_size: Expression,
    ) -> Result<GatedDeltaCoreOutput, DecoderError> {
        if convolved_qkv.graph_ref != projection.convolution_input.graph_ref
            || convolved_qkv.dims() != projection.convolution_input.dims()
        {
            return Err(DecoderError::InvalidGeometry(
                "gated-delta convolution output",
            ));
        }
        let recurrent = gated_delta_projected_step(
            GatedDeltaProjectedInputs {
                convolved_qkv: (*convolved_qkv).cast(DType::F32),
                output_gate: projection.output_gate,
                update_gate_logits: projection.update_gate_logits,
                decay_gate_logits: projection.decay_gate_logits,
                decay_log_rates: self.decay_log_rates,
                decay_bias: self.decay_bias,
                output_norm: self.output_norm,
                previous_state: *previous_state,
                batch_size,
            },
            GatedDeltaGeometry {
                key_heads: self.geometry.key_heads,
                value_heads: self.geometry.value_heads,
                key_width: self.geometry.key_width,
                value_width: self.geometry.value_width,
                normalization_epsilon: 1e-6,
            },
            self.normalization_epsilon,
        )?;
        let values = recurrent
            .values
            .merge_dims(1, 2)
            .cast(self.output.output_dtype());
        Ok(GatedDeltaCoreOutput {
            hidden: self.output.forward(&values),
            next_recurrent_state: recurrent.next_state,
        })
    }

    /// Executes the complete one-token GDN core including minimal convolution history.
    ///
    /// # Errors
    ///
    /// Rejects incompatible projection, convolution, or recurrent-state geometry.
    pub fn forward_decode(
        &self,
        hidden: &GraphTensor,
        previous_recurrent_state: &GraphTensor,
        previous_convolution_history: &GraphTensor,
        batch_size: Expression,
    ) -> Result<GatedDeltaDecodeOutput, DecoderError> {
        let projection = self.project(hidden)?;
        let convolution = causal_convolution_step(
            CausalConvolutionStepInputs {
                input: projection.convolution_input,
                weights: self.convolution_weight.squeeze(1),
                previous_history: *previous_convolution_history,
                batch_size,
            },
            CausalConvolutionGeometry {
                channels: self.geometry.convolution_channels().ok_or(
                    DecoderError::InvalidGeometry("gated-delta convolution width"),
                )?,
                kernel_width: self.geometry.convolution_kernel_width,
            },
        )?;
        let recurrent = self.finish_after_convolution(
            &projection,
            &convolution.values,
            previous_recurrent_state,
            batch_size,
        )?;
        Ok(GatedDeltaDecodeOutput {
            hidden: recurrent.hidden,
            next_recurrent_state: recurrent.next_recurrent_state,
            next_convolution_history: convolution.next_history,
        })
    }

    /// Executes a ragged packed sequence and returns every token value plus
    /// one final recurrent and convolution state per request.
    ///
    /// # Errors
    ///
    /// Rejects incompatible projection or persistent-state geometry.
    pub fn forward_packed(
        &self,
        hidden: &GraphTensor,
        previous_recurrent_state: &GraphTensor,
        previous_convolution_history: &GraphTensor,
        query_indptr: GraphTensor,
    ) -> Result<GatedDeltaDecodeOutput, DecoderError> {
        let projection = self.project(hidden)?;
        if projection.convolution_input.dtype != DType::Bf16 {
            return Err(DecoderError::InvalidGeometry(
                "packed gated-delta activation dtype",
            ));
        }
        let convolution = packed_causal_convolution(
            PackedConvolutionPlan {
                input: projection.convolution_input,
                weights: self.convolution_weight.squeeze(1),
                history: *previous_convolution_history,
                query_indptr,
            },
            PackedConvolutionSpec {
                channels: self.geometry.convolution_channels().ok_or(
                    DecoderError::InvalidGeometry("gated-delta convolution width"),
                )?,
                kernel_width: self.geometry.convolution_kernel_width,
            },
        );
        let key_elements = self
            .geometry
            .key_elements()
            .ok_or(DecoderError::InvalidGeometry("gated-delta key width"))?;
        let convolved = convolution.values.cast(DType::F32);
        let query = convolved
            .slice((.., ..key_elements))
            .split_dims(1, self.geometry.key_width);
        let key = convolved
            .slice((.., key_elements..key_elements * 2))
            .split_dims(1, self.geometry.key_width);
        let value = convolved
            .slice((.., key_elements * 2..))
            .split_dims(1, self.geometry.value_width);
        let tokens = hidden.dims()[0];
        let decay_argument = projection.decay_gate_logits + self.decay_bias.expand_dim(0, tokens);
        let log_decay =
            -self.decay_log_rates.exp().expand_dim(0, tokens) * softplus(&decay_argument);
        let recurrent = packed_delta_scan(
            PackedDeltaScanPlan {
                query,
                key,
                value,
                log_decay,
                update_gate: projection.update_gate_logits.sigmoid(),
                state: *previous_recurrent_state,
                query_indptr,
            },
            PackedDeltaScanSpec {
                key_heads: self.geometry.key_heads,
                value_heads: self.geometry.value_heads,
                key_width: self.geometry.key_width,
                value_width: self.geometry.value_width,
                normalization_epsilon: 1e-6,
            },
        );
        let values = recurrent.values.std_norm(2, self.normalization_epsilon)
            * self
                .output_norm
                .expand_lhs([tokens, Expression::from(self.geometry.value_heads)])
            * projection
                .output_gate
                .split_dims(1, self.geometry.value_width)
                .swish();
        Ok(GatedDeltaDecodeOutput {
            hidden: self
                .output
                .forward(&values.merge_dims(1, 2).cast(DType::Bf16)),
            next_recurrent_state: recurrent.state,
            next_convolution_history: convolution.history,
        })
    }
}

impl GatedDeltaStateGraph {
    /// Builds both manager-owned state arenas from the validated decoder topology.
    ///
    /// # Errors
    ///
    /// Rejects missing, duplicate, or incompatible state classes and registrations.
    pub fn new(
        graph: &mut Graph,
        config: &DecoderConfig,
        plan: &ExecutorPlan,
        registrations: &[FixedStateArenaRegistration],
        batch_size: Expression,
    ) -> Result<Self, DecoderError> {
        let topology = DecoderTopology::compile(config, plan)?;
        if registrations.len() != plan.fixed_states.len() {
            return Err(DecoderError::UnsupportedPlan);
        }
        let (recurrent_state_id, convolution_state_id, geometry) = topology
            .gated_delta_state()
            .ok_or(DecoderError::UnsupportedPlan)?;
        let recurrent_class = unique_state_class(plan, recurrent_state_id)?;
        let convolution_class = unique_state_class(plan, convolution_state_id)?;
        let recurrent_registration = unique_registration(registrations, recurrent_state_id)?;
        let convolution_registration = unique_registration(registrations, convolution_state_id)?;
        Ok(Self {
            geometry,
            recurrent: RecurrentStateGraphArena::new(
                graph,
                recurrent_class,
                recurrent_registration,
                batch_size,
            )
            .map_err(|_| DecoderError::UnsupportedPlan)?,
            convolution: ConvolutionStateGraphArena::new(
                graph,
                convolution_class,
                convolution_registration,
                batch_size,
            )
            .map_err(|_| DecoderError::UnsupportedPlan)?,
        })
    }

    /// Applies one stateful layer core and commits both updated state versions.
    ///
    /// # Errors
    ///
    /// Rejects a non-stateful layer or incompatible hidden/state geometry.
    pub fn apply_packed_core(
        &mut self,
        graph: &mut Graph,
        config: &DecoderConfig,
        layer: u32,
        hidden: &GraphTensor,
        query_indptr: GraphTensor,
    ) -> Result<GraphTensor, DecoderError> {
        let recurrent_geometry = GatedDeltaGeometry {
            key_heads: self.geometry.key_heads,
            value_heads: self.geometry.value_heads,
            key_width: self.geometry.key_width,
            value_width: self.geometry.value_width,
            normalization_epsilon: 1e-6,
        };
        let convolution_geometry = CausalConvolutionGeometry {
            channels: self
                .geometry
                .convolution_channels()
                .ok_or(DecoderError::UnsupportedPlan)?,
            kernel_width: self.geometry.convolution_kernel_width,
        };
        let previous_recurrent = self.recurrent.layer_state(layer, recurrent_geometry)?;
        let previous_convolution = self.convolution.layer_state(layer, convolution_geometry)?;
        let layer_index =
            usize::try_from(layer).map_err(|_| DecoderError::InvalidGeometry("layer index"))?;
        let core = GatedDeltaCore::new(graph, config, layer_index, hidden.dtype)?;
        let output = core.forward_packed(
            hidden,
            &previous_recurrent,
            &previous_convolution,
            query_indptr,
        )?;
        self.recurrent
            .commit_layer(layer, recurrent_geometry, output.next_recurrent_state)?;
        self.convolution.commit_layer(
            layer,
            convolution_geometry,
            output.next_convolution_history,
        )?;
        Ok(output.hidden)
    }

    #[must_use]
    pub fn finish(self) -> GatedDeltaStateBindings {
        GatedDeltaStateBindings {
            recurrent: self.recurrent.finish(),
            convolution: self.convolution.finish(),
        }
    }
}

impl GatedDeltaStateBindings {
    #[must_use]
    pub fn graph_bindings(self) -> [FixedStateGraphBinding; 2] {
        [self.recurrent, self.convolution]
    }

    #[must_use]
    pub fn write_policies() -> [crate::FixedStateWritePolicy; 2] {
        [
            crate::FixedStateWritePolicy::RequiredInPlace,
            crate::FixedStateWritePolicy::RequiredInPlace,
        ]
    }

    #[must_use]
    pub fn resources(self) -> [crate::FixedStateGraphResource; 2] {
        let [recurrent, convolution] = self.graph_bindings();
        let [recurrent_policy, convolution_policy] = Self::write_policies();
        [
            crate::FixedStateGraphResource {
                binding: recurrent,
                policy: recurrent_policy,
            },
            crate::FixedStateGraphResource {
                binding: convolution,
                policy: convolution_policy,
            },
        ]
    }
}

fn unique_state_class(
    plan: &ExecutorPlan,
    state_id: u16,
) -> Result<&FixedStateClass, DecoderError> {
    let mut states = plan
        .fixed_states
        .iter()
        .filter(|state| state.state_id == state_id);
    let state = states.next().ok_or(DecoderError::UnsupportedPlan)?;
    if states.next().is_some() {
        return Err(DecoderError::UnsupportedPlan);
    }
    Ok(state)
}

fn unique_registration(
    registrations: &[FixedStateArenaRegistration],
    state_id: u16,
) -> Result<FixedStateArenaRegistration, DecoderError> {
    let mut registrations = registrations
        .iter()
        .filter(|registration| registration.state_id == state_id);
    let registration = registrations
        .next()
        .copied()
        .ok_or(DecoderError::UnsupportedPlan)?;
    if registrations.next().is_some() {
        return Err(DecoderError::UnsupportedPlan);
    }
    Ok(registration)
}

#[cfg(test)]
#[path = "../../tests/unit/model/recurrent_layer/mod.rs"]
mod tests;
