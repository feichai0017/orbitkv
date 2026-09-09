//! Checkpoint-backed gated-delta graph components.

use luminal::{
    dtype::DType,
    prelude::{Expression, Graph, GraphTensor},
};

use super::{DecoderConfig, DecoderError, GatedDeltaConfig, topology::DecoderTopology, weight};
use crate::{
    CausalConvolutionGeometry, CausalConvolutionStepInputs, ConvolutionStateGraphArena,
    ExecutorPlan, FixedStateArenaRegistration, FixedStateClass, FixedStateGraphBinding,
    GatedDeltaGeometry, GatedDeltaProjectedInputs, RecurrentStateGraphArena,
    causal_convolution_step, gated_delta_projected_step,
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

/// Complete single-token GDN result including both persistent state updates.
#[derive(Clone, Copy)]
pub struct GatedDeltaDecodeOutput {
    pub hidden: GraphTensor,
    pub next_recurrent_state: GraphTensor,
    pub next_convolution_history: GraphTensor,
}

/// Weight-backed GDN core shared by decode and future chunked-prefill builders.
pub struct GatedDeltaCore {
    geometry: GatedDeltaConfig,
    normalization_epsilon: f32,
    input_qkv: GraphTensor,
    input_z: GraphTensor,
    input_b: GraphTensor,
    input_a: GraphTensor,
    convolution_weight: GraphTensor,
    decay_log_rates: GraphTensor,
    decay_bias: GraphTensor,
    output_norm: GraphTensor,
    output: GraphTensor,
}

/// Joint graph builder for the recurrent and convolution arenas used by a
/// gated-delta decoder stack.
pub struct GatedDeltaStateGraph {
    geometry: GatedDeltaConfig,
    batch_size: Expression,
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
            input_qkv: weight(
                graph,
                format!("{prefix}.in_proj_qkv.weight"),
                (convolution_channels, config.hidden_size),
                activation_dtype,
            ),
            input_z: weight(
                graph,
                format!("{prefix}.in_proj_z.weight"),
                (value_elements, config.hidden_size),
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
            output: weight(
                graph,
                format!("{prefix}.out_proj.weight"),
                (config.hidden_size, value_elements),
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
            convolution_input: (*hidden).matmul(self.input_qkv.t()),
            output_gate: (*hidden).matmul(self.input_z.t()).cast(DType::F32),
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
        let values = recurrent.values.merge_dims(1, 2).cast(self.output.dtype);
        Ok(GatedDeltaCoreOutput {
            hidden: values.matmul(self.output.t()),
            next_recurrent_state: recurrent.next_state,
        })
    }

    /// Executes the complete one-token GDN core including its minimal
    /// persistent causal-convolution history.
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
            batch_size,
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
    pub fn apply_decode_core(
        &mut self,
        graph: &mut Graph,
        config: &DecoderConfig,
        layer: u32,
        hidden: &GraphTensor,
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
        let output = core.forward_decode(
            hidden,
            &previous_recurrent,
            &previous_convolution,
            self.batch_size,
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
            crate::FixedStateWritePolicy::CopyBackAllowed,
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
mod tests {
    use luminal::prelude::{CompileOptions, ReferenceRuntime, Runtime};

    use super::*;
    use crate::model::{
        DecoderActivation, DecoderBlockLayout, DecoderLayerKind, DecoderNormWeights,
        DecoderWeightFormat,
    };
    use crate::{
        AttentionClass, AttentionVisibility, CausalConvolutionGeometry, FixedStateClass,
        FixedStateStorage, GatedDeltaReferenceInput, causal_convolution_reference,
        gated_delta_reference,
    };

    #[test]
    fn decode_core_updates_convolution_and_recurrent_state() {
        let mut graph = Graph::new();
        let config = config();
        let core = GatedDeltaCore::new(&mut graph, &config, 0, DType::F32).unwrap();
        let hidden = graph.named_tensor("hidden", (1, 4));
        let recurrent_state = graph.named_tensor("recurrent_state", (1, 2, 2, 1));
        let convolution_history = graph.named_tensor("convolution_history", (1, 6, 2));
        let output = core
            .forward_decode(&hidden, &recurrent_state, &convolution_history, 1.into())
            .unwrap();
        let hidden_out = output.hidden.output();
        let recurrent_out = output.next_recurrent_state.output();
        let convolution_out = output.next_convolution_history.output();
        let mut runtime = graph.compile(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        runtime.set_data(hidden, vec![1.0_f32, 2.0, 3.0, 4.0]);
        runtime.set_data(core.input_qkv, projection_weights());
        runtime.set_data(
            core.input_z,
            vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        );
        runtime.set_data(core.input_b, vec![0.0_f32; 8]);
        runtime.set_data(core.input_a, vec![0.0_f32; 8]);
        runtime.set_data(core.convolution_weight, vec![1.0_f32; 18]);
        runtime.set_data(core.decay_log_rates, vec![0.0_f32; 2]);
        runtime.set_data(core.decay_bias, vec![0.0_f32; 2]);
        runtime.set_data(core.output_norm, vec![1.0_f32]);
        runtime.set_data(
            core.output,
            vec![1.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        );
        runtime.set_data(recurrent_state, vec![0.0_f32; 4]);
        runtime.set_data(convolution_history, vec![0.0_f32; 12]);
        runtime.execute(&graph.dyn_map);

        let convolution = causal_convolution_reference(
            CausalConvolutionGeometry {
                channels: 6,
                kernel_width: 3,
            },
            &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
            &[1.0; 18],
            &[0.0; 12],
        )
        .unwrap();
        let expected_recurrent = gated_delta_reference(
            GatedDeltaGeometry {
                key_heads: 1,
                value_heads: 2,
                key_width: 2,
                value_width: 1,
                normalization_epsilon: 1e-6,
            },
            GatedDeltaReferenceInput {
                batch_size: 1,
                sequence_tokens: 1,
                query: &convolution.values[..2],
                key: &convolution.values[2..4],
                value: &convolution.values[4..],
                log_decay: &[-std::f32::consts::LN_2; 2],
                update_gate: &[0.5; 2],
                initial_state: &[0.0; 4],
            },
        )
        .unwrap();
        let output_gate = 1.0_f32 / (1.0 + (-1.0_f32).exp());
        let normalized = expected_recurrent
            .values
            .iter()
            .map(|value| value / (value * value + 1e-6).sqrt() * output_gate)
            .collect::<Vec<_>>();
        let combined = normalized[0] + normalized[1];
        let expected_hidden = [combined, 0.0, 0.0, combined];
        assert_close(runtime.get_f32(hidden_out), &expected_hidden);
        assert_close(runtime.get_f32(recurrent_out), &expected_recurrent.state);
        assert_eq!(
            runtime.get_f32(convolution_out).as_slice(),
            convolution.history.as_ref()
        );
    }

    #[test]
    fn state_graph_composes_both_manager_owned_state_classes() {
        use luminal_cuda_lite::runtime::CudaRuntime;
        use orbitkv::RecurrentFamily;

        let config = config();
        let plan = crate::test_support::executor_plan_with_fixed_states(
            "stateful",
            16,
            vec![AttentionClass {
                class_id: 0,
                name: "full".into(),
                layers: vec![1].into_boxed_slice(),
                page_tokens: 16,
                key_bytes_per_token_per_layer: 128,
                value_bytes_per_token_per_layer: 128,
                visibility: AttentionVisibility::Full,
            }],
            vec![
                FixedStateClass {
                    state_id: 1,
                    name: "recurrent".into(),
                    layers: vec![0].into_boxed_slice(),
                    storage: FixedStateStorage::Recurrent {
                        family: RecurrentFamily::Gdn,
                        bytes_per_layer: 16,
                        slots_per_request: 2,
                        bytes_per_request: 32,
                    },
                },
                FixedStateClass {
                    state_id: 2,
                    name: "convolution".into(),
                    layers: vec![0].into_boxed_slice(),
                    storage: FixedStateStorage::Convolution {
                        bytes_per_layer: 24,
                        kernel_width: 3,
                        slots_per_request: 2,
                        bytes_per_request: 48,
                    },
                },
            ],
        );
        let registrations = [
            FixedStateArenaRegistration {
                state_id: 1,
                engine_epoch: 1,
                pool_epoch: 2,
                pool_id: 3,
                slot_count: 4,
                slot_bytes: 16,
            },
            FixedStateArenaRegistration {
                state_id: 2,
                engine_epoch: 1,
                pool_epoch: 3,
                pool_id: 4,
                slot_count: 4,
                slot_bytes: 24,
            },
        ];
        let mut graph = Graph::new();
        let mut states =
            GatedDeltaStateGraph::new(&mut graph, &config, &plan, &registrations, 'b'.into())
                .unwrap();
        let hidden = graph.named_tensor("hidden", ('b', 4)).as_dtype(DType::Bf16);
        let output = states
            .apply_decode_core(&mut graph, &config, 0, &hidden)
            .unwrap()
            .output();
        let bindings = states.finish();
        assert_eq!(bindings.recurrent.state_id, 1);
        assert_eq!(bindings.convolution.state_id, 2);
        assert_eq!(
            GatedDeltaStateBindings::write_policies(),
            [
                crate::FixedStateWritePolicy::RequiredInPlace,
                crate::FixedStateWritePolicy::CopyBackAllowed,
            ]
        );
        assert_eq!(
            bindings.graph_bindings().map(|binding| binding.state_id),
            [1, 2]
        );
        assert_eq!(output.dtype, DType::Bf16);
        graph.set_dim('b', 1);
        graph.build_search_space::<CudaRuntime>(CompileOptions::default());
        assert!(egraph_has_kernel(&graph, "KernelDeltaStateUpdate"));
    }

    fn projection_weights() -> Vec<f32> {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
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

    fn config() -> DecoderConfig {
        DecoderConfig {
            layers: 2,
            hidden_size: 4,
            intermediate_size: 8,
            query_heads: 1,
            kv_heads: 1,
            head_dim: 64,
            vocabulary_size: 16,
            tensor_prefix: "model".into(),
            rope_theta: 10_000.0,
            rotary_dimensions: 64,
            rms_epsilon: 1e-6,
            tied_embeddings: true,
            embedding_scale: 1.0,
            activation: DecoderActivation::Silu,
            block_layout: DecoderBlockLayout::PreNorm,
            norm_weights: DecoderNormWeights::Direct,
            local_rope_theta: None,
            attention_softmax_scale: 0.0,
            attention_output_gate: false,
            layer_kinds: Some(
                vec![DecoderLayerKind::Linear, DecoderLayerKind::Full].into_boxed_slice(),
            ),
            gated_delta: Some(GatedDeltaConfig {
                key_heads: 1,
                value_heads: 2,
                key_width: 2,
                value_width: 1,
                convolution_kernel_width: 3,
            }),
            weight_format: DecoderWeightFormat::Float,
        }
    }
}
