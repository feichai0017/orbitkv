//! Decoder layer traversal and state-specific graph construction.

use orbitkv_compiler::{dtype::DType, prelude::GraphTensor};

use super::{
    DecoderCacheState, DecoderClassDimensions, DecoderConfig, DecoderDimensions, DecoderError,
    DecoderInputs, DecoderWeightFeatures, GatedDeltaStateGraph,
    block::{DecoderLayerEnvelope, TokenAttentionInputs, TokenAttentionLayer},
    topology::{self, DecoderTopology},
};
use crate::{ExecutorPlan, cuda::KvCacheBinding};

pub(super) struct DecoderLayerGraphBuilder<'a> {
    pub(super) graph: &'a mut orbitkv_compiler::prelude::Graph,
    pub(super) config: &'a DecoderConfig,
    pub(super) weights: DecoderWeightFeatures,
    pub(super) plan: &'a ExecutorPlan,
    pub(super) inputs: &'a DecoderInputs,
    pub(super) dimensions: DecoderDimensions,
    pub(super) class_dimensions: &'a [DecoderClassDimensions],
    #[cfg(test)]
    pub(super) diagnostic_boundary: DecoderLayerDiagnosticBoundary,
}

pub(super) struct DecoderLayerBuild {
    pub(super) hidden: GraphTensor,
    pub(super) cache: Vec<DecoderCacheState>,
    #[cfg(test)]
    pub(super) layer_hidden: Option<(usize, GraphTensor)>,
}

struct TokenLayerBuild {
    hidden: GraphTensor,
    state: DecoderCacheState,
    #[cfg(test)]
    observed: Option<GraphTensor>,
}

struct DecoderLayerStep {
    hidden: GraphTensor,
    cache: Option<DecoderCacheState>,
    #[cfg(test)]
    observed: Option<GraphTensor>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum DecoderLayerDiagnosticBoundary {
    #[default]
    Output,
    Input,
    Normalized,
    State,
    Residual,
    FeedForwardNormalized,
    Gate,
    Up,
    Activated,
    Product,
    Down,
    AddOperands,
    AddOperandsAndOutput,
    InputAndOutput,
    AttentionNormalized,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionReadout,
    AttentionProjected,
    AttentionReadoutAndProjected,
}

impl DecoderLayerGraphBuilder<'_> {
    pub(super) fn build(
        &mut self,
        topology: &DecoderTopology,
        fixed_state: &mut Option<GatedDeltaStateGraph>,
        mut hidden: GraphTensor,
        #[cfg(test)] diagnostic_layer: Option<usize>,
    ) -> Result<DecoderLayerBuild, DecoderError> {
        let mut cache = Vec::with_capacity(topology.token_layers().len());
        #[cfg(test)]
        let mut layer_hidden = None;
        #[cfg(test)]
        let layer_count = diagnostic_layer.map_or(self.config.layers, |layer| layer + 1);
        #[cfg(not(test))]
        let layer_count = self.config.layers;
        for layer_index in 0..layer_count {
            let layer = u32::try_from(layer_index)
                .map_err(|_| DecoderError::InvalidGeometry("layer index"))?;
            #[cfg(test)]
            if diagnostic_layer == Some(layer_index)
                && self.diagnostic_boundary == DecoderLayerDiagnosticBoundary::Input
            {
                hidden.output();
                layer_hidden = Some((layer_index, hidden));
                break;
            }
            #[cfg(test)]
            let layer_input = hidden;
            let step = self.build_layer(
                topology,
                fixed_state,
                layer_index,
                layer,
                &hidden,
                #[cfg(test)]
                (diagnostic_layer == Some(layer_index)).then_some(self.diagnostic_boundary),
            )?;
            hidden = step.hidden;
            if let Some(state) = step.cache {
                cache.push(state);
            }
            #[cfg(test)]
            if let Some(observed) = step.observed {
                observed.output();
                layer_hidden = Some((layer_index, observed));
                break;
            }
            #[cfg(test)]
            if diagnostic_layer == Some(layer_index)
                && self.diagnostic_boundary == DecoderLayerDiagnosticBoundary::InputAndOutput
            {
                let values = layer_input.flatten().concat_along(hidden.flatten(), 0);
                values.output();
                layer_hidden = Some((layer_index, values));
                break;
            }
            #[cfg(test)]
            if diagnostic_layer == Some(layer_index)
                && self.diagnostic_boundary == DecoderLayerDiagnosticBoundary::Output
                && layer_hidden.is_none()
            {
                hidden.output();
                layer_hidden = Some((layer_index, hidden));
            }
        }
        Ok(DecoderLayerBuild {
            hidden,
            cache,
            #[cfg(test)]
            layer_hidden,
        })
    }

    fn build_layer(
        &mut self,
        topology: &DecoderTopology,
        fixed_state: &mut Option<GatedDeltaStateGraph>,
        layer_index: usize,
        layer: u32,
        hidden: &GraphTensor,
        #[cfg(test)] diagnostic: Option<DecoderLayerDiagnosticBoundary>,
    ) -> Result<DecoderLayerStep, DecoderError> {
        match topology
            .layer(layer_index)
            .ok_or(DecoderError::UnsupportedPlan)?
        {
            topology::DecoderLayerState::TokenKv { class_id } => {
                let output = self.token_layer(
                    layer,
                    class_id,
                    hidden,
                    #[cfg(test)]
                    diagnostic,
                )?;
                Ok(DecoderLayerStep {
                    hidden: output.hidden,
                    cache: Some(output.state),
                    #[cfg(test)]
                    observed: output.observed,
                })
            }
            topology::DecoderLayerState::GatedDelta { .. } => self.gated_delta_layer(
                fixed_state,
                layer_index,
                layer,
                hidden,
                #[cfg(test)]
                diagnostic,
            ),
        }
    }

    fn gated_delta_layer(
        &mut self,
        fixed_state: &mut Option<GatedDeltaStateGraph>,
        layer_index: usize,
        layer: u32,
        hidden: &GraphTensor,
        #[cfg(test)] diagnostic: Option<DecoderLayerDiagnosticBoundary>,
    ) -> Result<DecoderLayerStep, DecoderError> {
        let envelope = DecoderLayerEnvelope::new(self.graph, self.config, layer_index);
        let normalized = envelope.state_input(hidden);
        #[cfg(test)]
        if diagnostic == Some(DecoderLayerDiagnosticBoundary::Normalized) {
            return Ok(diagnostic_layer_step(&normalized));
        }
        let state_output = fixed_state
            .as_mut()
            .ok_or(DecoderError::UnsupportedPlan)?
            .apply_packed_core(
                self.graph,
                self.config,
                layer,
                &normalized,
                self.inputs.query_indptr,
            )?;
        #[cfg(test)]
        if diagnostic == Some(DecoderLayerDiagnosticBoundary::State) {
            return Ok(diagnostic_layer_step(&state_output));
        }
        #[cfg(test)]
        let (output, observed) = envelope.finish_with_diagnostic(hidden, &state_output, diagnostic);
        #[cfg(not(test))]
        let output = envelope.finish(hidden, &state_output);
        Ok(DecoderLayerStep {
            hidden: output,
            cache: None,
            #[cfg(test)]
            observed,
        })
    }

    fn token_layer(
        &mut self,
        layer: u32,
        class_id: u16,
        hidden: &GraphTensor,
        #[cfg(test)] diagnostic: Option<DecoderLayerDiagnosticBoundary>,
    ) -> Result<TokenLayerBuild, DecoderError> {
        let class = self
            .plan
            .classes
            .get(usize::from(class_id))
            .filter(|class| class.class_id == class_id)
            .ok_or(DecoderError::UnsupportedPlan)?;
        let dimensions = self
            .class_dimensions
            .get(usize::from(class_id))
            .filter(|dimensions| dimensions.class_id == class_id)
            .ok_or(DecoderError::UnsupportedPlan)?;
        let inputs = self
            .inputs
            .classes
            .get(usize::from(class_id))
            .filter(|inputs| inputs.class_id == class_id)
            .ok_or(DecoderError::UnsupportedPlan)?;
        let cache = |graph: &mut orbitkv_compiler::prelude::Graph, component: &str| {
            graph
                .named_tensor(
                    format!("kv.{layer}.{component}"),
                    (dimensions.cache_slots, self.dimensions.kv_width),
                )
                .persist()
                .as_dtype(DType::Bf16)
        };
        let key = cache(self.graph, "key");
        let value = cache(self.graph, "value");
        let block = TokenAttentionLayer::new(self.graph, self.config, self.weights, layer as usize);
        #[cfg(test)]
        let (hidden, key_update, value_update, observed) = block.forward_with_diagnostic(
            &TokenAttentionInputs {
                hidden,
                positions: &self.inputs.positions,
                write_slots: &inputs.write_slots,
                metadata: &inputs.attention,
                k_cache: &key,
                v_cache: &value,
            },
            class,
            self.config,
            self.dimensions,
            *dimensions,
            diagnostic,
        )?;
        #[cfg(not(test))]
        let (hidden, key_update, value_update) = block.forward(
            &TokenAttentionInputs {
                hidden,
                positions: &self.inputs.positions,
                write_slots: &inputs.write_slots,
                metadata: &inputs.attention,
                k_cache: &key,
                v_cache: &value,
            },
            class,
            self.config,
            self.dimensions,
            *dimensions,
        )?;
        Ok(TokenLayerBuild {
            hidden,
            state: DecoderCacheState {
                binding: KvCacheBinding {
                    class_id,
                    layer,
                    key,
                    value,
                },
                key_update: key_update.output(),
                value_update: value_update.output(),
            },
            #[cfg(test)]
            observed,
        })
    }
}

#[cfg(test)]
fn diagnostic_layer_step(output: &GraphTensor) -> DecoderLayerStep {
    DecoderLayerStep {
        hidden: *output,
        cache: None,
        observed: Some(*output),
    }
}
