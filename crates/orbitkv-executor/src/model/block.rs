use luminal::{
    dtype::DType,
    prelude::{Expression, F32Pow, Graph, GraphTensor},
};
use luminal_nn::scatter_rows;

use super::{
    DecoderActivation, DecoderBlockLayout, DecoderClassDimensions, DecoderConfig,
    DecoderDimensions, DecoderError, DecoderLinearWeight, DecoderNormWeights,
    DecoderWeightFeatures, linear_weight, weight,
};
use crate::cuda::{
    AttentionGeometry, PagedAttentionInputs, PagedAttentionMetadata, paged_attention,
};

pub(super) struct TokenAttentionLayer {
    envelope: DecoderLayerEnvelope,
    q_weight: DecoderLinearWeight,
    k_weight: DecoderLinearWeight,
    v_weight: DecoderLinearWeight,
    o_weight: DecoderLinearWeight,
    q_bias: Option<GraphTensor>,
    k_bias: Option<GraphTensor>,
    v_bias: Option<GraphTensor>,
    q_norm: Option<GraphTensor>,
    k_norm: Option<GraphTensor>,
}

/// Residual, normalization, and feed-forward structure shared by every
/// decoder-layer state implementation.
pub(super) struct DecoderLayerEnvelope {
    input_norm: DecoderNorm,
    post_state_norm: Option<DecoderNorm>,
    feed_forward_norm: DecoderNorm,
    post_feed_forward_norm: Option<DecoderNorm>,
    gate: DecoderLinearWeight,
    up: DecoderLinearWeight,
    down: DecoderLinearWeight,
    activation: DecoderActivation,
}

pub(super) struct DecoderNorm {
    weight: GraphTensor,
    epsilon: f32,
    weights: DecoderNormWeights,
}

impl DecoderNorm {
    pub(super) fn new(graph: &mut Graph, config: &DecoderConfig, name: &str) -> Self {
        Self {
            weight: weight(graph, name, config.hidden_size, DType::F32),
            epsilon: config.rms_epsilon,
            weights: config.norm_weights,
        }
    }

    pub(super) fn forward(&self, input: &GraphTensor) -> GraphTensor {
        let normalized = (*input)
            .cast(DType::F32)
            .std_norm(input.shape.last_axis(), self.epsilon);
        let weight = match self.weights {
            DecoderNormWeights::Direct => self.weight,
            DecoderNormWeights::UnitOffset => self.weight + 1.0,
        };
        let output = normalized * weight.expand_lhs(&input.dims()[..input.dims().len() - 1]);
        output.cast(input.dtype)
    }
}

pub(super) struct TokenAttentionInputs<'a> {
    pub(super) hidden: &'a GraphTensor,
    pub(super) positions: &'a GraphTensor,
    pub(super) write_slots: &'a GraphTensor,
    pub(super) metadata: &'a PagedAttentionMetadata,
    pub(super) k_cache: &'a GraphTensor,
    pub(super) v_cache: &'a GraphTensor,
}

impl TokenAttentionLayer {
    pub(super) fn new(
        graph: &mut Graph,
        config: &DecoderConfig,
        weights: DecoderWeightFeatures,
        layer: usize,
    ) -> Self {
        let prefix = format!("{}.layers.{layer}", config.tensor_prefix);
        let q_width = config.query_heads * config.head_dim;
        let q_projection_width = if config.attention_output_gate {
            q_width * 2
        } else {
            q_width
        };
        let kv_width = config.kv_heads * config.head_dim;
        let projection = |graph: &mut Graph, name: &str, width| {
            linear_weight(
                graph,
                config,
                &format!("{prefix}.self_attn.{name}.weight"),
                width,
                config.hidden_size,
                DType::Bf16,
            )
        };
        let q_bias = projection_bias(
            graph,
            weights.qkv_bias,
            &prefix,
            "q_proj",
            q_projection_width,
        );
        let k_bias = projection_bias(graph, weights.qkv_bias, &prefix, "k_proj", kv_width);
        let v_bias = projection_bias(graph, weights.qkv_bias, &prefix, "v_proj", kv_width);
        let q_norm = qk_weight(graph, weights.qk_norm, &prefix, "q_norm", config.head_dim);
        let k_norm = qk_weight(graph, weights.qk_norm, &prefix, "k_norm", config.head_dim);
        Self {
            envelope: DecoderLayerEnvelope::new(graph, config, layer),
            q_weight: projection(graph, "q_proj", q_projection_width),
            k_weight: projection(graph, "k_proj", kv_width),
            v_weight: projection(graph, "v_proj", kv_width),
            o_weight: linear_weight(
                graph,
                config,
                &format!("{prefix}.self_attn.o_proj.weight"),
                config.hidden_size,
                q_width,
                DType::Bf16,
            ),
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
        }
    }

    pub(super) fn forward(
        &self,
        inputs: &TokenAttentionInputs<'_>,
        class: &crate::AttentionClass,
        config: &DecoderConfig,
        dimensions: DecoderDimensions,
        class_dimensions: DecoderClassDimensions,
    ) -> Result<(GraphTensor, GraphTensor, GraphTensor), DecoderError> {
        let normalized = self.envelope.state_input(inputs.hidden);
        let project = |weight: DecoderLinearWeight, bias: Option<GraphTensor>| {
            let output = weight.forward(&normalized);
            bias.map_or(output, |bias| bias.expand_lhs(&output.dims()[..1]) + output)
        };
        let projected_q = project(self.q_weight, self.q_bias);
        let (q, output_gate) = if config.attention_output_gate {
            let q_gate = projected_q.split_dims(1, config.head_dim * 2);
            let q = q_gate.slice((.., .., ..config.head_dim));
            let gate = q_gate.slice((.., .., config.head_dim..)).merge_dims(1, 2);
            (q, Some(gate))
        } else {
            (projected_q.split_dims(1, config.head_dim), None)
        };
        let mut q = q;
        let mut k = project(self.k_weight, self.k_bias).split_dims(1, config.head_dim);
        if let Some(norm) = self.q_norm {
            q = qk_norm(&q, &norm, config);
        }
        if let Some(norm) = self.k_norm {
            k = qk_norm(&k, &norm, config);
        }
        let rope_theta = match class.visibility {
            crate::AttentionVisibility::Sliding { .. } => {
                config.local_rope_theta.unwrap_or(config.rope_theta)
            }
            crate::AttentionVisibility::Full | crate::AttentionVisibility::Chunked { .. } => {
                config.rope_theta
            }
        };
        q = rotary(
            &q,
            inputs.positions,
            rope_theta,
            config.rotary_dimensions,
            config.head_dim,
        );
        k = rotary(
            &k,
            inputs.positions,
            rope_theta,
            config.rotary_dimensions,
            config.head_dim,
        );
        let value = project(self.v_weight, self.v_bias);
        let key_update = scatter_rows(
            k.merge_dims(1, 2),
            *inputs.write_slots,
            *inputs.k_cache,
            config.kv_heads * config.head_dim,
        );
        let value_update = scatter_rows(
            value,
            *inputs.write_slots,
            *inputs.v_cache,
            config.kv_heads * config.head_dim,
        );
        let attention = paged_attention(
            PagedAttentionInputs {
                q,
                k_cache: key_update,
                v_cache: value_update,
                query_tokens: dimensions.query_tokens,
                context_pages: Expression::from(class_dimensions.context_pages),
            },
            *inputs.metadata,
            class,
            AttentionGeometry {
                query_heads: config.query_heads,
                kv_heads: config.kv_heads,
                head_dim: config.head_dim,
                dtype: DType::Bf16,
                softmax_scale: config.attention_softmax_scale,
            },
        )?;
        let attention = attention.transpose(0, 1).merge_dims(1, 2);
        let attention = output_gate.map_or(attention, |gate| attention * gate.sigmoid());
        let state_output = self.o_weight.forward(&attention);
        let hidden = self.envelope.finish(inputs.hidden, state_output);
        Ok((hidden, key_update, value_update))
    }
}

impl DecoderLayerEnvelope {
    pub(super) fn new(graph: &mut Graph, config: &DecoderConfig, layer: usize) -> Self {
        let prefix = format!("{}.layers.{layer}", config.tensor_prefix);
        let sandwich = config.block_layout == DecoderBlockLayout::SandwichNorm;
        let input_norm =
            DecoderNorm::new(graph, config, &format!("{prefix}.input_layernorm.weight"));
        let post_state_norm = sandwich.then(|| {
            DecoderNorm::new(
                graph,
                config,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )
        });
        let feed_forward_name = if sandwich {
            "pre_feedforward_layernorm"
        } else {
            "post_attention_layernorm"
        };
        Self {
            input_norm,
            post_state_norm,
            feed_forward_norm: DecoderNorm::new(
                graph,
                config,
                &format!("{prefix}.{feed_forward_name}.weight"),
            ),
            post_feed_forward_norm: sandwich.then(|| {
                DecoderNorm::new(
                    graph,
                    config,
                    &format!("{prefix}.post_feedforward_layernorm.weight"),
                )
            }),
            gate: linear_weight(
                graph,
                config,
                &format!("{prefix}.mlp.gate_proj.weight"),
                config.intermediate_size,
                config.hidden_size,
                DType::Bf16,
            ),
            up: linear_weight(
                graph,
                config,
                &format!("{prefix}.mlp.up_proj.weight"),
                config.intermediate_size,
                config.hidden_size,
                DType::Bf16,
            ),
            down: linear_weight(
                graph,
                config,
                &format!("{prefix}.mlp.down_proj.weight"),
                config.hidden_size,
                config.intermediate_size,
                DType::Bf16,
            ),
            activation: config.activation,
        }
    }

    pub(super) fn state_input(&self, hidden: &GraphTensor) -> GraphTensor {
        self.input_norm.forward(hidden)
    }

    pub(super) fn finish(
        &self,
        residual: &GraphTensor,
        mut state_output: GraphTensor,
    ) -> GraphTensor {
        if let Some(norm) = &self.post_state_norm {
            state_output = norm.forward(&state_output);
        }
        let hidden = *residual + state_output;
        let normalized = self.feed_forward_norm.forward(&hidden);
        let gate = self.gate.forward(&normalized).cast(DType::F32);
        let up = self.up.forward(&normalized).cast(DType::F32);
        let activated = match self.activation {
            DecoderActivation::Silu => gate.swish(),
            DecoderActivation::GeluTanh => gelu_tanh(&gate),
        };
        let mut feed_forward = self.down.forward(&(activated * up).cast(DType::Bf16));
        if let Some(norm) = &self.post_feed_forward_norm {
            feed_forward = norm.forward(&feed_forward);
        }
        hidden + feed_forward
    }
}

fn projection_bias(
    graph: &mut Graph,
    enabled: bool,
    prefix: &str,
    projection: &str,
    width: usize,
) -> Option<GraphTensor> {
    enabled.then(|| {
        weight(
            graph,
            format!("{prefix}.self_attn.{projection}.bias"),
            width,
            DType::Bf16,
        )
    })
}

fn qk_weight(
    graph: &mut Graph,
    enabled: bool,
    prefix: &str,
    projection: &str,
    head_dim: usize,
) -> Option<GraphTensor> {
    enabled.then(|| {
        weight(
            graph,
            format!("{prefix}.self_attn.{projection}.weight"),
            head_dim,
            DType::F32,
        )
    })
}

fn qk_norm(input: &GraphTensor, weight: &GraphTensor, config: &DecoderConfig) -> GraphTensor {
    let dtype = input.dtype;
    let weight = match config.norm_weights {
        DecoderNormWeights::Direct => *weight,
        DecoderNormWeights::UnitOffset => *weight + 1.0,
    };
    let normalized = (*input).cast(DType::F32).std_norm(2, config.rms_epsilon)
        * weight.expand_lhs(&input.dims()[..2]);
    normalized.cast(dtype)
}

#[allow(clippy::excessive_precision)]
fn gelu_tanh(input: &GraphTensor) -> GraphTensor {
    let scaled = 1.595_769_1 * *input * (1.0 + 0.044_715 * *input * *input);
    *input * scaled.sigmoid()
}

fn rotary(
    input: &GraphTensor,
    positions: &GraphTensor,
    theta: f32,
    rotary_dimensions: usize,
    head_dim: usize,
) -> GraphTensor {
    let frequencies = input
        .graph()
        .arange_options(0, rotary_dimensions, 2)
        .cast(DType::F32)
        / rotary_dimensions;
    let inverse = theta.pow(frequencies).reciprocal();
    let angles = (*positions)
        .cast(DType::F32)
        .expand_dim(1, 1)
        .matmul(inverse.expand_dim(0, 1));
    let rotary = input.slice((.., .., ..rotary_dimensions));
    let first = rotary.slice((.., .., ..rotary_dimensions / 2));
    let second = rotary.slice((.., .., rotary_dimensions / 2..));
    let cosine = angles
        .cos()
        .cast(input.dtype)
        .expand_dim(1, input.dims()[1]);
    let sine = angles
        .sin()
        .cast(input.dtype)
        .expand_dim(1, input.dims()[1]);
    let rotated = (first * cosine - second * sine).concat_along(first * sine + second * cosine, 2);
    if rotary_dimensions == head_dim {
        rotated
    } else {
        rotated.concat_along(input.slice((.., .., rotary_dimensions..)), 2)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/model/block/mod.rs"]
mod tests;
