//! Configuration-driven decoder graph using OrbitKV-managed paged attention.

use std::collections::BTreeSet;

use luminal::{
    dtype::DType,
    prelude::{Expression, F32Pow, Graph, GraphTensor},
    shape::ToShape,
};
use luminal_nn::{LayerNorm, scatter_rows};
use serde::Deserialize;
use thiserror::Error;

use crate::{
    ExecutorPlan,
    cuda::{AttentionKernel, PagedAttentionInputs, PagedAttentionMetadata, paged_attention},
};

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderConfig {
    pub layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocabulary_size: usize,
    pub rope_theta: f32,
    pub rms_epsilon: f32,
    pub tied_embeddings: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DecoderWeightLayout {
    pub qkv_bias: bool,
    pub qk_norm: bool,
}

#[derive(Debug, Error)]
pub enum DecoderError {
    #[error("invalid decoder config JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("decoder geometry is invalid: {0}")]
    InvalidGeometry(&'static str),
    #[error("decoder currently requires one token-KV attention class covering every layer")]
    UnsupportedPlan,
    #[error(transparent)]
    Executor(#[from] crate::ExecutorError),
}

#[derive(Deserialize)]
struct DecoderConfigInput {
    num_hidden_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    vocab_size: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
}

impl DecoderConfig {
    /// Parses the common decoder geometry required by the graph builder.
    ///
    /// # Errors
    ///
    /// Rejects missing JSON fields, zero dimensions, inconsistent head
    /// geometry, unsupported head dimensions, or invalid floating constants.
    pub fn from_json(bytes: &[u8]) -> Result<Self, DecoderError> {
        let input = serde_json::from_slice::<DecoderConfigInput>(bytes)?;
        let head_dim = input.head_dim.unwrap_or_else(|| {
            input
                .hidden_size
                .checked_div(input.num_attention_heads.max(1))
                .unwrap_or_default()
        });
        if input.layers_or_width_is_zero()
            || head_dim == 0
            || input.num_attention_heads.checked_mul(head_dim) != Some(input.hidden_size)
            || !input
                .num_attention_heads
                .is_multiple_of(input.num_key_value_heads)
            || !matches!(head_dim, 64 | 128 | 256)
        {
            return Err(DecoderError::InvalidGeometry("dimensions"));
        }
        if !input.rope_theta.is_finite()
            || input.rope_theta <= 0.0
            || !input.rms_norm_eps.is_finite()
            || input.rms_norm_eps <= 0.0
        {
            return Err(DecoderError::InvalidGeometry("floating constants"));
        }
        Ok(Self {
            layers: input.num_hidden_layers,
            hidden_size: input.hidden_size,
            intermediate_size: input.intermediate_size,
            query_heads: input.num_attention_heads,
            kv_heads: input.num_key_value_heads,
            head_dim,
            vocabulary_size: input.vocab_size,
            rope_theta: input.rope_theta,
            rms_epsilon: input.rms_norm_eps,
            tied_embeddings: input.tie_word_embeddings,
        })
    }
}

impl DecoderConfigInput {
    fn layers_or_width_is_zero(&self) -> bool {
        self.num_hidden_layers == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.vocab_size == 0
    }
}

#[derive(Clone, Copy)]
pub struct DecoderInputs {
    pub token_ids: GraphTensor,
    pub positions: GraphTensor,
    pub write_slots: GraphTensor,
    pub attention: PagedAttentionMetadata,
}

pub struct DecoderOutputs {
    pub logits: GraphTensor,
    pub cache_inputs: Vec<(GraphTensor, GraphTensor)>,
    pub cache_updates: Vec<(GraphTensor, GraphTensor)>,
}

pub struct DecoderGraph {
    pub inputs: DecoderInputs,
    pub outputs: DecoderOutputs,
}

#[derive(Clone, Copy)]
struct DecoderDimensions {
    query_tokens: Expression,
    context_pages: Expression,
    cache_slots: usize,
    kv_width: usize,
}

impl DecoderGraph {
    /// Builds a decoder-only transformer whose KV state is addressed solely by
    /// one `ExecutorPlan` attention class.
    ///
    /// # Errors
    ///
    /// Rejects multi-class or incomplete layer coverage in this first native
    /// model path, and propagates paged-attention geometry failures.
    pub fn build(
        graph: &mut Graph,
        config: &DecoderConfig,
        weights: DecoderWeightLayout,
        plan: &ExecutorPlan,
        physical_pages: usize,
    ) -> Result<Self, DecoderError> {
        let dimensions = validate_plan(config, plan, physical_pages)?;
        let inputs = decoder_inputs(graph, plan.classes[0].class_id, dimensions);
        let embedding = weight(
            graph,
            "model.embed_tokens.weight",
            (config.vocabulary_size, config.hidden_size),
            DType::Bf16,
        );
        let mut hidden = token_embedding(&embedding, &inputs.token_ids, config.hidden_size);
        let mut cache_inputs = Vec::with_capacity(config.layers);
        let mut cache_updates = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            let k_cache = graph
                .named_tensor(
                    format!("kv.{layer}.key"),
                    (dimensions.cache_slots, dimensions.kv_width),
                )
                .as_dtype(DType::Bf16);
            let v_cache = graph
                .named_tensor(
                    format!("kv.{layer}.value"),
                    (dimensions.cache_slots, dimensions.kv_width),
                )
                .as_dtype(DType::Bf16);
            let block = DecoderLayer::new(graph, config, weights, layer);
            let (next, key_update, value_update) = block.forward(
                &LayerInputs {
                    hidden: &hidden,
                    positions: &inputs.positions,
                    write_slots: &inputs.write_slots,
                    metadata: &inputs.attention,
                    k_cache: &k_cache,
                    v_cache: &v_cache,
                },
                &plan.classes[0],
                config,
                dimensions,
            )?;
            hidden = next;
            cache_inputs.push((k_cache, v_cache));
            cache_updates.push((key_update.output(), value_update.output()));
        }
        let norm = LayerNorm::new(
            config.hidden_size,
            Some("model.norm.weight"),
            None,
            false,
            config.rms_epsilon,
            graph,
        );
        let normalized = norm.forward(hidden.cast(DType::F32)).cast(DType::Bf16);
        let lm_head = if config.tied_embeddings {
            embedding
        } else {
            weight(
                graph,
                "lm_head.weight",
                (config.vocabulary_size, config.hidden_size),
                DType::Bf16,
            )
        };
        let logits = normalized.matmul(lm_head.t()).cast(DType::F32).output();
        Ok(Self {
            inputs,
            outputs: DecoderOutputs {
                logits,
                cache_inputs,
                cache_updates,
            },
        })
    }
}

fn validate_plan(
    config: &DecoderConfig,
    plan: &ExecutorPlan,
    physical_pages: usize,
) -> Result<DecoderDimensions, DecoderError> {
    let layer_count =
        u32::try_from(config.layers).map_err(|_| DecoderError::InvalidGeometry("layer count"))?;
    if plan.classes.len() != 1
        || physical_pages == 0
        || plan.classes[0]
            .layers
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            != (0..layer_count).collect()
    {
        return Err(DecoderError::UnsupportedPlan);
    }
    Ok(DecoderDimensions {
        query_tokens: Expression::from('s'),
        context_pages: Expression::from('c'),
        cache_slots: physical_pages
            .checked_mul(plan.classes[0].page_tokens as usize)
            .ok_or(DecoderError::InvalidGeometry("cache slots"))?,
        kv_width: config
            .kv_heads
            .checked_mul(config.head_dim)
            .ok_or(DecoderError::InvalidGeometry("KV width"))?,
    })
}

fn decoder_inputs(
    graph: &mut Graph,
    class_id: u16,
    dimensions: DecoderDimensions,
) -> DecoderInputs {
    let token_ids = graph
        .named_tensor("tokens", dimensions.query_tokens)
        .as_dtype(DType::Int);
    let positions = graph
        .named_tensor("positions", dimensions.query_tokens)
        .as_dtype(DType::Int);
    let write_slots = graph
        .named_tensor("kv.write_slots", dimensions.query_tokens)
        .as_dtype(DType::Int);
    let attention = PagedAttentionMetadata::new(
        graph,
        class_id,
        Expression::from('b'),
        dimensions.context_pages,
    );
    DecoderInputs {
        token_ids,
        positions,
        write_slots,
        attention,
    }
}

struct DecoderLayer {
    attention_norm: LayerNorm,
    feed_forward_norm: LayerNorm,
    q_weight: GraphTensor,
    k_weight: GraphTensor,
    v_weight: GraphTensor,
    o_weight: GraphTensor,
    q_bias: Option<GraphTensor>,
    k_bias: Option<GraphTensor>,
    v_bias: Option<GraphTensor>,
    q_norm: Option<GraphTensor>,
    k_norm: Option<GraphTensor>,
    gate: GraphTensor,
    up: GraphTensor,
    down: GraphTensor,
}

struct LayerInputs<'a> {
    hidden: &'a GraphTensor,
    positions: &'a GraphTensor,
    write_slots: &'a GraphTensor,
    metadata: &'a PagedAttentionMetadata,
    k_cache: &'a GraphTensor,
    v_cache: &'a GraphTensor,
}

impl DecoderLayer {
    fn new(
        graph: &mut Graph,
        config: &DecoderConfig,
        layout: DecoderWeightLayout,
        layer: usize,
    ) -> Self {
        let prefix = format!("model.layers.{layer}");
        let q_width = config.query_heads * config.head_dim;
        let kv_width = config.kv_heads * config.head_dim;
        let projection = |graph: &mut Graph, name: &str, width| {
            weight(
                graph,
                format!("{prefix}.self_attn.{name}.weight"),
                (width, config.hidden_size),
                DType::Bf16,
            )
        };
        Self {
            attention_norm: LayerNorm::new(
                config.hidden_size,
                Some(&format!("{prefix}.input_layernorm.weight")),
                None,
                false,
                config.rms_epsilon,
                graph,
            ),
            feed_forward_norm: LayerNorm::new(
                config.hidden_size,
                Some(&format!("{prefix}.post_attention_layernorm.weight")),
                None,
                false,
                config.rms_epsilon,
                graph,
            ),
            q_weight: projection(graph, "q_proj", q_width),
            k_weight: projection(graph, "k_proj", kv_width),
            v_weight: projection(graph, "v_proj", kv_width),
            o_weight: weight(
                graph,
                format!("{prefix}.self_attn.o_proj.weight"),
                (config.hidden_size, q_width),
                DType::Bf16,
            ),
            q_bias: layout.qkv_bias.then(|| {
                weight(
                    graph,
                    format!("{prefix}.self_attn.q_proj.bias"),
                    q_width,
                    DType::Bf16,
                )
            }),
            k_bias: layout.qkv_bias.then(|| {
                weight(
                    graph,
                    format!("{prefix}.self_attn.k_proj.bias"),
                    kv_width,
                    DType::Bf16,
                )
            }),
            v_bias: layout.qkv_bias.then(|| {
                weight(
                    graph,
                    format!("{prefix}.self_attn.v_proj.bias"),
                    kv_width,
                    DType::Bf16,
                )
            }),
            q_norm: layout.qk_norm.then(|| {
                weight(
                    graph,
                    format!("{prefix}.self_attn.q_norm.weight"),
                    config.head_dim,
                    DType::F32,
                )
            }),
            k_norm: layout.qk_norm.then(|| {
                weight(
                    graph,
                    format!("{prefix}.self_attn.k_norm.weight"),
                    config.head_dim,
                    DType::F32,
                )
            }),
            gate: weight(
                graph,
                format!("{prefix}.mlp.gate_proj.weight"),
                (config.intermediate_size, config.hidden_size),
                DType::Bf16,
            ),
            up: weight(
                graph,
                format!("{prefix}.mlp.up_proj.weight"),
                (config.intermediate_size, config.hidden_size),
                DType::Bf16,
            ),
            down: weight(
                graph,
                format!("{prefix}.mlp.down_proj.weight"),
                (config.hidden_size, config.intermediate_size),
                DType::Bf16,
            ),
        }
    }

    fn forward(
        &self,
        inputs: &LayerInputs<'_>,
        class: &crate::AttentionClass,
        config: &DecoderConfig,
        dimensions: DecoderDimensions,
    ) -> Result<(GraphTensor, GraphTensor, GraphTensor), DecoderError> {
        let normalized = self
            .attention_norm
            .forward((*inputs.hidden).cast(DType::F32))
            .cast(DType::Bf16);
        let project = |weight: GraphTensor, bias: Option<GraphTensor>| {
            let output = normalized.matmul(weight.t());
            bias.map_or(output, |bias| bias.expand_lhs(&output.dims()[..1]) + output)
        };
        let mut q = project(self.q_weight, self.q_bias)
            .split_dims(1, config.head_dim)
            .transpose(0, 1);
        let mut k = project(self.k_weight, self.k_bias)
            .split_dims(1, config.head_dim)
            .transpose(0, 1);
        if let Some(norm) = self.q_norm {
            q = qk_norm(&q, &norm);
        }
        if let Some(norm) = self.k_norm {
            k = qk_norm(&k, &norm);
        }
        q = rotary(&q, inputs.positions, config.rope_theta);
        k = rotary(&k, inputs.positions, config.rope_theta);
        let value = project(self.v_weight, self.v_bias);
        let key_rows = k.transpose(0, 1).merge_dims(1, 2);
        let key_update = scatter_rows(
            key_rows,
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
                context_pages: dimensions.context_pages,
            },
            *inputs.metadata,
            class,
            AttentionKernel {
                query_heads: config.query_heads,
                kv_heads: config.kv_heads,
                head_dim: config.head_dim,
                dtype: DType::Bf16,
                softmax_scale: 0.0,
            },
        )?;
        let attention = attention
            .transpose(0, 1)
            .merge_dims(1, 2)
            .matmul(self.o_weight.t());
        let hidden = *inputs.hidden + attention;
        let normalized = self
            .feed_forward_norm
            .forward(hidden.cast(DType::F32))
            .cast(DType::Bf16);
        let gate = normalized.matmul(self.gate.t()).cast(DType::F32);
        let up = normalized.matmul(self.up.t()).cast(DType::F32);
        let feed_forward = (gate.swish() * up).cast(DType::Bf16).matmul(self.down.t());
        Ok((hidden + feed_forward, key_update, value_update))
    }
}

fn qk_norm(input: &GraphTensor, weight: &GraphTensor) -> GraphTensor {
    let dtype = input.dtype;
    let normalized =
        (*input).cast(DType::F32).std_norm(2, 1e-6) * (*weight).expand_lhs(&input.dims()[..2]);
    normalized.cast(dtype)
}

fn rotary(input: &GraphTensor, positions: &GraphTensor, theta: f32) -> GraphTensor {
    let head_dim = input.dims()[2];
    let frequencies = input
        .graph()
        .arange_options(0, head_dim, 2)
        .cast(DType::F32)
        / head_dim;
    let inverse = theta.pow(frequencies).reciprocal();
    let angles = (*positions)
        .cast(DType::F32)
        .expand_dim(1, 1)
        .matmul(inverse.expand_dim(0, 1));
    let first = input.slice((.., .., ..head_dim / 2));
    let second = input.slice((.., .., head_dim / 2..));
    let cosine = angles
        .cos()
        .cast(input.dtype)
        .expand_dim(0, input.dims()[0]);
    let sine = angles
        .sin()
        .cast(input.dtype)
        .expand_dim(0, input.dims()[0]);
    (first * cosine - second * sine).concat_along(first * sine + second * cosine, 2)
}

fn token_embedding(table: &GraphTensor, tokens: &GraphTensor, hidden: usize) -> GraphTensor {
    let count = tokens.dims1();
    table.gather(
        (*tokens * hidden).expand_dim(1, hidden)
            + tokens.graph().arange(hidden).expand_dim(0, count),
    )
}

fn weight(
    graph: &mut Graph,
    name: impl ToString,
    shape: impl ToShape,
    dtype: DType,
) -> GraphTensor {
    graph.named_tensor(name, shape).persist().as_dtype(dtype)
}
