use serde::Deserialize;

use super::DecoderError;

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
    pub embedding_scale: f32,
    pub activation: DecoderActivation,
    pub block_layout: DecoderBlockLayout,
    pub norm_weights: DecoderNormWeights,
    pub local_rope_theta: Option<f32>,
    pub attention_softmax_scale: f64,
    pub layer_attention: Option<Box<[DecoderAttentionKind]>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderActivation {
    Silu,
    GeluTanh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderBlockLayout {
    PreNorm,
    SandwichNorm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderNormWeights {
    Direct,
    UnitOffset,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderAttentionKind {
    Full,
    Sliding,
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
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    hidden_activation: Option<String>,
    #[serde(default)]
    query_pre_attn_scalar: Option<f64>,
    #[serde(default)]
    rope_local_base_freq: Option<f32>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    attn_logit_softcapping: Option<f32>,
    #[serde(default)]
    final_logit_softcapping: Option<f32>,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    num_local_experts: Option<usize>,
    #[serde(default)]
    quantization_config: Option<serde_json::Value>,
}

const fn default_true() -> bool {
    true
}

impl DecoderConfig {
    /// Parses the model geometry and semantic operations required by the graph.
    ///
    /// # Errors
    ///
    /// Rejects incomplete, ambiguous, or unsupported decoder semantics.
    pub fn from_json(bytes: &[u8]) -> Result<Self, DecoderError> {
        let input = serde_json::from_slice::<DecoderConfigInput>(bytes)?;
        input.validate()?;
        let head_dim = input
            .head_dim
            .unwrap_or(input.hidden_size / input.num_attention_heads);
        let activation = input.activation()?;
        let layer_attention = input
            .layer_types
            .as_ref()
            .map(|layers| parse_layer_attention(layers, input.num_hidden_layers))
            .transpose()?;
        let sandwich_norm = input.rope_local_base_freq.is_some();
        if sandwich_norm && (activation != DecoderActivation::GeluTanh || layer_attention.is_none())
        {
            return Err(DecoderError::InvalidGeometry(
                "incomplete sandwich-norm decoder semantics",
            ));
        }
        let attention_softmax_scale = input
            .query_pre_attn_scalar
            .map_or(0.0, |scalar| scalar.sqrt().recip());
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
            embedding_scale: if sandwich_norm {
                round_to_bfloat16(
                    f32::from(
                        u16::try_from(input.hidden_size)
                            .map_err(|_| DecoderError::InvalidGeometry("embedding scale"))?,
                    )
                    .sqrt(),
                )
            } else {
                1.0
            },
            activation,
            block_layout: if sandwich_norm {
                DecoderBlockLayout::SandwichNorm
            } else {
                DecoderBlockLayout::PreNorm
            },
            norm_weights: if sandwich_norm {
                DecoderNormWeights::UnitOffset
            } else {
                DecoderNormWeights::Direct
            },
            local_rope_theta: input.rope_local_base_freq,
            attention_softmax_scale,
            layer_attention,
        })
    }
}

fn round_to_bfloat16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff_u32 + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(rounding_bias) & 0xffff_0000)
}

impl DecoderConfigInput {
    fn validate(&self) -> Result<(), DecoderError> {
        let head_dim = self.head_dim.unwrap_or_else(|| {
            self.hidden_size
                .checked_div(self.num_attention_heads.max(1))
                .unwrap_or_default()
        });
        if self.layers_or_width_is_zero()
            || head_dim == 0
            || self.num_attention_heads.checked_mul(head_dim).is_none()
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || !matches!(head_dim, 64 | 128 | 256 | 512)
        {
            return Err(DecoderError::InvalidGeometry("dimensions"));
        }
        if !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || self
                .query_pre_attn_scalar
                .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(DecoderError::InvalidGeometry("floating constants"));
        }
        if self.num_local_experts.is_some() || self.quantization_config.is_some() {
            return Err(DecoderError::InvalidGeometry("unsupported decoder state"));
        }
        if self.attn_logit_softcapping.is_some()
            || self.final_logit_softcapping.is_some()
            || self
                .rope_scaling
                .as_ref()
                .is_some_and(|value| !value.is_null())
        {
            return Err(DecoderError::InvalidGeometry(
                "unsupported decoder semantics",
            ));
        }
        Ok(())
    }

    fn activation(&self) -> Result<DecoderActivation, DecoderError> {
        match (
            self.hidden_act.as_deref(),
            self.hidden_activation.as_deref(),
        ) {
            (Some(left), Some(right)) if left != right => {
                Err(DecoderError::InvalidGeometry("ambiguous activation"))
            }
            (Some("silu"), None | Some("silu")) | (None, Some("silu")) => {
                Ok(DecoderActivation::Silu)
            }
            (None, Some("gelu_pytorch_tanh")) => Ok(DecoderActivation::GeluTanh),
            _ => Err(DecoderError::InvalidGeometry("unsupported activation")),
        }
    }

    fn layers_or_width_is_zero(&self) -> bool {
        self.num_hidden_layers == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.vocab_size == 0
    }
}

fn parse_layer_attention(
    layers: &[String],
    expected: usize,
) -> Result<Box<[DecoderAttentionKind]>, DecoderError> {
    if layers.len() != expected {
        return Err(DecoderError::InvalidGeometry("attention layer count"));
    }
    layers
        .iter()
        .map(|layer| match layer.as_str() {
            "full_attention" => Ok(DecoderAttentionKind::Full),
            "sliding_attention" => Ok(DecoderAttentionKind::Sliding),
            _ => Err(DecoderError::InvalidGeometry("unsupported attention layer")),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}
