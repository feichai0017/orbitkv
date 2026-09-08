use serde::{Deserialize, Serialize};

use super::DecoderError;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DecoderConfig {
    pub layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocabulary_size: usize,
    /// Canonical weight namespace containing embeddings, layers, and final norm.
    pub tensor_prefix: String,
    pub rope_theta: f32,
    pub rotary_dimensions: usize,
    pub rms_epsilon: f32,
    pub tied_embeddings: bool,
    pub embedding_scale: f32,
    pub activation: DecoderActivation,
    pub block_layout: DecoderBlockLayout,
    pub norm_weights: DecoderNormWeights,
    pub local_rope_theta: Option<f32>,
    pub attention_softmax_scale: f64,
    pub attention_output_gate: bool,
    pub layer_kinds: Option<Box<[DecoderLayerKind]>>,
    pub gated_delta: Option<GatedDeltaConfig>,
    pub weight_format: DecoderWeightFormat,
}

/// Static state and projection geometry for one gated-delta decoder layer.
///
/// This is derived from structural checkpoint fields. It deliberately carries
/// no model identity and is shared by graph construction, state-layout
/// validation, and weight inspection.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct GatedDeltaConfig {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_width: usize,
    pub value_width: usize,
    pub convolution_kernel_width: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum DecoderActivation {
    Silu,
    GeluTanh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum DecoderBlockLayout {
    PreNorm,
    SandwichNorm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum DecoderNormWeights {
    Direct,
    UnitOffset,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum DecoderLayerKind {
    Full,
    Sliding,
    Linear,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum DecoderWeightFormat {
    Float,
    Fp8E4M3Block { rows: usize, columns: usize },
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
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
    rms_norm_eps: f32,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    hidden_activation: Option<String>,
    #[serde(default)]
    query_pre_attn_scalar: Option<f64>,
    #[serde(default)]
    attn_output_gate: Option<bool>,
    #[serde(default)]
    output_gate_type: Option<String>,
    #[serde(default)]
    rope_local_base_freq: Option<f32>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    linear_num_key_heads: Option<usize>,
    #[serde(default)]
    linear_num_value_heads: Option<usize>,
    #[serde(default)]
    linear_key_head_dim: Option<usize>,
    #[serde(default)]
    linear_value_head_dim: Option<usize>,
    #[serde(default)]
    linear_conv_kernel_dim: Option<usize>,
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

#[derive(Clone, Debug, Deserialize)]
struct RopeParameters {
    #[serde(default)]
    rope_type: Option<String>,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
    #[serde(default)]
    mrope_interleaved: Option<bool>,
    #[serde(default)]
    mrope_section: Option<[usize; 3]>,
}

#[derive(Clone, Debug, Deserialize)]
struct QuantizationConfig {
    quant_method: String,
    #[serde(default)]
    fmt: Option<String>,
    #[serde(default)]
    activation_scheme: Option<String>,
    #[serde(default)]
    weight_block_size: Option<[usize; 2]>,
}

impl DecoderConfig {
    pub(super) fn layer_kind(&self, layer: usize) -> DecoderLayerKind {
        self.layer_kinds
            .as_deref()
            .and_then(|layers| layers.get(layer))
            .copied()
            .unwrap_or(DecoderLayerKind::Full)
    }

    /// Parses the model geometry and semantic operations required by the graph.
    ///
    /// # Errors
    ///
    /// Rejects incomplete, ambiguous, or unsupported decoder semantics.
    pub fn from_json(bytes: &[u8]) -> Result<Self, DecoderError> {
        let document = serde_json::from_slice::<serde_json::Value>(bytes)?;
        let nested = document
            .get("text_config")
            .filter(|value| value.is_object());
        let mut input = serde_json::from_value::<DecoderConfigInput>(
            nested.cloned().unwrap_or_else(|| document.clone()),
        )?;
        if input.tie_word_embeddings.is_none() {
            input.tie_word_embeddings = document
                .get("tie_word_embeddings")
                .and_then(serde_json::Value::as_bool);
        }
        if input.quantization_config.is_none() {
            input.quantization_config = document.get("quantization_config").cloned();
        }
        input.validate()?;
        let head_dim = input
            .head_dim
            .unwrap_or(input.hidden_size / input.num_attention_heads);
        let activation = input.activation()?;
        let layer_kinds = input
            .layer_types
            .as_ref()
            .map(|layers| parse_layer_kinds(layers, input.num_hidden_layers))
            .transpose()?;
        let gated_delta = input.gated_delta_config(layer_kinds.as_deref())?;
        let sandwich_norm = input.rope_local_base_freq.is_some();
        let unit_offset_norm =
            sandwich_norm || (input.attn_output_gate == Some(true) && gated_delta.is_some());
        if sandwich_norm && (activation != DecoderActivation::GeluTanh || layer_kinds.is_none()) {
            return Err(DecoderError::InvalidGeometry(
                "incomplete sandwich-norm decoder semantics",
            ));
        }
        #[allow(clippy::cast_precision_loss)]
        let attention_softmax_scale = input.query_pre_attn_scalar.map_or_else(
            || (head_dim as f64).sqrt().recip(),
            |scalar| scalar.sqrt().recip(),
        );
        let rope_theta = input.rope_theta()?;
        let rotary_dimensions = input.rotary_dimensions(head_dim)?;
        let weight_format = input.weight_format()?;
        Ok(Self {
            layers: input.num_hidden_layers,
            hidden_size: input.hidden_size,
            intermediate_size: input.intermediate_size,
            query_heads: input.num_attention_heads,
            kv_heads: input.num_key_value_heads,
            head_dim,
            vocabulary_size: input.vocab_size,
            tensor_prefix: if nested.is_some() {
                "model.language_model".into()
            } else {
                "model".into()
            },
            rope_theta,
            rotary_dimensions,
            rms_epsilon: input.rms_norm_eps,
            tied_embeddings: input.tie_word_embeddings.unwrap_or(true),
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
            norm_weights: if unit_offset_norm {
                DecoderNormWeights::UnitOffset
            } else {
                DecoderNormWeights::Direct
            },
            local_rope_theta: input.rope_local_base_freq,
            attention_softmax_scale,
            attention_output_gate: input.attn_output_gate.unwrap_or(false),
            layer_kinds,
            gated_delta,
            weight_format,
        })
    }

    pub(super) fn require_executable(&self) -> Result<(), DecoderError> {
        if self
            .layer_kinds
            .as_deref()
            .is_some_and(|layers| layers.contains(&DecoderLayerKind::Linear))
        {
            return Err(DecoderError::UnsupportedExecution(
                "linear-attention state execution",
            ));
        }
        if self.weight_format != DecoderWeightFormat::Float {
            return Err(DecoderError::UnsupportedExecution(
                "quantized weight execution",
            ));
        }
        Ok(())
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
            || (self.attn_output_gate == Some(true)
                && self
                    .num_attention_heads
                    .checked_mul(head_dim)
                    .and_then(|width| width.checked_mul(2))
                    .is_none())
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || !matches!(head_dim, 64 | 128 | 256 | 512)
        {
            return Err(DecoderError::InvalidGeometry("dimensions"));
        }
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || self
                .query_pre_attn_scalar
                .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(DecoderError::InvalidGeometry("floating constants"));
        }
        if self.num_local_experts.is_some() {
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

    fn rope_theta(&self) -> Result<f32, DecoderError> {
        let nested = self
            .rope_parameters
            .as_ref()
            .and_then(|parameters| parameters.rope_theta);
        let theta = match (self.rope_theta, nested) {
            (Some(left), Some(right)) if left.to_bits() != right.to_bits() => {
                return Err(DecoderError::InvalidGeometry("ambiguous RoPE theta"));
            }
            (Some(value), _) | (None, Some(value)) => value,
            (None, None) => return Err(DecoderError::InvalidGeometry("missing RoPE theta")),
        };
        if !theta.is_finite() || theta <= 0.0 {
            return Err(DecoderError::InvalidGeometry("RoPE theta"));
        }
        Ok(theta)
    }

    fn rotary_dimensions(&self, head_dim: usize) -> Result<usize, DecoderError> {
        let parameters = self.rope_parameters.as_ref();
        if parameters
            .and_then(|parameters| parameters.rope_type.as_deref())
            .is_some_and(|rope_type| rope_type != "default")
        {
            return Err(DecoderError::InvalidGeometry("unsupported RoPE type"));
        }
        let factor = parameters
            .and_then(|parameters| parameters.partial_rotary_factor)
            .unwrap_or(1.0);
        if !(factor.is_finite() && 0.0 < factor && factor <= 1.0) {
            return Err(DecoderError::InvalidGeometry("partial rotary factor"));
        }
        #[allow(clippy::cast_precision_loss)]
        let dimensions = head_dim as f64 * factor;
        if dimensions.fract() != 0.0 {
            return Err(DecoderError::InvalidGeometry("partial rotary dimensions"));
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let dimensions = dimensions as usize;
        if dimensions == 0 || !dimensions.is_multiple_of(2) {
            return Err(DecoderError::InvalidGeometry("partial rotary dimensions"));
        }
        if let Some(section) = parameters.and_then(|parameters| parameters.mrope_section) {
            let frequency_dimensions = section.into_iter().try_fold(0_usize, usize::checked_add);
            if frequency_dimensions != Some(dimensions / 2) {
                return Err(DecoderError::InvalidGeometry("MRoPE sections"));
            }
            if parameters.and_then(|parameters| parameters.mrope_interleaved) != Some(true) {
                return Err(DecoderError::InvalidGeometry("MRoPE layout"));
            }
        }
        Ok(dimensions)
    }

    fn weight_format(&self) -> Result<DecoderWeightFormat, DecoderError> {
        let Some(value) = self.quantization_config.as_ref() else {
            return Ok(DecoderWeightFormat::Float);
        };
        let config = serde_json::from_value::<QuantizationConfig>(value.clone())?;
        match (
            config.quant_method.as_str(),
            config.fmt.as_deref(),
            config.activation_scheme.as_deref(),
            config.weight_block_size,
        ) {
            ("fp8", Some("e4m3"), Some("dynamic"), Some([rows, columns]))
                if rows > 0 && columns > 0 =>
            {
                Ok(DecoderWeightFormat::Fp8E4M3Block { rows, columns })
            }
            _ => Err(DecoderError::InvalidGeometry("unsupported weight format")),
        }
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

    fn gated_delta_config(
        &self,
        layer_kinds: Option<&[DecoderLayerKind]>,
    ) -> Result<Option<GatedDeltaConfig>, DecoderError> {
        let has_stateful_layers =
            layer_kinds.is_some_and(|layers| layers.contains(&DecoderLayerKind::Linear));
        if self
            .output_gate_type
            .as_deref()
            .is_some_and(|activation| activation != "swish")
        {
            return Err(DecoderError::InvalidGeometry(
                "unsupported gated-delta output gate",
            ));
        }
        let fields = [
            self.linear_num_key_heads,
            self.linear_num_value_heads,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
            self.linear_conv_kernel_dim,
        ];
        if !has_stateful_layers {
            return if fields.iter().all(Option::is_none) {
                Ok(None)
            } else {
                Err(DecoderError::InvalidGeometry(
                    "state geometry without stateful layers",
                ))
            };
        }
        let [
            Some(key_heads),
            Some(value_heads),
            Some(key_width),
            Some(value_width),
            Some(convolution_kernel_width),
        ] = fields
        else {
            return Err(DecoderError::InvalidGeometry(
                "incomplete gated-delta geometry",
            ));
        };
        let geometry = GatedDeltaConfig {
            key_heads,
            value_heads,
            key_width,
            value_width,
            convolution_kernel_width,
        };
        geometry.validate()?;
        Ok(Some(geometry))
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

impl GatedDeltaConfig {
    fn validate(self) -> Result<(), DecoderError> {
        if self.key_heads == 0
            || self.value_heads == 0
            || !self.value_heads.is_multiple_of(self.key_heads)
            || self.key_width == 0
            || self.value_width == 0
            || self.convolution_kernel_width < 2
            || self.recurrent_elements().is_none()
            || self.convolution_elements().is_none()
        {
            return Err(DecoderError::InvalidGeometry("gated-delta dimensions"));
        }
        Ok(())
    }

    pub(super) fn recurrent_bytes(self) -> Option<u64> {
        self.recurrent_elements()?.checked_mul(4)?.try_into().ok()
    }

    pub(super) fn convolution_bytes(self) -> Option<u64> {
        self.convolution_elements()?.checked_mul(2)?.try_into().ok()
    }

    pub(super) fn key_elements(self) -> Option<usize> {
        self.key_heads.checked_mul(self.key_width)
    }

    pub(super) fn value_elements(self) -> Option<usize> {
        self.value_heads.checked_mul(self.value_width)
    }

    fn recurrent_elements(self) -> Option<usize> {
        self.value_heads
            .checked_mul(self.key_width)?
            .checked_mul(self.value_width)
    }

    pub(super) fn convolution_channels(self) -> Option<usize> {
        self.key_elements()?
            .checked_mul(2)?
            .checked_add(self.value_elements()?)
    }

    fn convolution_elements(self) -> Option<usize> {
        self.convolution_channels()?
            .checked_mul(self.convolution_kernel_width.checked_sub(1)?)
    }
}

fn parse_layer_kinds(
    layers: &[String],
    expected: usize,
) -> Result<Box<[DecoderLayerKind]>, DecoderError> {
    if layers.len() != expected {
        return Err(DecoderError::InvalidGeometry("attention layer count"));
    }
    layers
        .iter()
        .map(|layer| match layer.as_str() {
            "full_attention" => Ok(DecoderLayerKind::Full),
            "sliding_attention" => Ok(DecoderLayerKind::Sliding),
            "linear_attention" => Ok(DecoderLayerKind::Linear),
            _ => Err(DecoderError::InvalidGeometry("unsupported attention layer")),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}
