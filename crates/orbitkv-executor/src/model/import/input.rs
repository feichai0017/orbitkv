//! Checkpoint fields and numeric validation shared by admitted importers.
use super::super::{
    DecoderActivation, DecoderError, DecoderLayerKind, DecoderWeightFormat, GatedDeltaConfig,
};
use serde::Deserialize;

#[derive(Deserialize)]
pub(super) struct DecoderConfigInput {
    pub(super) num_hidden_layers: usize,
    pub(super) hidden_size: usize,
    pub(super) intermediate_size: usize,
    pub(super) num_attention_heads: usize,
    pub(super) num_key_value_heads: usize,
    #[serde(default)]
    pub(super) head_dim: Option<usize>,
    pub(super) vocab_size: usize,
    #[serde(default)]
    pub(super) rope_theta: Option<f32>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
    pub(super) rms_norm_eps: f32,
    #[serde(default)]
    pub(super) tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub(super) hidden_act: Option<String>,
    #[serde(default)]
    pub(super) hidden_activation: Option<String>,
    #[serde(default)]
    pub(super) query_pre_attn_scalar: Option<f64>,
    #[serde(default)]
    pub(super) attn_output_gate: Option<bool>,
    #[serde(default)]
    pub(super) output_gate_type: Option<String>,
    #[serde(default)]
    pub(super) rope_local_base_freq: Option<f32>,
    #[serde(default)]
    pub(super) layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub(super) linear_num_key_heads: Option<usize>,
    #[serde(default)]
    pub(super) linear_num_value_heads: Option<usize>,
    #[serde(default)]
    pub(super) linear_key_head_dim: Option<usize>,
    #[serde(default)]
    pub(super) linear_value_head_dim: Option<usize>,
    #[serde(default)]
    pub(super) linear_conv_kernel_dim: Option<usize>,
    #[serde(default)]
    pub(super) attn_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub(super) final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub(super) rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub(super) num_local_experts: Option<usize>,
    #[serde(default)]
    pub(super) quantization_config: Option<serde_json::Value>,
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

impl DecoderConfigInput {
    pub(super) fn validate(&self) -> Result<(), DecoderError> {
        let head_dim = self.head_dim.unwrap_or_else(|| {
            self.hidden_size
                .checked_div(self.num_attention_heads.max(1))
                .unwrap_or_default()
        });
        if self.layers_or_width_is_zero()
            || (self.head_dim.is_none()
                && !self
                    .hidden_size
                    .is_multiple_of(self.num_attention_heads.max(1)))
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

    pub(super) fn rope_theta(&self) -> Result<f32, DecoderError> {
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

    pub(super) fn rotary_dimensions(&self, head_dim: usize) -> Result<usize, DecoderError> {
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

    pub(super) fn weight_format(&self) -> Result<DecoderWeightFormat, DecoderError> {
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

    pub(super) fn activation(&self) -> Result<DecoderActivation, DecoderError> {
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

    pub(super) fn gated_delta_config(
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

    pub(super) fn layers_or_width_is_zero(&self) -> bool {
        self.num_hidden_layers == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.vocab_size == 0
    }
}

pub(super) fn parse_layer_kinds(
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
