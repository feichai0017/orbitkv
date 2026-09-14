//! Checkpoint frontends normalize architecture-specific conventions once.
//! Kernel selection and graph construction consume only `DecoderConfig`.
mod input;
use super::{
    DecoderActivation, DecoderBlockLayout, DecoderConfig, DecoderError, DecoderLayerKind,
    DecoderNormWeights, GatedDeltaConfig,
};
use input::{DecoderConfigInput, parse_layer_kinds};

pub(super) fn import_decoder(bytes: &[u8]) -> Result<DecoderConfig, DecoderError> {
    let document = serde_json::from_slice::<serde_json::Value>(bytes)?;
    let (architecture, nested, tensor_prefix) = Architecture::select(&document)?;
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
    architecture.validate(&input, activation, layer_kinds.as_deref(), gated_delta)?;
    let sandwich_norm = architecture == Architecture::Gemma3;
    let unit_offset_norm = matches!(architecture, Architecture::Gemma3 | Architecture::Qwen35);
    #[allow(clippy::cast_precision_loss)]
    let attention_softmax_scale = input.query_pre_attn_scalar.map_or_else(
        || (head_dim as f64).sqrt().recip(),
        |scalar| scalar.sqrt().recip(),
    );
    let rope_theta = input.rope_theta()?;
    let rotary_dimensions = input.rotary_dimensions(head_dim)?;
    let weight_format = input.weight_format()?;
    Ok(DecoderConfig {
        layers: input.num_hidden_layers,
        hidden_size: input.hidden_size,
        intermediate_size: input.intermediate_size,
        query_heads: input.num_attention_heads,
        kv_heads: input.num_key_value_heads,
        head_dim,
        vocabulary_size: input.vocab_size,
        tensor_prefix: tensor_prefix.to_owned(),
        rope_theta,
        rotary_dimensions,
        rms_epsilon: input.rms_norm_eps,
        tied_embeddings: input
            .tie_word_embeddings
            .unwrap_or(architecture == Architecture::Gemma3),
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
        attention_output_gate: architecture == Architecture::Qwen35,
        layer_kinds,
        gated_delta,
        weight_format,
    })
}

fn round_to_bfloat16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff_u32 + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(rounding_bias) & 0xffff_0000)
}

/// Known serialization/architecture contracts, never a backend dispatch key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Architecture {
    Qwen2,
    Mistral,
    Gemma3,
    Qwen35,
}

impl Architecture {
    fn select(
        document: &serde_json::Value,
    ) -> Result<(Self, Option<&serde_json::Value>, &'static str), DecoderError> {
        let model_type = document
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| DecoderError::UnsupportedCheckpoint("model_type is required".into()))?;
        let (architecture, class) = match model_type {
            "qwen2" => (Self::Qwen2, "Qwen2ForCausalLM"),
            "mistral" => (Self::Mistral, "MistralForCausalLM"),
            "gemma3_text" => (Self::Gemma3, "Gemma3ForCausalLM"),
            "qwen3_5_text" => (Self::Qwen35, "Qwen3_5ForCausalLM"),
            "qwen3_5" => {
                Self::validate_class(document, "Qwen3_5ForConditionalGeneration")?;
                let text = document
                    .get("text_config")
                    .filter(|value| value.is_object())
                    .ok_or_else(|| {
                        DecoderError::UnsupportedCheckpoint("qwen3_5 requires text_config".into())
                    })?;
                let (inner, nested, _) = Self::select(text)?;
                if inner != Self::Qwen35 || nested.is_some() {
                    return Err(DecoderError::UnsupportedCheckpoint(
                        "inconsistent text architecture".into(),
                    ));
                }
                return Ok((Self::Qwen35, Some(text), "model.language_model"));
            }
            _ => {
                return Err(DecoderError::UnsupportedCheckpoint(format!(
                    "unsupported model_type {model_type:?}"
                )));
            }
        };
        if document.get("text_config").is_some() {
            return Err(DecoderError::UnsupportedCheckpoint(
                "unexpected text_config for a text-only architecture".into(),
            ));
        }
        Self::validate_class(document, class)?;
        Ok((architecture, None, "model"))
    }

    fn validate_class(document: &serde_json::Value, expected: &str) -> Result<(), DecoderError> {
        if let Some(value) = document.get("architectures") {
            let valid = value
                .as_array()
                .is_some_and(|classes| classes.len() == 1 && classes[0].as_str() == Some(expected));
            if !valid {
                return Err(DecoderError::UnsupportedCheckpoint(format!(
                    "architectures must identify {expected}"
                )));
            }
        }
        Ok(())
    }

    fn validate(
        self,
        input: &DecoderConfigInput,
        activation: DecoderActivation,
        layers: Option<&[DecoderLayerKind]>,
        state: Option<GatedDeltaConfig>,
    ) -> Result<(), DecoderError> {
        let expected_activation = if self == Self::Gemma3 {
            DecoderActivation::GeluTanh
        } else {
            DecoderActivation::Silu
        };
        if activation != expected_activation {
            return Err(DecoderError::InvalidGeometry(
                "activation conflicts with checkpoint architecture",
            ));
        }
        if self == Self::Gemma3 {
            if input
                .rope_local_base_freq
                .is_none_or(|theta| !theta.is_finite() || theta <= 0.0)
                || layers.is_none()
            {
                return Err(DecoderError::InvalidGeometry(
                    "incomplete sandwich-norm decoder semantics",
                ));
            }
        } else if input.rope_local_base_freq.is_some() || input.query_pre_attn_scalar.is_some() {
            return Err(DecoderError::InvalidGeometry(
                "attention constants conflict with checkpoint architecture",
            ));
        }
        if self == Self::Qwen35 {
            if input.attn_output_gate == Some(false)
                || layers.is_none()
                || layers.is_some_and(|layers| layers.contains(&DecoderLayerKind::Sliding))
            {
                return Err(DecoderError::InvalidGeometry(
                    "incomplete gated decoder semantics",
                ));
            }
        } else if state.is_some()
            || input.attn_output_gate == Some(true)
            || input.output_gate_type.is_some()
        {
            return Err(DecoderError::InvalidGeometry(
                "state or gating conflicts with checkpoint architecture",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/model/import/mod.rs"]
mod tests;
