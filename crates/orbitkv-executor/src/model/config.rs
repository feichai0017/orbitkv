use serde::Serialize;

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

impl DecoderConfig {
    pub(super) fn layer_kind(&self, layer: usize) -> DecoderLayerKind {
        self.layer_kinds
            .as_deref()
            .and_then(|layers| layers.get(layer))
            .copied()
            .unwrap_or(DecoderLayerKind::Full)
    }

    /// Imports a supported checkpoint architecture into explicit decoder semantics.
    ///
    /// # Errors
    /// Rejects missing/unknown architecture metadata and unsupported semantics.
    pub fn from_json(bytes: &[u8]) -> Result<Self, DecoderError> {
        super::import::import_decoder(bytes)
    }

    pub(super) fn require_executable(&self) -> Result<(), DecoderError> {
        if matches!(
            self.weight_format,
            DecoderWeightFormat::Fp8E4M3Block { rows, columns } if rows != orbitkv_ops::ops::linear::FP8_SCALE_BLOCK || columns != orbitkv_ops::ops::linear::FP8_SCALE_BLOCK
        ) {
            return Err(DecoderError::UnsupportedExecution(
                "non-128x128 block-scaled weight execution",
            ));
        }
        Ok(())
    }
}

impl GatedDeltaConfig {
    pub(super) fn validate(self) -> Result<(), DecoderError> {
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
