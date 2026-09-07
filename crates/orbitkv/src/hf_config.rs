use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::attention_state::{
    AttentionStateError, AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage,
    CompiledAttentionStatePlan, RecurrentFamily, compile_attention_state_plan,
};
use crate::plan::{KvPlanInput, PlanError, RetentionKind, compile_plan, compile_retention_program};
use crate::retention::{IntExpr, Predicate, RetentionProgramInput, RetentionStateDecl};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HfRetentionOptions {
    pub page_tokens: u64,
    pub kv_dtype_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HfLayerInference {
    ExplicitLayerTypes,
    UniformFull,
    UniformSliding,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HfRetentionCompilation {
    pub schema: &'static str,
    pub config_sha256: String,
    pub layer_inference: HfLayerInference,
    pub num_hidden_layers: u64,
    pub num_key_value_heads: u64,
    pub head_dim: u64,
    pub bytes_per_token_per_layer: u64,
    pub program: RetentionProgramInput,
}

#[derive(Clone, Debug, Deserialize)]
struct HfModelConfig {
    num_hidden_layers: u64,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    sliding_window: Option<u64>,
    #[serde(default)]
    use_sliding_window: Option<bool>,
    num_key_value_heads: u64,
    #[serde(default)]
    head_dim: Option<u64>,
    #[serde(default)]
    hidden_size: Option<u64>,
    #[serde(default)]
    num_attention_heads: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
struct HfConfigEnvelope {
    #[serde(default)]
    text_config: Option<serde_json::Value>,
    #[serde(default)]
    num_hidden_layers: Option<u64>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    use_sliding_window: Option<bool>,
    #[serde(default)]
    sliding_window: Option<u64>,
    #[serde(flatten)]
    structural_fields: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
struct HybridLinearAttentionConfig {
    #[serde(default)]
    dtype: Option<String>,
    #[serde(default)]
    mamba_ssm_dtype: Option<String>,
    #[serde(default)]
    num_hidden_layers: Option<u64>,
    #[serde(default)]
    full_attention_interval: Option<u64>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    num_key_value_heads: Option<u64>,
    #[serde(default)]
    head_dim: Option<u64>,
    #[serde(default)]
    linear_num_key_heads: Option<u64>,
    #[serde(default)]
    linear_num_value_heads: Option<u64>,
    #[serde(default)]
    linear_key_head_dim: Option<u64>,
    #[serde(default)]
    linear_value_head_dim: Option<u64>,
    #[serde(default)]
    linear_conv_kernel_dim: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct HybridLinearAttentionGeometry {
    full_layers: Vec<u32>,
    linear_layers: Vec<u32>,
    key_bytes_per_token_per_layer: u64,
    value_bytes_per_token_per_layer: u64,
    recurrent_state_bytes_per_layer: u64,
    convolution_state_bytes_per_layer: u64,
    convolution_kernel_width: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AttentionStateProfile {
    HybridFixedState,
    LatentKv,
}

const BF16_BYTES: u64 = 2;
const FP32_BYTES: u64 = 4;
const CHECKPOINT_SLOTS_PER_REQUEST: u32 = 2;
const HYBRID_TOKEN_DTYPE: &str = "bfloat16";
const HYBRID_RECURRENT_DTYPE: &str = "float32";
const HYBRID_FIXED_STATE_MARKERS: &[&str] = &[
    "full_attention_interval",
    "mamba_ssm_dtype",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
];

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum HfConfigError {
    #[error("HF frontend page_tokens must be positive")]
    ZeroPageTokens,
    #[error("HF frontend kv_dtype_bytes must be positive")]
    ZeroKvDtypeBytes,
    #[error("HF config num_hidden_layers must be positive")]
    ZeroLayers,
    #[error("HF config num_key_value_heads must be positive")]
    ZeroKvHeads,
    #[error("HF config head_dim must be positive")]
    ZeroHeadDim,
    #[error(
        "HF config must declare head_dim or an exactly divisible hidden_size/num_attention_heads"
    )]
    MissingHeadGeometry,
    #[error("HF config does not prove each layer's attention retention semantics")]
    MissingLayerSemantics,
    #[error("HF config layer_types has {actual} entries but num_hidden_layers is {expected}")]
    LayerTypeCountMismatch { expected: u64, actual: usize },
    #[error("HF config layer {layer} uses unsupported type {layer_type:?}")]
    UnsupportedLayerType { layer: u32, layer_type: String },
    #[error("HF config does not declare a heterogeneous attention-state structure")]
    NotHeterogeneousAttentionState,
    #[error("HF config declares ambiguous attention-state structures")]
    AmbiguousAttentionStructure,
    #[error("hybrid linear-attention config must contain a nested text_config object")]
    MissingHybridLinearAttentionTextConfig,
    #[error("hybrid linear-attention text_config is missing required field {0}")]
    MissingHybridLinearAttentionField(&'static str),
    #[error("hybrid linear-attention config field {field} must be positive, got {actual}")]
    InvalidHybridLinearAttentionGeometry { field: &'static str, actual: u64 },
    #[error(
        "hybrid linear-attention config layer {layer} must be {expected:?} for full_attention_interval {interval}, got {actual:?}"
    )]
    HybridLinearAttentionLayerScheduleMismatch {
        layer: u32,
        interval: u64,
        expected: &'static str,
        actual: String,
    },
    #[error(
        "hybrid linear-attention config field {field} does not fit its compiled representation"
    )]
    HybridLinearAttentionGeometryOutOfRange { field: &'static str },
    #[error(
        "hybrid linear-attention config field {field} must match the supported structural schema, got {actual:?}"
    )]
    UnsupportedHybridLinearAttentionDtype {
        field: &'static str,
        actual: Option<String>,
    },
    #[error(
        "hybrid linear-attention token and convolution state requires --kv-dtype-bytes 2, got {actual}"
    )]
    HybridLinearAttentionKvDtypeBytesMismatch { actual: u64 },
    #[error("latent-KV config is missing a positive {0}")]
    MissingLatentGeometry(&'static str),
    #[error("latent-KV config does not support sparse index_topk storage")]
    UnsupportedLatentSparseIndexing,
    #[error("HF config sliding_attention layers require a positive sliding_window")]
    MissingSlidingWindow,
    #[error("HF config sliding_window does not fit the Retention IR constant type")]
    SlidingWindowOutOfRange,
    #[error("HF config layer index does not fit u32")]
    LayerIndexOutOfRange,
    #[error("integer overflow while deriving {0}")]
    ArithmeticOverflow(&'static str),
    #[error("invalid HF config JSON: {0}")]
    Json(String),
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum HfManagerPlanError {
    #[error(transparent)]
    Config(#[from] HfConfigError),
    #[error(transparent)]
    AttentionState(#[from] AttentionStateError),
    #[error(transparent)]
    Plan(#[from] PlanError),
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum HfStatePlanError {
    #[error(transparent)]
    Config(#[from] HfConfigError),
    #[error(transparent)]
    AttentionState(#[from] AttentionStateError),
}

/// Compiles explicit Full/SWA layer semantics, or an explicit uniform
/// attention declaration, into declarative Retention IR.
///
/// Configs without either proof fail closed; this frontend never guesses an
/// all-Full compatibility plan from a model-wide field.
///
/// # Errors
///
/// Returns an error for unknown layer semantics, invalid attention geometry,
/// unsupported explicit layer types, or checked arithmetic failure.
pub fn compile_hf_config(
    config_json: &[u8],
    options: HfRetentionOptions,
) -> Result<HfRetentionCompilation, HfConfigError> {
    validate_options(options)?;
    match classify_attention_state_profile(&parse_envelope(config_json)?) {
        Err(HfConfigError::NotHeterogeneousAttentionState) => {}
        Ok(_) => return Err(HfConfigError::AmbiguousAttentionStructure),
        Err(error) => return Err(error),
    }
    let config = serde_json::from_slice::<HfModelConfig>(config_json)
        .map_err(|error| HfConfigError::Json(error.to_string()))?;
    if config.num_hidden_layers == 0 {
        return Err(HfConfigError::ZeroLayers);
    }
    if config.num_key_value_heads == 0 {
        return Err(HfConfigError::ZeroKvHeads);
    }
    let head_dim = derive_head_dim(&config)?;
    let bytes_per_token_per_layer = 2_u64
        .checked_mul(config.num_key_value_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .and_then(|value| value.checked_mul(options.kv_dtype_bytes))
        .ok_or(HfConfigError::ArithmeticOverflow(
            "KV bytes per token per layer",
        ))?;
    let (layer_inference, full_layers, sliding_layers) = derive_layers(&config)?;
    let mut states = Vec::with_capacity(2);
    if !full_layers.is_empty() {
        states.push(RetentionStateDecl {
            name: "full".into(),
            layers: full_layers,
            kv_head_range: None,
            bytes_per_token_per_layer,
            may_read: Predicate::True,
        });
    }
    if !sliding_layers.is_empty() {
        let window = config
            .sliding_window
            .filter(|window| *window > 0)
            .ok_or(HfConfigError::MissingSlidingWindow)?;
        let value = i64::try_from(window).map_err(|_| HfConfigError::SlidingWindowOutOfRange)?;
        states.push(RetentionStateDecl {
            name: "swa".into(),
            layers: sliding_layers,
            kv_head_range: None,
            bytes_per_token_per_layer,
            may_read: Predicate::LessThan {
                lhs: IntExpr::Sub {
                    lhs: Box::new(IntExpr::QueryPosition),
                    rhs: Box::new(IntExpr::KeyPosition),
                },
                rhs: IntExpr::Constant { value },
            },
        });
    }
    Ok(HfRetentionCompilation {
        schema: "orbitkv.hf-retention-compilation.v1",
        config_sha256: format!("sha256:{:x}", Sha256::digest(config_json)),
        layer_inference,
        num_hidden_layers: config.num_hidden_layers,
        num_key_value_heads: config.num_key_value_heads,
        head_dim,
        bytes_per_token_per_layer,
        program: RetentionProgramInput {
            schema: "orbitkv.retention-ir.v1".into(),
            page_tokens: options.page_tokens,
            states,
        },
    })
}

/// Lowers a supported HF config to the heterogeneous attention-state input
/// embedded as the declarative source of a [`RuntimeManifest`](crate::RuntimeManifest).
///
/// This frontend supports a manifest-bound nested hybrid linear-attention
/// profile.
/// All cache geometry is derived from explicit text decoder fields.
///
/// # Errors
///
/// Returns an error when the structural markers, dtype, layer semantics, or
/// any geometry field is missing or unsupported.
pub fn compile_hf_attention_state_input(
    config_json: &[u8],
    options: HfRetentionOptions,
) -> Result<AttentionStatePlanInput, HfConfigError> {
    validate_options(options)?;
    let envelope = parse_envelope(config_json)?;
    match classify_attention_state_profile(&envelope)? {
        AttentionStateProfile::HybridFixedState => {
            compile_hybrid_linear_attention_state_input(envelope, options)
        }
        AttentionStateProfile::LatentKv => compile_latent_attention_state_input(&envelope, options),
    }
}

/// Compiles a supported HF config into backend-specific heterogeneous
/// attention-state contracts.
///
/// This frontend supports a manifest-bound nested hybrid linear-attention
/// profile.
/// Use [`compile_hf_attention_state_input`] when constructing a canonical
/// runtime manifest; this function returns the compiled `backend` schema
/// instead.
///
/// # Errors
///
/// Returns an error when the structural markers, dtype, layer semantics, or
/// any geometry field is missing or unsupported.
pub fn compile_hf_attention_state_plan(
    config_json: &[u8],
    options: HfRetentionOptions,
) -> Result<CompiledAttentionStatePlan, HfStatePlanError> {
    let input = compile_hf_attention_state_input(config_json, options)?;
    Ok(compile_attention_state_plan(input)?)
}

/// Produces the token-addressable projection consumed by the canonical token
/// manager.
///
/// For heterogeneous hybrid linear-attention profiles, recurrent and
/// convolution state is deliberately absent. Use
/// [`compile_hf_attention_state_input`] for the complete state ownership input.
///
/// # Errors
///
/// Returns an error when the HF semantics cannot be proven or the generated
/// token-manager plan fails canonical plan validation.
pub fn compile_hf_token_manager_plan(
    config_json: &[u8],
    options: HfRetentionOptions,
) -> Result<KvPlanInput, HfManagerPlanError> {
    validate_options(options)?;
    let envelope = parse_envelope(config_json)?;
    match classify_attention_state_profile(&envelope) {
        Ok(AttentionStateProfile::HybridFixedState) => {
            let input = compile_hybrid_linear_attention_state_input(envelope, options)?;
            return compile_attention_state_token_projection(input);
        }
        Ok(AttentionStateProfile::LatentKv) => {
            let input = compile_latent_attention_state_input(&envelope, options)?;
            return compile_attention_state_token_projection(input);
        }
        Err(HfConfigError::NotHeterogeneousAttentionState) => {}
        Err(error) => return Err(error.into()),
    }
    let compilation = compile_hf_config(config_json, options)?;
    let compiled = compile_retention_program(compilation.program)?;
    let input = KvPlanInput {
        page_tokens: compiled.page_tokens,
        classes: compiled
            .classes
            .into_iter()
            .map(|class| class.spec)
            .collect(),
    };
    compile_plan(input.clone())?;
    Ok(input)
}

fn validate_options(options: HfRetentionOptions) -> Result<(), HfConfigError> {
    if options.page_tokens == 0 {
        return Err(HfConfigError::ZeroPageTokens);
    }
    if options.kv_dtype_bytes == 0 {
        return Err(HfConfigError::ZeroKvDtypeBytes);
    }
    Ok(())
}

fn parse_envelope(config_json: &[u8]) -> Result<HfConfigEnvelope, HfConfigError> {
    serde_json::from_slice(config_json).map_err(|error| HfConfigError::Json(error.to_string()))
}

fn classify_attention_state_profile(
    envelope: &HfConfigEnvelope,
) -> Result<AttentionStateProfile, HfConfigError> {
    let hybrid = match envelope.text_config.as_ref() {
        Some(serde_json::Value::Object(text)) => {
            HYBRID_FIXED_STATE_MARKERS
                .iter()
                .any(|field| text.contains_key(*field))
                || text
                    .get("layer_types")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|layers| layers.iter().any(|layer| layer == "linear_attention"))
        }
        Some(_) | None => false,
    };
    let latent = ["kv_lora_rank", "qk_rope_head_dim", "index_topk"]
        .iter()
        .any(|field| envelope.structural_fields.contains_key(*field));
    match (hybrid, latent) {
        (true, true) => Err(HfConfigError::AmbiguousAttentionStructure),
        (true, false) => Ok(AttentionStateProfile::HybridFixedState),
        (false, true) => Ok(AttentionStateProfile::LatentKv),
        (false, false) => Err(HfConfigError::NotHeterogeneousAttentionState),
    }
}

fn compile_hybrid_linear_attention_state_input(
    envelope: HfConfigEnvelope,
    options: HfRetentionOptions,
) -> Result<AttentionStatePlanInput, HfConfigError> {
    if options.kv_dtype_bytes != BF16_BYTES {
        return Err(HfConfigError::HybridLinearAttentionKvDtypeBytesMismatch {
            actual: options.kv_dtype_bytes,
        });
    }
    let text_value = envelope
        .text_config
        .filter(serde_json::Value::is_object)
        .ok_or(HfConfigError::MissingHybridLinearAttentionTextConfig)?;
    let text = serde_json::from_value::<HybridLinearAttentionConfig>(text_value)
        .map_err(|error| HfConfigError::Json(error.to_string()))?;
    let geometry =
        derive_hybrid_linear_attention_geometry(&text, HYBRID_TOKEN_DTYPE, HYBRID_RECURRENT_DTYPE)?;
    Ok(AttentionStatePlanInput {
        page_tokens: options.page_tokens,
        states: vec![
            AttentionStateSpec {
                name: "full_attention_kv".into(),
                layers: geometry.full_layers,
                storage: AttentionStateStorage::TokenKv {
                    key_bytes_per_token_per_layer: geometry.key_bytes_per_token_per_layer,
                    value_bytes_per_token_per_layer: geometry.value_bytes_per_token_per_layer,
                    retention: RetentionKind::Full,
                    window_tokens: None,
                },
            },
            AttentionStateSpec {
                name: "gdn_recurrent".into(),
                layers: geometry.linear_layers.clone(),
                storage: AttentionStateStorage::Recurrent {
                    family: RecurrentFamily::Gdn,
                    state_bytes_per_layer: geometry.recurrent_state_bytes_per_layer,
                    checkpoint_slots_per_request: CHECKPOINT_SLOTS_PER_REQUEST,
                },
            },
            AttentionStateSpec {
                name: "gdn_convolution".into(),
                layers: geometry.linear_layers,
                storage: AttentionStateStorage::Convolution {
                    state_bytes_per_layer: geometry.convolution_state_bytes_per_layer,
                    kernel_width: geometry.convolution_kernel_width,
                    checkpoint_slots_per_request: CHECKPOINT_SLOTS_PER_REQUEST,
                },
            },
        ],
    })
}

fn compile_latent_attention_state_input(
    envelope: &HfConfigEnvelope,
    options: HfRetentionOptions,
) -> Result<AttentionStatePlanInput, HfConfigError> {
    if envelope.layer_types.is_some()
        || envelope.use_sliding_window == Some(true)
        || envelope.sliding_window.is_some()
    {
        return Err(HfConfigError::AmbiguousAttentionStructure);
    }
    if envelope
        .structural_fields
        .get("index_topk")
        .is_some_and(|value| !value.is_null())
    {
        return Err(HfConfigError::UnsupportedLatentSparseIndexing);
    }
    let layers = envelope
        .num_hidden_layers
        .filter(|value| *value > 0)
        .ok_or(HfConfigError::MissingLatentGeometry("num_hidden_layers"))?;
    let latent_rank = required_latent_value(
        envelope.structural_fields.get("kv_lora_rank"),
        "kv_lora_rank",
    )?;
    let rope_dim = required_latent_value(
        envelope.structural_fields.get("qk_rope_head_dim"),
        "qk_rope_head_dim",
    )?;
    let latent_bytes = latent_rank.checked_mul(options.kv_dtype_bytes).ok_or(
        HfConfigError::ArithmeticOverflow("latent-KV payload bytes per token per layer"),
    )?;
    let rope_bytes =
        rope_dim
            .checked_mul(options.kv_dtype_bytes)
            .ok_or(HfConfigError::ArithmeticOverflow(
                "latent-KV RoPE bytes per token per layer",
            ))?;
    Ok(AttentionStatePlanInput {
        page_tokens: options.page_tokens,
        states: vec![AttentionStateSpec {
            name: "latent_mla".into(),
            layers: layer_range(layers)?,
            storage: AttentionStateStorage::LatentKv {
                latent_bytes_per_token_per_layer: latent_bytes,
                rope_bytes_per_token_per_layer: rope_bytes,
                retention: RetentionKind::Full,
                window_tokens: None,
            },
        }],
    })
}

fn compile_attention_state_token_projection(
    input: AttentionStatePlanInput,
) -> Result<KvPlanInput, HfManagerPlanError> {
    let compiled = compile_attention_state_plan(input)?;
    let input = compiled.token_manager_plan()?;
    compile_plan(input.clone())?;
    Ok(input)
}

fn derive_hybrid_linear_attention_geometry(
    text: &HybridLinearAttentionConfig,
    token_dtype: &'static str,
    recurrent_dtype: &'static str,
) -> Result<HybridLinearAttentionGeometry, HfConfigError> {
    require_dtype("dtype", text.dtype.as_deref(), token_dtype)?;
    require_dtype(
        "mamba_ssm_dtype",
        text.mamba_ssm_dtype.as_deref(),
        recurrent_dtype,
    )?;
    let num_hidden_layers = required_positive(text.num_hidden_layers, "num_hidden_layers")?;
    let full_attention_interval =
        required_positive(text.full_attention_interval, "full_attention_interval")?;
    let layer_types =
        text.layer_types
            .as_ref()
            .ok_or(HfConfigError::MissingHybridLinearAttentionField(
                "layer_types",
            ))?;
    let layer_count =
        u64::try_from(layer_types.len()).map_err(|_| HfConfigError::LayerIndexOutOfRange)?;
    if layer_count != num_hidden_layers {
        return Err(HfConfigError::LayerTypeCountMismatch {
            expected: num_hidden_layers,
            actual: layer_types.len(),
        });
    }

    let head_dim = required_positive(text.head_dim, "head_dim")?;
    let num_key_value_heads = required_positive(text.num_key_value_heads, "num_key_value_heads")?;
    let linear_num_key_heads =
        required_positive(text.linear_num_key_heads, "linear_num_key_heads")?;
    let linear_num_value_heads =
        required_positive(text.linear_num_value_heads, "linear_num_value_heads")?;
    let linear_key_head_dim = required_positive(text.linear_key_head_dim, "linear_key_head_dim")?;
    let linear_value_head_dim =
        required_positive(text.linear_value_head_dim, "linear_value_head_dim")?;
    let convolution_kernel_width_u64 =
        required_positive(text.linear_conv_kernel_dim, "linear_conv_kernel_dim")?;
    if convolution_kernel_width_u64 < 2 {
        return Err(HfConfigError::InvalidHybridLinearAttentionGeometry {
            field: "linear_conv_kernel_dim",
            actual: convolution_kernel_width_u64,
        });
    }
    let convolution_kernel_width = u32::try_from(convolution_kernel_width_u64).map_err(|_| {
        HfConfigError::HybridLinearAttentionGeometryOutOfRange {
            field: "linear_conv_kernel_dim",
        }
    })?;

    let (full_layers, linear_layers) =
        derive_hybrid_linear_attention_layers(layer_types, full_attention_interval)?;
    if full_layers.is_empty() || linear_layers.is_empty() {
        return Err(HfConfigError::MissingLayerSemantics);
    }
    let key_bytes_per_token_per_layer = checked_product(
        &[num_key_value_heads, head_dim, BF16_BYTES],
        "hybrid linear-attention key bytes per token per layer",
    )?;
    let value_bytes_per_token_per_layer = key_bytes_per_token_per_layer;
    let key_channels = checked_product(
        &[linear_num_key_heads, linear_key_head_dim],
        "hybrid linear-attention key channels",
    )?;
    let doubled_key_channels =
        key_channels
            .checked_mul(2)
            .ok_or(HfConfigError::ArithmeticOverflow(
                "hybrid linear-attention doubled key channels",
            ))?;
    let value_channels = checked_product(
        &[linear_num_value_heads, linear_value_head_dim],
        "hybrid linear-attention value channels",
    )?;
    let convolution_channels = doubled_key_channels.checked_add(value_channels).ok_or(
        HfConfigError::ArithmeticOverflow("hybrid linear-attention convolution channels"),
    )?;
    let recurrent_state_bytes_per_layer = checked_product(
        &[
            linear_num_value_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            FP32_BYTES,
        ],
        "hybrid linear-attention recurrent bytes per layer",
    )?;
    // The causal convolution weight has K taps, but only K - 1 history
    // positions survive between decode steps.
    let convolution_state_bytes_per_layer = checked_product(
        &[
            convolution_channels,
            convolution_kernel_width_u64 - 1,
            BF16_BYTES,
        ],
        "hybrid linear-attention convolution state bytes per layer",
    )?;

    Ok(HybridLinearAttentionGeometry {
        full_layers,
        linear_layers,
        key_bytes_per_token_per_layer,
        value_bytes_per_token_per_layer,
        recurrent_state_bytes_per_layer,
        convolution_state_bytes_per_layer,
        convolution_kernel_width,
    })
}

fn derive_hybrid_linear_attention_layers(
    layer_types: &[String],
    full_attention_interval: u64,
) -> Result<(Vec<u32>, Vec<u32>), HfConfigError> {
    let mut full_layers = Vec::new();
    let mut linear_layers = Vec::new();
    for (index, layer_type) in layer_types.iter().enumerate() {
        let layer = u32::try_from(index).map_err(|_| HfConfigError::LayerIndexOutOfRange)?;
        if !matches!(layer_type.as_str(), "full_attention" | "linear_attention") {
            return Err(HfConfigError::UnsupportedLayerType {
                layer,
                layer_type: layer_type.clone(),
            });
        }
        let expected = if (u64::from(layer) + 1).is_multiple_of(full_attention_interval) {
            "full_attention"
        } else {
            "linear_attention"
        };
        if layer_type != expected {
            return Err(HfConfigError::HybridLinearAttentionLayerScheduleMismatch {
                layer,
                interval: full_attention_interval,
                expected,
                actual: layer_type.clone(),
            });
        }
        match layer_type.as_str() {
            "full_attention" => full_layers.push(layer),
            "linear_attention" => linear_layers.push(layer),
            _ => unreachable!("supported hybrid linear-attention layer type was checked above"),
        }
    }
    Ok((full_layers, linear_layers))
}

fn required_positive(value: Option<u64>, field: &'static str) -> Result<u64, HfConfigError> {
    let value = value.ok_or(HfConfigError::MissingHybridLinearAttentionField(field))?;
    if value == 0 {
        return Err(HfConfigError::InvalidHybridLinearAttentionGeometry {
            field,
            actual: value,
        });
    }
    Ok(value)
}

fn required_latent_value(
    value: Option<&serde_json::Value>,
    field: &'static str,
) -> Result<u64, HfConfigError> {
    value
        .and_then(serde_json::Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or(HfConfigError::MissingLatentGeometry(field))
}

fn require_dtype(
    field: &'static str,
    actual: Option<&str>,
    expected: &'static str,
) -> Result<(), HfConfigError> {
    if actual != Some(expected) {
        return Err(HfConfigError::UnsupportedHybridLinearAttentionDtype {
            field,
            actual: actual.map(str::to_owned),
        });
    }
    Ok(())
}

fn checked_product(values: &[u64], label: &'static str) -> Result<u64, HfConfigError> {
    values.iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(HfConfigError::ArithmeticOverflow(label))
    })
}

fn derive_head_dim(config: &HfModelConfig) -> Result<u64, HfConfigError> {
    if let Some(head_dim) = config.head_dim {
        return if head_dim == 0 {
            Err(HfConfigError::ZeroHeadDim)
        } else {
            Ok(head_dim)
        };
    }
    let (Some(hidden_size), Some(attention_heads)) =
        (config.hidden_size, config.num_attention_heads)
    else {
        return Err(HfConfigError::MissingHeadGeometry);
    };
    if attention_heads == 0 || hidden_size == 0 || !hidden_size.is_multiple_of(attention_heads) {
        return Err(HfConfigError::MissingHeadGeometry);
    }
    Ok(hidden_size / attention_heads)
}

fn derive_layers(
    config: &HfModelConfig,
) -> Result<(HfLayerInference, Vec<u32>, Vec<u32>), HfConfigError> {
    let Some(layer_types) = &config.layer_types else {
        if config.use_sliding_window == Some(true)
            && config.sliding_window.is_some_and(|window| window > 0)
        {
            return Ok((
                HfLayerInference::UniformSliding,
                Vec::new(),
                layer_range(config.num_hidden_layers)?,
            ));
        }
        if config.use_sliding_window == Some(false) {
            return Ok((
                HfLayerInference::UniformFull,
                layer_range(config.num_hidden_layers)?,
                Vec::new(),
            ));
        }
        return Err(HfConfigError::MissingLayerSemantics);
    };
    if layer_types.len() as u64 != config.num_hidden_layers {
        return Err(HfConfigError::LayerTypeCountMismatch {
            expected: config.num_hidden_layers,
            actual: layer_types.len(),
        });
    }
    let mut full_layers = Vec::new();
    let mut sliding_layers = Vec::new();
    for (index, layer_type) in layer_types.iter().enumerate() {
        let layer = u32::try_from(index).map_err(|_| HfConfigError::LayerIndexOutOfRange)?;
        match layer_type.as_str() {
            "full_attention" => full_layers.push(layer),
            "sliding_attention" => sliding_layers.push(layer),
            _ => {
                return Err(HfConfigError::UnsupportedLayerType {
                    layer,
                    layer_type: layer_type.clone(),
                });
            }
        }
    }
    Ok((
        HfLayerInference::ExplicitLayerTypes,
        full_layers,
        sliding_layers,
    ))
}

fn layer_range(count: u64) -> Result<Vec<u32>, HfConfigError> {
    (0..count)
        .map(|index| u32::try_from(index).map_err(|_| HfConfigError::LayerIndexOutOfRange))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention_state::AttentionStateBackend;
    use crate::plan::AddressProgram;
    use crate::retention::{InferredRetention, analyze_state};

    const OPTIONS: HfRetentionOptions = HfRetentionOptions {
        page_tokens: 16,
        kv_dtype_bytes: 2,
    };
    const HYBRID_FIXED_STATE_FIXTURE: &[u8] =
        include_bytes!("../fixtures/hybrid-fixed-state/config.json");

    fn hybrid_linear_attention_fixture_with_text_fields(fields: &[(&str, u64)]) -> Vec<u8> {
        let mut config =
            serde_json::from_slice::<serde_json::Value>(HYBRID_FIXED_STATE_FIXTURE).unwrap();
        let text = config["text_config"].as_object_mut().unwrap();
        for &(field, value) in fields {
            text.insert(field.to_owned(), value.into());
        }
        serde_json::to_vec(&config).unwrap()
    }

    #[test]
    fn hybrid_fixed_state_fixture_compiles_exact_heterogeneous_geometry() {
        let input = compile_hf_attention_state_input(HYBRID_FIXED_STATE_FIXTURE, OPTIONS).unwrap();
        assert!(matches!(
            input.states[2].storage,
            AttentionStateStorage::Convolution {
                // The recurrent convolution state persists 6,144 BF16 channels
                // across K - 1 positions.
                state_bytes_per_layer: 36_864,
                kernel_width: 4,
                ..
            }
        ));
        let plan = compile_hf_attention_state_plan(HYBRID_FIXED_STATE_FIXTURE, OPTIONS).unwrap();
        assert_eq!(plan.page_tokens, 16);
        assert_eq!(plan.states.len(), 3);
        assert_eq!(plan.states[0].name, "full_attention_kv");
        assert_eq!(plan.states[0].layers, vec![3, 7, 11, 15, 19, 23]);
        assert_eq!(plan.states[1].layers.len(), 18);
        assert_eq!(plan.states[2].layers, plan.states[1].layers);

        let AttentionStateBackend::TokenSlots {
            components,
            bytes_per_token_per_layer,
            ..
        } = &plan.states[0].backend
        else {
            panic!("full-attention state must use token slots");
        };
        assert_eq!(components[0].bytes_per_token_per_layer, 1_024);
        assert_eq!(components[1].bytes_per_token_per_layer, 1_024);
        assert_eq!(*bytes_per_token_per_layer, 2_048);

        assert!(matches!(
            plan.states[1].backend,
            AttentionStateBackend::RecurrentCheckpoints {
                family: RecurrentFamily::Gdn,
                state_bytes_per_layer: 1_048_576,
                checkpoint_slots_per_request: 2,
                ..
            }
        ));
        assert!(matches!(
            plan.states[2].backend,
            AttentionStateBackend::ConvolutionRing {
                state_bytes_per_layer: 36_864,
                kernel_width: 4,
                checkpoint_slots_per_request: 2,
                ..
            }
        ));
    }

    #[test]
    fn hybrid_fixed_state_manager_projection_contains_only_full_token_kv() {
        let manager = compile_hf_token_manager_plan(HYBRID_FIXED_STATE_FIXTURE, OPTIONS).unwrap();
        assert_eq!(manager.classes.len(), 1);
        assert_eq!(manager.classes[0].name, "full_attention_kv");
        assert_eq!(manager.classes[0].layers, vec![3, 7, 11, 15, 19, 23]);
        assert_eq!(manager.classes[0].bytes_per_token_per_layer, 2_048);
        assert_eq!(manager.classes[0].components[0].name, "key");
        assert_eq!(
            manager.classes[0].components[0].bytes_per_token_per_layer,
            1_024
        );
        compile_plan(manager).unwrap();
    }

    #[test]
    fn hybrid_linear_attention_missing_or_unsupported_contracts_fail_closed() {
        let wrong_dtype = String::from_utf8(HYBRID_FIXED_STATE_FIXTURE.to_vec())
            .unwrap()
            .replacen(
                "\"mamba_ssm_dtype\": \"float32\"",
                "\"mamba_ssm_dtype\": \"bfloat16\"",
                1,
            );
        assert!(matches!(
            compile_hf_attention_state_plan(wrong_dtype.as_bytes(), OPTIONS),
            Err(HfStatePlanError::Config(
                HfConfigError::UnsupportedHybridLinearAttentionDtype {
                    field: "mamba_ssm_dtype",
                    ..
                }
            ))
        ));

        let missing_layers = String::from_utf8(HYBRID_FIXED_STATE_FIXTURE.to_vec())
            .unwrap()
            .replacen("\"layer_types\"", "\"unproven_layer_types\"", 1);
        assert_eq!(
            compile_hf_attention_state_plan(missing_layers.as_bytes(), OPTIONS),
            Err(HfStatePlanError::Config(
                HfConfigError::MissingHybridLinearAttentionField("layer_types")
            ))
        );

        assert_eq!(
            compile_hf_attention_state_plan(
                HYBRID_FIXED_STATE_FIXTURE,
                HfRetentionOptions {
                    kv_dtype_bytes: 4,
                    ..OPTIONS
                }
            ),
            Err(HfStatePlanError::Config(
                HfConfigError::HybridLinearAttentionKvDtypeBytesMismatch { actual: 4 }
            ))
        );
    }

    #[test]
    fn hybrid_linear_attention_schedule_must_match_the_declared_interval() {
        let config =
            hybrid_linear_attention_fixture_with_text_fields(&[("full_attention_interval", 3)]);
        assert_eq!(
            compile_hf_attention_state_input(&config, OPTIONS),
            Err(HfConfigError::HybridLinearAttentionLayerScheduleMismatch {
                layer: 2,
                interval: 3,
                expected: "full_attention",
                actual: "linear_attention".into(),
            })
        );

        let config =
            hybrid_linear_attention_fixture_with_text_fields(&[("full_attention_interval", 0)]);
        assert_eq!(
            compile_hf_attention_state_input(&config, OPTIONS),
            Err(HfConfigError::InvalidHybridLinearAttentionGeometry {
                field: "full_attention_interval",
                actual: 0,
            })
        );
    }

    #[test]
    fn hybrid_linear_attention_admission_is_structural_and_has_priority() {
        let expected =
            compile_hf_attention_state_input(HYBRID_FIXED_STATE_FIXTURE, OPTIONS).unwrap();
        let expected_tokens =
            compile_hf_token_manager_plan(HYBRID_FIXED_STATE_FIXTURE, OPTIONS).unwrap();
        let mut renamed =
            serde_json::from_slice::<serde_json::Value>(HYBRID_FIXED_STATE_FIXTURE).unwrap();
        renamed["architectures"] = serde_json::json!(["RenamedArchitecture"]);
        renamed["model_type"] = serde_json::json!("renamed_envelope");
        renamed["text_config"]["model_type"] = serde_json::json!("renamed_text");
        renamed["num_hidden_layers"] = serde_json::json!(1);
        renamed["layer_types"] = serde_json::json!(["full_attention"]);
        renamed["num_key_value_heads"] = serde_json::json!(1);
        renamed["head_dim"] = serde_json::json!(1);
        let renamed = serde_json::to_vec(&renamed).unwrap();
        assert_eq!(
            compile_hf_attention_state_input(&renamed, OPTIONS).unwrap(),
            expected
        );
        assert_eq!(
            compile_hf_token_manager_plan(&renamed, OPTIONS).unwrap(),
            expected_tokens
        );

        let mut anonymous =
            serde_json::from_slice::<serde_json::Value>(HYBRID_FIXED_STATE_FIXTURE).unwrap();
        anonymous.as_object_mut().unwrap().remove("architectures");
        anonymous.as_object_mut().unwrap().remove("model_type");
        anonymous["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("model_type");
        let anonymous = serde_json::to_vec(&anonymous).unwrap();
        assert_eq!(
            compile_hf_attention_state_input(&anonymous, OPTIONS).unwrap(),
            expected
        );
    }

    #[test]
    fn partial_hybrid_marker_fails_closed_in_both_frontends() {
        let config = br#"{
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
            "num_key_value_heads": 1,
            "head_dim": 1,
            "text_config": {"full_attention_interval": 4}
        }"#;
        assert!(matches!(
            compile_hf_attention_state_input(config, OPTIONS),
            Err(HfConfigError::UnsupportedHybridLinearAttentionDtype {
                field: "dtype",
                actual: None
            })
        ));
        assert!(matches!(
            compile_hf_token_manager_plan(config, OPTIONS),
            Err(HfManagerPlanError::Config(
                HfConfigError::UnsupportedHybridLinearAttentionDtype { .. }
            ))
        ));
    }

    #[test]
    fn latent_admission_is_structural_and_rejects_partial_or_ambiguous_markers() {
        let config = serde_json::to_vec(&serde_json::json!({
            "architectures": ["RenamedLatentArchitecture"],
            "num_hidden_layers": 2,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "index_topk": null
        }))
        .unwrap();
        let input = compile_hf_attention_state_input(&config, OPTIONS).unwrap();
        assert_eq!(input.states.len(), 1);
        assert_eq!(input.states[0].layers, vec![0, 1]);
        assert!(matches!(
            input.states[0].storage,
            AttentionStateStorage::LatentKv {
                latent_bytes_per_token_per_layer: 1_024,
                rope_bytes_per_token_per_layer: 128,
                retention: RetentionKind::Full,
                window_tokens: None,
            }
        ));
        let manager = compile_hf_token_manager_plan(&config, OPTIONS).unwrap();
        assert_eq!(manager.classes.len(), 1);
        assert_eq!(
            manager.classes[0].storage,
            crate::plan::TokenStorageKind::LatentKv
        );

        let partial = serde_json::to_vec(&serde_json::json!({
            "num_hidden_layers": 2,
            "kv_lora_rank": 512,
            "use_sliding_window": false,
            "num_key_value_heads": 8,
            "head_dim": 64
        }))
        .unwrap();
        assert_eq!(
            compile_hf_token_manager_plan(&partial, OPTIONS),
            Err(HfManagerPlanError::Config(
                HfConfigError::MissingLatentGeometry("qk_rope_head_dim")
            ))
        );

        let sparse = serde_json::to_vec(&serde_json::json!({
            "num_hidden_layers": 2,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "index_topk": 2048
        }))
        .unwrap();
        assert_eq!(
            compile_hf_attention_state_input(&sparse, OPTIONS),
            Err(HfConfigError::UnsupportedLatentSparseIndexing)
        );
    }

    #[test]
    fn hybrid_linear_attention_arithmetic_reports_the_exact_failed_derivation() {
        let cases = [
            (
                vec![
                    ("linear_num_key_heads", u64::MAX),
                    ("linear_key_head_dim", 2),
                ],
                "hybrid linear-attention key channels",
            ),
            (
                vec![
                    ("linear_num_key_heads", u64::MAX / 2 + 1),
                    ("linear_key_head_dim", 1),
                ],
                "hybrid linear-attention doubled key channels",
            ),
            (
                vec![
                    ("linear_num_value_heads", u64::MAX),
                    ("linear_value_head_dim", 2),
                ],
                "hybrid linear-attention value channels",
            ),
            (
                vec![
                    ("linear_num_key_heads", (u64::MAX - 1) / 2),
                    ("linear_key_head_dim", 1),
                    ("linear_num_value_heads", 2),
                    ("linear_value_head_dim", 1),
                ],
                "hybrid linear-attention convolution channels",
            ),
            (
                vec![
                    ("linear_num_key_heads", u64::from(u32::MAX)),
                    ("linear_key_head_dim", 1),
                    ("linear_conv_kernel_dim", u64::from(u32::MAX)),
                ],
                "hybrid linear-attention convolution state bytes per layer",
            ),
        ];
        for (fields, expected) in cases {
            assert_eq!(
                compile_hf_attention_state_input(
                    &hybrid_linear_attention_fixture_with_text_fields(fields.as_slice()),
                    OPTIONS,
                ),
                Err(HfConfigError::ArithmeticOverflow(expected))
            );
        }
    }

    #[test]
    fn explicit_hybrid_config_compiles_lifetime_classes() {
        let config = serde_json::to_vec(&serde_json::json!({
            "architectures": ["RenamedHybridArchitecture"],
            "num_hidden_layers": 4,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ],
            "sliding_window": 128,
            "num_key_value_heads": 8,
            "head_dim": 64
        }))
        .unwrap();
        let compilation = compile_hf_config(&config, OPTIONS).unwrap();
        assert_eq!(
            compilation.layer_inference,
            HfLayerInference::ExplicitLayerTypes
        );
        assert_eq!(compilation.bytes_per_token_per_layer, 2048);
        assert_eq!(compilation.program.states[0].layers, vec![1, 3]);
        assert_eq!(compilation.program.states[1].layers, vec![0, 2]);
        assert_eq!(
            analyze_state(&compilation.program.states[1])
                .unwrap()
                .inferred,
            InferredRetention::FixedWindow { window_tokens: 128 }
        );
        let layout = compile_retention_program(compilation.program)
            .unwrap()
            .layout_program()
            .unwrap();
        assert_eq!(
            layout.classes[1].address,
            AddressProgram::Periodic { period_blocks: 9 }
        );
    }

    #[test]
    fn explicit_uniform_sliding_profile_is_inferred() {
        let config = serde_json::to_vec(&serde_json::json!({
            "architectures": ["RenamedSlidingArchitecture"],
            "num_hidden_layers": 2,
            "sliding_window": 4096,
            "use_sliding_window": true,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "num_attention_heads": 32
        }))
        .unwrap();
        let compilation = compile_hf_config(&config, OPTIONS).unwrap();
        assert_eq!(
            compilation.layer_inference,
            HfLayerInference::UniformSliding
        );
        assert_eq!(compilation.program.states[0].name, "swa");
        assert_eq!(compilation.program.states[0].layers, vec![0, 1]);
    }

    #[test]
    fn explicit_uniform_full_profile_compiles() {
        let config = serde_json::to_vec(&serde_json::json!({
            "architectures": ["RenamedFullArchitecture"],
            "num_hidden_layers": 2,
            "sliding_window": 4096,
            "use_sliding_window": false,
            "num_key_value_heads": 8,
            "head_dim": 64
        }))
        .unwrap();
        let compilation = compile_hf_config(&config, OPTIONS).unwrap();
        assert_eq!(compilation.layer_inference, HfLayerInference::UniformFull);
        assert_eq!(compilation.program.states[0].name, "full");
        assert_eq!(compilation.program.states[0].layers, vec![0, 1]);

        let input = compile_hf_token_manager_plan(&config, OPTIONS).unwrap();
        assert_eq!(input.classes.len(), 1);
        assert_eq!(input.classes[0].retention, crate::plan::RetentionKind::Full);
        assert_eq!(input.classes[0].window_tokens, None);
        let compiled = compile_plan(input).unwrap();
        let layout = compiled.layout_program().unwrap();
        assert_eq!(layout.classes[0].address, AddressProgram::AppendOnly);
    }

    #[test]
    fn missing_layer_semantics_fail_closed() {
        let config = serde_json::to_vec(&serde_json::json!({
            "architectures": ["RenamedFullArchitecture"],
            "num_hidden_layers": 2,
            "sliding_window": 4096,
            "num_key_value_heads": 8,
            "head_dim": 64
        }))
        .unwrap();
        assert_eq!(
            compile_hf_config(&config, OPTIONS),
            Err(HfConfigError::MissingLayerSemantics)
        );
    }

    #[test]
    fn unknown_explicit_layer_type_fails_closed() {
        let config = br#"{
            "num_hidden_layers": 1,
            "layer_types": ["mamba"],
            "num_key_value_heads": 8,
            "head_dim": 64
        }"#;
        assert!(matches!(
            compile_hf_config(config, OPTIONS),
            Err(HfConfigError::UnsupportedLayerType { layer: 0, layer_type })
                if layer_type == "mamba"
        ));
    }

    #[test]
    fn manager_plan_is_the_strict_canonical_source_shape() {
        let config = serde_json::to_vec(&serde_json::json!({
            "architectures": ["RenamedSlidingArchitecture"],
            "num_hidden_layers": 2,
            "sliding_window": 18,
            "use_sliding_window": true,
            "num_key_value_heads": 8,
            "head_dim": 64
        }))
        .unwrap();
        let input = compile_hf_token_manager_plan(&config, OPTIONS).unwrap();
        assert_eq!(input.page_tokens, 16);
        assert_eq!(input.classes.len(), 1);
        assert_eq!(input.classes[0].name, "swa");
        assert_eq!(
            input.classes[0].retention,
            crate::plan::RetentionKind::Sliding
        );
        assert_eq!(input.classes[0].window_tokens, Some(18));
        compile_plan(input).unwrap();
    }
}
