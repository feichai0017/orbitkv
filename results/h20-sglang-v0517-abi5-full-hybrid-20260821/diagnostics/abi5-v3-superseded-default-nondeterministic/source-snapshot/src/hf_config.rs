use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::plan::{KvPlanInput, PlanError, compile_plan, compile_retention_program};
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
    ArchitectureUniformFull,
    ArchitectureUniformSliding,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HfRetentionCompilation {
    pub schema: &'static str,
    pub config_sha256: String,
    pub architecture: Option<String>,
    pub layer_inference: HfLayerInference,
    pub num_hidden_layers: u64,
    pub num_key_value_heads: u64,
    pub head_dim: u64,
    pub bytes_per_token_per_layer: u64,
    pub program: RetentionProgramInput,
}

#[derive(Clone, Debug, Deserialize)]
struct HfModelConfig {
    #[serde(default)]
    architectures: Vec<String>,
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
    Plan(#[from] PlanError),
}

/// Compiles explicitly declared Full/SWA layer semantics, or a narrowly
/// allowlisted uniform-attention architecture contract, into declarative
/// Retention IR.
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
    if options.page_tokens == 0 {
        return Err(HfConfigError::ZeroPageTokens);
    }
    if options.kv_dtype_bytes == 0 {
        return Err(HfConfigError::ZeroKvDtypeBytes);
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
        architecture: config.architectures.first().cloned(),
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

/// Produces the strict `KvPlanInput` consumed by the canonical manager.
///
/// # Errors
///
/// Returns an error when the HF semantics cannot be proven or the generated
/// manager plan fails canonical plan validation.
pub fn compile_hf_manager_plan(
    config_json: &[u8],
    options: HfRetentionOptions,
) -> Result<KvPlanInput, HfManagerPlanError> {
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
        if is_uniform_sliding_architecture(config) {
            return Ok((
                HfLayerInference::ArchitectureUniformSliding,
                Vec::new(),
                layer_range(config.num_hidden_layers)?,
            ));
        }
        if is_uniform_full_architecture(config) {
            return Ok((
                HfLayerInference::ArchitectureUniformFull,
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

fn is_uniform_sliding_architecture(config: &HfModelConfig) -> bool {
    config.architectures.as_slice() == ["MistralForCausalLM"]
        && config.use_sliding_window != Some(false)
        && config.sliding_window.is_some_and(|window| window > 0)
}

fn is_uniform_full_architecture(config: &HfModelConfig) -> bool {
    config.architectures.as_slice() == ["Qwen2ForCausalLM"]
        && config.use_sliding_window == Some(false)
}

fn layer_range(count: u64) -> Result<Vec<u32>, HfConfigError> {
    (0..count)
        .map(|index| u32::try_from(index).map_err(|_| HfConfigError::LayerIndexOutOfRange))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::AddressProgram;
    use crate::retention::{InferredRetention, analyze_state};

    const OPTIONS: HfRetentionOptions = HfRetentionOptions {
        page_tokens: 16,
        kv_dtype_bytes: 2,
    };

    #[test]
    fn explicit_hybrid_config_compiles_lifetime_classes() {
        let config = br#"{
            "architectures": ["GptOssForCausalLM"],
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
        }"#;
        let compilation = compile_hf_config(config, OPTIONS).unwrap();
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
    fn allowlisted_mistral_config_infers_uniform_sliding() {
        let config = br#"{
            "architectures": ["MistralForCausalLM"],
            "num_hidden_layers": 2,
            "sliding_window": 4096,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "num_attention_heads": 32
        }"#;
        let compilation = compile_hf_config(config, OPTIONS).unwrap();
        assert_eq!(
            compilation.layer_inference,
            HfLayerInference::ArchitectureUniformSliding
        );
        assert_eq!(compilation.program.states[0].name, "swa");
        assert_eq!(compilation.program.states[0].layers, vec![0, 1]);
    }

    #[test]
    fn allowlisted_qwen2_config_with_explicit_full_contract_compiles() {
        let config = br#"{
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 2,
            "sliding_window": 4096,
            "use_sliding_window": false,
            "num_key_value_heads": 8,
            "head_dim": 64
        }"#;
        let compilation = compile_hf_config(config, OPTIONS).unwrap();
        assert_eq!(
            compilation.layer_inference,
            HfLayerInference::ArchitectureUniformFull
        );
        assert_eq!(compilation.program.states[0].name, "full");
        assert_eq!(compilation.program.states[0].layers, vec![0, 1]);

        let input = compile_hf_manager_plan(config, OPTIONS).unwrap();
        assert_eq!(input.classes.len(), 1);
        assert_eq!(input.classes[0].retention, crate::plan::RetentionKind::Full);
        assert_eq!(input.classes[0].window_tokens, None);
        let compiled = compile_plan(input).unwrap();
        let layout = compiled.layout_program().unwrap();
        assert_eq!(layout.classes[0].address, AddressProgram::AppendOnly);
    }

    #[test]
    fn missing_layer_semantics_fail_closed() {
        let config = br#"{
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 2,
            "sliding_window": 4096,
            "num_key_value_heads": 8,
            "head_dim": 64
        }"#;
        assert_eq!(
            compile_hf_config(config, OPTIONS),
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
        let config = br#"{
            "architectures": ["MistralForCausalLM"],
            "num_hidden_layers": 2,
            "sliding_window": 18,
            "num_key_value_heads": 8,
            "head_dim": 64
        }"#;
        let input = compile_hf_manager_plan(config, OPTIONS).unwrap();
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
