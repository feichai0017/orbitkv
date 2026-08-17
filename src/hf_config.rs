use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    ApplicabilityReport, IntExpr, LayoutProgram, PlanError, Predicate, RetentionProgramInput,
    RetentionStateDecl, compile_retention_program,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HfRetentionOptions {
    pub page_tokens: u64,
    pub kv_dtype_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SglangUniformSwaOptions {
    pub maximum_running_requests: u64,
    pub chunked_prefill_tokens: u64,
    pub eviction_interval_tokens: u64,
    pub decode_headroom_tokens: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HfStatePlanOptions {
    pub retention: HfRetentionOptions,
    pub boundary_tokens: u64,
    pub sglang_uniform_swa: SglangUniformSwaOptions,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HfLayerInference {
    ExplicitLayerTypes,
    ArchitectureUniformSliding,
    FallbackAllFull,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SglangUniformSwaContract {
    pub schema: &'static str,
    pub plan_fingerprint: String,
    pub contract_fingerprint: String,
    pub config_sha256: String,
    pub architecture: String,
    pub num_hidden_layers: u64,
    pub num_key_value_heads: u64,
    pub head_dim: u64,
    pub window_tokens: u64,
    pub kernel_window_left: u64,
    pub page_tokens: u64,
    pub maximum_running_requests: u64,
    pub chunked_prefill_tokens: u64,
    pub eviction_interval_tokens: u64,
    pub decode_headroom_tokens: u64,
    pub per_request_resident_tokens: u64,
    pub global_staging_tokens: u64,
    pub minimum_pool_tokens: u64,
    pub scheduler_admission: &'static str,
    pub required_disabled_features: [&'static str; 5],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum HfSglangLowering {
    Enabled {
        kind: &'static str,
        contract: Box<SglangUniformSwaContract>,
    },
    Unsupported {
        reason: &'static str,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HfStatePlan {
    pub schema: &'static str,
    pub compilation: HfRetentionCompilation,
    pub layout: LayoutProgram,
    pub applicability: ApplicabilityReport,
    pub sglang_lowering: HfSglangLowering,
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
pub enum HfStatePlanError {
    #[error(transparent)]
    Config(#[from] HfConfigError),
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error("SGLang uniform-SWA {0} must be positive")]
    ZeroRuntimeOption(&'static str),
    #[error("integer overflow while deriving SGLang uniform-SWA {0}")]
    RuntimeArithmeticOverflow(&'static str),
}

/// Compiles a constrained Hugging Face model config into declarative
/// Retention IR.
///
/// Explicit `full_attention` and `sliding_attention` entries are grouped by
/// lifetime. A small architecture allowlist recognizes model families whose
/// config contract defines one uniform sliding window across every layer.
/// Other configs without `layer_types` safely fall back to unbounded Full
/// retention instead of guessing from a model-wide `sliding_window` field.
///
/// # Errors
///
/// Returns an error for invalid attention geometry, unsupported explicit layer
/// types, missing sliding-window geometry, or checked arithmetic failure.
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

/// Compiles one HF config into a portable state plan and an optional strict
/// `SGLang` lowering contract.
///
/// # Errors
///
/// Returns an error when HF retention compilation or plan synthesis fails.
pub fn compile_hf_state_plan(
    config_json: &[u8],
    options: HfStatePlanOptions,
) -> Result<HfStatePlan, HfStatePlanError> {
    validate_state_plan_options(options.sglang_uniform_swa)?;
    let compilation = compile_hf_config(config_json, options.retention)?;
    let plan = compile_retention_program(compilation.program.clone())?;
    let layout = plan.layout_program()?;
    let applicability = plan.applicability_report(options.boundary_tokens)?;
    let sglang_lowering =
        lower_uniform_swa_to_sglang(&compilation, &layout, options.sglang_uniform_swa)?;
    Ok(HfStatePlan {
        schema: "orbitkv.hf-state-plan.v2",
        compilation,
        layout,
        applicability,
        sglang_lowering,
    })
}

fn validate_state_plan_options(options: SglangUniformSwaOptions) -> Result<(), HfStatePlanError> {
    for (name, value) in [
        ("maximum_running_requests", options.maximum_running_requests),
        ("chunked_prefill_tokens", options.chunked_prefill_tokens),
        ("eviction_interval_tokens", options.eviction_interval_tokens),
        ("decode_headroom_tokens", options.decode_headroom_tokens),
    ] {
        if value == 0 {
            return Err(HfStatePlanError::ZeroRuntimeOption(name));
        }
    }
    Ok(())
}

fn lower_uniform_swa_to_sglang(
    compilation: &HfRetentionCompilation,
    layout: &LayoutProgram,
    options: SglangUniformSwaOptions,
) -> Result<HfSglangLowering, HfStatePlanError> {
    if compilation.layer_inference != HfLayerInference::ArchitectureUniformSliding {
        return Ok(HfSglangLowering::Unsupported {
            reason: "SGLang uniform-SWA lowering requires architecture-proven all-layer SWA",
        });
    }
    if layout.page_tokens != 1 {
        return Ok(HfSglangLowering::Unsupported {
            reason: "SGLang PureSWA allocator requires page_tokens=1",
        });
    }
    let Some(architecture) = compilation.architecture.as_deref() else {
        return Ok(HfSglangLowering::Unsupported {
            reason: "SGLang uniform-SWA lowering requires an explicit architecture",
        });
    };
    if architecture != "MistralForCausalLM" {
        return Ok(HfSglangLowering::Unsupported {
            reason: "architecture is not qualified for SGLang uniform-SWA lowering",
        });
    }
    let Some(class) = layout.classes.first() else {
        return Ok(HfSglangLowering::Unsupported {
            reason: "uniform-SWA lowering requires one generated class",
        });
    };
    if layout.classes.len() != 1 || class.name != "swa" {
        return Ok(HfSglangLowering::Unsupported {
            reason: "uniform-SWA lowering requires exactly one SWA class",
        });
    }
    let Some(window_tokens) =
        compilation
            .program
            .states
            .first()
            .and_then(|state| match &state.may_read {
                Predicate::LessThan {
                    rhs: IntExpr::Constant { value },
                    ..
                } => u64::try_from(*value).ok(),
                _ => None,
            })
    else {
        return Ok(HfSglangLowering::Unsupported {
            reason: "uniform-SWA lowering requires a constant positive window",
        });
    };
    let per_request_resident_tokens = window_tokens
        .checked_add(options.eviction_interval_tokens)
        .and_then(|value| value.checked_add(layout.page_tokens))
        .and_then(|value| value.checked_add(options.decode_headroom_tokens))
        .ok_or(HfStatePlanError::RuntimeArithmeticOverflow(
            "per-request resident tokens",
        ))?;
    let global_staging_tokens = options
        .chunked_prefill_tokens
        .checked_add(layout.page_tokens)
        .ok_or(HfStatePlanError::RuntimeArithmeticOverflow(
            "global staging tokens",
        ))?;
    let minimum_pool_tokens = per_request_resident_tokens
        .checked_mul(options.maximum_running_requests)
        .and_then(|value| value.checked_add(global_staging_tokens))
        .ok_or(HfStatePlanError::RuntimeArithmeticOverflow(
            "minimum pool tokens",
        ))?;
    let contract_fingerprint =
        uniform_swa_contract_fingerprint(compilation, layout, window_tokens, options);
    Ok(HfSglangLowering::Enabled {
        kind: "uniform_swa",
        contract: Box::new(SglangUniformSwaContract {
            schema: "orbitkv.sglang-uniform-swa-contract.v2",
            plan_fingerprint: layout.plan_fingerprint.clone(),
            contract_fingerprint,
            config_sha256: compilation.config_sha256.clone(),
            architecture: architecture.to_owned(),
            num_hidden_layers: compilation.num_hidden_layers,
            num_key_value_heads: compilation.num_key_value_heads,
            head_dim: compilation.head_dim,
            window_tokens,
            kernel_window_left: window_tokens - 1,
            page_tokens: layout.page_tokens,
            maximum_running_requests: options.maximum_running_requests,
            chunked_prefill_tokens: options.chunked_prefill_tokens,
            eviction_interval_tokens: options.eviction_interval_tokens,
            decode_headroom_tokens: options.decode_headroom_tokens,
            per_request_resident_tokens,
            global_staging_tokens,
            minimum_pool_tokens,
            scheduler_admission: "pure_swa_live_state",
            required_disabled_features: [
                "radix_cache",
                "overlap_schedule",
                "speculative_decoding",
                "disaggregation",
                "cuda_graph",
            ],
        }),
    })
}

fn uniform_swa_contract_fingerprint(
    compilation: &HfRetentionCompilation,
    layout: &LayoutProgram,
    window_tokens: u64,
    options: SglangUniformSwaOptions,
) -> String {
    let mut hash = Sha256::new();
    for value in [
        layout.plan_fingerprint.as_bytes(),
        compilation.config_sha256.as_bytes(),
        compilation.architecture.as_deref().unwrap_or("").as_bytes(),
    ] {
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value);
    }
    for value in [
        compilation.num_hidden_layers,
        compilation.num_key_value_heads,
        compilation.head_dim,
        window_tokens,
        layout.page_tokens,
        options.maximum_running_requests,
        options.chunked_prefill_tokens,
        options.eviction_interval_tokens,
        options.decode_headroom_tokens,
    ] {
        hash.update(value.to_le_bytes());
    }
    format!("sha256:{:x}", hash.finalize())
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
        return Ok((
            HfLayerInference::FallbackAllFull,
            layer_range(config.num_hidden_layers)?,
            Vec::new(),
        ));
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
    config.use_sliding_window != Some(false)
        && config.sliding_window.is_some_and(|window| window > 0)
        && matches!(
            config.architectures.first().map(String::as_str),
            Some("MistralForCausalLM")
        )
}

fn layer_range(count: u64) -> Result<Vec<u32>, HfConfigError> {
    (0..count)
        .map(|index| u32::try_from(index).map_err(|_| HfConfigError::LayerIndexOutOfRange))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AddressProgram, InferredRetention, analyze_state, compile_retention_program};

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
        let compilation = compile_hf_config(
            config,
            HfRetentionOptions {
                page_tokens: 16,
                kv_dtype_bytes: 2,
            },
        )
        .unwrap();
        assert_eq!(
            compilation.layer_inference,
            HfLayerInference::ExplicitLayerTypes
        );
        assert_eq!(compilation.bytes_per_token_per_layer, 2048);
        assert_eq!(compilation.program.states[0].name, "full");
        assert_eq!(compilation.program.states[0].layers, vec![1, 3]);
        assert_eq!(compilation.program.states[1].name, "swa");
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
    fn missing_layer_types_falls_back_to_full() {
        let config = br#"{
            "num_hidden_layers": 2,
            "sliding_window": 4096,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "num_attention_heads": 32
        }"#;
        let compilation = compile_hf_config(
            config,
            HfRetentionOptions {
                page_tokens: 16,
                kv_dtype_bytes: 2,
            },
        )
        .unwrap();
        assert_eq!(
            compilation.layer_inference,
            HfLayerInference::FallbackAllFull
        );
        assert_eq!(compilation.head_dim, 128);
        assert_eq!(compilation.program.states.len(), 1);
        assert_eq!(compilation.program.states[0].layers, vec![0, 1]);
        assert_eq!(compilation.program.states[0].may_read, Predicate::True);
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
        let compilation = compile_hf_config(
            config,
            HfRetentionOptions {
                page_tokens: 16,
                kv_dtype_bytes: 2,
            },
        )
        .unwrap();
        assert_eq!(
            compilation.layer_inference,
            HfLayerInference::ArchitectureUniformSliding
        );
        assert_eq!(compilation.program.states.len(), 1);
        assert_eq!(compilation.program.states[0].name, "swa");
        assert_eq!(compilation.program.states[0].layers, vec![0, 1]);
        assert_eq!(
            analyze_state(&compilation.program.states[0])
                .unwrap()
                .inferred,
            InferredRetention::FixedWindow {
                window_tokens: 4096
            }
        );
    }

    #[test]
    fn uniform_swa_state_plan_enables_only_qualified_page_one_lowering() {
        let config = br#"{
            "architectures": ["MistralForCausalLM"],
            "num_hidden_layers": 2,
            "sliding_window": 4096,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "num_attention_heads": 32
        }"#;
        let enabled = compile_hf_state_plan(
            config,
            HfStatePlanOptions {
                retention: HfRetentionOptions {
                    page_tokens: 1,
                    kv_dtype_bytes: 2,
                },
                boundary_tokens: 8192,
                sglang_uniform_swa: SglangUniformSwaOptions {
                    maximum_running_requests: 4,
                    chunked_prefill_tokens: 2048,
                    eviction_interval_tokens: 128,
                    decode_headroom_tokens: 32,
                },
            },
        )
        .unwrap();
        let HfSglangLowering::Enabled { kind, contract } = enabled.sglang_lowering else {
            panic!("qualified Mistral plan must lower");
        };
        assert_eq!(kind, "uniform_swa");
        assert_eq!(contract.window_tokens, 4096);
        assert_eq!(contract.kernel_window_left, 4095);
        assert_eq!(contract.page_tokens, 1);
        assert_eq!(contract.maximum_running_requests, 4);
        assert_eq!(contract.per_request_resident_tokens, 4257);
        assert_eq!(contract.global_staging_tokens, 2049);
        assert_eq!(contract.minimum_pool_tokens, 19_077);
        assert!(contract.contract_fingerprint.starts_with("sha256:"));
        assert!(contract.config_sha256.starts_with("sha256:"));

        let unsupported = compile_hf_state_plan(
            config,
            HfStatePlanOptions {
                retention: HfRetentionOptions {
                    page_tokens: 16,
                    kv_dtype_bytes: 2,
                },
                boundary_tokens: 8192,
                sglang_uniform_swa: SglangUniformSwaOptions {
                    maximum_running_requests: 4,
                    chunked_prefill_tokens: 2048,
                    eviction_interval_tokens: 128,
                    decode_headroom_tokens: 32,
                },
            },
        )
        .unwrap();
        assert_eq!(
            unsupported.sglang_lowering,
            HfSglangLowering::Unsupported {
                reason: "SGLang PureSWA allocator requires page_tokens=1"
            }
        );
    }

    #[test]
    fn uniform_swa_state_plan_rejects_zero_runtime_budget() {
        let config = br#"{
            "architectures": ["MistralForCausalLM"],
            "num_hidden_layers": 2,
            "sliding_window": 4096,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "num_attention_heads": 32
        }"#;
        assert_eq!(
            compile_hf_state_plan(
                config,
                HfStatePlanOptions {
                    retention: HfRetentionOptions {
                        page_tokens: 1,
                        kv_dtype_bytes: 2,
                    },
                    boundary_tokens: 8192,
                    sglang_uniform_swa: SglangUniformSwaOptions {
                        maximum_running_requests: 0,
                        chunked_prefill_tokens: 2048,
                        eviction_interval_tokens: 128,
                        decode_headroom_tokens: 32,
                    },
                },
            ),
            Err(HfStatePlanError::ZeroRuntimeOption(
                "maximum_running_requests"
            ))
        );
    }

    #[test]
    fn non_allowlisted_window_field_still_falls_back_to_full() {
        let config = br#"{
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 2,
            "sliding_window": 131072,
            "use_sliding_window": false,
            "num_key_value_heads": 4,
            "hidden_size": 3584,
            "num_attention_heads": 28
        }"#;
        let compilation = compile_hf_config(
            config,
            HfRetentionOptions {
                page_tokens: 16,
                kv_dtype_bytes: 2,
            },
        )
        .unwrap();
        assert_eq!(
            compilation.layer_inference,
            HfLayerInference::FallbackAllFull
        );
        assert_eq!(compilation.program.states[0].may_read, Predicate::True);
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
            compile_hf_config(
                config,
                HfRetentionOptions {
                    page_tokens: 16,
                    kv_dtype_bytes: 2
                }
            ),
            Err(HfConfigError::UnsupportedLayerType {
                layer: 0,
                layer_type
            }) if layer_type == "mamba"
        ));
    }

    #[test]
    fn layer_type_count_must_match_geometry() {
        let config = br#"{
            "num_hidden_layers": 2,
            "layer_types": ["full_attention"],
            "num_key_value_heads": 8,
            "head_dim": 64
        }"#;
        assert_eq!(
            compile_hf_config(
                config,
                HfRetentionOptions {
                    page_tokens: 16,
                    kv_dtype_bytes: 2
                }
            ),
            Err(HfConfigError::LayerTypeCountMismatch {
                expected: 2,
                actual: 1
            })
        );
    }
}
