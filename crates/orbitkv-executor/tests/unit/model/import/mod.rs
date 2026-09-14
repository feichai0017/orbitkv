use super::super::DecoderWeightFormat;
use super::*;

#[test]
fn imports_sandwich_norm_semantics_from_architecture() {
    let config = DecoderConfig::from_json(
        br#"{
            "model_type": "gemma3_text",
            "num_hidden_layers": 6,
            "hidden_size": 640,
            "intermediate_size": 2048,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "vocab_size": 262144,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "rms_norm_eps": 0.000001,
            "tie_word_embeddings": true,
            "attention_bias": false,
            "hidden_activation": "gelu_pytorch_tanh",
            "query_pre_attn_scalar": 256,
            "layer_types": [
                "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention"
            ],
            "attn_logit_softcapping": null,
            "final_logit_softcapping": null,
            "rope_scaling": null
        }"#,
    )
    .unwrap();
    assert_eq!(config.query_heads * config.head_dim, 1024);
    assert_eq!(config.hidden_size, 640);
    assert!((config.embedding_scale - 25.25).abs() < f32::EPSILON);
    assert_eq!(config.activation, DecoderActivation::GeluTanh);
    assert_eq!(config.block_layout, DecoderBlockLayout::SandwichNorm);
    assert_eq!(config.norm_weights, DecoderNormWeights::UnitOffset);
    assert_eq!(config.local_rope_theta, Some(10_000.0));
    assert!((config.attention_softmax_scale - 1.0 / 16.0).abs() < f64::EPSILON);
    assert_eq!(
        config.layer_kinds.as_deref(),
        Some(
            &[
                DecoderLayerKind::Sliding,
                DecoderLayerKind::Sliding,
                DecoderLayerKind::Sliding,
                DecoderLayerKind::Sliding,
                DecoderLayerKind::Sliding,
                DecoderLayerKind::Full,
            ][..]
        )
    );
}

#[test]
fn config_accepts_non_hidden_query_width_and_rejects_unimplemented_semantics() {
    let hybrid = br#"{
        "model_type": "gemma3_text",
        "num_hidden_layers": 2, "hidden_size": 640,
        "intermediate_size": 2048, "num_attention_heads": 4,
        "num_key_value_heads": 1, "head_dim": 256,
        "vocab_size": 262144, "rope_theta": 1000000.0,
        "rope_local_base_freq": 10000.0, "rms_norm_eps": 0.000001,
        "tie_word_embeddings": true, "hidden_activation": "gelu_pytorch_tanh",
        "query_pre_attn_scalar": 256,
        "layer_types": ["sliding_attention", "full_attention"]
    }"#;
    assert!(DecoderConfig::from_json(hybrid).is_ok());

    let unsupported = String::from_utf8(hybrid.to_vec()).unwrap().replace(
        "\"query_pre_attn_scalar\": 256",
        "\"query_pre_attn_scalar\": 256, \"final_logit_softcapping\": 30.0",
    );
    assert!(matches!(
        DecoderConfig::from_json(unsupported.as_bytes()),
        Err(DecoderError::InvalidGeometry(
            "unsupported decoder semantics"
        ))
    ));
}

#[test]
fn omitted_tied_embedding_field_uses_architecture_default() {
    let config = DecoderConfig::from_json(
        br#"{
            "model_type": "qwen2",
            "num_hidden_layers": 1, "hidden_size": 128,
            "intermediate_size": 256, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 64,
            "vocab_size": 320, "rope_theta": 10000.0,
            "rms_norm_eps": 0.000001, "hidden_act": "silu"
        }"#,
    )
    .unwrap();
    assert!(!config.tied_embeddings);
    assert!((config.attention_softmax_scale - 0.125).abs() < f64::EPSILON);
}

#[test]
fn imports_nested_hybrid_semantics_without_checkpoint_path_dispatch() {
    let config = DecoderConfig::from_json(
        br#"{
            "model_type": "qwen3_5",
            "tie_word_embeddings": false,
            "quantization_config": {
                "quant_method": "fp8", "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128]
            },
            "text_config": {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 4, "hidden_size": 1024,
                "intermediate_size": 3584, "num_attention_heads": 8,
                "num_key_value_heads": 2, "head_dim": 256,
                "attn_output_gate": true,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 248320, "rms_norm_eps": 0.000001,
                "hidden_act": "silu",
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000000.0,
                    "partial_rotary_factor": 0.25,
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10]
                },
                "layer_types": [
                    "linear_attention", "linear_attention",
                    "linear_attention", "full_attention"
                ]
            }
        }"#,
    )
    .unwrap();
    assert_eq!(config.tensor_prefix, "model.language_model");
    assert!((config.rope_theta - 10_000_000.0).abs() < f32::EPSILON);
    assert_eq!(config.rotary_dimensions, 64);
    assert!(config.attention_output_gate);
    assert_eq!(config.norm_weights, DecoderNormWeights::UnitOffset);
    assert_eq!(
        config.gated_delta,
        Some(GatedDeltaConfig {
            key_heads: 16,
            value_heads: 16,
            key_width: 128,
            value_width: 128,
            convolution_kernel_width: 4,
        })
    );
    assert!(!config.tied_embeddings);
    assert_eq!(
        config.layer_kinds.as_deref(),
        Some(
            &[
                DecoderLayerKind::Linear,
                DecoderLayerKind::Linear,
                DecoderLayerKind::Linear,
                DecoderLayerKind::Full,
            ][..]
        )
    );
    assert_eq!(
        config.weight_format,
        DecoderWeightFormat::Fp8E4M3Block {
            rows: 128,
            columns: 128,
        }
    );
    assert!(config.require_executable().is_ok());
}

#[test]
fn nested_rope_contract_rejects_unknown_or_inconsistent_layouts() {
    let base = r#"{
        "model_type": "qwen3_5",
        "text_config": {
                "model_type": "qwen3_5_text",
            "num_hidden_layers": 1, "hidden_size": 128,
            "intermediate_size": 256, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 64,
            "vocab_size": 320, "rms_norm_eps": 0.000001,
            "hidden_act": "silu", "layer_types": ["full_attention"],
            "rope_parameters": {
                "rope_type": "TYPE", "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_interleaved": true, "mrope_section": SECTION
            }
        }
    }"#;
    let unknown = base
        .replace("\"TYPE\"", "\"yarn\"")
        .replace("SECTION", "[6, 5, 5]");
    assert!(matches!(
        DecoderConfig::from_json(unknown.as_bytes()),
        Err(DecoderError::InvalidGeometry("unsupported RoPE type"))
    ));
    let inconsistent = base
        .replace("\"TYPE\"", "\"default\"")
        .replace("SECTION", "[5, 5, 5]");
    assert!(matches!(
        DecoderConfig::from_json(inconsistent.as_bytes()),
        Err(DecoderError::InvalidGeometry("MRoPE sections"))
    ));
}

#[test]
fn nested_dense_decoder_inherits_top_level_embedding_policy() {
    let config = DecoderConfig::from_json(
        br#"{
            "model_type": "qwen3_5",
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 1, "hidden_size": 128,
                "intermediate_size": 256, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 64,
                "vocab_size": 320, "rope_theta": 10000.0,
                "rms_norm_eps": 0.000001, "hidden_act": "silu",
                "layer_types": ["full_attention"]
            }
        }"#,
    )
    .unwrap();
    assert!(!config.tied_embeddings);
    assert_eq!(config.tensor_prefix, "model.language_model");
    assert_eq!(config.rotary_dimensions, 64);
    assert!(config.require_executable().is_ok());
}

fn dense_document() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"],
        "num_hidden_layers": 1, "hidden_size": 128, "intermediate_size": 256,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
        "vocab_size": 320, "rope_theta": 10000.0, "rms_norm_eps": 0.000_001,
        "hidden_act": "silu"
    })
}

fn import(document: &serde_json::Value) -> Result<DecoderConfig, DecoderError> {
    DecoderConfig::from_json(&serde_json::to_vec(document).unwrap())
}

#[test]
fn rejects_missing_unknown_or_contradictory_architecture_metadata() {
    let mut missing = dense_document();
    missing.as_object_mut().unwrap().remove("model_type");
    let mut unknown = dense_document();
    unknown["model_type"] = "future_decoder".into();
    let mut contradictory = dense_document();
    contradictory["architectures"] = serde_json::json!(["Gemma3ForCausalLM"]);
    let mut nested = dense_document();
    nested["text_config"] = dense_document();
    for document in [missing, unknown, contradictory, nested] {
        assert!(matches!(
            import(&document),
            Err(DecoderError::UnsupportedCheckpoint(_))
        ));
    }
}

#[test]
fn unrelated_rope_field_cannot_select_another_norm_architecture() {
    let mut document = dense_document();
    document["rope_local_base_freq"] = 10000.0.into();
    assert!(matches!(
        import(&document),
        Err(DecoderError::InvalidGeometry(_))
    ));
}

#[test]
fn checkpoint_name_and_source_do_not_change_normalized_semantics() {
    let mut document = dense_document();
    let expected = import(&document).unwrap();
    document["_name_or_path"] = "another-org/another-size".into();
    document["transformers_version"] = "future-version".into();
    assert_eq!(import(&document).unwrap(), expected);
}

#[test]
fn normalization_is_not_limited_to_cuda_attention_head_sizes() {
    let mut document = dense_document();
    document["head_dim"] = 32.into();
    assert_eq!(import(&document).unwrap().head_dim, 32);
    document.as_object_mut().unwrap().remove("head_dim");
    document["hidden_size"] = 127.into();
    assert!(matches!(
        import(&document),
        Err(DecoderError::InvalidGeometry(_))
    ));
}

#[test]
fn nested_text_architecture_must_match_the_envelope() {
    let document = serde_json::json!({"model_type": "qwen3_5", "text_config": dense_document()});
    assert!(matches!(
        import(&document),
        Err(DecoderError::UnsupportedCheckpoint(_))
    ));
}
