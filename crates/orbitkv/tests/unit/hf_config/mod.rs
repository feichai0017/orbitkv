use super::*;
use crate::attention_state::AttentionStateBackend;
use crate::plan::AddressProgram;
use crate::retention::{InferredRetention, analyze_state};

const OPTIONS: HfRetentionOptions = HfRetentionOptions {
    page_tokens: 16,
    kv_dtype_bytes: 2,
};
const HYBRID_FIXED_STATE_FIXTURE: &[u8] =
    include_bytes!("../../fixtures/hybrid-fixed-state/config.json");

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
    let expected = compile_hf_attention_state_input(HYBRID_FIXED_STATE_FIXTURE, OPTIONS).unwrap();
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
