use super::*;
use orbitkv::{
    AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage, CacheSharingPolicy,
    EngineAppendIntent, EngineRequestId, RuntimeSession, compile_attention_state_plan,
    compile_plan, compile_runtime_manifest,
    kv_manager::{BackendArenaRegistration, CanonicalKvManager, ManagerConfig},
    plan::RetentionKind,
};

#[test]
fn compile_config_requires_distinct_decode_and_prefill_ranges() {
    let valid = DecoderCompileConfig {
        maximum_query_tokens: 32,
        representative_prefill_tokens: 8,
        maximum_batch_size: 4,
        maximum_context_pages: 64,
        representative_context_pages: 8,
        search_graphs: 2,
        search_seed: 1,
    };
    assert!(valid.validate().is_ok());
    assert!(
        DecoderCompileConfig {
            maximum_query_tokens: 1,
            ..valid
        }
        .validate()
        .is_err()
    );
    assert!(
        DecoderCompileConfig {
            search_graphs: 1,
            ..valid
        }
        .validate()
        .is_err()
    );
}

#[test]
fn decoder_artifact_identity_covers_plan_arena_and_compile_geometry() {
    let config = test_config(4);
    let plan = hybrid_executor_plan();
    let arenas = hybrid_arenas();
    let compile = DecoderCompileConfig {
        maximum_query_tokens: 32,
        representative_prefill_tokens: 8,
        maximum_batch_size: 4,
        maximum_context_pages: 64,
        representative_context_pages: 8,
        search_graphs: 2,
        search_seed: 1,
    };
    let identity = decoder_artifact_identity(
        &config,
        &plan,
        &arenas,
        DecoderWeightFeatures::default(),
        compile,
        "facts-a",
    )
    .unwrap();
    let mut changed_arena = arenas;
    changed_arena[1].page_count += 1;
    let arena_identity = decoder_artifact_identity(
        &config,
        &plan,
        &changed_arena,
        DecoderWeightFeatures::default(),
        compile,
        "facts-a",
    )
    .unwrap();
    let compile_identity = decoder_artifact_identity(
        &config,
        &plan,
        &arenas,
        DecoderWeightFeatures::default(),
        DecoderCompileConfig {
            maximum_batch_size: 2,
            ..compile
        },
        "facts-a",
    )
    .unwrap();
    let facts_identity = decoder_artifact_identity(
        &config,
        &plan,
        &arenas,
        DecoderWeightFeatures::default(),
        compile,
        "facts-b",
    )
    .unwrap();

    assert_ne!(identity, arena_identity);
    assert_ne!(identity, compile_identity);
    assert_ne!(identity, facts_identity);
}

#[test]
fn step_validation_enforces_compiled_capacities() {
    let compile = DecoderCompileConfig {
        maximum_query_tokens: 4,
        representative_prefill_tokens: 2,
        maximum_batch_size: 1,
        maximum_context_pages: 2,
        representative_context_pages: 1,
        search_graphs: 2,
        search_seed: 1,
    };
    let attention = crate::AttentionBatch {
        class_id: 0,
        query_indptr: vec![0, 2].into_boxed_slice(),
        page_indptr: vec![0, 1].into_boxed_slice(),
        page_indices: vec![0].into_boxed_slice(),
        last_page_len: vec![2].into_boxed_slice(),
    };
    let classes = [DecoderClassDimensions {
        class_id: 0,
        context_pages: sym("c_0"),
        backend_base_index: 0,
        page_count: 4,
        cache_slots: 64,
    }];
    let class_steps = [DecoderClassStep {
        class_id: 0,
        write_slots: &[0, 1],
        attention: &attention,
    }];
    let valid = DecoderStep {
        tokens: &[1, 2],
        positions: &[0, 1],
        classes: &class_steps,
    };
    assert!(validate_step(valid, compile, &classes, 16, 32).is_ok());
    assert!(
        validate_step(
            DecoderStep {
                tokens: &[1, 2],
                positions: &[0],
                ..valid
            },
            compile,
            &classes,
            16,
            32,
        )
        .is_err()
    );
}

#[test]
fn decode_capture_signature_freezes_planner_geometry_only() {
    let first_attention = crate::AttentionBatch {
        class_id: 0,
        query_indptr: vec![0, 1].into_boxed_slice(),
        page_indptr: vec![0, 2].into_boxed_slice(),
        page_indices: vec![3, 7].into_boxed_slice(),
        last_page_len: vec![4].into_boxed_slice(),
    };
    let second_attention = crate::AttentionBatch {
        class_id: 0,
        query_indptr: vec![0, 1].into_boxed_slice(),
        page_indptr: vec![0, 2].into_boxed_slice(),
        page_indices: vec![5, 9].into_boxed_slice(),
        last_page_len: vec![8].into_boxed_slice(),
    };
    let first_classes = [DecoderClassStep {
        class_id: 0,
        write_slots: &[16],
        attention: &first_attention,
    }];
    let first = DecodeCaptureSignature::from_step(DecoderStep {
        tokens: &[1],
        positions: &[16],
        classes: &first_classes,
    })
    .unwrap();
    let second_classes = [DecoderClassStep {
        class_id: 0,
        write_slots: &[17],
        attention: &second_attention,
    }];
    let second = DecodeCaptureSignature::from_step(DecoderStep {
        tokens: &[2],
        positions: &[17],
        classes: &second_classes,
    })
    .unwrap();
    assert_eq!(first, second);

    let changed_indptr = crate::AttentionBatch {
        page_indptr: vec![0, 1].into_boxed_slice(),
        page_indices: vec![5].into_boxed_slice(),
        ..second_attention
    };
    let changed_classes = [DecoderClassStep {
        class_id: 0,
        write_slots: &[17],
        attention: &changed_indptr,
    }];
    let changed = DecodeCaptureSignature::from_step(DecoderStep {
        tokens: &[2],
        positions: &[17],
        classes: &changed_classes,
    })
    .unwrap();
    assert_ne!(first, changed);
}

#[test]
fn decode_capture_signature_covers_every_attention_class() {
    let full = crate::AttentionBatch {
        class_id: 0,
        query_indptr: vec![0, 1].into_boxed_slice(),
        page_indptr: vec![0, 2].into_boxed_slice(),
        page_indices: vec![2, 3].into_boxed_slice(),
        last_page_len: vec![1].into_boxed_slice(),
    };
    let sliding = crate::AttentionBatch {
        class_id: 1,
        query_indptr: vec![0, 1].into_boxed_slice(),
        page_indptr: vec![0, 1].into_boxed_slice(),
        page_indices: vec![9].into_boxed_slice(),
        last_page_len: vec![1].into_boxed_slice(),
    };
    let classes = [
        DecoderClassStep {
            class_id: 0,
            write_slots: &[48],
            attention: &full,
        },
        DecoderClassStep {
            class_id: 1,
            write_slots: &[144],
            attention: &sliding,
        },
    ];
    let first = DecodeCaptureSignature::from_step(DecoderStep {
        tokens: &[1],
        positions: &[16],
        classes: &classes,
    })
    .unwrap();

    let changed_sliding = crate::AttentionBatch {
        class_id: sliding.class_id,
        query_indptr: sliding.query_indptr.clone(),
        page_indptr: vec![0, 2].into_boxed_slice(),
        page_indices: vec![8, 9].into_boxed_slice(),
        last_page_len: sliding.last_page_len.clone(),
    };
    let changed_classes = [
        classes[0],
        DecoderClassStep {
            attention: &changed_sliding,
            ..classes[1]
        },
    ];
    let changed = DecodeCaptureSignature::from_step(DecoderStep {
        tokens: &[2],
        positions: &[17],
        classes: &changed_classes,
    })
    .unwrap();
    assert_ne!(first, changed);
}

#[test]
fn decode_capture_signature_rejects_prefill() {
    let attention = crate::AttentionBatch {
        class_id: 0,
        query_indptr: vec![0, 2].into_boxed_slice(),
        page_indptr: vec![0, 1].into_boxed_slice(),
        page_indices: vec![0].into_boxed_slice(),
        last_page_len: vec![2].into_boxed_slice(),
    };
    assert!(matches!(
        DecodeCaptureSignature::from_step(DecoderStep {
            tokens: &[1, 2],
            positions: &[0, 1],
            classes: &[DecoderClassStep {
                class_id: 0,
                write_slots: &[0, 1],
                attention: &attention,
            }],
        }),
        Err(DecoderError::CaptureRequiresDecode)
    ));
}

#[test]
fn decode_capture_signature_accepts_one_token_per_request() {
    let attention = crate::AttentionBatch {
        class_id: 0,
        query_indptr: vec![0, 1, 2].into_boxed_slice(),
        page_indptr: vec![0, 2, 5].into_boxed_slice(),
        page_indices: vec![2, 3, 7, 8, 9].into_boxed_slice(),
        last_page_len: vec![4, 8].into_boxed_slice(),
    };
    let classes = [DecoderClassStep {
        class_id: 0,
        write_slots: &[48, 144],
        attention: &attention,
    }];
    let signature = DecodeCaptureSignature::from_step(DecoderStep {
        tokens: &[1, 2],
        positions: &[16, 32],
        classes: &classes,
    })
    .unwrap();

    assert_eq!(signature.query_tokens, 2);
    assert_eq!(signature.batch_size, 2);
}

fn test_config(layers: usize) -> DecoderConfig {
    DecoderConfig {
        layers,
        hidden_size: 128,
        intermediate_size: 256,
        query_heads: 2,
        kv_heads: 1,
        head_dim: 64,
        vocabulary_size: 320,
        tensor_prefix: "model".into(),
        rope_theta: 10_000.0,
        rotary_dimensions: 64,
        rms_epsilon: 1e-6,
        tied_embeddings: false,
        embedding_scale: 1.0,
        activation: DecoderActivation::Silu,
        block_layout: DecoderBlockLayout::PreNorm,
        norm_weights: DecoderNormWeights::Direct,
        local_rope_theta: None,
        attention_softmax_scale: 0.0,
        attention_output_gate: false,
        layer_kinds: None,
        gated_delta: None,
        weight_format: DecoderWeightFormat::Float,
    }
}

#[test]
fn parses_structural_decoder_semantics_without_model_name_dispatch() {
    let config = DecoderConfig::from_json(
        br#"{
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
fn omitted_tied_embedding_field_uses_the_hf_default() {
    let config = DecoderConfig::from_json(
        br#"{
            "num_hidden_layers": 1, "hidden_size": 128,
            "intermediate_size": 256, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 64,
            "vocab_size": 320, "rope_theta": 10000.0,
            "rms_norm_eps": 0.000001, "hidden_act": "silu"
        }"#,
    )
    .unwrap();
    assert!(config.tied_embeddings);
    assert!((config.attention_softmax_scale - 0.125).abs() < f64::EPSILON);
}

#[test]
fn parses_nested_hybrid_decoder_without_checkpoint_name_dispatch() {
    let config = DecoderConfig::from_json(
        br#"{
            "tie_word_embeddings": false,
            "quantization_config": {
                "quant_method": "fp8", "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128]
            },
            "text_config": {
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
    assert!(matches!(
        config.require_executable(),
        Err(DecoderError::UnsupportedExecution(
            "linear-attention state execution"
        ))
    ));
}

#[test]
fn nested_rope_contract_rejects_unknown_or_inconsistent_layouts() {
    let base = r#"{
        "text_config": {
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
            "tie_word_embeddings": false,
            "text_config": {
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

#[test]
#[ignore = "requires ORBITKV_MODEL_DIR containing an external hybrid-state checkpoint"]
fn external_hybrid_checkpoint_reaches_the_explicit_execution_gate() {
    let directory = std::env::var_os("ORBITKV_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .expect("ORBITKV_MODEL_DIR is required");
    let bytes = std::fs::read(directory.join("config.json")).unwrap();
    let config = DecoderConfig::from_json(&bytes).unwrap();
    let mut weight_files = std::fs::read_dir(&directory)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .collect::<Vec<_>>();
    weight_files.sort();
    assert!(!weight_files.is_empty());
    inspect_weight_features(&weight_files, &config).unwrap();
    assert_eq!(config.tensor_prefix, "model.language_model");
    assert!(config.rotary_dimensions < config.head_dim);
    assert!(
        config
            .layer_kinds
            .as_deref()
            .is_some_and(|layers| layers.contains(&DecoderLayerKind::Linear))
    );
    assert!(matches!(
        config.require_executable(),
        Err(DecoderError::UnsupportedExecution(
            "linear-attention state execution"
        ))
    ));
}

#[test]
fn graph_rejects_manifest_layer_semantics_that_disagree_with_model_config() {
    let mut config = test_config(4);
    config.layer_kinds = Some(
        vec![
            DecoderLayerKind::Sliding,
            DecoderLayerKind::Full,
            DecoderLayerKind::Sliding,
            DecoderLayerKind::Full,
        ]
        .into_boxed_slice(),
    );
    assert!(matches!(
        DecoderGraph::build(
            &mut Graph::default(),
            &config,
            DecoderWeightFeatures::default(),
            &hybrid_executor_plan(),
            &hybrid_arenas(),
        ),
        Err(DecoderError::UnsupportedPlan)
    ));
}

fn hybrid_executor_plan() -> ExecutorPlan {
    crate::test_executor_plan(
        "hybrid",
        16,
        vec![
            crate::AttentionClass {
                class_id: 0,
                name: "global".into(),
                layers: vec![0, 2].into_boxed_slice(),
                page_tokens: 16,
                key_bytes_per_token_per_layer: 128,
                value_bytes_per_token_per_layer: 128,
                visibility: crate::AttentionVisibility::Full,
            },
            crate::AttentionClass {
                class_id: 1,
                name: "local".into(),
                layers: vec![1, 3].into_boxed_slice(),
                page_tokens: 16,
                key_bytes_per_token_per_layer: 128,
                value_bytes_per_token_per_layer: 128,
                visibility: crate::AttentionVisibility::Sliding { window_tokens: 64 },
            },
        ],
    )
}

fn hybrid_arenas() -> [ExecutorArena; 2] {
    [
        ExecutorArena {
            engine_epoch: 1,
            pool_epoch: 1,
            pool_id: 1,
            class_id: 0,
            backend_domain: 10,
            first_page_id: 1,
            page_count: 8,
            backend_base_index: 0,
        },
        ExecutorArena {
            engine_epoch: 1,
            pool_epoch: 1,
            pool_id: 2,
            class_id: 1,
            backend_domain: 11,
            first_page_id: 9,
            page_count: 4,
            backend_base_index: 0,
        },
    ]
}

fn hybrid_attention_input() -> AttentionStatePlanInput {
    AttentionStatePlanInput {
        page_tokens: 16,
        states: vec![
            AttentionStateSpec {
                name: "global".into(),
                layers: vec![0, 2],
                storage: AttentionStateStorage::TokenKv {
                    key_bytes_per_token_per_layer: 128,
                    value_bytes_per_token_per_layer: 128,
                    retention: RetentionKind::Full,
                    window_tokens: None,
                },
            },
            AttentionStateSpec {
                name: "local".into(),
                layers: vec![1, 3],
                storage: AttentionStateStorage::TokenKv {
                    key_bytes_per_token_per_layer: 128,
                    value_bytes_per_token_per_layer: 128,
                    retention: RetentionKind::Sliding,
                    window_tokens: Some(18),
                },
            },
        ],
    }
}

fn hybrid_registrations() -> [BackendArenaRegistration; 2] {
    [
        BackendArenaRegistration {
            pool_id: 1,
            class_id: 0,
            backend_domain: 10,
            page_count: 8,
            reserved: 0,
            backend_base_index: 0,
        },
        BackendArenaRegistration {
            pool_id: 2,
            class_id: 1,
            backend_domain: 11,
            page_count: 4,
            reserved: 0,
            backend_base_index: 8,
        },
    ]
}

fn hybrid_runtime_session(
    plan: &orbitkv::CompiledKvPlan,
    registrations: &[BackendArenaRegistration],
) -> RuntimeSession {
    RuntimeSession::new(
        CanonicalKvManager::new(
            plan,
            ManagerConfig {
                maximum_requests: 1,
                maximum_operations: 2,
                maximum_prefixes: 1,
                maximum_reclamations: 12,
                maximum_step_tokens: 64,
            },
            registrations,
        )
        .unwrap(),
        CacheSharingPolicy::RequestPrivate,
    )
}

#[test]
fn graph_builds_layers_from_independent_full_and_sliding_classes() {
    let mut graph = Graph::default();
    let plan = hybrid_executor_plan();
    let decoder = DecoderGraph::build(
        &mut graph,
        &test_config(4),
        DecoderWeightFeatures::default(),
        &plan,
        &hybrid_arenas(),
    )
    .unwrap();

    assert_eq!(decoder.inputs.classes.len(), 2);
    assert_eq!(decoder.inputs.classes[0].class_id, 0);
    assert_eq!(decoder.inputs.classes[1].class_id, 1);
    assert_ne!(
        decoder.class_dimensions[0].context_pages,
        decoder.class_dimensions[1].context_pages
    );
    assert_eq!(decoder.class_dimensions[0].cache_slots, 128);
    assert_eq!(decoder.class_dimensions[1].cache_slots, 64);
    assert_eq!(
        decoder
            .cache_bindings(&plan)
            .unwrap()
            .iter()
            .map(|binding| binding.class_id)
            .collect::<Vec<_>>(),
        vec![0, 1, 0, 1]
    );
}

#[test]
fn graph_rejects_duplicate_or_missing_layer_ownership() {
    let mut duplicate = hybrid_executor_plan();
    duplicate.classes[1].layers[0] = 0;
    assert!(matches!(
        DecoderGraph::build(
            &mut Graph::default(),
            &test_config(4),
            DecoderWeightFeatures::default(),
            &duplicate,
            &hybrid_arenas(),
        ),
        Err(DecoderError::UnsupportedPlan)
    ));

    let mut missing = hybrid_executor_plan();
    missing.classes[1].layers = vec![1].into_boxed_slice();
    assert!(matches!(
        DecoderGraph::build(
            &mut Graph::default(),
            &test_config(4),
            DecoderWeightFeatures::default(),
            &missing,
            &hybrid_arenas(),
        ),
        Err(DecoderError::UnsupportedPlan)
    ));
}

#[test]
fn multi_class_step_validates_each_absolute_arena() {
    let compile = DecoderCompileConfig {
        maximum_query_tokens: 4,
        representative_prefill_tokens: 2,
        maximum_batch_size: 1,
        maximum_context_pages: 4,
        representative_context_pages: 1,
        search_graphs: 2,
        search_seed: 1,
    };
    let full = crate::AttentionBatch {
        class_id: 0,
        query_indptr: vec![0, 1].into_boxed_slice(),
        page_indptr: vec![0, 2].into_boxed_slice(),
        page_indices: vec![10, 11].into_boxed_slice(),
        last_page_len: vec![1].into_boxed_slice(),
    };
    let sliding = crate::AttentionBatch {
        class_id: 1,
        query_indptr: vec![0, 1].into_boxed_slice(),
        page_indptr: vec![0, 1].into_boxed_slice(),
        page_indices: vec![30].into_boxed_slice(),
        last_page_len: vec![1].into_boxed_slice(),
    };
    let dimensions = [
        DecoderClassDimensions {
            class_id: 0,
            context_pages: sym("c_0"),
            backend_base_index: 10,
            page_count: 4,
            cache_slots: 224,
        },
        DecoderClassDimensions {
            class_id: 1,
            context_pages: sym("c_1"),
            backend_base_index: 30,
            page_count: 2,
            cache_slots: 512,
        },
    ];
    let steps = [
        DecoderClassStep {
            class_id: 0,
            write_slots: &[160],
            attention: &full,
        },
        DecoderClassStep {
            class_id: 1,
            write_slots: &[480],
            attention: &sliding,
        },
    ];
    let valid = DecoderStep {
        tokens: &[1],
        positions: &[64],
        classes: &steps,
    };
    assert!(validate_step(valid, compile, &dimensions, 16, 32).is_ok());

    let bad_steps = [
        steps[0],
        DecoderClassStep {
            write_slots: &[160],
            ..steps[1]
        },
    ];
    assert!(matches!(
        validate_step(
            DecoderStep {
                classes: &bad_steps,
                ..valid
            },
            compile,
            &dimensions,
            16,
            32,
        ),
        Err(DecoderError::InputCapacity)
    ));
}

#[test]
fn runtime_session_hybrid_plan_feeds_one_multi_class_decoder_step() {
    let input = hybrid_attention_input();
    let manifest = compile_runtime_manifest(input.clone()).unwrap();
    let manager_plan = compile_plan(
        compile_attention_state_plan(input)
            .unwrap()
            .token_manager_plan()
            .unwrap(),
    )
    .unwrap();
    let registrations = hybrid_registrations();
    let mut session = hybrid_runtime_session(&manager_plan, &registrations);
    let request_id = EngineRequestId(91);
    session.acquire_requests(&[request_id]).unwrap();
    let source = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 17,
        }])
        .unwrap();
    let view = session.prepared_execution_view(source.batch_id).unwrap();
    let executor = ExecutorPlan::compile(&manifest).unwrap();
    let arenas = session
        .arena_stats()
        .iter()
        .copied()
        .zip(registrations)
        .map(|(stats, registration)| ExecutorArena::bind(stats, registration).unwrap())
        .collect::<Vec<_>>();
    let prepared = executor.lower_prepared(source, &arenas).unwrap();
    let attention = executor.attention_batches(&view).unwrap();
    let graph = DecoderGraph::build(
        &mut Graph::default(),
        &test_config(4),
        DecoderWeightFeatures::default(),
        &executor,
        &arenas,
    )
    .unwrap();
    let class_steps = prepared.steps()[0]
        .classes
        .iter()
        .zip(&attention)
        .map(|(class, attention)| DecoderClassStep {
            class_id: class.class_id,
            write_slots: &class.write_slots,
            attention,
        })
        .collect::<Vec<_>>();
    assert!(
        validate_step(
            DecoderStep {
                tokens: &(0..17).collect::<Vec<_>>(),
                positions: &(0..17).collect::<Vec<_>>(),
                classes: &class_steps,
            },
            DecoderCompileConfig {
                maximum_query_tokens: 32,
                representative_prefill_tokens: 17,
                maximum_batch_size: 1,
                maximum_context_pages: 8,
                representative_context_pages: 2,
                search_graphs: 2,
                search_seed: 1,
            },
            &graph.class_dimensions,
            16,
            320,
        )
        .is_ok()
    );
    assert_eq!(attention[0].page_indices.len(), 2);
    assert_eq!(attention[1].page_indices.len(), 2);
    assert!(attention[1].page_indices.iter().all(|&page| page >= 8));
}
