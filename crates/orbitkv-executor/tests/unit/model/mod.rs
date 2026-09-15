use super::*;
use orbitkv::{
    AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage, CacheSharingPolicy,
    EngineAppendIntent, EngineRequestId, RecurrentFamily, RuntimeSession, StatePoolIdentity,
    compile_attention_state_plan, compile_plan, compile_runtime_manifest,
    kv_manager::{BackendArenaRegistration, CanonicalKvManager, ManagerConfig},
    plan::RetentionKind,
};

mod compilation;
mod tuning;

#[test]
fn compile_config_requires_distinct_decode_and_prefill_ranges() {
    let valid = DecoderCompileConfig {
        output_rows: crate::model::DecoderOutputRows::AllTokens,
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
            search_graphs: 1,
            ..valid
        }
        .validate()
        .is_ok()
    );
    assert!(
        DecoderCompileConfig {
            output_rows: crate::model::DecoderOutputRows::AllTokens,
            maximum_query_tokens: 1,
            ..valid
        }
        .validate()
        .is_err()
    );
    assert!(
        DecoderCompileConfig {
            search_graphs: 0,
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
        output_rows: crate::model::DecoderOutputRows::AllTokens,
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
        &[],
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
        &[],
        DecoderWeightFeatures::default(),
        compile,
        "facts-a",
    )
    .unwrap();
    let compile_identity = decoder_artifact_identity(
        &config,
        &plan,
        &arenas,
        &[],
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
        &[],
        DecoderWeightFeatures::default(),
        compile,
        "facts-b",
    )
    .unwrap();

    assert_ne!(identity, arena_identity);
    assert_ne!(identity, compile_identity);
    assert_ne!(identity, facts_identity);

    let mut fixed_state = [FixedStateArenaRegistration {
        state_id: 1,
        engine_epoch: 1,
        pool_epoch: 2,
        pool_id: 3,
        slot_count: 4,
        slot_bytes: 64,
    }];
    let fixed_identity = decoder_artifact_identity(
        &config,
        &plan,
        &arenas,
        &fixed_state,
        DecoderWeightFeatures::default(),
        compile,
        "facts-a",
    )
    .unwrap();
    fixed_state[0].slot_count = 6;
    let changed_fixed_identity = decoder_artifact_identity(
        &config,
        &plan,
        &arenas,
        &fixed_state,
        DecoderWeightFeatures::default(),
        compile,
        "facts-a",
    )
    .unwrap();
    assert_ne!(fixed_identity, changed_fixed_identity);
}

#[test]
fn fp8_linear_declares_checkpoint_scale_and_searchable_deepgemm_candidates() {
    // Tile eligibility needs occupancy as well as architecture. This is a
    // synthetic Hopper target; the test does not query a local device.
    let simulated_sm_count = 78;
    let mut config = test_config(1);
    config.weight_format = DecoderWeightFormat::Fp8E4M3Block {
        rows: 128,
        columns: 128,
    };
    let mut graph = Graph::default();
    let input = graph
        .named_tensor("input", (1usize, config.hidden_size))
        .as_dtype(DType::Bf16);
    let linear = linear_weight(
        &mut graph,
        &config,
        "model.layers.0.mlp.gate_proj.weight",
        config.intermediate_size,
        config.hidden_size,
        DType::Bf16,
    );
    linear.forward(&input).output();

    assert!(graph.input_meta.values().any(|(name, dtype)| {
        name == "model.layers.0.mlp.gate_proj.weight" && *dtype == DType::F8E4M3
    }));
    assert!(graph.input_meta.values().any(|(name, dtype)| {
        name == "model.layers.0.mlp.gate_proj.weight_scale_inv" && *dtype == DType::F32
    }));
    graph.build_search_space::<CudaRuntime>(
        orbitkv_compiler::prelude::CompileOptions::default().compiler_facts(format!(
            "{}\n(set (cuda-target-sm-count) {simulated_sm_count})",
            orbitkv_cuda::target::CudaTarget { major: 9, minor: 0 }.compiler_facts(),
        )),
    );
    assert!(
        graph
            .egraph()
            .unwrap()
            .enodes
            .values()
            .any(|(label, _)| label == "DeepGemm")
    );
}

#[test]
fn step_validation_enforces_compiled_capacities() {
    let compile = DecoderCompileConfig {
        output_rows: crate::model::DecoderOutputRows::AllTokens,
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
fn stateful_step_requires_one_state_plan_per_request() {
    let attention = crate::AttentionBatch {
        class_id: 0,
        query_indptr: vec![0, 1, 2].into_boxed_slice(),
        page_indptr: vec![0, 1, 2].into_boxed_slice(),
        page_indices: vec![0, 1].into_boxed_slice(),
        last_page_len: vec![1, 1].into_boxed_slice(),
    };
    let classes = [DecoderClassStep {
        class_id: 0,
        write_slots: &[0, 16],
        attention: &attention,
    }];
    let step = DecoderStep {
        tokens: &[1, 2],
        positions: &[0, 0],
        classes: &classes,
    };
    let state = orbitkv::EngineFixedStatePlan {
        state_id: 1,
        source: None,
        destination: orbitkv::StateSlotLease {
            engine_epoch: 1,
            pool_epoch: 2,
            generation: 1,
            slot_id: 0,
            pool_id: 3,
        },
        byte_count: 64,
    };
    let states = [
        DecoderFixedStateStep {
            request_id: 7,
            states: std::slice::from_ref(&state),
        },
        DecoderFixedStateStep {
            request_id: 8,
            states: std::slice::from_ref(&state),
        },
    ];
    assert!(validate_stateful_step(step, &states).is_ok());

    let prefill_attention = crate::AttentionBatch {
        class_id: 0,
        query_indptr: vec![0, 2].into_boxed_slice(),
        page_indptr: vec![0, 1].into_boxed_slice(),
        page_indices: vec![0].into_boxed_slice(),
        last_page_len: vec![2].into_boxed_slice(),
    };
    let prefill_classes = [DecoderClassStep {
        class_id: 0,
        write_slots: &[0, 1],
        attention: &prefill_attention,
    }];
    assert!(
        validate_stateful_step(
            DecoderStep {
                classes: &prefill_classes,
                ..step
            },
            &states[..1],
        )
        .is_ok()
    );
    assert!(matches!(
        validate_stateful_step(
            DecoderStep {
                classes: &prefill_classes,
                ..step
            },
            &states,
        ),
        Err(DecoderError::InputCapacity)
    ));
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
        attention_softmax_scale: 64_f64.sqrt().recip(),
        attention_output_gate: false,
        layer_kinds: None,
        gated_delta: None,
        weight_format: DecoderWeightFormat::Float,
    }
}

#[test]
#[ignore = "requires ORBITKV_MODEL_DIR containing an external hybrid-state checkpoint"]
fn external_hybrid_checkpoint_passes_structural_execution_admission() {
    let directory = std::env::var_os("ORBITKV_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .expect("ORBITKV_MODEL_DIR is required");
    let bytes = std::fs::read(directory.join("config.json")).unwrap();
    let config = DecoderConfig::from_json(&bytes).unwrap();
    let manifest = orbitkv::compile_hf_runtime_manifest(
        &bytes,
        orbitkv::HfRetentionOptions {
            page_tokens: 16,
            kv_dtype_bytes: 2,
        },
    )
    .unwrap();
    let plan = ExecutorPlan::compile(&manifest).unwrap();
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
    let weights = inspect_weight_features(&weight_files, &config).unwrap();
    assert_eq!(config.tensor_prefix, "model.language_model");
    assert!(config.rotary_dimensions < config.head_dim);
    assert!(
        config
            .layer_kinds
            .as_deref()
            .is_some_and(|layers| layers.contains(&DecoderLayerKind::Linear))
    );
    assert!(config.require_executable().is_ok());

    let arenas = plan
        .classes
        .iter()
        .map(|class| ExecutorArena {
            engine_epoch: 1,
            pool_epoch: u64::from(class.class_id) + 1,
            pool_id: u32::from(class.class_id) + 1,
            class_id: class.class_id,
            backend_domain: class.class_id + 1,
            first_page_id: 1,
            page_count: 1,
            backend_base_index: 0,
        })
        .collect::<Vec<_>>();
    let identities = plan
        .fixed_states
        .iter()
        .map(|class| {
            let (slots, bytes) = match class.storage {
                crate::FixedStateStorage::Recurrent {
                    slots_per_request,
                    bytes_per_request,
                    ..
                }
                | crate::FixedStateStorage::Convolution {
                    slots_per_request,
                    bytes_per_request,
                    ..
                } => (slots_per_request, bytes_per_request),
            };
            (
                class.state_id,
                StatePoolIdentity {
                    engine_epoch: 1,
                    pool_epoch: u64::from(class.state_id) + 100,
                    byte_count: bytes / u64::from(slots),
                    pool_id: u32::from(class.state_id) + 100,
                    slot_count: slots,
                },
            )
        })
        .collect::<Vec<_>>();
    let registrations = plan.fixed_state_registrations(&identities).unwrap();
    let mut graph = Graph::default();
    let decoder = DecoderGraph::build(
        &mut graph,
        &config,
        weights,
        &plan,
        &arenas,
        &registrations,
        DecoderOutputRows::AllTokens,
    )
    .unwrap();
    assert_fp8_weights_have_scales(&graph);
    compilation::export_saturation_fixture(&mut graph, &decoder, &plan, &arenas);
}

fn assert_fp8_weights_have_scales(graph: &Graph) {
    let fp8_weights = graph
        .input_meta
        .values()
        .filter(|(name, dtype)| name.ends_with(".weight") && *dtype == DType::F8E4M3)
        .count();
    let scales = graph
        .input_meta
        .values()
        .filter(|(name, dtype)| name.ends_with(".weight_scale_inv") && *dtype == DType::F32)
        .count();
    assert!(fp8_weights > 0);
    assert_eq!(fp8_weights, scales);
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
            &[],
            DecoderOutputRows::AllTokens
        ),
        Err(DecoderError::UnsupportedPlan)
    ));
}

fn hybrid_executor_plan() -> ExecutorPlan {
    crate::tests::support::executor_plan(
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

fn stateful_decoder_contract() -> (
    DecoderConfig,
    ExecutorPlan,
    Vec<FixedStateArenaRegistration>,
) {
    let mut config = test_config(2);
    config.hidden_size = 4;
    config.intermediate_size = 8;
    config.query_heads = 1;
    config.head_dim = 64;
    config.layer_kinds =
        Some(vec![DecoderLayerKind::Linear, DecoderLayerKind::Full].into_boxed_slice());
    config.gated_delta = Some(GatedDeltaConfig {
        key_heads: 1,
        value_heads: 2,
        key_width: 2,
        value_width: 1,
        convolution_kernel_width: 3,
    });
    let plan = crate::tests::support::executor_plan_with_fixed_states(
        "stateful-decoder",
        16,
        vec![crate::AttentionClass {
            class_id: 0,
            name: "attention".into(),
            layers: vec![1].into_boxed_slice(),
            page_tokens: 16,
            key_bytes_per_token_per_layer: 128,
            value_bytes_per_token_per_layer: 128,
            visibility: crate::AttentionVisibility::Full,
        }],
        vec![
            crate::FixedStateClass {
                state_id: 1,
                name: "recurrent".into(),
                layers: vec![0].into_boxed_slice(),
                storage: crate::FixedStateStorage::Recurrent {
                    family: RecurrentFamily::Gdn,
                    bytes_per_layer: 16,
                    slots_per_request: 2,
                    bytes_per_request: 32,
                },
            },
            crate::FixedStateClass {
                state_id: 2,
                name: "convolution".into(),
                layers: vec![0].into_boxed_slice(),
                storage: crate::FixedStateStorage::Convolution {
                    bytes_per_layer: 24,
                    kernel_width: 3,
                    slots_per_request: 2,
                    bytes_per_request: 48,
                },
            },
        ],
    );
    let identities = [(1, (16, 2)), (2, (24, 3))].map(|(state_id, (byte_count, pool_id))| {
        (
            state_id,
            StatePoolIdentity {
                engine_epoch: 1,
                pool_epoch: u64::from(pool_id),
                byte_count,
                pool_id,
                slot_count: 4,
            },
        )
    });
    let registrations = plan
        .fixed_state_registrations(&identities)
        .unwrap()
        .into_vec();
    (config, plan, registrations)
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
        &[],
        DecoderOutputRows::AllTokens,
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
fn graph_composes_token_attention_and_fixed_state_layers() {
    let (config, plan, registrations) = stateful_decoder_contract();
    let decoder = DecoderGraph::build(
        &mut Graph::default(),
        &config,
        DecoderWeightFeatures::default(),
        &plan,
        &[ExecutorArena {
            engine_epoch: 1,
            pool_epoch: 1,
            pool_id: 1,
            class_id: 0,
            backend_domain: 1,
            first_page_id: 1,
            page_count: 4,
            backend_base_index: 0,
        }],
        &registrations,
        DecoderOutputRows::AllTokens,
    )
    .unwrap();

    assert_eq!(decoder.outputs.cache.len(), 1);
    assert_eq!(decoder.outputs.cache[0].binding.layer, 1);
    assert_eq!(
        decoder
            .outputs
            .fixed_states
            .iter()
            .map(|state| (state.binding.state_id, state.policy))
            .collect::<Vec<_>>(),
        vec![
            (1, crate::FixedStateWritePolicy::RequiredInPlace),
            (2, crate::FixedStateWritePolicy::RequiredInPlace),
        ]
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
            &[],
            DecoderOutputRows::AllTokens
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
            &[],
            DecoderOutputRows::AllTokens
        ),
        Err(DecoderError::UnsupportedPlan)
    ));
}

#[test]
fn multi_class_step_validates_each_absolute_arena() {
    let compile = DecoderCompileConfig {
        output_rows: crate::model::DecoderOutputRows::AllTokens,
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
        &[],
        DecoderOutputRows::AllTokens,
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
                output_rows: crate::model::DecoderOutputRows::AllTokens,
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
