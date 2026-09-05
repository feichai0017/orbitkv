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

fn test_config(layers: usize) -> DecoderConfig {
    DecoderConfig {
        layers,
        hidden_size: 128,
        intermediate_size: 256,
        query_heads: 2,
        kv_heads: 1,
        head_dim: 64,
        vocabulary_size: 320,
        rope_theta: 10_000.0,
        rms_epsilon: 1e-6,
        tied_embeddings: false,
    }
}

fn hybrid_executor_plan() -> ExecutorPlan {
    ExecutorPlan {
        manifest_fingerprint: "hybrid".into(),
        page_tokens: 16,
        classes: vec![
            crate::AttentionClass {
                class_id: 0,
                name: "global".into(),
                layers: vec![0, 2].into_boxed_slice(),
                page_tokens: 16,
                key_bytes_per_token_per_layer: 128,
                value_bytes_per_token_per_layer: 128,
                token_relocatable: true,
                visibility: crate::AttentionVisibility::Full,
            },
            crate::AttentionClass {
                class_id: 1,
                name: "local".into(),
                layers: vec![1, 3].into_boxed_slice(),
                page_tokens: 16,
                key_bytes_per_token_per_layer: 128,
                value_bytes_per_token_per_layer: 128,
                token_relocatable: true,
                visibility: crate::AttentionVisibility::Sliding { window_tokens: 64 },
            },
        ]
        .into_boxed_slice(),
    }
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
        DecoderWeightLayout::default(),
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
            DecoderWeightLayout::default(),
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
            DecoderWeightLayout::default(),
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
        DecoderWeightLayout::default(),
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
