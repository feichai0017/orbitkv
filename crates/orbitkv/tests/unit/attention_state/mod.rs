use super::*;

#[test]
#[allow(clippy::too_many_lines)]
fn compiles_token_latent_recurrent_and_convolution_backends() {
    let output = compile_attention_state_plan(AttentionStatePlanInput {
        page_tokens: 16,
        states: vec![
            AttentionStateSpec {
                name: "full".into(),
                layers: vec![0],
                storage: AttentionStateStorage::TokenKv {
                    key_bytes_per_token_per_layer: 256,
                    value_bytes_per_token_per_layer: 256,
                    retention: RetentionKind::Full,
                    window_tokens: None,
                },
            },
            AttentionStateSpec {
                name: "mla".into(),
                layers: vec![1],
                storage: AttentionStateStorage::LatentKv {
                    latent_bytes_per_token_per_layer: 1024,
                    rope_bytes_per_token_per_layer: 128,
                    retention: RetentionKind::Full,
                    window_tokens: None,
                },
            },
            AttentionStateSpec {
                name: "gdn".into(),
                layers: vec![2],
                storage: AttentionStateStorage::Recurrent {
                    family: RecurrentFamily::Gdn,
                    state_bytes_per_layer: 4096,
                    checkpoint_slots_per_request: 2,
                },
            },
            AttentionStateSpec {
                name: "shortconv".into(),
                layers: vec![2],
                storage: AttentionStateStorage::Convolution {
                    state_bytes_per_layer: 2048,
                    kernel_width: 4,
                    checkpoint_slots_per_request: 2,
                },
            },
        ],
    })
    .unwrap();
    assert_eq!(output.schema, "orbitkv.attention-state-plan.v1");
    let AttentionStateBackend::TokenSlots {
        components,
        bytes_per_token_per_layer,
        page_bytes_per_layer,
        ..
    } = &output.states[1].backend
    else {
        panic!("MLA state must lower to token slots");
    };
    assert_eq!(*bytes_per_token_per_layer, 1_152);
    assert_eq!(*page_bytes_per_layer, 18_432);
    assert_eq!(
        components,
        &[
            StateComponentGeometry {
                name: "latent",
                bytes_per_token_per_layer: 1_024,
            },
            StateComponentGeometry {
                name: "rope",
                bytes_per_token_per_layer: 128,
            },
        ]
    );
    let manager = output.token_manager_plan().unwrap();
    assert_eq!(manager.page_tokens, 16);
    assert_eq!(manager.classes.len(), 2);
    assert_eq!(manager.classes[0].name, "full");
    assert_eq!(manager.classes[0].bytes_per_token_per_layer, 512);
    assert_eq!(manager.classes[1].name, "mla");
    assert_eq!(manager.classes[1].bytes_per_token_per_layer, 1_152);
    assert_eq!(manager.classes[1].layers, vec![1]);
    assert_eq!(manager.classes[1].storage, TokenStorageKind::LatentKv);
    assert_eq!(
        manager.classes[1].components,
        vec![
            TokenComponentSpec {
                name: "latent".into(),
                bytes_per_token_per_layer: 1_024,
            },
            TokenComponentSpec {
                name: "rope".into(),
                bytes_per_token_per_layer: 128,
            },
        ]
    );
}

#[test]
fn page_geometry_uses_the_declared_page_size() {
    let output = compile_attention_state_plan(AttentionStatePlanInput {
        page_tokens: 7,
        states: vec![AttentionStateSpec {
            name: "full".into(),
            layers: vec![0],
            storage: AttentionStateStorage::TokenKv {
                key_bytes_per_token_per_layer: 3,
                value_bytes_per_token_per_layer: 5,
                retention: RetentionKind::Full,
                window_tokens: None,
            },
        }],
    })
    .unwrap();
    assert!(matches!(
        output.states[0].backend,
        AttentionStateBackend::TokenSlots {
            bytes_per_token_per_layer: 8,
            page_bytes_per_layer: 56,
            ..
        }
    ));
}

#[test]
fn same_role_overlap_and_single_checkpoint_fail_closed() {
    let duplicated = AttentionStatePlanInput {
        page_tokens: 16,
        states: vec![
            AttentionStateSpec {
                name: "mha".into(),
                layers: vec![0],
                storage: AttentionStateStorage::TokenKv {
                    key_bytes_per_token_per_layer: 1,
                    value_bytes_per_token_per_layer: 1,
                    retention: RetentionKind::Full,
                    window_tokens: None,
                },
            },
            AttentionStateSpec {
                name: "mla".into(),
                layers: vec![0],
                storage: AttentionStateStorage::LatentKv {
                    latent_bytes_per_token_per_layer: 1,
                    rope_bytes_per_token_per_layer: 1,
                    retention: RetentionKind::Full,
                    window_tokens: None,
                },
            },
        ],
    };
    assert!(matches!(
        compile_attention_state_plan(duplicated),
        Err(AttentionStateError::OverlappingRole { .. })
    ));
    let recurrent = AttentionStatePlanInput {
        page_tokens: 16,
        states: vec![AttentionStateSpec {
            name: "mamba".into(),
            layers: vec![0],
            storage: AttentionStateStorage::Recurrent {
                family: RecurrentFamily::Mamba,
                state_bytes_per_layer: 16,
                checkpoint_slots_per_request: 1,
            },
        }],
    };
    assert_eq!(
        compile_attention_state_plan(recurrent),
        Err(AttentionStateError::InsufficientCheckpointSlots)
    );
}

#[test]
fn state_only_plan_has_no_token_manager_projection() {
    let output = compile_attention_state_plan(AttentionStatePlanInput {
        page_tokens: 16,
        states: vec![AttentionStateSpec {
            name: "mamba".into(),
            layers: vec![0, 1, 2],
            storage: AttentionStateStorage::Recurrent {
                family: RecurrentFamily::Mamba,
                state_bytes_per_layer: 16,
                checkpoint_slots_per_request: 2,
            },
        }],
    })
    .unwrap();
    assert!(matches!(
        output.states[0].backend,
        AttentionStateBackend::RecurrentCheckpoints {
            checkpoint_bytes_per_request: 96,
            ..
        }
    ));
    assert_eq!(
        output.token_manager_plan(),
        Err(AttentionStateError::NoTokenState)
    );
}
