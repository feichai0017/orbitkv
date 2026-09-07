use super::*;
use crate::attention_state::{AttentionStateSpec, RecurrentFamily};
use crate::plan::RetentionKind;

fn mixed_input() -> AttentionStatePlanInput {
    AttentionStatePlanInput {
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
                name: "local".into(),
                layers: vec![1],
                storage: AttentionStateStorage::LatentKv {
                    latent_bytes_per_token_per_layer: 96,
                    rope_bytes_per_token_per_layer: 32,
                    retention: RetentionKind::Sliding,
                    window_tokens: Some(18),
                },
            },
            AttentionStateSpec {
                name: "recurrent".into(),
                layers: vec![2],
                storage: AttentionStateStorage::Recurrent {
                    family: RecurrentFamily::LinearAttention,
                    state_bytes_per_layer: 4_096,
                    checkpoint_slots_per_request: 2,
                },
            },
            AttentionStateSpec {
                name: "convolution".into(),
                layers: vec![2],
                storage: AttentionStateStorage::Convolution {
                    state_bytes_per_layer: 2_048,
                    kernel_width: 4,
                    checkpoint_slots_per_request: 2,
                },
            },
        ],
    }
}

fn sink_window_program() -> RetentionProgramInput {
    use crate::retention::{IntExpr, Predicate, RetentionStateDecl};

    RetentionProgramInput {
        schema: "orbitkv.retention-ir.v1".into(),
        page_tokens: 4,
        states: vec![RetentionStateDecl {
            name: "attention".into(),
            layers: vec![0],
            kv_head_range: None,
            bytes_per_token_per_layer: 128,
            may_read: Predicate::Or {
                terms: vec![
                    Predicate::LessThan {
                        lhs: IntExpr::KeyPosition,
                        rhs: IntExpr::Constant { value: 4 },
                    },
                    Predicate::LessThan {
                        lhs: IntExpr::Sub {
                            lhs: Box::new(IntExpr::QueryPosition),
                            rhs: Box::new(IntExpr::KeyPosition),
                        },
                        rhs: IntExpr::Constant { value: 8 },
                    },
                ],
            },
        }],
    }
}

fn chunked_program() -> RetentionProgramInput {
    use crate::retention::{IntExpr, Predicate, RetentionStateDecl};

    RetentionProgramInput {
        schema: "orbitkv.retention-ir.v1".into(),
        page_tokens: 4,
        states: vec![RetentionStateDecl {
            name: "chunked".into(),
            layers: vec![0],
            kv_head_range: None,
            bytes_per_token_per_layer: 128,
            may_read: Predicate::Equal {
                lhs: IntExpr::FloorDiv {
                    value: Box::new(IntExpr::QueryPosition),
                    divisor: 16,
                },
                rhs: IntExpr::FloorDiv {
                    value: Box::new(IntExpr::KeyPosition),
                    divisor: 16,
                },
            },
        }],
    }
}

fn head_partition_program() -> RetentionProgramInput {
    use crate::retention::{IntExpr, Predicate, RetentionStateDecl};

    let state = |name: &str, start: u32, end_exclusive: u32, window: i64| RetentionStateDecl {
        name: name.into(),
        layers: vec![0],
        kv_head_range: Some(KvHeadRange {
            start,
            end_exclusive,
        }),
        bytes_per_token_per_layer: u64::from(end_exclusive - start) * 64,
        may_read: Predicate::LessThan {
            lhs: IntExpr::Sub {
                lhs: Box::new(IntExpr::QueryPosition),
                rhs: Box::new(IntExpr::KeyPosition),
            },
            rhs: IntExpr::Constant { value: window },
        },
    };
    RetentionProgramInput {
        schema: "orbitkv.retention-ir.v1".into(),
        page_tokens: 16,
        states: vec![state("short", 0, 4, 128), state("long", 4, 8, 512)],
    }
}

#[test]
fn retention_manifest_preserves_region_head_and_chunked_topologies() {
    let sink = compile_retention_runtime_manifest(sink_window_program()).unwrap();
    assert_eq!(sink.version, RUNTIME_MANIFEST_VERSION);
    let sink_manager = sink.token_manager_plan.as_ref().unwrap();
    assert!(matches!(
        sink_manager.layout.classes[0].address,
        AddressProgram::Pinned
    ));
    assert!(matches!(
        sink_manager.layout.classes[1].address,
        AddressProgram::PeriodicFrom {
            period_blocks: 3,
            origin_block: 1
        }
    ));
    assert_eq!(
        sink.capability_requirements,
        vec![
            RuntimeCapability::BlockDomainPartitioning,
            RuntimeCapability::PeriodicFromAddressing,
            RuntimeCapability::PinnedAddressing,
            RuntimeCapability::SemanticRetirement,
            RuntimeCapability::TokenManager,
        ]
    );

    let chunked = compile_retention_runtime_manifest(chunked_program()).unwrap();
    let chunked_class = &chunked.token_manager_plan.as_ref().unwrap().layout.classes[0];
    assert!(matches!(
        chunked_class.address,
        AddressProgram::ResettableArena {
            blocks_per_epoch: 4
        }
    ));
    assert!(matches!(
        chunked_class.retirement,
        RetirementProgram::EpochEnd {
            blocks_per_epoch: 4
        }
    ));

    let heads = compile_retention_runtime_manifest(head_partition_program()).unwrap();
    assert_eq!(
        heads
            .token_manager_plan
            .as_ref()
            .unwrap()
            .layout
            .classes
            .iter()
            .map(|class| class.kv_head_range.clone().unwrap())
            .collect::<Vec<_>>(),
        vec![
            KvHeadRange {
                start: 0,
                end_exclusive: 4,
            },
            KvHeadRange {
                start: 4,
                end_exclusive: 8,
            },
        ]
    );
    assert!(
        heads
            .capability_requirements
            .contains(&RuntimeCapability::KvHeadPartitioning)
    );
}

#[test]
fn retention_manifest_round_trips_and_rejects_resealed_tampering() {
    let original = compile_retention_runtime_manifest(sink_window_program()).unwrap();
    let encoded = serde_json::to_vec(&original).unwrap();
    assert_eq!(RuntimeManifest::from_json(&encoded).unwrap(), original);

    let mut layout = original.clone();
    layout.token_manager_plan.as_mut().unwrap().layout.classes[0].address =
        AddressProgram::AppendOnly;
    layout.fingerprint = layout.computed_fingerprint().unwrap();
    assert!(matches!(
        layout.validate(),
        Err(RuntimeManifestError::TokenManagerLayoutMismatch)
    ));

    let mut capabilities = original.clone();
    capabilities.capability_requirements.swap(0, 1);
    capabilities.fingerprint = capabilities.computed_fingerprint().unwrap();
    assert!(matches!(
        capabilities.validate(),
        Err(RuntimeManifestError::CapabilityRequirementsMismatch)
    ));

    let mut mixed_source = original;
    mixed_source.attention_state_plan = Some(compile_attention_state_plan(mixed_input()).unwrap());
    mixed_source.fingerprint = mixed_source.computed_fingerprint().unwrap();
    assert!(matches!(
        mixed_source.validate(),
        Err(RuntimeManifestError::SourceMismatch)
    ));
}

#[test]
fn mixed_manifest_round_trips_with_canonical_capabilities() {
    let manifest = compile_runtime_manifest(mixed_input()).unwrap();
    manifest.validate().unwrap();
    assert_eq!(manifest.schema, RUNTIME_MANIFEST_SCHEMA);
    assert_eq!(manifest.version, RUNTIME_MANIFEST_VERSION);
    assert_eq!(
        manifest.capability_requirements,
        vec![
            RuntimeCapability::AppendOnlyAddressing,
            RuntimeCapability::ConvolutionState,
            RuntimeCapability::FixedStateCheckpoints,
            RuntimeCapability::PeriodicAddressing,
            RuntimeCapability::RecurrentState,
            RuntimeCapability::SemanticRetirement,
            RuntimeCapability::TokenComponentGeometry,
            RuntimeCapability::TokenManager,
        ]
    );
    let token = manifest.token_manager_plan.as_ref().unwrap();
    assert_eq!(
        manifest
            .token_manager_input()
            .unwrap()
            .unwrap()
            .classes
            .len(),
        2
    );
    assert_eq!(
        token.layout.classes[1].address,
        AddressProgram::Periodic { period_blocks: 3 }
    );

    let encoded = serde_json::to_vec(&manifest).unwrap();
    let decoded = RuntimeManifest::from_json(&encoded).unwrap();
    assert_eq!(decoded, manifest);
}

#[test]
fn fixed_only_manifest_uses_null_token_manager() {
    let manifest = compile_runtime_manifest(AttentionStatePlanInput {
        page_tokens: 16,
        states: vec![AttentionStateSpec {
            name: "state".into(),
            layers: vec![0, 1],
            storage: AttentionStateStorage::Recurrent {
                family: RecurrentFamily::Mamba,
                state_bytes_per_layer: 64,
                checkpoint_slots_per_request: 2,
            },
        }],
    })
    .unwrap();
    assert!(manifest.token_manager_plan.is_none());
    assert_eq!(
        manifest.capability_requirements,
        vec![
            RuntimeCapability::FixedStateCheckpoints,
            RuntimeCapability::RecurrentState,
        ]
    );
    RuntimeManifest::from_json(&serde_json::to_vec(&manifest).unwrap()).unwrap();
}

#[test]
fn fingerprint_is_stable_and_covers_array_order() {
    let manifest = compile_runtime_manifest(mixed_input()).unwrap();
    assert_eq!(
        manifest.fingerprint,
        manifest.computed_fingerprint().unwrap()
    );
    assert_eq!(
        manifest.fingerprint,
        "sha256:2182b85ff1da2391854fcd9927e4a0882ca06cc041a27199b3e207545dc5b082"
    );

    let mut reordered = manifest.clone();
    reordered
        .attention_state_plan
        .as_mut()
        .unwrap()
        .states
        .swap(0, 1);
    assert_ne!(
        reordered.computed_fingerprint().unwrap(),
        manifest.fingerprint
    );
}

#[test]
fn canonical_json_recursively_sorts_object_keys() {
    let value = serde_json::json!({
        "z": {"b": 2, "a": 1},
        "a": ["状态", {"y": false, "x": null}]
    });
    let mut encoded = Vec::new();
    write_canonical_json(&value, &mut encoded).unwrap();
    assert_eq!(
        String::from_utf8(encoded).unwrap(),
        r#"{"a":["状态",{"x":null,"y":false}],"z":{"a":1,"b":2}}"#
    );
}

#[test]
fn canonical_json_interoperability_vector_covers_escaping_and_unicode() {
    let value = serde_json::json!({
        "z": ["状态", "line\nquote\"slash\\", null, true, 42],
        "a": {"é": "值", "\u{0001}": "\u{0008}\u{000c}\n\r\t"}
    });
    let mut encoded = Vec::new();
    write_canonical_json(&value, &mut encoded).unwrap();
    assert_eq!(
        format!("sha256:{:x}", Sha256::digest(&encoded)),
        "sha256:95762173915ab37233eb16469bdb42ad0c2efcc5aad56052a87ec2ec33819b7f"
    );
}

#[test]
fn rejects_manifest_above_the_serialized_size_limit() {
    let oversized = vec![b' '; RUNTIME_MANIFEST_MAX_BYTES + 1];
    assert!(matches!(
        RuntimeManifest::from_json(&oversized),
        Err(RuntimeManifestError::ManifestTooLarge { .. })
    ));
}

#[test]
fn compiled_manifest_must_fit_the_emitted_size_limit() {
    let manifest = compile_runtime_manifest(mixed_input()).unwrap();
    assert!(matches!(
        manifest.validate_serialized_size_with_limit(64),
        Err(RuntimeManifestError::ManifestTooLarge { maximum: 64, .. })
    ));
}

#[test]
fn rejects_tampering_even_with_a_recomputed_fingerprint() {
    let original = compile_runtime_manifest(mixed_input()).unwrap();

    let mut state = original.clone();
    let AttentionStateBackend::TokenSlots {
        page_bytes_per_layer,
        ..
    } = &mut state.attention_state_plan.as_mut().unwrap().states[0].backend
    else {
        panic!("fixture begins with token state");
    };
    *page_bytes_per_layer += 1;
    state.fingerprint = state.computed_fingerprint().unwrap();
    assert!(matches!(
        state.validate(),
        Err(RuntimeManifestError::AttentionStatePlanMismatch)
    ));

    let mut layout = original.clone();
    layout.token_manager_plan.as_mut().unwrap().layout.classes[0].bytes_per_token_per_layer += 1;
    layout.fingerprint = layout.computed_fingerprint().unwrap();
    assert!(matches!(
        layout.validate(),
        Err(RuntimeManifestError::TokenManagerLayoutMismatch)
    ));

    let mut capabilities = original;
    capabilities.capability_requirements.swap(0, 1);
    capabilities.fingerprint = capabilities.computed_fingerprint().unwrap();
    assert!(matches!(
        capabilities.validate(),
        Err(RuntimeManifestError::CapabilityRequirementsMismatch)
    ));
}

#[test]
fn rejects_fingerprint_schema_version_and_section_presence_changes() {
    let original = compile_runtime_manifest(mixed_input()).unwrap();

    let mut malformed = original.clone();
    malformed.fingerprint = "SHA256:1234".into();
    assert!(matches!(
        malformed.validate(),
        Err(RuntimeManifestError::MalformedFingerprint)
    ));

    let mut stale = original.clone();
    stale.fingerprint.replace_range(7..8, "0");
    if stale.fingerprint == original.fingerprint {
        stale.fingerprint.replace_range(7..8, "1");
    }
    assert!(matches!(
        stale.validate(),
        Err(RuntimeManifestError::FingerprintMismatch)
    ));

    let mut schema = original.clone();
    schema.schema.push_str(".v1");
    assert!(matches!(
        schema.validate(),
        Err(RuntimeManifestError::UnsupportedSchema(_))
    ));

    let mut version = original.clone();
    version.version = 2;
    assert!(matches!(
        version.validate(),
        Err(RuntimeManifestError::UnsupportedVersion(2))
    ));

    let mut missing_token_plan = original;
    missing_token_plan.token_manager_plan = None;
    missing_token_plan.fingerprint = missing_token_plan.computed_fingerprint().unwrap();
    assert!(matches!(
        missing_token_plan.validate(),
        Err(RuntimeManifestError::TokenManagerPresenceMismatch)
    ));
}

#[test]
fn rejects_unknown_fields_at_every_manifest_layer() {
    let manifest = compile_runtime_manifest(mixed_input()).unwrap();
    let base = serde_json::to_value(manifest).unwrap();

    let mut top = base.clone();
    top.as_object_mut()
        .unwrap()
        .insert("unknown".into(), true.into());
    assert!(matches!(
        RuntimeManifest::from_json(&serde_json::to_vec(&top).unwrap()),
        Err(RuntimeManifestError::Json(_))
    ));

    let mut address = base.clone();
    address
        .pointer_mut("/token_manager_plan/layout/classes/0/address")
        .unwrap()
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), true.into());
    assert!(matches!(
        RuntimeManifest::from_json(&serde_json::to_vec(&address).unwrap()),
        Err(RuntimeManifestError::Json(_) | RuntimeManifestError::NonCanonicalJson)
    ));

    let mut retirement = base.clone();
    retirement
        .pointer_mut("/token_manager_plan/layout/classes/0/retirement")
        .unwrap()
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), true.into());
    assert!(matches!(
        RuntimeManifest::from_json(&serde_json::to_vec(&retirement).unwrap()),
        Err(RuntimeManifestError::Json(_) | RuntimeManifestError::NonCanonicalJson)
    ));

    let mut backend = base;
    backend
        .pointer_mut("/attention_state_plan/states/2/backend")
        .unwrap()
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), true.into());
    assert!(matches!(
        RuntimeManifest::from_json(&serde_json::to_vec(&backend).unwrap()),
        Err(RuntimeManifestError::Json(_) | RuntimeManifestError::NonCanonicalJson)
    ));
}

#[test]
fn rejects_duplicate_root_and_nested_fields() {
    let manifest = compile_runtime_manifest(mixed_input()).unwrap();
    let encoded = serde_json::to_string(&manifest).unwrap();
    let duplicate_root =
        encoded.replacen("{\"schema\":", "{\"schema\":\"duplicate\",\"schema\":", 1);
    assert!(matches!(
        RuntimeManifest::from_json(duplicate_root.as_bytes()),
        Err(RuntimeManifestError::Json(_))
    ));

    let duplicate_nested = encoded.replacen(
        "\"input\":{\"page_tokens\":",
        "\"input\":{\"page_tokens\":8,\"page_tokens\":",
        1,
    );
    assert!(matches!(
        RuntimeManifest::from_json(duplicate_nested.as_bytes()),
        Err(RuntimeManifestError::Json(_))
    ));
}
