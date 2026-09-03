fn token_state(
    name: &str,
    layers: Vec<u32>,
    storage: TokenStorageKind,
    retention: RetentionKind,
    window_tokens: Option<u64>,
) -> AttentionStateSpec {
    let storage = match storage {
        TokenStorageKind::TokenKv => AttentionStateStorage::TokenKv {
            key_bytes_per_token_per_layer: 64,
            value_bytes_per_token_per_layer: 64,
            retention,
            window_tokens,
        },
        TokenStorageKind::LatentKv => AttentionStateStorage::LatentKv {
            latent_bytes_per_token_per_layer: 96,
            rope_bytes_per_token_per_layer: 32,
            retention,
            window_tokens,
        },
    };
    AttentionStateSpec {
        name: name.into(),
        layers,
        storage,
    }
}

fn manifest(states: Vec<AttentionStateSpec>) -> RuntimeManifest {
    crate::runtime_manifest::compile_runtime_manifest(AttentionStatePlanInput {
        page_tokens: 16,
        states,
    })
    .unwrap()
}

fn target(topologies: Vec<ExecutionTopology>) -> RuntimeTarget {
    let packaged = RuntimeTarget::packaged_sglang().unwrap();
    RuntimeTarget::new(
        RuntimeTargetIdentity {
            id: "sglang".into(),
            contract_version: 4,
        },
        RuntimeAdmissionProfile {
            id: "eager-single-device-bf16-nhd".into(),
            version: 1,
        },
        16,
        vec![RUNTIME_MANIFEST_VERSION],
        REQUIRED_WIRE_VERSION,
        packaged.supported_capabilities,
        topologies,
    )
    .unwrap()
}

fn all_topologies() -> Vec<ExecutionTopology> {
    vec![
        ExecutionTopology::WholeDomainChunkedTokenKv,
        ExecutionTopology::WholeDomainFullLatentKv,
        ExecutionTopology::WholeDomainFullSlidingTokenKv,
        ExecutionTopology::WholeDomainFullTokenKv,
        ExecutionTopology::WholeDomainFullTokenKvGdnConvolution,
        ExecutionTopology::WholeDomainFullTokenKvMamba,
        ExecutionTopology::WholeDomainSlidingTokenKv,
    ]
}

#[test]
#[allow(clippy::too_many_lines)]
fn derives_and_admits_every_closed_topology() {
    let cases = [
        (
            manifest(vec![token_state(
                "latent",
                vec![0, 1],
                TokenStorageKind::LatentKv,
                RetentionKind::Full,
                None,
            )]),
            ExecutionTopology::WholeDomainFullLatentKv,
        ),
        (
            manifest(vec![token_state(
                "full",
                vec![0, 1],
                TokenStorageKind::TokenKv,
                RetentionKind::Full,
                None,
            )]),
            ExecutionTopology::WholeDomainFullTokenKv,
        ),
        (
            manifest(vec![token_state(
                "sliding",
                vec![0, 1],
                TokenStorageKind::TokenKv,
                RetentionKind::Sliding,
                Some(18),
            )]),
            ExecutionTopology::WholeDomainSlidingTokenKv,
        ),
        (
            manifest(vec![
                token_state(
                    "full",
                    vec![0, 2],
                    TokenStorageKind::TokenKv,
                    RetentionKind::Full,
                    None,
                ),
                token_state(
                    "sliding",
                    vec![1, 3],
                    TokenStorageKind::TokenKv,
                    RetentionKind::Sliding,
                    Some(18),
                ),
            ]),
            ExecutionTopology::WholeDomainFullSlidingTokenKv,
        ),
        (
            manifest(vec![
                token_state(
                    "full",
                    vec![0, 2],
                    TokenStorageKind::TokenKv,
                    RetentionKind::Full,
                    None,
                ),
                AttentionStateSpec {
                    name: "mamba".into(),
                    layers: vec![1, 3],
                    storage: AttentionStateStorage::Recurrent {
                        family: RecurrentFamily::Mamba,
                        state_bytes_per_layer: 256,
                        checkpoint_slots_per_request: 2,
                    },
                },
            ]),
            ExecutionTopology::WholeDomainFullTokenKvMamba,
        ),
        (
            manifest(vec![
                token_state(
                    "full",
                    vec![0, 3],
                    TokenStorageKind::TokenKv,
                    RetentionKind::Full,
                    None,
                ),
                AttentionStateSpec {
                    name: "gdn".into(),
                    layers: vec![1, 2],
                    storage: AttentionStateStorage::Recurrent {
                        family: RecurrentFamily::Gdn,
                        state_bytes_per_layer: 512,
                        checkpoint_slots_per_request: 2,
                    },
                },
                AttentionStateSpec {
                    name: "convolution".into(),
                    layers: vec![1, 2],
                    storage: AttentionStateStorage::Convolution {
                        state_bytes_per_layer: 128,
                        kernel_width: 4,
                        checkpoint_slots_per_request: 2,
                    },
                },
            ]),
            ExecutionTopology::WholeDomainFullTokenKvGdnConvolution,
        ),
    ];
    let target = target(all_topologies());
    for (manifest, expected) in cases {
        let signature = derive_execution_signature(&manifest).unwrap();
        signature.validate().unwrap();
        let binding = admit_runtime_manifest_for_target(&manifest, &target)
            .unwrap_or_else(|error| panic!("failed to admit {expected:?}: {error}"));
        assert_eq!(binding.execution_topology, expected);
        assert_eq!(binding.manifest_fingerprint, manifest.fingerprint);
        assert_eq!(binding.target_contract_fingerprint, target.fingerprint);
        binding
            .validate_against_target(&manifest, &target)
            .unwrap();
    }
}

#[test]
fn rejects_unsupported_storage_fixed_state_and_target_topology() {
    let latent_sliding = manifest(vec![token_state(
        "latent",
        vec![0],
        TokenStorageKind::LatentKv,
        RetentionKind::Sliding,
        Some(18),
    )]);
    assert!(matches!(
        admit_runtime_manifest_for_target(&latent_sliding, &target(all_topologies())),
        Err(TargetAdmissionError::UnsupportedManifestTopology)
    ));

    let unsupported_fixed = manifest(vec![
        token_state(
            "full",
            vec![0],
            TokenStorageKind::TokenKv,
            RetentionKind::Full,
            None,
        ),
        AttentionStateSpec {
            name: "linear".into(),
            layers: vec![1],
            storage: AttentionStateStorage::Recurrent {
                family: RecurrentFamily::LinearAttention,
                state_bytes_per_layer: 128,
                checkpoint_slots_per_request: 2,
            },
        },
    ]);
    assert!(matches!(
        admit_runtime_manifest_for_target(&unsupported_fixed, &target(all_topologies())),
        Err(TargetAdmissionError::UnsupportedManifestTopology)
    ));

    let full = manifest(vec![token_state(
        "full",
        vec![0],
        TokenStorageKind::TokenKv,
        RetentionKind::Full,
        None,
    )]);
    let sliding_only = target(vec![ExecutionTopology::WholeDomainSlidingTokenKv]);
    assert!(matches!(
        admit_runtime_manifest_for_target(&full, &sliding_only),
        Err(TargetAdmissionError::UnsupportedTargetTopology(
            ExecutionTopology::WholeDomainFullTokenKv
        ))
    ));
}

#[test]
fn rejects_noncanonical_target_and_tampered_signature_or_binding() {
    let duplicate = RuntimeTarget::new(
        RuntimeTargetIdentity {
            id: "sglang".into(),
            contract_version: 4,
        },
        RuntimeAdmissionProfile {
            id: "eager-single-device-bf16-nhd".into(),
            version: 1,
        },
        16,
        vec![RUNTIME_MANIFEST_VERSION],
        REQUIRED_WIRE_VERSION,
        RuntimeTarget::packaged_sglang()
            .unwrap()
            .supported_capabilities,
        vec![
            ExecutionTopology::WholeDomainFullTokenKv,
            ExecutionTopology::WholeDomainFullTokenKv,
        ],
    );
    assert!(matches!(
        duplicate,
        Err(TargetAdmissionError::NonCanonicalTargetLists)
    ));

    let manifest = manifest(vec![token_state(
        "full",
        vec![0],
        TokenStorageKind::TokenKv,
        RetentionKind::Full,
        None,
    )]);
    let target = target(all_topologies());
    let binding = admit_runtime_manifest_for_target(&manifest, &target).unwrap();
    let mut signature = binding.execution_signature.clone();
    signature.page_tokens = 8;
    signature.fingerprint = signature.computed_fingerprint().unwrap();
    assert!(matches!(
        signature.validate(),
        Err(TargetAdmissionError::InvalidExecutionSignature(_)
            | TargetAdmissionError::AttentionState(_)
            | TargetAdmissionError::Plan(_))
    ));

    let mut forged = binding.clone();
    forged.manifest_fingerprint = format!("sha256:{}", "0".repeat(64));
    forged.fingerprint = forged.computed_fingerprint().unwrap();
    assert!(matches!(
        forged.validate(),
        Err(TargetAdmissionError::BindingIdentityMismatch)
    ));
    let other_target = RuntimeTarget::new(
        RuntimeTargetIdentity {
            id: "other-engine".into(),
            contract_version: 1,
        },
        RuntimeAdmissionProfile {
            id: "test-profile".into(),
            version: 1,
        },
        16,
        vec![RUNTIME_MANIFEST_VERSION],
        REQUIRED_WIRE_VERSION,
        RuntimeTarget::packaged_sglang()
            .unwrap()
            .supported_capabilities,
        all_topologies(),
    )
    .unwrap();
    let other_binding =
        admit_runtime_manifest_for_target(&manifest, &other_target).unwrap();
    assert!(matches!(
        other_binding.validate(),
        Err(TargetAdmissionError::BindingIdentityMismatch)
    ));
    assert!(matches!(
        RuntimeBinding::from_json(&serde_json::to_vec(&other_binding).unwrap()),
        Err(TargetAdmissionError::BindingIdentityMismatch)
    ));
    let mut wrong_contract = target.clone();
    wrong_contract.target.contract_version += 1;
    wrong_contract.fingerprint = wrong_contract.computed_fingerprint().unwrap();
    assert!(matches!(
        binding.validate_against_target(&manifest, &wrong_contract),
        Err(TargetAdmissionError::BindingMismatch)
    ));
    let mut unsupported_program = binding.execution_signature.clone();
    unsupported_program.token_classes[0].address = ExecutionAddressProgram::Pinned;
    unsupported_program.fingerprint = unsupported_program.computed_fingerprint().unwrap();
    assert!(matches!(
        unsupported_program.validate(),
        Err(TargetAdmissionError::InvalidExecutionSignature(_))
    ));
}

#[test]
fn strict_wire_rejects_unknown_duplicate_unsorted_and_stale_fields() {
    let mut unsorted = target(all_topologies());
    unsorted.supported_topologies.swap(0, 1);
    unsorted.fingerprint = unsorted.computed_fingerprint().unwrap();
    assert!(matches!(
        unsorted.validate(),
        Err(TargetAdmissionError::NonCanonicalTargetLists)
    ));
    let mut stale = target(all_topologies());
    stale.page_tokens = 8;
    assert!(matches!(
        stale.validate(),
        Err(TargetAdmissionError::FingerprintMismatch { .. })
    ));

    let manifest = manifest(vec![token_state(
        "full",
        vec![0],
        TokenStorageKind::TokenKv,
        RetentionKind::Full,
        None,
    )]);
    let binding =
        admit_runtime_manifest_for_target(&manifest, &target(all_topologies())).unwrap();
    let binding_json = serde_json::to_string(&binding).unwrap();
    let duplicate_binding =
        binding_json.replacen("{\"schema\":", "{\"schema\":\"duplicate\",\"schema\":", 1);
    assert!(matches!(
        RuntimeBinding::from_json(duplicate_binding.as_bytes()),
        Err(TargetAdmissionError::Json(_))
    ));
    let mut unknown_binding = serde_json::to_value(binding).unwrap();
    unknown_binding
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), true.into());
    assert!(matches!(
        RuntimeBinding::from_json(&serde_json::to_vec(&unknown_binding).unwrap()),
        Err(TargetAdmissionError::Json(_))
    ));
}

#[test]
fn runtime_manifest_golden_fingerprint_is_stable() {
    let mixed = crate::runtime_manifest::compile_runtime_manifest(AttentionStatePlanInput {
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
    })
    .unwrap();
    assert_eq!(
        mixed.fingerprint,
        "sha256:2182b85ff1da2391854fcd9927e4a0882ca06cc041a27199b3e207545dc5b082"
    );
}

fn chunked_retention_program(chunk_tokens: i64) -> crate::retention::RetentionProgramInput {
    crate::retention::RetentionProgramInput {
        schema: "orbitkv.retention-ir.v1".into(),
        page_tokens: 16,
        states: vec![crate::retention::RetentionStateDecl {
            name: "chunked".into(),
            layers: vec![0, 1],
            kv_head_range: None,
            bytes_per_token_per_layer: 128,
            may_read: crate::retention::Predicate::Equal {
                lhs: crate::retention::IntExpr::FloorDiv {
                    value: Box::new(crate::retention::IntExpr::QueryPosition),
                    divisor: chunk_tokens,
                },
                rhs: crate::retention::IntExpr::FloorDiv {
                    value: Box::new(crate::retention::IntExpr::KeyPosition),
                    divisor: chunk_tokens,
                },
            },
        }],
    }
}

#[test]
fn derives_and_admits_exact_chunked_topology() {
    let manifest = crate::runtime_manifest::compile_retention_runtime_manifest(
        chunked_retention_program(32),
    )
    .unwrap();
    let signature = derive_execution_signature(&manifest).unwrap();
    assert_eq!(signature.manifest_version, RUNTIME_MANIFEST_VERSION);
    assert_eq!(signature.page_tokens, 16);
    assert_eq!(signature.token_classes.len(), 1);
    assert_eq!(signature.token_states.len(), 1);
    assert!(signature.fixed_states.is_empty());
    assert_eq!(
        classify_execution_signature(&signature).unwrap(),
        ExecutionTopology::WholeDomainChunkedTokenKv
    );

    let binding = admit_runtime_manifest(&manifest).unwrap();
    assert_eq!(binding.target.id, "sglang");
    assert_eq!(binding.admission_profile.version, 1);
    assert_eq!(binding.required_wire_version, REQUIRED_WIRE_VERSION);
    binding.validate_against(&manifest).unwrap();
}

#[test]
fn rejects_non_exact_chunked_projection_constraints() {
    let manifest = crate::runtime_manifest::compile_retention_runtime_manifest(
        chunked_retention_program(32),
    )
    .unwrap();
    let mut signature = derive_execution_signature(&manifest).unwrap();
    let ExecutionTokenBackend::TokenSlots { components, .. } =
        &mut signature.token_states[0].backend;
    components.push(ExecutionStateComponent {
        name: "key".into(),
        bytes_per_token_per_layer: 64,
    });
    signature.fingerprint = signature.computed_fingerprint().unwrap();
    assert!(matches!(
        signature.validate(),
        Err(TargetAdmissionError::InvalidExecutionSignature(_))
    ));
}

#[test]
fn rejects_targets_without_canonical_manifest_version_or_capability() {
    let manifest = crate::runtime_manifest::compile_retention_runtime_manifest(
        chunked_retention_program(32),
    )
    .unwrap();

    let mut stale_version = RuntimeTarget::packaged_sglang().unwrap();
    stale_version.supported_manifest_versions = vec![RUNTIME_MANIFEST_VERSION - 1];
    stale_version.fingerprint = stale_version.computed_fingerprint().unwrap();
    stale_version.validate().unwrap();
    let error = admit_runtime_manifest_for_target(&manifest, &stale_version).unwrap_err();
    assert_eq!(
        error.to_string(),
        "runtime target contract sglang@4 with admission profile \
         eager-single-device-bf16-nhd@1 does not support manifest version 3"
    );

    let mut missing_capability = RuntimeTarget::packaged_sglang().unwrap();
    missing_capability.supported_capabilities.retain(|capability| {
        *capability != RuntimeCapability::ResettableArenaAddressing
    });
    missing_capability.fingerprint = missing_capability.computed_fingerprint().unwrap();
    missing_capability.validate().unwrap();
    assert!(matches!(
        admit_runtime_manifest_for_target(&manifest, &missing_capability),
        Err(TargetAdmissionError::UnsupportedTargetCapability(
            RuntimeCapability::ResettableArenaAddressing
        ))
    ));
}

#[test]
fn packaged_sglang_target_wire_is_canonical() {
    let target = RuntimeTarget::packaged_sglang().unwrap();
    assert_eq!(target.target.id, "sglang");
    assert_eq!(target.target.contract_version, 4);
    assert_eq!(target.supported_manifest_versions, vec![RUNTIME_MANIFEST_VERSION]);
    assert_eq!(target.required_wire_version, REQUIRED_WIRE_VERSION);
    assert_eq!(
        target.fingerprint,
        "sha256:ac915458195e757e477cf04866dae76147cd71a7472661e0d791e9c4474173ba"
    );
    let value = serde_json::to_value(&target).unwrap();
    let object = value.as_object().unwrap();
    assert_eq!(object.len(), 10);
    for key in [
        "schema",
        "version",
        "fingerprint",
        "target",
        "admission_profile",
        "page_tokens",
        "supported_manifest_versions",
        "required_wire_version",
        "supported_capabilities",
        "supported_topologies",
    ] {
        assert!(object.contains_key(key));
    }

    let manifest = manifest(vec![token_state(
        "sliding",
        vec![0, 1],
        TokenStorageKind::TokenKv,
        RetentionKind::Sliding,
        Some(18),
    )]);
    admit_runtime_manifest(&manifest).unwrap();
}
