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

fn target(topologies: Vec<ExecutionTopologyV1>) -> RuntimeTargetContractV1 {
    RuntimeTargetContractV1::new(
        RuntimeTargetIdentityV1 {
            id: "sglang.abi8".into(),
            contract_version: 1,
        },
        RuntimeAdmissionProfileV1 {
            id: "eager-single-device-bf16-nhd".into(),
            version: 1,
        },
        16,
        topologies,
    )
    .unwrap()
}

fn all_topologies() -> Vec<ExecutionTopologyV1> {
    vec![
        ExecutionTopologyV1::WholeDomainFullLatentKv,
        ExecutionTopologyV1::WholeDomainFullSlidingTokenKv,
        ExecutionTopologyV1::WholeDomainFullTokenKv,
        ExecutionTopologyV1::WholeDomainFullTokenKvGdnConvolution,
        ExecutionTopologyV1::WholeDomainFullTokenKvMamba,
        ExecutionTopologyV1::WholeDomainSlidingTokenKv,
    ]
}

#[test]
#[allow(clippy::too_many_lines)]
fn derives_and_admits_every_closed_v1_topology() {
    let cases = [
        (
            manifest(vec![token_state(
                "latent",
                vec![0, 1],
                TokenStorageKind::LatentKv,
                RetentionKind::Full,
                None,
            )]),
            ExecutionTopologyV1::WholeDomainFullLatentKv,
        ),
        (
            manifest(vec![token_state(
                "full",
                vec![0, 1],
                TokenStorageKind::TokenKv,
                RetentionKind::Full,
                None,
            )]),
            ExecutionTopologyV1::WholeDomainFullTokenKv,
        ),
        (
            manifest(vec![token_state(
                "sliding",
                vec![0, 1],
                TokenStorageKind::TokenKv,
                RetentionKind::Sliding,
                Some(18),
            )]),
            ExecutionTopologyV1::WholeDomainSlidingTokenKv,
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
            ExecutionTopologyV1::WholeDomainFullSlidingTokenKv,
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
            ExecutionTopologyV1::WholeDomainFullTokenKvMamba,
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
            ExecutionTopologyV1::WholeDomainFullTokenKvGdnConvolution,
        ),
    ];
    let target = target(all_topologies());
    for (manifest, expected) in cases {
        let signature = derive_execution_signature(&manifest).unwrap();
        signature.validate().unwrap();
        let binding = admit_runtime_manifest(&manifest, &target)
            .unwrap_or_else(|error| panic!("failed to admit {expected:?}: {error}"));
        assert_eq!(binding.execution_topology, expected);
        assert_eq!(binding.manifest_fingerprint, manifest.fingerprint);
        assert_eq!(binding.target_contract_fingerprint, target.fingerprint);
        binding.validate_against(&manifest, &target).unwrap();
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
        admit_runtime_manifest(&latent_sliding, &target(all_topologies())),
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
        admit_runtime_manifest(&unsupported_fixed, &target(all_topologies())),
        Err(TargetAdmissionError::UnsupportedManifestTopology)
    ));

    let full = manifest(vec![token_state(
        "full",
        vec![0],
        TokenStorageKind::TokenKv,
        RetentionKind::Full,
        None,
    )]);
    let sliding_only = target(vec![ExecutionTopologyV1::WholeDomainSlidingTokenKv]);
    assert!(matches!(
        admit_runtime_manifest(&full, &sliding_only),
        Err(TargetAdmissionError::UnsupportedTargetTopology(
            ExecutionTopologyV1::WholeDomainFullTokenKv
        ))
    ));
}

#[test]
fn rejects_noncanonical_contract_and_tampered_signature_or_binding() {
    let duplicate = RuntimeTargetContractV1::new(
        RuntimeTargetIdentityV1 {
            id: "sglang.abi8".into(),
            contract_version: 1,
        },
        RuntimeAdmissionProfileV1 {
            id: "eager-single-device-bf16-nhd".into(),
            version: 1,
        },
        16,
        vec![
            ExecutionTopologyV1::WholeDomainFullTokenKv,
            ExecutionTopologyV1::WholeDomainFullTokenKv,
        ],
    );
    assert!(matches!(
        duplicate,
        Err(TargetAdmissionError::NonCanonicalTopologies)
    ));

    let manifest = manifest(vec![token_state(
        "full",
        vec![0],
        TokenStorageKind::TokenKv,
        RetentionKind::Full,
        None,
    )]);
    let target = target(all_topologies());
    let binding = admit_runtime_manifest(&manifest, &target).unwrap();
    let mut signature = binding.execution_signature.clone();
    signature.page_tokens = 8;
    signature.fingerprint = signature.computed_fingerprint().unwrap();
    assert!(matches!(
        signature.validate(),
        Err(
            TargetAdmissionError::InvalidExecutionSignature(_)
                | TargetAdmissionError::AttentionState(_)
                | TargetAdmissionError::Plan(_)
        )
    ));

    let mut forged = binding.clone();
    forged.manifest_fingerprint = format!("sha256:{}", "0".repeat(64));
    forged.fingerprint = forged.computed_fingerprint().unwrap();
    assert!(matches!(
        forged.validate(),
        Err(TargetAdmissionError::BindingIdentityMismatch)
    ));
    let mut wrong_contract = target.clone();
    wrong_contract.target.contract_version = 2;
    wrong_contract.fingerprint = wrong_contract.computed_fingerprint().unwrap();
    assert!(matches!(
        binding.validate_against(&manifest, &wrong_contract),
        Err(TargetAdmissionError::BindingMismatch)
    ));
    let mut unsupported_program = binding.execution_signature.clone();
    unsupported_program.token_classes[0].address = ExecutionAddressProgramV1::Pinned;
    unsupported_program.fingerprint = unsupported_program.computed_fingerprint().unwrap();
    assert!(matches!(
        unsupported_program.validate(),
        Err(TargetAdmissionError::InvalidExecutionSignature(_))
    ));
}

#[test]
fn strict_wire_rejects_unknown_duplicate_unsorted_and_stale_fields() {
    let contract = target(all_topologies());
    let encoded = serde_json::to_vec(&contract).unwrap();
    RuntimeTargetContractV1::from_json(&encoded).unwrap();
    let mut unknown = serde_json::to_value(&contract).unwrap();
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), true.into());
    assert!(matches!(
        RuntimeTargetContractV1::from_json(&serde_json::to_vec(&unknown).unwrap()),
        Err(TargetAdmissionError::Json(_))
    ));
    let duplicate = String::from_utf8(encoded.clone()).unwrap().replacen(
        "{\"schema\":",
        "{\"schema\":\"duplicate\",\"schema\":",
        1,
    );
    assert!(matches!(
        RuntimeTargetContractV1::from_json(duplicate.as_bytes()),
        Err(TargetAdmissionError::Json(_))
    ));
    let mut unsorted = contract.clone();
    unsorted.supported_topologies.swap(0, 1);
    unsorted.fingerprint = unsorted.computed_fingerprint().unwrap();
    assert!(matches!(
        RuntimeTargetContractV1::from_json(&serde_json::to_vec(&unsorted).unwrap()),
        Err(TargetAdmissionError::NonCanonicalTopologies)
    ));
    let mut stale = contract;
    stale.page_tokens = 8;
    assert!(matches!(
        RuntimeTargetContractV1::from_json(&serde_json::to_vec(&stale).unwrap()),
        Err(TargetAdmissionError::FingerprintMismatch { .. })
    ));

    let manifest = manifest(vec![token_state(
        "full",
        vec![0],
        TokenStorageKind::TokenKv,
        RetentionKind::Full,
        None,
    )]);
    let binding = admit_runtime_manifest(&manifest, &target(all_topologies())).unwrap();
    let binding_json = serde_json::to_string(&binding).unwrap();
    let duplicate_binding =
        binding_json.replacen("{\"schema\":", "{\"schema\":\"duplicate\",\"schema\":", 1);
    assert!(matches!(
        RuntimeTargetBindingV1::from_json(duplicate_binding.as_bytes()),
        Err(TargetAdmissionError::Json(_))
    ));
    let mut unknown_binding = serde_json::to_value(binding).unwrap();
    unknown_binding
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), true.into());
    assert!(matches!(
        RuntimeTargetBindingV1::from_json(&serde_json::to_vec(&unknown_binding).unwrap()),
        Err(TargetAdmissionError::Json(_))
    ));
}

#[test]
fn runtime_manifest_v1_golden_fingerprint_remains_unchanged() {
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
        "sha256:b936744c23c8fc874f477900bef47c45c2b027ea7bb175f943438eeacfe76946"
    );
}
