fn assert_no_manager_capability(value: &serde_json::Value) {
    match value {
        serde_json::Value::Array(items) => {
            for item in items {
                assert_no_manager_capability(item);
            }
        }
        serde_json::Value::Object(fields) => {
            assert!(
                !fields.contains_key("reclamation"),
                "public DTO leaked a reclamation capability: {value}"
            );
            assert!(
                !(fields.contains_key("engine_epoch")
                    && fields.contains_key("slot")
                    && fields.contains_key("generation")),
                "public DTO leaked a manager capability lease: {value}"
            );
            for item in fields.values() {
                assert_no_manager_capability(item);
            }
        }
        _ => {}
    }
}

#[test]
fn public_retirement_dtos_hide_manager_reclamation_capabilities() {
    let page = PageLease {
        engine_epoch: 7,
        pool_epoch: 8,
        generation: 9,
        page_id: 10,
        pool_id: 11,
    };
    let retirement = EngineRetirement {
        page,
        class_id: 12,
        backend_domain: 13,
        logical_ordinal: 14,
        backend_index: 15,
        token_begin: 16,
        token_end_exclusive: 17,
        completion_domain: 18,
        completion_value: 19,
    };
    let retirement_evidence = EngineRetirementEvidence {
        page,
        backend_domain: retirement.backend_domain,
        acknowledged: true,
        backend_index: retirement.backend_index,
    };
    let publication_id = EnginePublicationId::from_parts(7, 1);
    let release_id = EngineReleaseId::from_parts(7, 2);
    let values = [
        serde_json::to_value(retirement).expect("serialize retirement"),
        serde_json::to_value(EngineBatchPublication {
            publication_id,
            batch_id: EngineBatchId::from_parts(7, 3),
            steps: Box::new([]),
            retirements: Box::new([retirement]),
        })
        .expect("serialize publication"),
        serde_json::to_value(EnginePublicationEvidence {
            publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::new([retirement_evidence]),
        })
        .expect("serialize publication evidence"),
        serde_json::to_value(EngineReleasePlan {
            release_id,
            releases: Box::new([]),
            retirements: Box::new([retirement]),
        })
        .expect("serialize release plan"),
        serde_json::to_value(EngineReleaseEvidence {
            release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::new([retirement_evidence]),
        })
        .expect("serialize release evidence"),
    ];

    for value in values {
        assert_no_manager_capability(&value);
        let wire = value.to_string();
        for forbidden in [
            "ReclamationLease",
            "ReclamationCertificate",
            "ReclamationReceipt",
        ] {
            assert!(!wire.contains(forbidden), "public DTO leaked {forbidden}");
        }
    }
}

#[test]
fn public_fixed_state_dtos_hide_transition_capabilities() {
    let slot = crate::StateSlotLease {
        engine_epoch: 7,
        pool_epoch: 8,
        generation: 9,
        slot_id: 10,
        pool_id: 11,
    };
    let values = [
        serde_json::to_value(EngineFixedStatePlan {
            state_id: 2,
            source: None,
            destination: slot,
            byte_count: 64,
        })
        .unwrap(),
        serde_json::to_value(EngineFixedStateEvidence {
            state_id: 2,
            source: None,
            destination: slot,
            byte_count: 64,
            observed: true,
            written: true,
        })
        .unwrap(),
        serde_json::to_value(EngineFixedStatePublication {
            state_id: 2,
            request_id: EngineRequestId(12),
            slot,
        })
        .unwrap(),
    ];
    for value in values {
        let object = value.to_string();
        for forbidden in [
            "transition",
            "StateTransitionLease",
            "StateRetirementLease",
            "StateRetirementCertificate",
        ] {
            assert!(!object.contains(forbidden), "public DTO leaked {forbidden}");
        }
    }
}
