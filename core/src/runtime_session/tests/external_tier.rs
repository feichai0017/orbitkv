use super::*;

fn external_key(tag: u8, boundary: u64) -> ExternalObjectKey {
    ExternalObjectKey {
        namespace: [0x51; 32],
        digest: [tag; 32],
        boundary,
    }
}

fn export_receipts(plan: &ExternalExportPlan) -> Box<[ExternalExportReceipt]> {
    plan.copies
        .iter()
        .enumerate()
        .map(|(index, copy)| ExternalExportReceipt {
            copy: *copy,
            checksum: [u8::try_from(index + 1).unwrap(); 32],
            copied: true,
            durable: true,
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn ready_full_request(
    request_id: EngineRequestId,
    boundary: u64,
) -> (RuntimeSession, [BackendArenaRegistration; 1]) {
    let backends = [backend(0, 91, 8, 4_000)];
    let mut session = session(&full_plan(), &backends, 1);
    session.acquire_requests(&[request_id]).expect("acquire");
    let (_, publication) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id,
            target_boundary: boundary,
        }],
        31,
        1,
    );
    confirm_publication(&mut session, &publication);
    (session, backends)
}

#[test]
fn external_export_pins_exact_pages_and_publishes_durable_replica() {
    let request_id = EngineRequestId(9_001);
    let (mut session, backends) = ready_full_request(request_id, 32);
    let key = external_key(1, 32);
    let target = ExternalReplicaTarget {
        storage_domain: 7,
        object_index: 44,
        base_offset: 8_192,
    };
    let plan = session
        .prepare_external_export(request_id, key, target)
        .expect("prepare export");

    assert_eq!(plan.copies.len(), 2);
    assert_eq!(plan.total_bytes, 4_096);
    assert_eq!(plan.copies[0].byte_count, 2_048);
    assert_eq!(plan.copies[0].copy_index, 0);
    assert_eq!(plan.copies[1].destination_offset, 10_240);
    assert_eq!(session.stats().total_reader_pins, 2);
    let public_json = serde_json::to_string(&plan).expect("serialize public plan");
    for private_field in [
        "engine_epoch",
        "pool_epoch",
        "generation",
        "page_id",
        "pool_id",
    ] {
        assert!(!public_json.contains(private_field));
    }
    assert!(matches!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 33,
        }]),
        Err(RuntimeSessionError::RequestNotReady { .. })
    ));

    let replica = session
        .complete_external_export(
            ExternalTransferCompletion {
                transfer_id: plan.transfer_id,
                completion_domain: 41,
                completion_value: 1,
                confirmed: true,
            },
            &export_receipts(&plan),
        )
        .expect("complete export");
    assert_eq!(replica.total_bytes, 4_096);
    assert_eq!(replica.pages.len(), 2);
    assert_eq!(session.external_replica(key), Some(&replica));
    assert_eq!(session.stats().total_reader_pins, 0);

    let (_, publication) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id,
            target_boundary: 33,
        }],
        31,
        2,
    );
    confirm_publication(&mut session, &publication);
}

#[test]
fn malformed_external_receipt_preserves_pins_and_unobserved_abort_recovers() {
    let request_id = EngineRequestId(9_002);
    let (mut session, _) = ready_full_request(request_id, 16);
    let key = external_key(2, 16);
    let plan = session
        .prepare_external_export(
            request_id,
            key,
            ExternalReplicaTarget {
                storage_domain: 8,
                object_index: 45,
                base_offset: 0,
            },
        )
        .expect("prepare export");
    let mut receipts = export_receipts(&plan);
    receipts[0].durable = false;
    assert_eq!(
        session.complete_external_export(
            ExternalTransferCompletion {
                transfer_id: plan.transfer_id,
                completion_domain: 42,
                completion_value: 1,
                confirmed: true,
            },
            &receipts,
        ),
        Err(ExternalTierError::ReceiptMismatch)
    );
    assert_eq!(session.stats().total_reader_pins, 1);
    assert_eq!(session.external_replica(key), None);
    assert_eq!(
        session.abort_external_export(ExternalExportAbortEvidence {
            transfer_id: plan.transfer_id,
            backend_unobserved: false,
        }),
        Err(ExternalTierError::AbortObservationUnknown)
    );
    session
        .abort_external_export(ExternalExportAbortEvidence {
            transfer_id: plan.transfer_id,
            backend_unobserved: true,
        })
        .expect("abort unobserved export");
    assert_eq!(session.stats().total_reader_pins, 0);
}

#[test]
fn external_replica_catalog_requires_exact_deletion_ack() {
    let request_id = EngineRequestId(9_003);
    let (mut session, _) = ready_full_request(request_id, 16);
    let key = external_key(3, 16);
    let target = ExternalReplicaTarget {
        storage_domain: 9,
        object_index: 46,
        base_offset: 64,
    };
    let plan = session
        .prepare_external_export(request_id, key, target)
        .expect("prepare export");
    session
        .complete_external_export(
            ExternalTransferCompletion {
                transfer_id: plan.transfer_id,
                completion_domain: 43,
                completion_value: 1,
                confirmed: true,
            },
            &export_receipts(&plan),
        )
        .expect("complete export");

    assert_eq!(
        session.confirm_external_replica_deletion(ExternalReplicaDeletionEvidence {
            key,
            target,
            deleted: false,
        }),
        Err(ExternalTierError::DeletionNotConfirmed)
    );
    let deleted = session
        .confirm_external_replica_deletion(ExternalReplicaDeletionEvidence {
            key,
            target,
            deleted: true,
        })
        .expect("confirm deletion");
    assert_eq!(deleted.key, key);
    assert_eq!(session.external_replica(key), None);
}

#[test]
fn external_transfer_identity_and_completion_frontier_fail_closed() {
    let request_id = EngineRequestId(9_004);
    let (mut session, _) = ready_full_request(request_id, 16);
    let plan = session
        .prepare_external_export(
            request_id,
            external_key(4, 16),
            ExternalReplicaTarget {
                storage_domain: 10,
                object_index: 47,
                base_offset: 0,
            },
        )
        .expect("prepare export");
    let foreign = ExternalTransferId::from_parts(
        plan.transfer_id.session_epoch() + 1,
        plan.transfer_id.sequence(),
    );
    assert_eq!(
        session.abort_external_export(ExternalExportAbortEvidence {
            transfer_id: foreign,
            backend_unobserved: true,
        }),
        Err(ExternalTierError::ForeignTransfer)
    );
    assert_eq!(session.stats().total_reader_pins, 1);

    assert_eq!(
        session.complete_external_export(
            ExternalTransferCompletion {
                transfer_id: plan.transfer_id,
                completion_domain: 31,
                completion_value: 1,
                confirmed: true,
            },
            &export_receipts(&plan),
        ),
        Err(ExternalTierError::Manager(
            KvManagerError::CompletionFrontierDidNotAdvance {
                completion_domain: 31,
                previous: 1,
                received: 1,
            }
        ))
    );
    assert_eq!(session.stats().total_reader_pins, 1);
    session
        .abort_external_export(ExternalExportAbortEvidence {
            transfer_id: plan.transfer_id,
            backend_unobserved: true,
        })
        .expect("abort after rejected completion");
    assert_eq!(session.stats().total_reader_pins, 0);
    assert_eq!(
        session.abort_external_export(ExternalExportAbortEvidence {
            transfer_id: plan.transfer_id,
            backend_unobserved: true,
        }),
        Err(ExternalTierError::StaleTransfer)
    );
}

#[test]
fn external_catalog_rejects_duplicate_storage_target_until_deletion() {
    let first = EngineRequestId(9_005);
    let second = EngineRequestId(9_006);
    let backends = [backend(0, 92, 8, 5_000)];
    let mut session = session(&full_plan(), &backends, 2);
    session.acquire_requests(&[first, second]).expect("acquire");
    let (_, publication) = append(
        &mut session,
        &backends,
        &[
            EngineAppendIntent {
                request_id: first,
                target_boundary: 16,
            },
            EngineAppendIntent {
                request_id: second,
                target_boundary: 16,
            },
        ],
        51,
        1,
    );
    confirm_publication(&mut session, &publication);
    let target = ExternalReplicaTarget {
        storage_domain: 11,
        object_index: 48,
        base_offset: 0,
    };
    let first_plan = session
        .prepare_external_export(first, external_key(5, 16), target)
        .expect("first export");
    assert_eq!(
        session.prepare_external_export(
            second,
            external_key(6, 16),
            ExternalReplicaTarget {
                base_offset: 4_096,
                ..target
            },
        ),
        Err(ExternalTierError::DuplicateObject)
    );
    session
        .abort_external_export(ExternalExportAbortEvidence {
            transfer_id: first_plan.transfer_id,
            backend_unobserved: true,
        })
        .expect("abort first export");
    session
        .prepare_external_export(second, external_key(6, 16), target)
        .expect("target reusable after abort");
}
