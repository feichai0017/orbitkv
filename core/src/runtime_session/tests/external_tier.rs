use super::*;

fn external_key(tag: u8, boundary: u64) -> ExternalObjectKey {
    ExternalObjectKey {
        namespace: [0x51; 32],
        digest: [tag; 32],
        plan_fingerprint: full_plan().fingerprint_digest(),
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

fn restore_receipts(plan: &ExternalRestorePlan) -> Box<[ExternalRestoreReceipt]> {
    plan.copies
        .iter()
        .map(|copy| ExternalRestoreReceipt {
            copy: *copy,
            checksum: copy.expected_checksum,
            copied: true,
            ordered_before_publish: true,
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
    assert_eq!(session.external_tier_stats().pending_exports, 1);
    assert_eq!(session.external_tier_stats().pinned_export_pages, 2);
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
    assert_eq!(session.external_tier_stats().pending_exports, 0);
    assert_eq!(session.external_tier_stats().replicas, 1);

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

#[test]
fn external_object_key_rejects_incompatible_compiled_plan() {
    let request_id = EngineRequestId(9_014);
    let (mut session, _) = ready_full_request(request_id, 16);
    let mut key = external_key(11, 16);
    key.plan_fingerprint[0] ^= 0xff;
    assert_eq!(
        session.prepare_external_export(
            request_id,
            key,
            ExternalReplicaTarget {
                storage_domain: 16,
                object_index: 53,
                base_offset: 0,
            },
        ),
        Err(ExternalTierError::InvalidDescriptor)
    );
    assert_eq!(session.stats().total_reader_pins, 0);
}

#[test]
fn cross_session_catalog_admission_rejects_tampered_metadata() {
    let source = EngineRequestId(9_015);
    let (mut source_session, _) = ready_full_request(source, 16);
    let key = external_key(12, 16);
    let export = source_session
        .prepare_external_export(
            source,
            key,
            ExternalReplicaTarget {
                storage_domain: 17,
                object_index: 54,
                base_offset: 0,
            },
        )
        .unwrap();
    let replica = source_session
        .complete_external_export(
            ExternalTransferCompletion {
                transfer_id: export.transfer_id,
                completion_domain: 91,
                completion_value: 1,
                confirmed: true,
            },
            &export_receipts(&export),
        )
        .unwrap();

    let target_backends = [backend(0, 197, 4, 19_000)];
    let mut target_session = session(&full_plan(), &target_backends, 1);
    let mut wrong_plan = replica.clone();
    wrong_plan.key.plan_fingerprint[0] ^= 0xff;
    assert_eq!(
        target_session.admit_external_replica(wrong_plan),
        Err(ExternalTierError::InvalidDescriptor)
    );
    let mut wrong_offset = replica.clone();
    wrong_offset.pages[0].storage_offset += 1;
    assert_eq!(
        target_session.admit_external_replica(wrong_offset),
        Err(ExternalTierError::InvalidDescriptor)
    );
    target_session
        .admit_external_replica(replica)
        .expect("admit valid metadata");
}

#[test]
fn external_restore_uses_native_append_submission_and_publication() {
    let source = EngineRequestId(9_007);
    let target_request = EngineRequestId(9_008);
    let source_backends = [backend(0, 93, 4, 6_000)];
    let mut source_session = session(&full_plan(), &source_backends, 1);
    source_session
        .acquire_requests(&[source])
        .expect("acquire source");
    let (_, source_publication) = append(
        &mut source_session,
        &source_backends,
        &[EngineAppendIntent {
            request_id: source,
            target_boundary: 17,
        }],
        61,
        1,
    );
    confirm_publication(&mut source_session, &source_publication);

    let key = external_key(7, 17);
    let export = source_session
        .prepare_external_export(
            source,
            key,
            ExternalReplicaTarget {
                storage_domain: 12,
                object_index: 49,
                base_offset: 128,
            },
        )
        .expect("prepare export");
    let replica = source_session
        .complete_external_export(
            ExternalTransferCompletion {
                transfer_id: export.transfer_id,
                completion_domain: 62,
                completion_value: 1,
                confirmed: true,
            },
            &export_receipts(&export),
        )
        .expect("complete export");

    let target_backends = [backend(0, 193, 4, 16_000)];
    let mut target_session = session(&full_plan(), &target_backends, 1);
    target_session
        .acquire_requests(&[target_request])
        .expect("acquire target");
    target_session
        .admit_external_replica(replica)
        .expect("admit cross-session replica");

    let restore = target_session
        .prepare_external_restore(target_request, key)
        .expect("prepare restore");
    assert_eq!(restore.total_bytes, export.total_bytes);
    assert_eq!(restore.copies.len(), export.copies.len());
    assert_eq!(restore.total_bytes, 2_176);
    assert_eq!(restore.copies[1].byte_count, 128);
    assert_eq!(restore.copies[1].valid_token_count, 1);
    let public_json = serde_json::to_string(&restore).expect("serialize restore plan");
    for private_field in [
        "engine_epoch",
        "pool_epoch",
        "generation",
        "page_id",
        "pool_id",
    ] {
        assert!(!public_json.contains(private_field));
    }
    target_session
        .submit_external_restore(restore.transfer_id, &restore_receipts(&restore))
        .expect("submit restore");
    let publication = target_session
        .complete_external_restore(ExternalTransferCompletion {
            transfer_id: restore.transfer_id,
            completion_domain: 63,
            completion_value: 1,
            confirmed: true,
        })
        .expect("complete restore");
    assert_eq!(publication.steps[0].request_id, target_request);
    assert_eq!(publication.steps[0].boundary, 17);
    confirm_publication(&mut target_session, &publication);
    let view = target_session
        .manager
        .request_views_batch(&[target_session.requests[&target_request].view.request])
        .unwrap();
    assert_eq!(view[0].boundary, 17);
    assert_eq!(view[0].resident_count, 2);
}

#[test]
fn bad_restore_receipt_remains_abortable_and_releases_destinations() {
    let source = EngineRequestId(9_009);
    let target_request = EngineRequestId(9_010);
    let backends = [backend(0, 94, 4, 7_000)];
    let mut session = session(&full_plan(), &backends, 2);
    session
        .acquire_requests(&[source, target_request])
        .expect("acquire source and target");
    let (_, source_publication) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id: source,
            target_boundary: 16,
        }],
        71,
        1,
    );
    confirm_publication(&mut session, &source_publication);
    let key = external_key(8, 16);
    let export = session
        .prepare_external_export(
            source,
            key,
            ExternalReplicaTarget {
                storage_domain: 13,
                object_index: 50,
                base_offset: 0,
            },
        )
        .unwrap();
    session
        .complete_external_export(
            ExternalTransferCompletion {
                transfer_id: export.transfer_id,
                completion_domain: 72,
                completion_value: 1,
                confirmed: true,
            },
            &export_receipts(&export),
        )
        .unwrap();
    let free_before = session.stats().free_pages;
    let restore = session
        .prepare_external_restore(target_request, key)
        .expect("prepare restore");
    assert!(session.stats().free_pages < free_before);
    let mut receipts = restore_receipts(&restore);
    receipts[0].checksum = [0xEE; 32];
    assert_eq!(
        session.submit_external_restore(restore.transfer_id, &receipts),
        Err(ExternalTierError::RestoreReceiptMismatch)
    );
    session
        .abort_external_restore(ExternalRestoreAbortEvidence {
            transfer_id: restore.transfer_id,
            backend_unobserved: true,
        })
        .expect("abort restore");
    assert_eq!(session.stats().free_pages, free_before);
    let retry = session
        .prepare_external_restore(target_request, key)
        .expect("retry restore");
    session
        .submit_external_restore(retry.transfer_id, &restore_receipts(&retry))
        .expect("submit retry");
    assert_eq!(
        session.abort_external_restore(ExternalRestoreAbortEvidence {
            transfer_id: retry.transfer_id,
            backend_unobserved: true,
        }),
        Err(ExternalTierError::RestorePhaseMismatch)
    );
}

#[test]
fn ambiguous_external_export_keeps_pins_and_fail_stops_request() {
    let source = EngineRequestId(9_011);
    let backends = [backend(0, 95, 4, 8_000)];
    let mut export_session = session(&full_plan(), &backends, 1);
    export_session.acquire_requests(&[source]).expect("acquire");
    let (_, publication) = append(
        &mut export_session,
        &backends,
        &[EngineAppendIntent {
            request_id: source,
            target_boundary: 16,
        }],
        81,
        1,
    );
    confirm_publication(&mut export_session, &publication);
    let key = external_key(9, 16);
    let target = ExternalReplicaTarget {
        storage_domain: 14,
        object_index: 51,
        base_offset: 0,
    };
    let first = export_session
        .prepare_external_export(source, key, target)
        .expect("prepare ambiguous export");
    export_session
        .quarantine_external_export(first.transfer_id)
        .expect("quarantine export");
    assert_eq!(export_session.stats().total_reader_pins, 1);
    assert_eq!(export_session.external_tier_stats().pending_exports, 0);
    assert_eq!(export_session.external_tier_stats().quarantined_exports, 1);
    assert!(matches!(
        export_session.prepare_append_batch(&[EngineAppendIntent {
            request_id: source,
            target_boundary: 17,
        }]),
        Err(RuntimeSessionError::RequestNotReady { .. })
    ));
}

#[test]
fn ambiguous_external_restore_keeps_destinations_and_blocks_object_deletion() {
    let target_request = EngineRequestId(9_012);
    let restore_backends = [backend(0, 96, 4, 9_000)];
    let mut restore_session = session(&full_plan(), &restore_backends, 2);
    restore_session
        .acquire_requests(&[EngineRequestId(9_013), target_request])
        .expect("acquire restore source and target");
    let (_, publication) = append(
        &mut restore_session,
        &restore_backends,
        &[EngineAppendIntent {
            request_id: EngineRequestId(9_013),
            target_boundary: 16,
        }],
        83,
        1,
    );
    confirm_publication(&mut restore_session, &publication);
    let restore_key = external_key(10, 16);
    let export = restore_session
        .prepare_external_export(
            EngineRequestId(9_013),
            restore_key,
            ExternalReplicaTarget {
                storage_domain: 15,
                object_index: 52,
                base_offset: 0,
            },
        )
        .unwrap();
    restore_session
        .complete_external_export(
            ExternalTransferCompletion {
                transfer_id: export.transfer_id,
                completion_domain: 82,
                completion_value: 1,
                confirmed: true,
            },
            &export_receipts(&export),
        )
        .unwrap();
    let restore = restore_session
        .prepare_external_restore(target_request, restore_key)
        .unwrap();
    assert_eq!(
        restore_session.confirm_external_replica_deletion(ExternalReplicaDeletionEvidence {
            key: restore_key,
            target: ExternalReplicaTarget {
                storage_domain: 15,
                object_index: 52,
                base_offset: 0,
            },
            deleted: true,
        }),
        Err(ExternalTierError::ObjectBusy)
    );
    let free_after_prepare = restore_session.stats().free_pages;
    restore_session
        .quarantine_external_restore(restore.transfer_id)
        .expect("quarantine restore");
    assert_eq!(restore_session.stats().free_pages, free_after_prepare);
    assert_eq!(restore_session.external_tier_stats().pending_restores, 0);
    assert_eq!(
        restore_session.external_tier_stats().quarantined_restores,
        1
    );
    assert!(matches!(
        restore_session.prepare_append_batch(&[EngineAppendIntent {
            request_id: target_request,
            target_boundary: 1,
        }]),
        Err(RuntimeSessionError::RequestNotReady { .. })
    ));
}
