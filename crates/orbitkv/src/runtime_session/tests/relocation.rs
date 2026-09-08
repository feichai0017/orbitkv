use super::*;

fn relocation_policy() -> crate::kv_manager::RelocationPolicy {
    crate::kv_manager::RelocationPolicy::static_fragmentation(250, 8, 2, true)
}

fn prepare_relocatable(
    session: &mut RuntimeSession,
    backends: &[BackendArenaRegistration],
    requests: &[EngineRequestId],
) -> EnginePreparedRelocation {
    session.acquire_requests(requests).expect("acquire");
    let intents = requests
        .iter()
        .copied()
        .map(|request_id| EngineAppendIntent {
            request_id,
            target_boundary: 48,
        })
        .collect::<Vec<_>>();
    let (_, publication) = append(session, backends, &intents, 201, 1);
    confirm_publication(session, &publication);

    let updates = (0..48_u64)
        .filter(|token_id| token_id % 16 >= 8)
        .map(|token_id| EngineTokenDispositionUpdate {
            class_id: 0,
            token_id,
            disposition: crate::kv_manager::TokenDisposition::policy_evicted(91, 1, 7),
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let items = requests
        .iter()
        .copied()
        .map(|request_id| EngineTokenDispositionBatchItem {
            request_id,
            updates: updates.clone(),
        })
        .collect::<Vec<_>>();
    let marked = session
        .mark_token_dispositions_batch(&items)
        .expect("mark dispositions");
    assert_eq!(marked.len(), requests.len());
    assert!(marked.iter().all(|view| view.boundary == 48));

    session
        .prepare_relocation_batch(
            &requests
                .iter()
                .copied()
                .map(|request_id| EnginePrepareRelocationItem {
                    request_id,
                    class_id: 0,
                    policy: relocation_policy(),
                })
                .collect::<Vec<_>>(),
        )
        .expect("prepare relocation")
}

fn relocation_execution(prepared: &EnginePreparedRelocation) -> EngineRelocationExecutionEvidence {
    EngineRelocationExecutionEvidence {
        relocation_id: prepared.relocation_id,
        requests: prepared
            .plans
            .iter()
            .map(|plan| EngineRelocationRequestEvidence {
                request_id: plan.request_id,
                copies: plan
                    .moves
                    .iter()
                    .map(|movement| EngineRelocationCopyEvidence {
                        token_id: movement.token_id,
                        source: movement.source,
                        destination: movement.destination,
                        observed: true,
                        copied: true,
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn session_relocation_success_gates_until_exact_ack() {
    let backends = [backend(0, 401, 5, 40_000)];
    let request_id = EngineRequestId(1_001);
    let mut session = session(&full_plan(), &backends, 2);
    let prepared = prepare_relocatable(&mut session, &backends, &[request_id]);
    assert_eq!(prepared.plans.len(), 1);
    assert_eq!(prepared.plans[0].source_pages.len(), 3);
    assert_eq!(prepared.plans[0].destination_pages.len(), 2);
    assert_eq!(prepared.plans[0].moves.len(), 24);
    let wire = serde_json::to_string(&prepared).expect("serialize relocation plan");
    assert!(!wire.contains("base_snapshot"));
    assert!(!wire.contains("target_snapshot"));
    assert!(!wire.contains("\"relocation\":"));
    assert!(!wire.contains("\"request\":"));

    assert!(matches!(
        session.token_views_batch(&[EngineTokenViewQuery {
            request_id,
            class_id: 0,
            expected_boundary: 48,
        }]),
        Err(RuntimeSessionError::RequestNotReady { .. })
    ));
    let ticket = session
        .submit_relocation(&relocation_execution(&prepared))
        .expect("submit relocation");
    assert_eq!(
        session.abort_prepared_relocation(
            prepared.relocation_id,
            &[EngineRelocationAbortEvidence {
                request_id,
                backend_unobserved: true,
            }],
        ),
        Err(RuntimeSessionError::RelocationNotPrepared(
            prepared.relocation_id
        ))
    );
    assert_eq!(
        session.complete_relocation(
            prepared.relocation_id,
            EngineCompletionEvidence {
                completion_domain: 201,
                completion_value: 2,
                confirmed: false,
            },
        ),
        Err(RuntimeSessionError::Manager(
            KvManagerError::CompletionNotConfirmed
        ))
    );
    let publication = session
        .complete_relocation(
            ticket.relocation_id(),
            EngineCompletionEvidence {
                completion_domain: 201,
                completion_value: 2,
                confirmed: true,
            },
        )
        .expect("complete relocation");
    assert_eq!(publication.requests.len(), 1);
    assert_eq!(publication.retirements.len(), 3);
    assert!(matches!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 49,
        }]),
        Err(RuntimeSessionError::RequestNotReady { .. })
    ));

    let mut wrong = control_reclamation_evidence(&publication.retirements);
    wrong[0].backend_index += 1;
    assert_eq!(
        session.confirm_relocation_publication(&EngineRelocationPublicationEvidence {
            relocation_id: publication.relocation_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: wrong,
        }),
        Err(RuntimeSessionError::ReclamationReceiptMismatch)
    );
    session
        .confirm_relocation_publication(&EngineRelocationPublicationEvidence {
            relocation_id: publication.relocation_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&publication.retirements),
        })
        .expect("confirm relocation");
    let (_, appended) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id,
            target_boundary: 65,
        }],
        201,
        3,
    );
    confirm_publication(&mut session, &appended);

    let view = session
        .token_views_batch(&[EngineTokenViewQuery {
            request_id,
            class_id: 0,
            expected_boundary: 65,
        }])
        .expect("read packed view");
    assert_eq!(view[0].placements.len(), 65);
    assert_eq!(
        view[0]
            .placements
            .iter()
            .filter(|placement| placement.disposition.retained())
            .count(),
        41
    );
    assert!(view[0].placements.iter().all(|placement| {
        placement
            .location
            .is_none_or(|location| !prepared.plans[0].source_pages.contains(&location.page))
    }));
    let reused = view[0].placements[64]
        .location
        .expect("post-ACK append has a physical location")
        .page;
    assert!(prepared.plans[0]
        .source_pages
        .iter()
        .any(|source| source.page_id == reused.page_id && source.generation < reused.generation));
}

#[test]
fn session_relocation_unobserved_abort_is_ordered_and_reusable() {
    let backends = [backend(0, 402, 8, 41_000)];
    let request_id = EngineRequestId(1_002);
    let mut session = session(&full_plan(), &backends, 1);
    let prepared = prepare_relocatable(&mut session, &backends, &[request_id]);
    assert_eq!(
        session.abort_prepared_relocation(
            prepared.relocation_id,
            &[EngineRelocationAbortEvidence {
                request_id,
                backend_unobserved: false,
            }],
        ),
        Err(RuntimeSessionError::Manager(
            KvManagerError::BackendObservationUnknown
        ))
    );
    session
        .abort_prepared_relocation(
            prepared.relocation_id,
            &[EngineRelocationAbortEvidence {
                request_id,
                backend_unobserved: true,
            }],
        )
        .expect("abort unobserved relocation");
    assert!(matches!(
        session.abort_prepared_relocation(
            prepared.relocation_id,
            &[EngineRelocationAbortEvidence {
                request_id,
                backend_unobserved: true,
            }],
        ),
        Err(RuntimeSessionError::StaleRelocation(_))
    ));
    session
        .prepare_relocation_batch(&[EnginePrepareRelocationItem {
            request_id,
            class_id: 0,
            policy: relocation_policy(),
        }])
        .expect("reservation reusable after abort");
}

#[test]
fn failed_relocation_prepare_does_not_consume_an_engine_id() {
    let backends = [backend(0, 410, 8, 49_000)];
    let request_id = EngineRequestId(1_013);
    let mut session = session(&full_plan(), &backends, 1);
    let first = prepare_relocatable(&mut session, &backends, &[request_id]);
    session
        .abort_prepared_relocation(
            first.relocation_id,
            &[EngineRelocationAbortEvidence {
                request_id,
                backend_unobserved: true,
            }],
        )
        .expect("abort first relocation");
    let invalid = EnginePrepareRelocationItem {
        request_id,
        class_id: 0,
        policy: crate::kv_manager::RelocationPolicy {
            maximum_source_pages: 0,
            ..relocation_policy()
        },
    };
    assert_eq!(
        session.prepare_relocation_batch(&[invalid]),
        Err(RuntimeSessionError::Manager(
            KvManagerError::InvalidRelocationPolicy
        ))
    );
    let second = session
        .prepare_relocation_batch(&[EnginePrepareRelocationItem {
            request_id,
            class_id: 0,
            policy: relocation_policy(),
        }])
        .expect("prepare after rejected attempt");
    assert_eq!(second.relocation_id.sequence(), 2);
}

#[test]
fn token_view_invariant_mismatch_sticky_poisons_the_session() {
    let backends = [backend(0, 408, 4, 47_000)];
    let request_id = EngineRequestId(1_011);
    let mut session = session(&full_plan(), &backends, 1);
    session.acquire_requests(&[request_id]).expect("acquire");
    session.inject_test_fault(RuntimeSessionTestFault::TokenViewOrdering);
    assert_eq!(
        session.token_views_batch(&[EngineTokenViewQuery {
            request_id,
            class_id: 0,
            expected_boundary: 0,
        }]),
        Err(RuntimeSessionError::SessionPoisoned(
            "token view result ordering"
        ))
    );
    assert_eq!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 1,
        }]),
        Err(RuntimeSessionError::SessionPoisoned(
            "token view result ordering"
        ))
    );
}

#[test]
fn token_view_expected_boundary_is_checked_before_readback() {
    let backends = [backend(0, 409, 4, 48_000)];
    let request_id = EngineRequestId(1_012);
    let mut session = session(&full_plan(), &backends, 1);
    session.acquire_requests(&[request_id]).expect("acquire");
    assert_eq!(
        session.token_views_batch(&[EngineTokenViewQuery {
            request_id,
            class_id: 0,
            expected_boundary: 1,
        }]),
        Err(RuntimeSessionError::TokenViewBoundary {
            request_id,
            expected: 1,
            actual: 0,
        })
    );
    session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 1,
        }])
        .expect("boundary mismatch is retryable");
}

#[test]
fn session_relocation_rejects_foreign_stale_and_reordered_ids() {
    let backends = [backend(0, 403, 16, 42_000)];
    let requests = [EngineRequestId(1_003), EngineRequestId(1_004)];
    let mut session = session(&full_plan(), &backends, 2);
    let prepared = prepare_relocatable(&mut session, &backends, &requests);
    let foreign = EngineRelocationId::from_parts(
        prepared.relocation_id.session_epoch() + 1,
        prepared.relocation_id.sequence(),
    );
    assert_eq!(
        session.abort_prepared_relocation(
            foreign,
            &[EngineRelocationAbortEvidence {
                request_id: requests[0],
                backend_unobserved: true,
            }],
        ),
        Err(RuntimeSessionError::ForeignRelocation(foreign))
    );
    let mut execution = relocation_execution(&prepared);
    execution.requests.swap(0, 1);
    assert!(matches!(
        session.submit_relocation(&execution),
        Err(RuntimeSessionError::EvidenceRequest { index: 0, .. })
    ));
    session
        .abort_prepared_relocation(
            prepared.relocation_id,
            &[
                EngineRelocationAbortEvidence {
                    request_id: requests[0],
                    backend_unobserved: true,
                },
                EngineRelocationAbortEvidence {
                    request_id: requests[1],
                    backend_unobserved: true,
                },
            ],
        )
        .expect("abort original batch");
}

#[test]
fn bad_copy_evidence_quarantines_the_collective_batch() {
    let backends = [backend(0, 404, 16, 43_000)];
    let requests = [EngineRequestId(1_005), EngineRequestId(1_006)];
    let mut session = session(&full_plan(), &backends, 2);
    let prepared = prepare_relocatable(&mut session, &backends, &requests);
    let mut execution = relocation_execution(&prepared);
    execution.requests[1].copies[0].copied = false;
    assert!(matches!(
        session.submit_relocation(&execution),
        Err(RuntimeSessionError::Manager(
            KvManagerError::BatchQuarantined(_)
        ))
    ));
    assert_eq!(
        session.token_views_batch(&[EngineTokenViewQuery {
            request_id: requests[0],
            class_id: 0,
            expected_boundary: 48,
        }]),
        Err(RuntimeSessionError::SessionPoisoned(
            "relocation batch quarantined"
        ))
    );
}

#[test]
fn explicit_ambiguous_quarantine_retains_canonical_reservations() {
    let backends = [backend(0, 407, 8, 46_000)];
    let request_id = EngineRequestId(1_010);
    let mut session = session(&full_plan(), &backends, 1);
    let prepared = prepare_relocatable(&mut session, &backends, &[request_id]);
    let before = session.stats();
    assert_eq!(before.reserved_pages, 2);
    assert_eq!(
        session.quarantine_relocation(prepared.relocation_id),
        Err(RuntimeSessionError::SessionPoisoned(
            "relocation outcome is ambiguous"
        ))
    );
    let after = session.stats();
    assert_eq!(after.reserved_pages, before.reserved_pages);
    assert_eq!(after.free_pages, before.free_pages);
    assert!(matches!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 49,
        }]),
        Err(RuntimeSessionError::SessionPoisoned(
            "relocation outcome is ambiguous"
        ))
    ));
    assert_eq!(
        session.abort_prepared_relocation(
            prepared.relocation_id,
            &[EngineRelocationAbortEvidence {
                request_id,
                backend_unobserved: true,
            }],
        ),
        Err(RuntimeSessionError::SessionPoisoned(
            "relocation outcome is ambiguous"
        ))
    );
}

#[test]
fn multi_request_relocation_success_preserves_order() {
    let backends = [backend(0, 406, 10, 45_000)];
    let requests = [EngineRequestId(1_008), EngineRequestId(1_009)];
    let mut session = session(&full_plan(), &backends, 2);
    let prepared = prepare_relocatable(&mut session, &backends, &requests);
    assert_eq!(
        prepared
            .plans
            .iter()
            .map(|plan| plan.request_id)
            .collect::<Vec<_>>(),
        requests
    );
    let ticket = session
        .submit_relocation(&relocation_execution(&prepared))
        .expect("submit collective relocation");
    let publication = session
        .complete_relocation(
            ticket.relocation_id(),
            EngineCompletionEvidence {
                completion_domain: 201,
                completion_value: 2,
                confirmed: true,
            },
        )
        .expect("complete collective relocation");
    assert_eq!(
        publication
            .requests
            .iter()
            .map(|item| item.request_id)
            .collect::<Vec<_>>(),
        requests
    );
    assert_eq!(publication.retirements.len(), 6);
    session
        .confirm_relocation_publication(&EngineRelocationPublicationEvidence {
            relocation_id: publication.relocation_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&publication.retirements),
        })
        .expect("confirm collective relocation");
    assert_eq!(
        session
            .token_views_batch(&[
                EngineTokenViewQuery {
                    request_id: requests[0],
                    class_id: 0,
                    expected_boundary: 48,
                },
                EngineTokenViewQuery {
                    request_id: requests[1],
                    class_id: 0,
                    expected_boundary: 48,
                },
            ])
            .expect("read collective packed views")
            .iter()
            .map(|view| view.request_id)
            .collect::<Vec<_>>(),
        requests
    );
}

#[test]
fn shared_prefix_pages_are_not_relocatable() {
    let backends = [backend(0, 405, 12, 44_000)];
    let request_id = EngineRequestId(1_007);
    let mut session = session(&full_plan(), &backends, 2);
    session.acquire_requests(&[request_id]).expect("acquire");
    let (_, publication) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id,
            target_boundary: 48,
        }],
        205,
        1,
    );
    confirm_publication(&mut session, &publication);
    let key = prefix_key(45, 48);
    session
        .publish_prefix_batch(&[(request_id, key)])
        .expect("publish prefix");
    session
        .mark_token_dispositions_batch(&[EngineTokenDispositionBatchItem {
            request_id,
            updates: (0..48_u64)
                .filter(|token_id| token_id % 16 >= 8)
                .map(|token_id| EngineTokenDispositionUpdate {
                    class_id: 0,
                    token_id,
                    disposition: crate::kv_manager::TokenDisposition::policy_evicted(92, 1, 8),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }])
        .expect("mark");
    assert_eq!(
        session.prepare_relocation_batch(&[EnginePrepareRelocationItem {
            request_id,
            class_id: 0,
            policy: relocation_policy(),
        }]),
        Err(RuntimeSessionError::Manager(
            KvManagerError::InvalidRelocationPlan
        ))
    );
}
