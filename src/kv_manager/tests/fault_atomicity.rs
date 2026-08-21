use super::*;

#[test]
fn snapshot_staging_capacity_overflow_is_rejected_before_allocation() {
    let plan = full_plan(CANONICAL_PAGE_TOKENS);
    assert_eq!(
        CanonicalKvManager::new(
            &plan,
            ManagerConfig {
                maximum_requests: u32::MAX,
                maximum_operations: 1,
                maximum_prefixes: 1,
                maximum_reclamations: 1,
                maximum_step_tokens: 1,
            },
            &[backend(0, 192, 1, 10_000)],
        )
        .expect_err("overflow must fail before arena allocation"),
        KvManagerError::ArithmeticOverflow("snapshot capacity")
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn every_head_sensitive_command_cas_is_collective_and_zero_mutation() {
    let plan = full_plan(CANONICAL_PAGE_TOKENS);
    let mut manager = manager_for_plan(&plan, &[backend(0, 220, 12, 45_000)], 64, 12);
    let acquired = manager
        .acquire_request_views(3)
        .expect("acquire returns head-bearing views");
    assert!(acquired.iter().all(|view| {
        view.boundary == 0 && view.view_version == ViewVersion(0) && view.resident_count == 0
    }));
    let source_initial = acquired[0];
    let first_attach_initial = acquired[1];
    let second_attach_initial = acquired[2];
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request: source_initial.request,
            expected_head: source_initial.snapshot,
            target_boundary: 16,
        }])
        .expect("advance source");
    let submitted = submit(&mut manager, &prepared[0]);
    complete(&mut manager, &submitted, 43, 1);
    let source_current = manager
        .request_view(source_initial.request)
        .expect("advanced source view");
    assert_ne!(source_initial.snapshot, source_current.snapshot);

    let before = state_image(&manager);
    assert_eq!(
        manager.prepare_batch(&[
            PrepareBatchItem {
                request: second_attach_initial.request,
                expected_head: second_attach_initial.snapshot,
                target_boundary: 16,
            },
            PrepareBatchItem {
                request: source_initial.request,
                expected_head: source_initial.snapshot,
                target_boundary: 32,
            },
        ]),
        Err(KvManagerError::StaleView)
    );
    assert_eq!(state_image(&manager), before);

    let before = state_image(&manager);
    assert_eq!(
        manager.release_batch(&[
            ReleaseBatchItem {
                request: second_attach_initial.request,
                expected_head: second_attach_initial.snapshot,
            },
            ReleaseBatchItem {
                request: source_initial.request,
                expected_head: source_initial.snapshot,
            },
        ]),
        Err(KvManagerError::StaleView)
    );
    assert_eq!(state_image(&manager), before);

    let resident_key = prefix_key(50, 16);
    let publication = manager
        .publish_prefix_batch(&[PrefixPublishItem {
            request: source_current.request,
            expected_head: source_current.snapshot,
            key: resident_key,
        }])
        .expect("publish current head")[0];
    let before = state_image(&manager);
    assert_eq!(
        manager.publish_prefix_batch(&[PrefixPublishItem {
            request: source_initial.request,
            expected_head: source_initial.snapshot,
            key: prefix_key(51, 16),
        }]),
        Err(KvManagerError::StaleView)
    );
    assert_eq!(state_image(&manager), before);
    assert_eq!(
        manager.publish_prefix_and_release_batch(&[PrefixPublishItem {
            request: source_initial.request,
            expected_head: source_initial.snapshot,
            key: prefix_key(52, 16),
        }]),
        Err(KvManagerError::StaleView)
    );
    assert_eq!(state_image(&manager), before);

    let hint = manager
        .lookup_prefix_batch(&[resident_key])
        .expect("lookup current prefix")[0];
    manager
        .attach_prefix_batch(&[PrefixAttachItem {
            request: first_attach_initial.request,
            expected_empty_head: first_attach_initial.snapshot,
            hint,
        }])
        .expect("first attach advances empty head");
    let before = state_image(&manager);
    assert_eq!(
        manager.attach_prefix_batch(&[
            PrefixAttachItem {
                request: second_attach_initial.request,
                expected_empty_head: second_attach_initial.snapshot,
                hint,
            },
            PrefixAttachItem {
                request: first_attach_initial.request,
                expected_empty_head: first_attach_initial.snapshot,
                hint,
            },
        ]),
        Err(KvManagerError::StaleView)
    );
    assert_eq!(state_image(&manager), before);
    assert_eq!(
        manager
            .request_view(second_attach_initial.request)
            .expect("valid sibling remained empty")
            .snapshot,
        second_attach_initial.snapshot
    );
    assert_eq!(publication.prefix, hint.candidate.expect("resident hint"));
    assert_reference_census_matches_full_scan(&manager);
}

#[test]
fn joint_cow_cross_class_oom_is_collectively_zero_mutation() {
    let plan = hybrid_plan(18);
    let mut manager = manager_for_plan(
        &plan,
        &[backend(0, 225, 4, 90_000), backend(1, 226, 3, 100_000)],
        64,
        8,
    );
    let requests = manager
        .acquire_request_leases_for_test(2)
        .expect("shared requests");
    complete_initial_18(&mut manager, requests[0]);
    share_snapshot_for_cow(&mut manager, requests[0], requests[1]);
    let before_stats = manager.stats();
    let before_views = requests
        .iter()
        .map(|request| manager.request_view(*request).expect("before view"))
        .collect::<Vec<_>>();
    assert_eq!(
        manager.prepare_batch(&[
            PrepareBatchItem {
                request: requests[0],
                expected_head: manager.request(requests[0]).expect("expected head").head,
                target_boundary: 19,
            },
            PrepareBatchItem {
                request: requests[1],
                expected_head: manager.request(requests[1]).expect("expected head").head,
                target_boundary: 19,
            },
        ]),
        Err(KvManagerError::PageCapacityExhausted)
    );
    assert_eq!(manager.stats(), before_stats);
    assert_eq!(
        requests
            .iter()
            .map(|request| manager.request_view(*request).expect("after view"))
            .collect::<Vec<_>>(),
        before_views
    );
    assert_reference_census_matches_full_scan(&manager);
}

#[test]
fn copy_receipt_fault_quarantines_destinations_never_shared_sources() {
    let plan = hybrid_plan(18);
    let mut manager = manager_for_plan(
        &plan,
        &[backend(0, 227, 8, 110_000), backend(1, 228, 8, 120_000)],
        64,
        16,
    );
    let requests = manager
        .acquire_request_leases_for_test(2)
        .expect("shared requests");
    complete_initial_18(&mut manager, requests[0]);
    share_snapshot_for_cow(&mut manager, requests[0], requests[1]);
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request: requests[0],
            expected_head: manager.request(requests[0]).expect("expected head").head,
            target_boundary: 19,
        }])
        .expect("COW prepare");
    let sources = prepared[0]
        .copy_intents
        .iter()
        .map(|intent| intent.source)
        .collect::<Vec<_>>();
    let destinations = prepared[0]
        .copy_intents
        .iter()
        .map(|intent| intent.destination)
        .collect::<Vec<_>>();
    let target_snapshot = prepared[0].target_snapshot;
    let (items, binds, mut copies) = batch_submit_items(&manager, &prepared);
    copies[0].ordered_before_writes = 0;
    assert_eq!(
        manager.submit_batch(&items, &binds, &copies),
        Err(KvManagerError::BatchQuarantined(Box::new(
            KvManagerError::CopyOrderingUnknown,
        )))
    );
    assert!(manager.request(requests[0]).expect("initiator").quarantined);
    assert!(
        !manager
            .request(requests[1])
            .expect("shared peer")
            .quarantined
    );
    for source in sources {
        let page = manager.page(source.page_id).expect("shared source");
        assert_eq!(page.phase, PagePhase::Live);
        assert_eq!(page.request_refs, 2);
        assert_eq!(page.reader_pins, 0);
        assert!(page.writer.is_none());
    }
    for destination in destinations {
        let page = manager.page(destination.page_id).expect("COW destination");
        assert_eq!(page.phase, PagePhase::Quarantined);
        assert_eq!(page.request_refs, 0);
    }
    assert_eq!(
        manager
            .snapshots
            .get(target_snapshot.slot, target_snapshot.generation),
        Err(KvManagerError::StaleLease("snapshot"))
    );
    assert_reference_census_matches_full_scan(&manager);
}

#[test]
fn b4_semantic_submit_fault_quarantines_only_selected_destinations() {
    const CONTEXT_PAGES: u32 = 8;
    let plan = full_plan(CANONICAL_PAGE_TOKENS);
    let mut manager = manager_for_plan(
        &plan,
        &[backend(0, 96, (CONTEXT_PAGES + 1) * 4, 60_000)],
        CONTEXT_PAGES * 16 + 1,
        (CONTEXT_PAGES + 1) * 4,
    );
    let requests = manager
        .acquire_request_leases_for_test(4)
        .expect("B4 requests");
    let initial = requests
        .iter()
        .copied()
        .map(|request| PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: u64::from(CONTEXT_PAGES) * CANONICAL_PAGE_TOKENS,
        })
        .collect::<Vec<_>>();
    let prepared = manager.prepare_batch(&initial).expect("initial prepare");
    let (items, receipts, copies) = batch_submit_items(&manager, &prepared);
    let submitted = manager
        .submit_batch(&items, &receipts, &copies)
        .expect("initial submit");
    let submissions = submitted
        .iter()
        .map(|item| item.submission)
        .collect::<Vec<_>>();
    manager
        .complete_batch(
            BatchCompletionReceipt {
                engine_epoch: manager.engine_epoch,
                completion_domain: 1,
                completion_value: 1,
                confirmed: 1,
                reserved: 0,
            },
            &submissions,
        )
        .expect("initial completion");

    let extensions = requests
        .iter()
        .copied()
        .map(|request| PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: u64::from(CONTEXT_PAGES) * CANONICAL_PAGE_TOKENS + 1,
        })
        .collect::<Vec<_>>();
    let prepared = manager
        .prepare_batch(&extensions)
        .expect("extension prepare");
    let (items, mut receipts, copies) = batch_submit_items(&manager, &prepared);
    receipts.last_mut().expect("last receipt").backend_index += 1;
    assert_eq!(
        manager.submit_batch(&items, &receipts, &copies),
        Err(KvManagerError::BatchQuarantined(Box::new(
            KvManagerError::BindingReceiptMismatch,
        )))
    );
    assert!(
        requests
            .iter()
            .all(|&request| { manager.request(request).expect("request").quarantined })
    );
    assert_eq!(manager.stats().prepared_steps, 0);
    assert_eq!(manager.stats().active_pages, u64::from(CONTEXT_PAGES * 4));
    assert_eq!(manager.stats().reserved_pages, 0);
    assert_eq!(manager.stats().quarantined_pages, 4);
}

#[test]
fn batch_prepare_capacity_failure_is_collectively_zero_mutation() {
    let mut manager = manager_with(18, 3, 64, 3);
    let requests = manager
        .acquire_request_leases_for_test(2)
        .expect("batch acquire");
    let before = manager.stats();
    assert_eq!(
        manager.prepare_batch(&[
            PrepareBatchItem {
                request: requests[0],
                expected_head: manager.request(requests[0]).expect("expected head").head,
                target_boundary: 18,
            },
            PrepareBatchItem {
                request: requests[1],
                expected_head: manager.request(requests[1]).expect("expected head").head,
                target_boundary: 18,
            },
        ]),
        Err(KvManagerError::PageCapacityExhausted)
    );
    assert_eq!(manager.stats(), before);
    assert!(
        manager
            .prepare_batch(&[PrepareBatchItem {
                request: requests[0],
                expected_head: manager.request(requests[0]).expect("expected head").head,
                target_boundary: 18,
            }])
            .is_ok()
    );
}

#[test]
fn structural_submit_failure_is_retryable_but_semantic_failure_quarantines_all() {
    let mut manager = manager_with(18, 8, 64, 8);
    let requests = manager
        .acquire_request_leases_for_test(2)
        .expect("batch acquire");
    let prepare_items = requests
        .iter()
        .copied()
        .map(|request| PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 18,
        })
        .collect::<Vec<_>>();
    let prepared = manager
        .prepare_batch(&prepare_items)
        .expect("batch prepare");
    let (mut items, mut receipts, copies) = batch_submit_items(&manager, &prepared);
    let before = manager.stats();
    items[1].receipt_offset += 1;
    assert_eq!(
        manager.submit_batch(&items, &receipts, &copies),
        Err(KvManagerError::InvalidBatchRange)
    );
    assert_eq!(manager.stats(), before);

    let (items, _, copies) = batch_submit_items(&manager, &prepared);
    receipts.last_mut().expect("receipt").backend_index += 1;
    assert_eq!(
        manager.submit_batch(&items, &receipts, &copies),
        Err(KvManagerError::BatchQuarantined(Box::new(
            KvManagerError::BindingReceiptMismatch,
        )))
    );
    let stats = manager.stats();
    assert_eq!(stats.prepared_steps, 0);
    assert_eq!(stats.quarantined_pages, 4);
    for request in requests {
        assert!(manager.request(request).expect("request").quarantined);
    }
    assert_eq!(
        manager.abort_steps_batch(&[
            BackendUnobservedReceipt {
                step: prepared[0].step,
                backend_unobserved: 1,
                reserved: 0,
            },
            BackendUnobservedReceipt {
                step: prepared[1].step,
                backend_unobserved: 1,
                reserved: 0,
            },
        ]),
        Err(KvManagerError::StaleLease("operation"))
    );
}

#[test]
fn batch_complete_duplicate_is_zero_mutation_and_retryable() {
    let mut manager = manager_with(18, 8, 64, 8);
    let requests = manager
        .acquire_request_leases_for_test(2)
        .expect("batch acquire");
    let prepare_items = requests
        .iter()
        .copied()
        .map(|request| PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 18,
        })
        .collect::<Vec<_>>();
    let prepared = manager
        .prepare_batch(&prepare_items)
        .expect("batch prepare");
    let (items, receipts, copies) = batch_submit_items(&manager, &prepared);
    let submitted = manager
        .submit_batch(&items, &receipts, &copies)
        .expect("batch submit");
    let first = submitted[0].submission;
    let second = submitted[1].submission;
    let event = BatchCompletionReceipt {
        engine_epoch: manager.engine_epoch,
        completion_domain: 7,
        completion_value: 8,
        confirmed: 1,
        reserved: 0,
    };
    let before = manager.stats();
    assert_eq!(
        manager.complete_batch(event, &[first, first]),
        Err(KvManagerError::DuplicateSubmission)
    );
    assert_eq!(manager.stats(), before);
    assert!(manager.complete_batch(event, &[first, second]).is_ok());
}

#[test]
fn observed_submit_with_stale_candidate_quarantines_entire_batch() {
    let mut manager = manager_with(18, 8, 64, 8);
    let requests = manager
        .acquire_request_leases_for_test(2)
        .expect("batch acquire");
    let prepare_items = requests
        .iter()
        .copied()
        .map(|request| PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 18,
        })
        .collect::<Vec<_>>();
    let prepared = manager
        .prepare_batch(&prepare_items)
        .expect("batch prepare");
    let (items, receipts, copies) = batch_submit_items(&manager, &prepared);
    let stale_page = prepared[1].write_intents[0].page_id;
    manager.page_mut(stale_page).expect("page").generation += 1;
    assert_eq!(
        manager.submit_batch(&items, &receipts, &copies),
        Err(KvManagerError::BatchQuarantined(Box::new(
            KvManagerError::StalePage,
        )))
    );
    assert_eq!(manager.stats().prepared_steps, 0);
    assert_eq!(manager.stats().quarantined_pages, 4);
    assert!(
        requests
            .iter()
            .all(|&request| manager.request(request).expect("request").quarantined)
    );
}

#[test]
fn empty_reclamation_ack_is_rejected() {
    let mut manager = manager_with(18, 4, 64, 4);
    assert_eq!(
        manager.acknowledge_reclamations_batch(&[]),
        Err(KvManagerError::EmptyBatch)
    );
}

#[test]
fn hybrid_prepare_oom_is_atomic_across_class_pools() {
    let plan = hybrid_plan(18);
    let mut manager = manager_for_plan(&plan, &[backend(0, 41, 1, 0), backend(1, 42, 3, 0)], 64, 8);
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    let before = manager.stats();
    assert_eq!(
        manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                expected_head: manager.request(request).expect("expected head").head,
                target_boundary: 18
            }])
            .map(|items| items[0].clone()),
        Err(KvManagerError::PageCapacityExhausted)
    );
    assert_eq!(manager.stats(), before);
    assert!(
        manager
            .request_snapshot(request)
            .expect("request snapshot")
            .is_empty()
    );
    assert!(
        manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                expected_head: manager.request(request).expect("expected head").head,
                target_boundary: 1
            }])
            .map(|items| items[0].clone())
            .is_ok()
    );
}

#[test]
fn prepare_oom_is_zero_mutation() {
    let mut manager = manager_with(18, 3, 64, 8);
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    let before = manager.stats();
    assert_eq!(
        manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                expected_head: manager.request(request).expect("expected head").head,
                target_boundary: 51
            }])
            .map(|items| items[0].clone()),
        Err(KvManagerError::PageCapacityExhausted)
    );
    assert_eq!(manager.stats(), before);
    assert!(
        manager
            .request_snapshot(request)
            .expect("request snapshot")
            .is_empty()
    );
    assert!(
        manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                expected_head: manager.request(request).expect("expected head").head,
                target_boundary: 1
            }])
            .map(|items| items[0].clone())
            .is_ok()
    );
}

#[test]
fn publishes_only_after_confirmed_completion() {
    let mut manager = manager_with(18, 4, 64, 8);
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 18,
        }])
        .map(|items| items[0].clone())
        .expect("prepare");
    assert_eq!(
        manager
            .request_snapshot(request)
            .expect("current snapshot")
            .view_version,
        ViewVersion(0)
    );
    let submitted = submit(&mut manager, &prepared);
    assert_eq!(
        manager
            .request_snapshot(request)
            .expect("current snapshot")
            .view_version,
        ViewVersion(0)
    );
    let completion = complete(&mut manager, &submitted, 4, 12);
    assert_eq!(completion.publication.view_version, ViewVersion(1));
    let state = manager.request_snapshot(request).expect("current snapshot");
    assert_eq!(state.view_version, completion.publication.view_version);
    assert_eq!(state.boundary, completion.publication.boundary);
    assert_eq!(
        state.resident_count(),
        completion.publication.resident_count as usize
    );
    let before = manager.stats();
    assert!(
        manager
            .complete_batch(
                BatchCompletionReceipt {
                    engine_epoch: submitted.submission.engine_epoch,
                    completion_domain: 4,
                    completion_value: 12,
                    confirmed: 1,
                    reserved: 0,
                },
                &[submitted.submission],
            )
            .is_err()
    );
    assert_eq!(manager.stats(), before);
}

#[test]
fn abort_requires_pre_submit_unobserved_proof() {
    let mut manager = manager_with(18, 4, 64, 8);
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 18,
        }])
        .map(|items| items[0].clone())
        .expect("prepare");
    let before = manager.stats();
    assert_eq!(
        manager.abort_steps_batch(&[BackendUnobservedReceipt {
            step: prepared.step,
            backend_unobserved: 0,
            reserved: 0,
        }]),
        Err(KvManagerError::BackendObservationUnknown)
    );
    assert_eq!(manager.stats(), before);
    manager
        .abort_steps_batch(&[BackendUnobservedReceipt {
            step: prepared.step,
            backend_unobserved: 1,
            reserved: 0,
        }])
        .expect("safe abort");
    assert_eq!(manager.stats().free_pages, 4);

    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 18,
        }])
        .map(|items| items[0].clone())
        .expect("prepare retry");
    let _submitted = submit(&mut manager, &prepared);
    assert_eq!(
        manager.abort_steps_batch(&[BackendUnobservedReceipt {
            step: prepared.step,
            backend_unobserved: 1,
            reserved: 0,
        }]),
        Err(KvManagerError::StepAlreadySubmitted)
    );
    assert_eq!(manager.stats().writing_pages, 2);
}

#[test]
fn forged_binding_quarantines_pages_without_reuse() {
    let mut manager = manager_with(18, 3, 64, 8);
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 18,
        }])
        .map(|items| items[0].clone())
        .expect("prepare");
    let reserved_ids = prepared
        .write_intents
        .iter()
        .map(|intent| intent.page_id)
        .collect::<BTreeSet<_>>();
    let mut receipts = binding_receipts(&manager, &prepared);
    let copies = copy_receipts(&prepared);
    receipts[0].backend_index += 1;
    assert_eq!(
        manager.submit_batch(
            &[SubmitBatchItem {
                step: prepared.step,
                receipt_offset: 0,
                receipt_count: u32::try_from(receipts.len()).expect("receipt count"),
                copy_receipt_offset: 0,
                copy_receipt_count: u32::try_from(copies.len()).expect("copy receipt count"),
            }],
            &receipts,
            &copies,
        ),
        Err(KvManagerError::BatchQuarantined(Box::new(
            KvManagerError::BindingReceiptMismatch,
        )))
    );
    assert_eq!(manager.stats().quarantined_pages, 2);
    assert!(manager.request(request).expect("request").quarantined);

    let second = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("second request");
    let second_prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request: second,
            expected_head: manager.request(second).expect("expected head").head,
            target_boundary: 1,
        }])
        .map(|items| items[0].clone())
        .expect("second prepare");
    assert!(
        second_prepared
            .write_intents
            .iter()
            .all(|intent| !reserved_ids.contains(&intent.page_id))
    );
}

#[test]
fn ambiguous_pre_submit_lowering_has_an_explicit_quarantine_path() {
    let mut manager = manager_with(18, 4, 64, 8);
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 18,
        }])
        .map(|items| items[0].clone())
        .expect("prepare");
    manager
        .quarantine_steps_batch(&[prepared.step])
        .expect("quarantine prepared lowering");
    assert_eq!(manager.stats().prepared_steps, 0);
    assert_eq!(manager.stats().quarantined_pages, 2);
    assert!(manager.request(request).expect("request").quarantined);
    assert_eq!(
        manager.abort_steps_batch(&[BackendUnobservedReceipt {
            step: prepared.step,
            backend_unobserved: 1,
            reserved: 0,
        }]),
        Err(KvManagerError::StaleLease("operation"))
    );
}

#[test]
fn completion_preflight_failure_is_zero_mutation() {
    let mut manager = manager_with(18, 4, 64, 8);
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 18,
        }])
        .map(|items| items[0].clone())
        .expect("prepare");
    let submitted = submit(&mut manager, &prepared);
    let last_page = operation_entries(
        &manager,
        submitted.submission.slot,
        submitted.submission.generation,
    )
    .last()
    .expect("page")
    .page_id;
    manager.page_mut(last_page).expect("page").reader_pins = 2;
    let before = snapshot_entries(&manager, request);
    assert_eq!(
        manager.complete_batch(
            BatchCompletionReceipt {
                engine_epoch: submitted.submission.engine_epoch,
                completion_domain: 1,
                completion_value: 2,
                confirmed: 1,
                reserved: 0,
            },
            &[submitted.submission],
        ),
        Err(KvManagerError::StalePage)
    );
    assert_eq!(snapshot_entries(&manager, request), before);
    assert_eq!(manager.stats().submitted_steps, 1);
    assert_eq!(manager.stats().pending_reclamations, 0);
    assert_eq!(manager.page(last_page).expect("page").reader_pins, 2);
    manager.page_mut(last_page).expect("page").reader_pins = 1;
    assert_eq!(
        complete(&mut manager, &submitted, 1, 2)
            .publication
            .view_version,
        ViewVersion(1)
    );
}

#[test]
fn reclamation_arena_backpressure_is_zero_mutation_and_retryable() {
    let mut arena = Arena::new("reclamation", 1).expect("arena");
    let occupied = arena.plan_many(1).expect("initial allocation")[0];
    arena.insert_planned(occupied, 7_u8);
    assert_eq!(
        arena.plan_many(1),
        Err(KvManagerError::ArenaExhausted("reclamation"))
    );
    assert_eq!(arena.get(occupied.0, occupied.1), Ok(&7));
    assert_eq!(arena.remove(occupied.0, occupied.1), Ok(7));
    assert!(arena.plan_many(1).is_ok());
}

#[test]
fn ambiguous_submission_and_cross_manager_identities_fail_closed() {
    let mut first = manager_with(18, 4, 64, 8);
    let second = manager_with(18, 4, 64, 8);
    let request = first
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    assert!(matches!(
        second.request(request),
        Err(KvManagerError::WrongEngine)
    ));
    let prepared = first
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: first.request(request).expect("expected head").head,
            target_boundary: 18,
        }])
        .map(|items| items[0].clone())
        .expect("prepare");
    let submitted = submit(&mut first, &prepared);
    first
        .quarantine_submissions_batch(&[submitted.submission])
        .expect("quarantine");
    assert_eq!(first.stats().quarantined_pages, 2);
    assert!(first.request(request).expect("request").quarantined);
}

#[test]
fn release_invalidates_snapshot_without_version_allocation_and_page_exhaustion_is_safe() {
    let mut manager = manager_with(18, 3, 64, 8);
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    complete_initial_18(&mut manager, request);
    manager
        .request_snapshot_mut(request)
        .expect("request snapshot")
        .view_version = ViewVersion(u64::MAX);
    let old_snapshot = manager.request(request).expect("request").head;
    let release = manager
        .release_current_for_test(&[request])
        .expect("release does not allocate a replacement version");
    assert_eq!(release.retirements.len(), 2);
    assert_eq!(
        manager
            .snapshots
            .get(old_snapshot.slot, old_snapshot.generation),
        Err(KvManagerError::StaleLease("snapshot"))
    );

    let mut generation_manager = manager_with(18, 3, 64, 8);
    let request = generation_manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    generation_manager.pages[0].generation = u64::MAX - 1;
    let prepared = generation_manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: generation_manager
                .request(request)
                .expect("expected head")
                .head,
            target_boundary: 1,
        }])
        .map(|items| items[0].clone())
        .expect("prepare max gen");
    assert_eq!(prepared.write_intents[0].page_generation, u64::MAX);
    generation_manager
        .abort_steps_batch(&[BackendUnobservedReceipt {
            step: prepared.step,
            backend_unobserved: 1,
            reserved: 0,
        }])
        .expect("abort max gen");
    assert_eq!(generation_manager.stats().exhausted_pages, 1);
    assert_eq!(generation_manager.stats().free_pages, 2);
}

#[test]
fn generation_exhaustion_is_isolated_to_one_hybrid_arena_page() {
    let plan = hybrid_plan(18);
    let mut manager = manager_for_plan(
        &plan,
        &[backend(0, 101, 4, 1_000), backend(1, 102, 3, 2_000)],
        64,
        7,
    );
    manager.pages[4].generation = u64::MAX - 1;
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 1,
        }])
        .map(|items| items[0].clone())
        .expect("prepare max gen");
    let second_class = prepared
        .class_lowerings
        .iter()
        .find(|lowering| lowering.class_id == 1)
        .expect("second-arena class");
    let second =
        prepared.write_intents[usize::try_from(second_class.write_offset).expect("write offset")];
    assert_eq!(second.page_id, 5);
    assert_eq!(second.page_generation, u64::MAX);
    manager
        .abort_steps_batch(&[BackendUnobservedReceipt {
            step: prepared.step,
            backend_unobserved: 1,
            reserved: 0,
        }])
        .expect("abort max generation");
    assert_eq!(
        arena_counts(&manager),
        vec![
            (0, 101, 4, 4, 0, 0, 0, 0, 0, 0),
            (1, 102, 3, 2, 0, 0, 0, 0, 0, 1)
        ]
    );
}
