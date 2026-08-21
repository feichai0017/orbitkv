use super::*;

#[test]
fn batch_lifecycle_preserves_order_offsets_and_shared_completion() {
    let mut manager = manager_with(18, 8, 64, 8);
    let requests = manager
        .acquire_request_leases_for_test(2)
        .expect("batch acquire");
    assert_incremental_census_matches_full_scan(&manager);
    let prepared = manager
        .prepare_batch(&[
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
        ])
        .expect("batch prepare");
    assert_incremental_census_matches_full_scan(&manager);
    assert!(prepared.iter().all(|item| {
        item.class_lowerings.len() == 1
            && item.class_lowerings[0].flags == 0
            && item.class_lowerings[0].tail_offset == 0
            && item.class_lowerings[0].tail_count == 1
            && item.class_lowerings[0].copy_offset == 0
            && item.class_lowerings[0].copy_count == 0
            && item.class_lowerings[0].write_offset == 0
            && item.class_lowerings[0].write_count == 2
            && item.class_lowerings[0].reserved == 0
            && item.tail_actions.len() == 1
            && item.write_intents.len() == 2
    }));
    let (submit_items, receipts, copies) = batch_submit_items(&manager, &prepared);
    let submitted = manager
        .submit_batch(&submit_items, &receipts, &copies)
        .expect("batch submit");
    assert_incremental_census_matches_full_scan(&manager);
    assert_eq!(submitted[0].request, requests[0]);
    assert_eq!(submitted[1].request, requests[1]);
    let submissions = submitted
        .iter()
        .map(|item| item.submission)
        .collect::<Vec<_>>();
    let completed = manager
        .complete_batch(
            BatchCompletionReceipt {
                engine_epoch: manager.engine_epoch,
                completion_domain: 77,
                completion_value: 99,
                confirmed: 1,
                reserved: 0,
            },
            &submissions,
        )
        .expect("batch complete");
    assert_incremental_census_matches_full_scan(&manager);
    assert_eq!(completed.completions[0].request, requests[0]);
    assert_eq!(completed.completions[1].request, requests[1]);
    for ((completion, prepared), submitted) in completed
        .completions
        .iter()
        .zip(prepared.iter())
        .zip(submitted.iter())
    {
        assert_snapshot_transition_identities(&manager, prepared, submitted, completion);
    }
    assert!(
        completed
            .completions
            .iter()
            .all(|item| item.publication.resident_count == 2)
    );
    assert!(completed.retirements.is_empty());

    let released = manager
        .release_current_for_test(&requests)
        .expect("batch release");
    assert_incremental_census_matches_full_scan(&manager);
    assert_eq!(
        released
            .releases
            .iter()
            .map(|release| release.request)
            .collect::<Vec<_>>(),
        requests.as_ref()
    );
    let certificates = released.retirements.into_vec();
    assert!(certificates.iter().all(|certificate| {
        certificate.completion_domain == 77 && certificate.completion_value == 99
    }));
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&certificates))
        .expect("batch acknowledge");
    assert_incremental_census_matches_full_scan(&manager);
    manager
        .recycle_requests_batch(&requests)
        .expect("batch recycle");
    assert_incremental_census_matches_full_scan(&manager);
    assert_eq!(manager.stats().active_requests, 0);
}

#[test]
fn public_request_queries_are_collective_batches() {
    let mut manager = manager_with(18, 4, 32, 4);
    let acquired = manager
        .acquire_requests_batch(2)
        .expect("batch acquire views");
    let requests = acquired.iter().map(|view| view.request).collect::<Vec<_>>();
    assert_eq!(
        manager
            .request_views_batch(&requests)
            .expect("batch request views")
            .as_ref(),
        acquired.as_ref()
    );
    assert_eq!(
        manager.request_views_batch(&[]),
        Err(KvManagerError::EmptyBatch)
    );
    assert_eq!(
        manager.request_views_batch(&[requests[0], requests[0]]),
        Err(KvManagerError::DuplicateRequest)
    );
    let empty_materialized = manager
        .materialize_request_views_batch(&[
            (requests[0], acquired[0].snapshot),
            (requests[1], acquired[1].snapshot),
        ])
        .expect("empty views materialize");
    assert!(empty_materialized.iter().all(|item| item.pages.is_empty()));
    assert_eq!(
        manager.materialize_request_views_batch(&[]),
        Err(KvManagerError::EmptyBatch)
    );
    assert_eq!(
        manager.materialize_request_views_batch(&[
            (requests[0], acquired[0].snapshot),
            (requests[0], acquired[0].snapshot),
        ]),
        Err(KvManagerError::DuplicateRequest)
    );

    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request: requests[0],
            expected_head: acquired[0].snapshot,
            target_boundary: 16,
        }])
        .expect("query test prepare");
    let submitted = submit(&mut manager, &prepared[0]);
    complete(&mut manager, &submitted, 1, 1);
    let current = manager
        .request_views_batch(&requests)
        .expect("advanced batch views");
    assert_ne!(current[0].snapshot, acquired[0].snapshot);
    let before = state_image(&manager);
    assert_eq!(
        manager.materialize_request_views_batch(&[
            (requests[0], acquired[0].snapshot),
            (requests[1], current[1].snapshot),
        ]),
        Err(KvManagerError::StaleView)
    );
    assert_eq!(state_image(&manager), before);
    let materialized = manager
        .materialize_request_views_batch(&[
            (requests[0], current[0].snapshot),
            (requests[1], current[1].snapshot),
        ])
        .expect("current batch materializes");
    assert_eq!(materialized[0].view, current[0]);
    assert_eq!(materialized[0].pages.len(), 1);
    assert!(materialized[1].pages.is_empty());
}

#[test]
fn full_attention_is_append_only_until_request_release() {
    let plan = full_plan(CANONICAL_PAGE_TOKENS);
    let mut manager = manager_for_plan(&plan, &[backend(0, 21, 8, 1_000)], 64, 8);
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");

    let first = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 18,
        }])
        .map(|items| items[0].clone())
        .expect("prepare full");
    let first = submit(&mut manager, &first);
    let first = complete(&mut manager, &first, 2, 3);
    assert!(first.retirements.is_empty());
    assert_eq!(first.publication.resident_count, 2);

    let second = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 35,
        }])
        .map(|items| items[0].clone())
        .expect("extend full");
    let second = submit(&mut manager, &second);
    let second = complete(&mut manager, &second, 2, 4);
    assert!(second.retirements.is_empty());
    assert_eq!(
        published_entries(&manager, request)
            .iter()
            .map(|entry| (
                entry.class_id,
                entry.logical_ordinal,
                entry.pool_id,
                entry.temporal_cell_index,
                entry.temporal_cycle,
                entry.visible_token_offset,
                entry.visible_token_count,
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, 0, 21, 0, 0, 0, 16),
            (0, 1, 21, 1, 0, 0, 16),
            (0, 2, 21, 2, 0, 0, 3),
        ]
    );
    let release = manager
        .release_current_for_test(&[request])
        .expect("release full");
    assert_eq!(release.retirements.len(), 3);
}

#[test]
fn hybrid_classes_have_independent_pools_addresses_and_retirement() {
    let plan = hybrid_plan(18);
    let mut manager = manager_for_plan(
        &plan,
        &[backend(0, 31, 8, 1_000), backend(1, 32, 4, 2_000)],
        64,
        12,
    );
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    let first = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 18,
        }])
        .map(|items| items[0].clone())
        .expect("prepare hybrid");
    assert_eq!(
        operation_entries(&manager, first.step.slot, first.step.generation)
            .iter()
            .map(|entry| (entry.class_id, entry.pool_id, entry.backend_index))
            .collect::<Vec<_>>(),
        vec![
            (0, 31, 1_000),
            (0, 31, 1_001),
            (1, 32, 2_000),
            (1, 32, 2_001)
        ]
    );
    let first = submit(&mut manager, &first);
    assert!(complete(&mut manager, &first, 5, 6).retirements.is_empty());

    let second = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 35,
        }])
        .map(|items| items[0].clone())
        .expect("extend hybrid");
    let second = submit(&mut manager, &second);
    let completion = complete(&mut manager, &second, 5, 7);
    assert_eq!(completion.retirements.len(), 1);
    assert_eq!(completion.retirements[0].class_id, 1);
    assert_eq!(completion.retirements[0].page.pool_id, 32);
    assert_eq!(completion.retirements[0].logical_ordinal, 0);
    assert_eq!(
        published_entries(&manager, request)
            .iter()
            .map(|entry| (
                entry.class_id,
                entry.logical_ordinal,
                entry.pool_id,
                entry.temporal_cell_index,
                entry.temporal_cycle,
                entry.visible_token_offset,
                entry.visible_token_count,
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, 0, 31, 0, 0, 0, 16),
            (0, 1, 31, 1, 0, 0, 16),
            (0, 2, 31, 2, 0, 0, 3),
            (1, 1, 32, 1, 0, 2, 14),
            (1, 2, 32, 2, 0, 0, 3),
        ]
    );
    assert_eq!(manager.stats().active_pages, 5);
    assert_eq!(manager.stats().retiring_pages, 1);
}

#[test]
#[allow(clippy::too_many_lines)]
fn hybrid_arena_census_is_pure_and_tracks_every_lifecycle_phase() {
    let plan = hybrid_plan(18);
    let mut manager = manager_for_plan(
        &plan,
        &[backend(0, 81, 8, 1_000), backend(1, 82, 4, 2_000)],
        64,
        12,
    );
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    assert_eq!(
        arena_counts(&manager),
        vec![
            (0, 81, 8, 8, 0, 0, 0, 0, 0, 0),
            (1, 82, 4, 4, 0, 0, 0, 0, 0, 0),
        ]
    );
    let first_read = manager.arena_stats();
    assert_eq!(manager.arena_stats(), first_read);
    assert!(
        first_read
            .iter()
            .all(|stats| stats.engine_epoch == manager.engine_epoch
                && stats.pool_epoch == manager.pool_epoch)
    );
    assert_eq!(first_read[1].first_page_id, 9);

    let first = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 18,
        }])
        .map(|items| items[0].clone())
        .expect("prepare first");
    assert_eq!(
        arena_counts(&manager),
        vec![
            (0, 81, 8, 6, 2, 0, 0, 0, 0, 0),
            (1, 82, 4, 2, 2, 0, 0, 0, 0, 0),
        ]
    );
    let first = submit(&mut manager, &first);
    assert_eq!(
        arena_counts(&manager),
        vec![
            (0, 81, 8, 6, 0, 2, 0, 0, 0, 0),
            (1, 82, 4, 2, 0, 2, 0, 0, 0, 0),
        ]
    );
    complete(&mut manager, &first, 10, 1);
    assert_eq!(
        arena_counts(&manager),
        vec![
            (0, 81, 8, 6, 0, 0, 2, 0, 0, 0),
            (1, 82, 4, 2, 0, 0, 2, 0, 0, 0),
        ]
    );

    let second = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 35,
        }])
        .map(|items| items[0].clone())
        .expect("prepare second");
    assert_eq!(
        arena_counts(&manager),
        vec![
            (0, 81, 8, 5, 1, 0, 2, 0, 0, 0),
            (1, 82, 4, 1, 1, 0, 2, 0, 0, 0),
        ]
    );
    let second = submit(&mut manager, &second);
    assert_eq!(
        arena_counts(&manager),
        vec![
            (0, 81, 8, 5, 0, 2, 1, 0, 0, 0),
            (1, 82, 4, 1, 0, 2, 1, 0, 0, 0),
        ]
    );
    let second = complete(&mut manager, &second, 10, 2);
    assert_eq!(
        arena_counts(&manager),
        vec![
            (0, 81, 8, 5, 0, 0, 3, 0, 0, 0),
            (1, 82, 4, 1, 0, 0, 2, 1, 0, 0),
        ]
    );
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&second.retirements))
        .expect("ack sliding retirement");
    assert_eq!(
        arena_counts(&manager),
        vec![
            (0, 81, 8, 5, 0, 0, 3, 0, 0, 0),
            (1, 82, 4, 2, 0, 0, 2, 0, 0, 0),
        ]
    );

    let release = manager
        .release_current_for_test(&[request])
        .expect("release");
    assert_eq!(
        arena_counts(&manager),
        vec![
            (0, 81, 8, 5, 0, 0, 0, 3, 0, 0),
            (1, 82, 4, 2, 0, 0, 0, 2, 0, 0),
        ]
    );
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&release.retirements))
        .expect("ack release");
    assert_eq!(
        arena_counts(&manager),
        vec![
            (0, 81, 8, 8, 0, 0, 0, 0, 0, 0),
            (1, 82, 4, 4, 0, 0, 0, 0, 0, 0),
        ]
    );
}

#[test]
fn reclamation_capacity_must_cover_every_registered_page() {
    let plan = hybrid_plan(18);
    let backends = [backend(0, 51, 4, 0), backend(1, 52, 4, 0)];
    assert!(matches!(
        CanonicalKvManager::new(
            &plan,
            ManagerConfig {
                maximum_requests: 4,
                maximum_operations: 4,
                maximum_prefixes: 4,
                maximum_reclamations: 7,
                maximum_step_tokens: 64,
            },
            &backends,
        ),
        Err(KvManagerError::InvalidConfiguration)
    ));

    manager_for_plan(&plan, &backends, 64, 8);
}

#[test]
fn minimum_reclamation_capacity_releases_a_full_arena_root() {
    let plan = full_plan(CANONICAL_PAGE_TOKENS);
    let mut manager = manager_for_plan(&plan, &[backend(0, 53, 4, 0)], 64, 4);
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 64,
        }])
        .map(|items| items[0].clone())
        .expect("prepare full root");
    let submitted = submit(&mut manager, &prepared);
    complete(&mut manager, &submitted, 8, 1);

    let release = manager
        .release_current_for_test(&[request])
        .expect("release full root");
    assert_eq!(release.retirements.len(), 4);
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&release.retirements))
        .expect("ack full root");
    manager
        .recycle_requests_batch(&[request])
        .expect("recycle request");
    assert_eq!(manager.stats().free_pages, 4);
}

#[test]
fn reclamation_batch_is_atomic_and_release_inherits_completion() {
    let mut manager = manager_with(18, 8, 64, 8);
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    complete_initial_18(&mut manager, request);
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 51,
        }])
        .map(|items| items[0].clone())
        .expect("prepare long step");
    let submitted = submit(&mut manager, &prepared);
    let completion = complete(&mut manager, &submitted, 17, 29);
    assert_eq!(completion.retirements.len(), 2);
    let mut receipts = reclamation_receipts(&completion.retirements);
    receipts[1].backend_index += 1;
    let before = manager.stats();
    assert_eq!(
        manager.acknowledge_reclamations_batch(&receipts),
        Err(KvManagerError::ReclamationMismatch)
    );
    assert_eq!(manager.stats(), before);
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&completion.retirements))
        .expect("atomic acknowledgement");

    let release = manager
        .release_current_for_test(&[request])
        .expect("release");
    assert!(release.retirements.iter().all(
        |certificate| certificate.completion_domain == 17 && certificate.completion_value == 29
    ));
}

#[test]
fn hybrid_release_acknowledgement_is_atomic_across_pools() {
    let plan = hybrid_plan(18);
    let mut manager = manager_for_plan(
        &plan,
        &[backend(0, 91, 4, 1_000), backend(1, 92, 3, 2_000)],
        64,
        7,
    );
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
        .expect("prepare hybrid");
    let submitted = submit(&mut manager, &prepared);
    complete(&mut manager, &submitted, 17, 29);
    let release = manager
        .release_current_for_test(&[request])
        .expect("release hybrid");
    assert_eq!(release.retirements.len(), 4);
    assert!(release.retirements.iter().any(|item| item.class_id == 0));
    assert!(release.retirements.iter().any(|item| item.class_id == 1));

    let mut forged = reclamation_receipts(&release.retirements);
    let second_pool = forged
        .iter_mut()
        .find(|item| item.page.pool_id == 92)
        .expect("second-pool receipt");
    second_pool.backend_index += 1;
    let before = arena_counts(&manager);
    assert_eq!(
        manager.acknowledge_reclamations_batch(&forged),
        Err(KvManagerError::ReclamationMismatch)
    );
    assert_eq!(arena_counts(&manager), before);

    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&release.retirements))
        .expect("retry exact hybrid acknowledgement");
    manager
        .recycle_requests_batch(&[request])
        .expect("recycle hybrid request");
    assert_eq!(manager.stats().free_pages, 7);
}

#[test]
fn release_ack_recycle_advances_request_and_page_generations() {
    let mut manager = manager_with(18, 4, 64, 8);
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    complete_initial_18(&mut manager, request);
    let old_snapshot = manager.request(request).expect("request").head;
    let old_pages = manager
        .materialize_request_view(request, old_snapshot)
        .expect("old snapshot materialization");
    let release = manager
        .release_current_for_test(&[request])
        .expect("release");
    assert_eq!(release.releases[0].detached_snapshot, old_snapshot);
    assert_eq!(old_pages.len(), 2);
    assert_eq!(
        manager
            .snapshots
            .get(old_snapshot.slot, old_snapshot.generation),
        Err(KvManagerError::StaleLease("snapshot"))
    );
    assert_eq!(
        manager.materialize_request_view(request, old_snapshot),
        Err(KvManagerError::RequestUnavailable)
    );
    let first_page = release.retirements[0].page;
    manager
        .recycle_requests_batch(&[request])
        .expect("page-owned reclamation does not retain request");
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&release.retirements))
        .expect("ack release");
    let next = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("next request");
    assert_eq!(next.slot, request.slot);
    assert_eq!(next.generation, request.generation + 1);
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request: next,
            expected_head: manager.request(next).expect("expected head").head,
            target_boundary: 1,
        }])
        .map(|items| items[0].clone())
        .expect("prepare next");
    assert_eq!(prepared.write_intents[0].page_id, first_page.page_id);
    assert_eq!(
        prepared.write_intents[0].page_generation,
        first_page.generation + 1
    );
    assert!(manager.request(request).is_err());
}
