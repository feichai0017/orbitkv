use super::*;

#[test]
#[allow(clippy::too_many_lines)]
fn public_b4_partial_fork_reaches_joint_cow_and_last_reference_certification() {
    let (mut manager, source, targets) = partial_fork_fixture(4);
    let old_target_heads = targets
        .iter()
        .map(|target| manager.request(*target).expect("old target head").head)
        .collect::<Vec<_>>();
    let forked = manager
        .fork_requests_batch(&fork_items(&manager, source, &targets))
        .expect("public B4 partial fork");
    assert_eq!(forked.len(), 4);
    assert_eq!(
        forked
            .iter()
            .map(|item| item.target.view.snapshot)
            .collect::<BTreeSet<_>>()
            .len(),
        4
    );
    assert!(forked.iter().all(|item| {
        item.source == source
            && item.target.view.boundary == 18
            && item.target.pages.len() == 4
            && item
                .target
                .pages
                .iter()
                .map(|page| page.class_id)
                .collect::<BTreeSet<_>>()
                == BTreeSet::from([0, 1])
    }));
    for item in &forked {
        assert_materialization_matches_snapshot(&manager, &item.target);
    }
    assert!(old_target_heads.iter().all(|head| {
        manager.snapshots.get(head.slot, head.generation)
            == Err(KvManagerError::StaleLease("snapshot"))
    }));
    let source_roots = &manager
        .request_snapshot(source)
        .expect("source snapshot")
        .roots;
    assert!(targets.iter().all(|target| {
        Arc::ptr_eq(
            source_roots,
            &manager
                .request_snapshot(*target)
                .expect("forked target snapshot")
                .roots,
        )
    }));
    assert_reference_census_matches_full_scan(&manager);

    let source_release = manager
        .release_current_for_test(&[source])
        .expect("release fork source");
    assert!(source_release.retirements.is_empty());
    assert_eq!(source_release.releases[0].detached.len(), 4);
    assert!(source_release.releases[0].detached.iter().all(|binding| {
        binding.action == DetachedAction::Clear
            && binding.reason == DetachedReason::RequestRelease
            && binding.replacement == PageLease::default()
            && binding.reserved == 0
    }));
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request: targets[0],
            expected_head: manager.request(targets[0]).expect("fork head").head,
            target_boundary: 19,
        }])
        .expect("publicly reachable shared-tail append");
    assert!(
        prepared[0]
            .tail_actions
            .iter()
            .all(|action| action.kind == TailActionKind::CopyOnWrite)
    );
    assert_eq!(prepared[0].copy_intents.len(), 2);
    let submitted = submit(&mut manager, &prepared[0]);
    let completion = complete(&mut manager, &submitted, 13, 19);
    assert!(completion.retirements.is_empty());
    assert_eq!(completion.step.detached.len(), 2);
    assert!(completion.step.detached.iter().all(|binding| {
        binding.action == DetachedAction::Replace
            && binding.reason == DetachedReason::CopyOnWrite
            && binding.logical_ordinal == 1
            && binding.token_begin == 16
            && binding.token_end_exclusive == 18
            && binding.old != binding.replacement
            && binding.reserved == 0
    }));
    for copy in &prepared[0].copy_intents {
        let binding = completion
            .step
            .detached
            .iter()
            .find(|binding| binding.class_id == copy.class_id)
            .expect("COW detach for every class");
        assert_eq!(binding.old, copy.source);
        assert_eq!(binding.replacement, copy.destination);
        assert_eq!(binding.old_backend_index, copy.source_backend_index);
        assert_eq!(
            binding.replacement_backend_index,
            copy.destination_backend_index
        );
    }
    assert_reference_census_matches_full_scan(&manager);
    for sibling in &targets[1..] {
        let view = manager
            .request_view(*sibling)
            .expect("sibling survives COW");
        assert_eq!(view.boundary, 18);
        assert_eq!(
            manager
                .materialize_request_view(*sibling, view.snapshot)
                .expect("sibling materialization")
                .len(),
            4
        );
    }

    let intermediate = manager
        .release_current_for_test(&targets[1..3])
        .expect("release two shared siblings");
    assert!(intermediate.retirements.is_empty());
    assert!(
        intermediate
            .releases
            .iter()
            .all(|release| release.detached.len() == 4)
    );
    let last_old_tail = manager
        .release_current_for_test(&targets[3..4])
        .expect("release last old-tail sibling");
    assert_eq!(last_old_tail.retirements.len(), 2);
    assert_eq!(last_old_tail.releases[0].detached.len(), 4);
    assert!(last_old_tail.retirements.iter().all(|certificate| {
        certificate.logical_ordinal == 1
            && certificate.token_begin == 16
            && certificate.token_end_exclusive == 18
    }));
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&last_old_tail.retirements))
        .expect("ack last old tails");
    let extended_release = manager
        .release_current_for_test(&targets[..1])
        .expect("release COW request");
    assert_eq!(extended_release.retirements.len(), 4);
    assert_eq!(extended_release.releases[0].detached.len(), 4);
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&extended_release.retirements))
        .expect("ack extended request");
    assert_reference_census_matches_full_scan(&manager);
    let mut requests = vec![source];
    requests.extend(targets.iter().copied());
    manager
        .recycle_requests_batch(&requests)
        .expect("recycle fork lifecycle requests");
    assert_eq!(manager.stats().free_pages, 64);
}

#[test]
#[allow(clippy::too_many_lines)]
fn request_fork_batch_stale_and_staging_oom_are_zero_mutation() {
    let (mut manager, source, targets) = partial_fork_fixture(1);
    let valid = fork_items(&manager, source, &targets);
    let mut stale = valid.clone();
    stale[2].expected_target_head.generation = stale[2]
        .expected_target_head
        .generation
        .checked_add(1)
        .expect("target generation");
    let before = state_image(&manager);
    assert_eq!(
        manager.fork_requests_batch(&stale),
        Err(KvManagerError::StaleView)
    );
    assert_eq!(state_image(&manager), before);
    assert_reference_census_matches_full_scan(&manager);

    assert_eq!(
        manager.fork_requests_batch(&[]),
        Err(KvManagerError::EmptyBatch)
    );
    let before = state_image(&manager);
    assert_eq!(
        manager.fork_requests_batch(&[valid[0], valid[0]]),
        Err(KvManagerError::DuplicateRequest)
    );
    assert_eq!(state_image(&manager), before);

    let overlap = RequestForkItem {
        source_request: source,
        expected_source_head: manager.request(source).expect("source head").head,
        target_empty_request: source,
        expected_target_head: manager.request(source).expect("source head").head,
    };
    let before = state_image(&manager);
    assert_eq!(
        manager.fork_requests_batch(&[overlap]),
        Err(KvManagerError::DuplicateRequest)
    );
    assert_eq!(state_image(&manager), before);

    let busy = manager
        .prepare_batch(&[PrepareBatchItem {
            request: targets[0],
            expected_head: manager.request(targets[0]).expect("target head").head,
            target_boundary: 1,
        }])
        .expect("make fork target busy");
    let before = state_image(&manager);
    assert_eq!(
        manager.fork_requests_batch(&valid),
        Err(KvManagerError::RequestBusy)
    );
    assert_eq!(state_image(&manager), before);
    manager
        .abort_steps_batch(&[BackendUnobservedReceipt {
            step: busy[0].step,
            backend_unobserved: 1,
            reserved: 0,
        }])
        .expect("clear busy target");

    manager
        .request_mut(targets[1])
        .expect("quarantine target")
        .quarantined = true;
    let before = state_image(&manager);
    assert_eq!(
        manager.fork_requests_batch(&valid),
        Err(KvManagerError::RequestUnavailable)
    );
    assert_eq!(state_image(&manager), before);
    manager
        .request_mut(targets[1])
        .expect("restore target")
        .quarantined = false;

    let shared_page = manager
        .request_snapshot(source)
        .expect("source snapshot")
        .roots[0]
        .entries
        .front()
        .expect("source page")
        .page;
    let original_refs = manager
        .page(shared_page.page_id)
        .expect("source page state")
        .request_refs;
    manager
        .page_mut(shared_page.page_id)
        .expect("source page state")
        .request_refs = u32::MAX;
    let before = state_image(&manager);
    assert_eq!(
        manager.fork_requests_batch(&valid),
        Err(KvManagerError::ReferenceCountOverflow(shared_page.page_id))
    );
    assert_eq!(state_image(&manager), before);
    manager
        .page_mut(shared_page.page_id)
        .expect("restore source page")
        .request_refs = original_refs;
    assert_reference_census_matches_full_scan(&manager);

    while manager.snapshots.free.len() > targets.len() - 1 {
        let slot = manager.snapshots.free.pop().expect("free snapshot slot");
        manager.snapshots.slots[slot as usize].generation = u32::MAX;
    }
    let before = state_image(&manager);
    assert_eq!(
        manager.fork_requests_batch(&valid),
        Err(KvManagerError::ArenaExhausted("snapshot"))
    );
    assert_eq!(state_image(&manager), before);
    assert_reference_census_matches_full_scan(&manager);
}

#[test]
fn request_fork_aggregates_distinct_shared_sources_and_empty_roots_exactly() {
    let (mut manager, source, targets) = partial_fork_fixture(2);
    let first = manager
        .fork_requests_batch(&[fork_items(&manager, source, &targets[..1])[0]])
        .expect("make a second source sharing the partial root");
    assert_eq!(first[0].target.pages.len(), 4);
    let items = [
        RequestForkItem {
            source_request: source,
            expected_source_head: manager.request(source).expect("first source head").head,
            target_empty_request: targets[1],
            expected_target_head: manager.request(targets[1]).expect("target head").head,
        },
        RequestForkItem {
            source_request: targets[0],
            expected_source_head: manager
                .request(targets[0])
                .expect("second source head")
                .head,
            target_empty_request: targets[2],
            expected_target_head: manager.request(targets[2]).expect("target head").head,
        },
    ];
    let forked = manager
        .fork_requests_batch(&items)
        .expect("aggregate two distinct sources sharing pages");
    assert_eq!(forked[0].source, source);
    assert_eq!(forked[1].source, targets[0]);
    assert!(forked.iter().all(|item| item.target.pages.len() == 4));
    for entry in snapshot_entries(&manager, source) {
        assert_eq!(
            manager
                .page(entry.page.page_id)
                .expect("shared source page")
                .request_refs,
            4
        );
    }
    assert_reference_census_matches_full_scan(&manager);
    assert_incremental_census_matches_full_scan(&manager);

    let plan = full_plan(CANONICAL_PAGE_TOKENS);
    let mut empty = CanonicalKvManager::new(
        &plan,
        ManagerConfig {
            maximum_requests: 2,
            maximum_operations: 1,
            maximum_prefixes: 1,
            maximum_reclamations: 1,
            maximum_step_tokens: 16,
        },
        &[backend(0, 195, 1, 13_000)],
    )
    .expect("empty fork manager");
    let requests = empty
        .acquire_requests_batch(2)
        .expect("empty fork requests");
    let old_target = requests[1].snapshot;
    let forked = empty
        .fork_requests_batch(&[RequestForkItem {
            source_request: requests[0].request,
            expected_source_head: requests[0].snapshot,
            target_empty_request: requests[1].request,
            expected_target_head: old_target,
        }])
        .expect("empty snapshot fork");
    assert_eq!(forked[0].target.view.boundary, 0);
    assert_eq!(forked[0].target.view.resident_count, 0);
    assert!(forked[0].target.pages.is_empty());
    assert_ne!(forked[0].target.view.snapshot, old_target);
    assert_eq!(empty.stats().total_request_page_refs, 0);
    assert_reference_census_matches_full_scan(&empty);
}

#[test]
#[allow(clippy::too_many_lines)]
fn b4_prefix_attach_evict_and_last_close_are_reference_exact() {
    let plan = hybrid_plan(18);
    let mut manager = CanonicalKvManager::new(
        &plan,
        ManagerConfig {
            maximum_requests: 8,
            maximum_operations: 8,
            maximum_prefixes: 16,
            maximum_reclamations: 32,
            maximum_step_tokens: 64,
        },
        &[backend(0, 201, 16, 10_000), backend(1, 202, 16, 20_000)],
    )
    .expect("prefix manager");
    let source = manager.acquire_request_leases_for_test(1).expect("source")[0];
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request: source,
            expected_head: manager.request(source).expect("expected head").head,
            target_boundary: 32,
        }])
        .expect("source prepare");
    let submitted = submit(&mut manager, &prepared[0]);
    complete(&mut manager, &submitted, 9, 11);
    let key = prefix_key(1, 32);
    let published = manager
        .publish_prefix_batch(&[PrefixPublishItem {
            request: source,
            expected_head: manager.request(source).expect("expected head").head,
            key,
        }])
        .expect("publish prefix")[0];
    assert_reference_census_matches_full_scan(&manager);

    let attached_requests = manager
        .acquire_request_leases_for_test(4)
        .expect("B4 requests");
    let hints = manager
        .lookup_prefix_batch(&[key; 4])
        .expect("prefix lookup");
    let attach_items = attached_requests
        .iter()
        .copied()
        .zip(hints.iter().copied())
        .map(|(request, hint)| PrefixAttachItem {
            request,
            expected_empty_head: manager.request(request).expect("empty head").head,
            hint,
        })
        .collect::<Vec<_>>();
    let attached = manager
        .attach_prefix_batch(&attach_items)
        .expect("B4 attach");
    assert_eq!(attached.len(), 4);
    assert!(
        attached
            .iter()
            .zip(attached_requests.iter())
            .all(|(item, request)| {
                item.target.view.request == *request
                    && item.target.view.boundary == 32
                    && item.target.pages.len() == 4
                    && item.target.pages.iter().all(|page| {
                        page.valid_token_count
                            == u32::try_from(CANONICAL_PAGE_TOKENS).expect("page tokens")
                            && page.visible_token_count != 0
                    })
            })
    );
    for item in &attached {
        assert_materialization_matches_snapshot(&manager, &item.target);
    }
    assert_eq!(
        attached
            .iter()
            .map(|item| item.target.view.snapshot)
            .collect::<BTreeSet<_>>()
            .len(),
        4
    );
    let first_roots = &manager
        .request_snapshot(attached_requests[0])
        .expect("first attached")
        .roots;
    assert!(attached_requests.iter().skip(1).all(|request| {
        Arc::ptr_eq(
            first_roots,
            &manager
                .request_snapshot(*request)
                .expect("attached snapshot")
                .roots,
        )
    }));
    assert_reference_census_matches_full_scan(&manager);

    let stale_hint = hints[0];
    let miss_request = manager
        .acquire_request_leases_for_test(1)
        .expect("miss request")[0];
    let eviction = manager
        .evict_prefix_batch(&[published.prefix])
        .expect("evict while requests retain roots");
    assert!(eviction.retirements.is_empty());
    assert_eq!(
        manager.attach_prefix_batch(&[PrefixAttachItem {
            request: miss_request,
            expected_empty_head: manager
                .request(miss_request)
                .expect("expected empty head")
                .head,
            hint: stale_hint,
        }]),
        Err(KvManagerError::PrefixHintStale)
    );
    assert_eq!(
        manager
            .request_view(miss_request)
            .expect("unchanged miss")
            .boundary,
        0
    );
    manager
        .recycle_prefixes_batch(&[published.prefix])
        .expect("recycle evicted prefix");
    assert_reference_census_matches_full_scan(&manager);

    let source_release = manager
        .release_current_for_test(&[source])
        .expect("release source");
    assert!(source_release.retirements.is_empty());
    let batch_release = manager
        .release_current_for_test(&attached_requests)
        .expect("last shared close");
    assert_eq!(batch_release.retirements.len(), 4);
    assert_eq!(
        batch_release
            .retirements
            .iter()
            .map(|certificate| certificate.page)
            .collect::<BTreeSet<_>>()
            .len(),
        4
    );
    let miss_release = manager
        .release_current_for_test(&[miss_request])
        .expect("release miss request");
    assert!(miss_release.retirements.is_empty());
    assert_reference_census_matches_full_scan(&manager);
    let mut all_requests = vec![source, miss_request];
    all_requests.extend(attached_requests.iter().copied());
    manager
        .recycle_requests_batch(&all_requests)
        .expect("page transactions do not retain requests");
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&batch_release.retirements))
        .expect("ack unique last-close certificates");
    assert_eq!(manager.stats().free_pages, 32);
}

#[test]
fn attach_staging_is_independent_of_operation_capacity_and_oom_is_atomic() {
    let (mut manager, requests, hints) = staged_prefix_attach_fixture();
    assert_eq!(manager.requests.active_len(), 4);
    assert_eq!(manager.operations.slots.len(), 1);
    let old_heads = requests
        .iter()
        .map(|request| manager.request(*request).expect("old head").head)
        .collect::<Vec<_>>();
    let attached = manager
        .attach_prefix_batch(&attach_items(&manager, &requests, &hints))
        .expect("B3 attach does not depend on maximum_operations");
    assert_eq!(attached.len(), 3);
    assert!(old_heads.iter().all(|head| {
        manager.snapshots.get(head.slot, head.generation)
            == Err(KvManagerError::StaleLease("snapshot"))
    }));
    assert_reference_census_matches_full_scan(&manager);

    let (mut exhausted, requests, hints) = staged_prefix_attach_fixture();
    while exhausted.snapshots.free.len() > 2 {
        let slot = exhausted.snapshots.free.pop().expect("free snapshot slot");
        exhausted.snapshots.slots[slot as usize].generation = u32::MAX;
    }
    let items = attach_items(&exhausted, &requests, &hints);
    let before = state_image(&exhausted);
    assert_eq!(
        exhausted.attach_prefix_batch(&items),
        Err(KvManagerError::ArenaExhausted("snapshot"))
    );
    assert_eq!(state_image(&exhausted), before);
    for (request, item) in requests.iter().zip(items.iter()) {
        let view = exhausted
            .request_view(*request)
            .expect("unchanged empty view");
        assert_eq!(view.snapshot, item.expected_empty_head);
        assert_eq!(view.boundary, 0);
    }
    assert_reference_census_matches_full_scan(&exhausted);
}

#[test]
#[allow(clippy::too_many_lines)]
fn prefix_publish_gates_and_joint_publish_release_are_atomic() {
    let plan = hybrid_plan(18);
    let mut manager = CanonicalKvManager::new(
        &plan,
        ManagerConfig {
            maximum_requests: 4,
            maximum_operations: 4,
            maximum_prefixes: 8,
            maximum_reclamations: 16,
            maximum_step_tokens: 64,
        },
        &[backend(0, 211, 8, 30_000), backend(1, 212, 8, 40_000)],
    )
    .expect("prefix manager");
    let requests = manager.acquire_request_leases_for_test(2).expect("sources");
    let prepared = manager
        .prepare_batch(
            &requests
                .iter()
                .copied()
                .map(|request| PrepareBatchItem {
                    request,
                    expected_head: manager.request(request).expect("expected head").head,
                    target_boundary: 32,
                })
                .collect::<Vec<_>>(),
        )
        .expect("prepare sources");
    let (submit_items, bind_receipts, copy_receipts) = batch_submit_items(&manager, &prepared);
    let submitted = manager
        .submit_batch(&submit_items, &bind_receipts, &copy_receipts)
        .expect("submit sources");
    manager
        .complete_batch(
            BatchCompletionReceipt {
                engine_epoch: manager.engine_epoch,
                completion_domain: 5,
                completion_value: 7,
                confirmed: 1,
                reserved: 0,
            },
            &submitted
                .iter()
                .map(|item| item.submission)
                .collect::<Vec<_>>(),
        )
        .expect("complete sources");
    let before_prefixes = manager.prefixes.active_len();
    assert_eq!(
        manager.publish_prefix_batch(&[PrefixPublishItem {
            request: requests[0],
            expected_head: manager.request(requests[0]).expect("expected head").head,
            key: prefix_key(2, 31),
        }]),
        Err(KvManagerError::PrefixBoundaryNotPageAligned)
    );
    let duplicate = prefix_key(3, 32);
    assert_eq!(
        manager.publish_prefix_batch(&[
            PrefixPublishItem {
                request: requests[0],
                expected_head: manager.request(requests[0]).expect("expected head").head,
                key: duplicate,
            },
            PrefixPublishItem {
                request: requests[1],
                expected_head: manager.request(requests[1]).expect("expected head").head,
                key: duplicate,
            },
        ]),
        Err(KvManagerError::DuplicatePrefixKey)
    );
    assert_eq!(manager.prefixes.active_len(), before_prefixes);
    assert_reference_census_matches_full_scan(&manager);

    let key = prefix_key(4, 32);
    let detached_view = manager.request_view(requests[0]).expect("source view");
    let detached_pages = manager
        .materialize_request_view(requests[0], detached_view.snapshot)
        .expect("source materialization");
    let transfer = manager
        .publish_prefix_and_release_batch(&[PrefixPublishItem {
            request: requests[0],
            expected_head: manager.request(requests[0]).expect("expected head").head,
            key,
        }])
        .expect("joint publish release");
    assert_eq!(transfer[0].release.request, requests[0]);
    assert_eq!(
        transfer[0].release.detached_snapshot,
        detached_view.snapshot
    );
    assert_eq!(detached_pages.len(), 4);
    assert_eq!(transfer[0].release.detached.len(), detached_pages.len());
    assert!(transfer[0].release.detached.iter().all(|binding| {
        binding.action == DetachedAction::Clear
            && binding.reason == DetachedReason::PrefixTransfer
            && binding.replacement == PageLease::default()
            && detached_pages.iter().any(|page| {
                page.page == binding.old
                    && page.logical_ordinal == binding.logical_ordinal
                    && page.backend_index == binding.old_backend_index
                    && u64::from(page.valid_token_count)
                        == binding.token_end_exclusive - binding.token_begin
            })
    }));
    assert_eq!(
        manager.snapshots.get(
            detached_view.snapshot.slot,
            detached_view.snapshot.generation
        ),
        Err(KvManagerError::StaleLease("snapshot"))
    );
    assert_eq!(
        manager.materialize_request_view(requests[0], detached_view.snapshot),
        Err(KvManagerError::RequestUnavailable)
    );
    assert_reference_census_matches_full_scan(&manager);
    manager
        .recycle_requests_batch(&[requests[0]])
        .expect("jointly released request recycles before prefix");
    let eviction = manager
        .evict_prefix_batch(&[transfer[0].publication.prefix])
        .expect("evict transferred prefix");
    assert_eq!(eviction.retirements.len(), 4);
    manager
        .recycle_prefixes_batch(&[transfer[0].publication.prefix])
        .expect("recycle transferred prefix");
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&eviction.retirements))
        .expect("ack transferred prefix pages");
    let release = manager
        .release_current_for_test(&[requests[1]])
        .expect("last source release");
    assert_eq!(release.retirements.len(), 4);
    assert!(
        release.releases[0]
            .detached
            .iter()
            .all(|binding| binding.reason == DetachedReason::RequestRelease)
    );
    manager
        .recycle_requests_batch(&[requests[1]])
        .expect("recycle last source");
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&release.retirements))
        .expect("ack last source pages");
    assert_eq!(manager.stats().free_pages, 16);
}

#[test]
fn private_partial_tail_is_in_place_but_shared_hybrid_tail_is_joint_cow() {
    let plan = hybrid_plan(18);
    let mut private = manager_for_plan(
        &plan,
        &[backend(0, 221, 8, 50_000), backend(1, 222, 8, 60_000)],
        64,
        16,
    );
    let private_request = private
        .acquire_request_leases_for_test(1)
        .expect("private request")[0];
    complete_initial_18(&mut private, private_request);
    let private_step = private
        .prepare_batch(&[PrepareBatchItem {
            request: private_request,
            expected_head: private
                .request(private_request)
                .expect("expected head")
                .head,
            target_boundary: 19,
        }])
        .expect("private tail prepare");
    assert!(private_step[0].tail_actions.iter().all(|action| action.kind
        == TailActionKind::InPlace
        && action.source == action.destination));
    assert!(private_step[0].copy_intents.is_empty());
    assert!(private_step[0].write_intents.is_empty());
    private
        .abort_steps_batch(&[BackendUnobservedReceipt {
            step: private_step[0].step,
            backend_unobserved: 1,
            reserved: 0,
        }])
        .expect("abort private tail");

    let mut shared = manager_for_plan(
        &plan,
        &[backend(0, 223, 8, 70_000), backend(1, 224, 8, 80_000)],
        64,
        16,
    );
    let requests = shared
        .acquire_request_leases_for_test(2)
        .expect("shared requests");
    complete_initial_18(&mut shared, requests[0]);
    share_snapshot_for_cow(&mut shared, requests[0], requests[1]);
    assert!(Arc::ptr_eq(
        &shared.request_snapshot(requests[0]).expect("source").roots,
        &shared.request_snapshot(requests[1]).expect("fork").roots,
    ));
    assert_reference_census_matches_full_scan(&shared);

    let prepared = shared
        .prepare_batch(&[PrepareBatchItem {
            request: requests[0],
            expected_head: shared.request(requests[0]).expect("expected head").head,
            target_boundary: 19,
        }])
        .expect("shared tail prepare");
    assert_eq!(prepared[0].tail_actions.len(), 2);
    assert_eq!(prepared[0].copy_intents.len(), 2);
    assert!(prepared[0].write_intents.is_empty());
    assert!(prepared[0].tail_actions.iter().all(|action| {
        action.kind == TailActionKind::CopyOnWrite
            && action.source != action.destination
            && action.valid_token_count == 2
    }));
    let sources = prepared[0]
        .copy_intents
        .iter()
        .map(|intent| intent.source)
        .collect::<Vec<_>>();
    let submitted = submit(&mut shared, &prepared[0]);
    let completion = complete(&mut shared, &submitted, 13, 17);
    assert!(completion.retirements.is_empty());
    for source in sources {
        let page = shared.page(source.page_id).expect("shared COW source");
        assert_eq!(page.phase, PagePhase::Live);
        assert_eq!(page.request_refs, 1);
        assert_eq!(page.reader_pins, 0);
        assert!(page.writer.is_none());
    }
    assert_reference_census_matches_full_scan(&shared);
}

#[test]
#[allow(clippy::too_many_lines)]
fn shared_cow_completion_aggregates_sources_and_emits_one_cert_per_page() {
    let plan = hybrid_plan(18);
    let mut manager = manager_for_plan(
        &plan,
        &[backend(0, 229, 6, 130_000), backend(1, 230, 6, 140_000)],
        64,
        16,
    );
    let requests = manager
        .acquire_request_leases_for_test(2)
        .expect("shared requests");
    complete_initial_18(&mut manager, requests[0]);
    share_snapshot_for_cow(&mut manager, requests[0], requests[1]);
    let prepared = manager
        .prepare_batch(&[
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
        ])
        .expect("B2 joint COW prepare");
    assert!(prepared.iter().all(|step| {
        step.tail_actions
            .iter()
            .all(|action| action.kind == TailActionKind::CopyOnWrite)
            && step.copy_intents.len() == 2
    }));
    let old_sources = prepared[0]
        .copy_intents
        .iter()
        .map(|intent| intent.source)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        prepared[1]
            .copy_intents
            .iter()
            .map(|intent| intent.source)
            .collect::<BTreeSet<_>>(),
        old_sources
    );
    let (items, binds, copies) = batch_submit_items(&manager, &prepared);
    let submitted = manager
        .submit_batch(&items, &binds, &copies)
        .expect("B2 joint COW submit");
    for source in &old_sources {
        assert_eq!(
            manager
                .page(source.page_id)
                .expect("pinned source")
                .reader_pins,
            2
        );
    }
    let completion = manager
        .complete_batch(
            BatchCompletionReceipt {
                engine_epoch: manager.engine_epoch,
                completion_domain: 23,
                completion_value: 29,
                confirmed: 1,
                reserved: 0,
            },
            &submitted
                .iter()
                .map(|item| item.submission)
                .collect::<Vec<_>>(),
        )
        .expect("B2 joint COW completion");
    assert_eq!(completion.completions.len(), 2);
    assert_eq!(completion.retirements.len(), 2);
    assert!(
        completion
            .retirements
            .iter()
            .all(|certificate| certificate.token_begin == 16
                && certificate.token_end_exclusive == 18)
    );
    assert_eq!(
        completion
            .retirements
            .iter()
            .map(|certificate| certificate.page)
            .collect::<BTreeSet<_>>(),
        old_sources
    );
    for certificate in &completion.retirements {
        let page = manager
            .page(certificate.page.page_id)
            .expect("retiring source");
        assert_eq!(page.request_refs, 0);
        assert_eq!(page.reader_pins, 0);
        assert_eq!(
            page.phase,
            PagePhase::Retiring {
                reclamation: certificate.reclamation
            }
        );
    }
    assert_reference_census_matches_full_scan(&manager);
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&completion.retirements))
        .expect("ack unique source certificates");
}

#[test]
#[allow(clippy::too_many_lines)]
fn snapshot_and_prefix_generation_exhaustion_never_reuses_an_old_lease() {
    let plan = full_plan(CANONICAL_PAGE_TOKENS);
    let mut snapshots = CanonicalKvManager::new(
        &plan,
        ManagerConfig {
            maximum_requests: 1,
            maximum_operations: 1,
            maximum_prefixes: 1,
            maximum_reclamations: 2,
            maximum_step_tokens: 16,
        },
        &[backend(0, 232, 2, 160_000)],
    )
    .expect("snapshot exhaustion manager");
    let mut exhausted_snapshot_leases = Vec::new();
    let snapshot_capacity = snapshots.snapshots.slots.len();
    for _ in 0..snapshot_capacity {
        let request = snapshots
            .acquire_request_leases_for_test(1)
            .expect("request")[0];
        let head = snapshots.request(request).expect("request state").head;
        snapshots.snapshots.slots[head.slot as usize].generation = u32::MAX;
        snapshots
            .request_mut(request)
            .expect("request state")
            .head
            .generation = u32::MAX;
        let exhausted = SnapshotLease {
            generation: u32::MAX,
            ..head
        };
        snapshots
            .release_current_for_test(&[request])
            .expect("release exhausted snapshot slot");
        assert_eq!(
            snapshots
                .snapshots
                .get(exhausted.slot, exhausted.generation),
            Err(KvManagerError::StaleLease("snapshot"))
        );
        snapshots
            .recycle_requests_batch(&[request])
            .expect("recycle request after snapshot exhaustion");
        exhausted_snapshot_leases.push(exhausted);
    }
    assert_eq!(
        snapshots.acquire_request_leases_for_test(1),
        Err(KvManagerError::ArenaExhausted("snapshot"))
    );
    assert_eq!(snapshots.requests.active_len(), 0);
    assert_eq!(
        exhausted_snapshot_leases
            .iter()
            .map(|lease| lease.slot)
            .collect::<BTreeSet<_>>()
            .len(),
        snapshot_capacity
    );

    let mut prefixes = CanonicalKvManager::new(
        &plan,
        ManagerConfig {
            maximum_requests: 2,
            maximum_operations: 2,
            maximum_prefixes: 1,
            maximum_reclamations: 2,
            maximum_step_tokens: 16,
        },
        &[backend(0, 233, 2, 170_000)],
    )
    .expect("prefix exhaustion manager");
    let source = prefixes.acquire_request_leases_for_test(1).expect("source")[0];
    let prepared = prefixes
        .prepare_batch(&[PrepareBatchItem {
            request: source,
            expected_head: prefixes.request(source).expect("expected head").head,
            target_boundary: 16,
        }])
        .expect("source prepare");
    let submitted = submit(&mut prefixes, &prepared[0]);
    complete(&mut prefixes, &submitted, 37, 1);
    let first = prefixes
        .publish_prefix_batch(&[PrefixPublishItem {
            request: source,
            expected_head: prefixes.request(source).expect("expected head").head,
            key: prefix_key(40, 16),
        }])
        .expect("first prefix")[0]
        .prefix;
    prefixes
        .evict_prefix_batch(&[first])
        .expect("evict first prefix");
    prefixes
        .recycle_prefixes_batch(&[first])
        .expect("recycle first prefix");
    prefixes.prefixes.slots[first.slot as usize].generation = u32::MAX - 1;
    let exhausted = prefixes
        .publish_prefix_batch(&[PrefixPublishItem {
            request: source,
            expected_head: prefixes.request(source).expect("expected head").head,
            key: prefix_key(41, 16),
        }])
        .expect("maximum-generation prefix")[0]
        .prefix;
    assert_eq!(exhausted.generation, u32::MAX);
    assert!(matches!(
        prefixes.prefixes.get(first.slot, first.generation),
        Err(KvManagerError::StaleLease("prefix"))
    ));
    prefixes
        .evict_prefix_batch(&[exhausted])
        .expect("evict maximum-generation prefix");
    prefixes
        .recycle_prefixes_batch(&[exhausted])
        .expect("exhaust prefix slot");
    assert_eq!(
        prefixes.publish_prefix_batch(&[PrefixPublishItem {
            request: source,
            expected_head: prefixes.request(source).expect("expected head").head,
            key: prefix_key(42, 16),
        }]),
        Err(KvManagerError::ArenaExhausted("prefix"))
    );
    assert_reference_census_matches_full_scan(&prefixes);
}
