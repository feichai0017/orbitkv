use std::collections::BTreeSet;

use super::*;

const PAGE_TOKENS: u32 = 4;

fn page(page_id: u32) -> PageLease {
    PageLease {
        engine_epoch: 7,
        pool_epoch: 11,
        generation: 1,
        page_id,
        pool_id: 3,
    }
}

fn location(page_id: u32, offset: u32) -> TokenLocation {
    TokenLocation {
        page: page(page_id),
        backend_index: u64::from(page_id - 1),
        offset,
        reserved: 0,
    }
}

fn page_state(page_id: u32) -> RelocationPageState {
    RelocationPageState {
        page: page(page_id),
        backend_index: u64::from(page_id - 1),
        request_refs: 1,
        prefix_refs: 0,
        reader_pins: 0,
        writer_present: false,
    }
}

fn destination(page_id: u32) -> RelocationDestination {
    RelocationDestination {
        page: page(page_id),
        backend_index: u64::from(page_id - 1),
    }
}

fn placement(token_id: u64, page_id: u32, offset: u32, live: bool) -> TokenPlacement {
    TokenPlacement {
        token_id,
        disposition: if live {
            TokenDisposition::RETAINED
        } else if token_id.is_multiple_of(2) {
            TokenDisposition::semantically_dead(1, 1)
        } else {
            TokenDisposition::policy_evicted(2, 1, 9)
        },
        location: Some(location(page_id, offset)),
    }
}

fn fragmented_view() -> TokenView {
    let mut placements = Vec::new();
    for token_id in 0..12_u64 {
        let page_id = u32::try_from(token_id / u64::from(PAGE_TOKENS)).unwrap() + 1;
        let offset = u32::try_from(token_id % u64::from(PAGE_TOKENS)).unwrap();
        let live = page_id == 1 || offset == 0;
        placements.push(placement(token_id, page_id, offset, live));
    }
    TokenView {
        class_id: 0,
        version: ViewVersion(4),
        page_tokens: PAGE_TOKENS,
        placements: placements.into_boxed_slice(),
    }
}

#[test]
fn profitable_plan_conserves_tokens_and_reclaims_fragmented_pages() {
    let view = fragmented_view();
    let plan = plan_token_relocation(
        &view,
        &[page_state(1), page_state(2), page_state(3)],
        &[destination(4)],
        RelocationPolicy::default(),
    )
    .unwrap()
    .expect("two sparse pages pack into one destination");

    assert_eq!(&*plan.source_pages, &[page(2), page(3)]);
    assert_eq!(&*plan.destination_pages, &[page(4)]);
    assert_eq!(plan.projected_reclaimed_pages, 1);
    assert_eq!(plan.fragmentation_milli, 500);
    assert_eq!(
        plan.moves
            .iter()
            .map(|item| (item.token_id, item.destination.offset))
            .collect::<Vec<_>>(),
        vec![(4, 0), (8, 1)]
    );

    let next = apply_token_relocation(&view, &plan).unwrap();
    assert_eq!(next.version, ViewVersion(5));
    let before_live = retained_tokens(&view);
    let after_live = retained_tokens(&next);
    assert_eq!(before_live, after_live);
    assert!(next.placements.iter().all(|item| {
        item.location
            .is_none_or(|value| !plan.source_pages.contains(&value.page))
    }));
}

#[test]
fn planner_defers_for_shared_pages_headroom_and_nonpositive_gain() {
    let view = fragmented_view();
    let mut shared = page_state(2);
    shared.prefix_refs = 1;
    assert!(
        plan_token_relocation(
            &view,
            &[page_state(1), shared, page_state(3)],
            &[destination(4)],
            RelocationPolicy::default(),
        )
        .unwrap()
        .is_none()
    );
    assert!(
        plan_token_relocation(
            &view,
            &[page_state(1), page_state(2), page_state(3)],
            &[],
            RelocationPolicy::default(),
        )
        .unwrap()
        .is_none()
    );

    let dense = TokenView {
        placements: view.placements[..8].to_vec().into_boxed_slice(),
        ..view
    };
    assert!(
        plan_token_relocation(
            &dense,
            &[page_state(1), page_state(2)],
            &[destination(4)],
            RelocationPolicy::default(),
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn stale_forged_and_cross_pool_plans_fail_closed() {
    let view = fragmented_view();
    let pages = [page_state(1), page_state(2), page_state(3)];
    let plan = plan_token_relocation(
        &view,
        &pages,
        &[destination(4)],
        RelocationPolicy::default(),
    )
    .unwrap()
    .unwrap();
    let mut stale = view.clone();
    stale.version = ViewVersion(5);
    assert_eq!(
        apply_token_relocation(&stale, &plan),
        Err(KvManagerError::StaleTokenView)
    );

    let mut forged = plan.clone();
    let extra = TokenMove {
        token_id: 999,
        source: location(2, 1),
        destination: TokenLocation {
            offset: 2,
            ..location(4, 0)
        },
    };
    forged.moves = forged
        .moves
        .iter()
        .copied()
        .chain([extra])
        .collect::<Vec<_>>()
        .into_boxed_slice();
    assert_eq!(
        apply_token_relocation(&view, &forged),
        Err(KvManagerError::InvalidRelocationPlan)
    );

    let mut wrong_pool = view.clone();
    wrong_pool.placements[8]
        .location
        .as_mut()
        .unwrap()
        .page
        .pool_id = 4;
    assert_eq!(
        plan_token_relocation(
            &wrong_pool,
            &pages,
            &[destination(4)],
            RelocationPolicy::default(),
        ),
        Err(KvManagerError::WrongPageArena)
    );
}

#[test]
fn naive_evict_and_relocation_share_the_exact_victim_set() {
    let view = fragmented_view();
    let updates = [
        TokenDispositionUpdate {
            token_id: 1,
            disposition: TokenDisposition::policy_evicted(41, 3, 99),
        },
        TokenDispositionUpdate {
            token_id: 2,
            disposition: TokenDisposition::policy_evicted(41, 3, 99),
        },
    ];
    let marked = mark_token_dispositions(&view, &updates).unwrap();
    assert_eq!(marked.version, ViewVersion(5));
    assert_eq!(
        marked
            .placements
            .iter()
            .filter(|item| !item.disposition.retained())
            .map(|item| item.token_id)
            .collect::<BTreeSet<_>>(),
        view.placements
            .iter()
            .filter(|item| !item.disposition.retained())
            .map(|item| item.token_id)
            .chain([1, 2])
            .collect::<BTreeSet<_>>()
    );
    assert!(marked.placements[1].location.is_some());
    assert!(marked.placements[2].location.is_some());

    let duplicate = [updates[0], updates[0]];
    assert_eq!(
        mark_token_dispositions(&view, &duplicate),
        Err(KvManagerError::InvalidTokenView)
    );
    assert_eq!(
        mark_token_dispositions(
            &marked,
            &[TokenDispositionUpdate {
                token_id: 1,
                disposition: TokenDisposition::semantically_dead(7, 1),
            }],
        ),
        Err(KvManagerError::InvalidTokenView)
    );
}

#[test]
fn canonical_disposition_batch_is_snapshot_atomic_and_page_neutral() {
    let plan = full_plan(CANONICAL_PAGE_TOKENS);
    let mut manager = manager_for_plan(&plan, &[backend(0, 1, 32, 0)], 64, 32);
    let requests = manager.acquire_request_leases_for_test(2).unwrap();
    let left = complete_initial_18(&mut manager, requests[0]);
    let right = complete_initial_18(&mut manager, requests[1]);
    let before = manager.arena_stats();
    let image = state_image(&manager);
    let update = |request, snapshot, token_id| TokenDispositionBatchItem {
        request,
        expected_snapshot: snapshot,
        updates: vec![ClassTokenDispositionUpdate {
            class_id: 0,
            token_id,
            disposition: TokenDisposition::policy_evicted(51, 1, 77),
        }]
        .into_boxed_slice(),
    };
    let mut stale = right.publication.snapshot;
    stale.generation += 1;
    assert_eq!(
        manager.mark_token_dispositions_batch(&[
            update(requests[0], left.publication.snapshot, 2),
            update(requests[1], stale, 3),
        ]),
        Err(KvManagerError::StaleTokenView)
    );
    assert_eq!(state_image(&manager), image);

    let outputs = manager
        .mark_token_dispositions_batch(&[
            update(requests[0], left.publication.snapshot, 2),
            update(requests[1], right.publication.snapshot, 3),
        ])
        .unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(manager.arena_stats(), before);
    for (output, token_id) in outputs.iter().zip([2, 3]) {
        assert_eq!(output.view_version, ViewVersion(2));
        let token_view = manager
            .token_views_batch(&[TokenViewQuery {
                request: output.request,
                expected_snapshot: output.snapshot,
                class_id: 0,
            }])
            .unwrap()[0]
            .clone();
        let token = &token_view.placements[token_id];
        assert!(matches!(
            token.disposition.kind,
            TokenDispositionKind::PolicyEvicted
        ));
        assert!(token.location.is_some());
    }
}

#[test]
fn randomized_plans_preserve_every_retained_token_and_unique_slot() {
    let mut seed = 0x7265_6c6f_6361_7465_u64;
    for _case in 0..512 {
        let page_count = 3 + usize::try_from(next(&mut seed) % 6).unwrap();
        let mut placements = Vec::with_capacity(page_count * PAGE_TOKENS as usize);
        for index in 0..page_count * PAGE_TOKENS as usize {
            let token_id = u64::try_from(index).unwrap();
            let page_id = u32::try_from(index / PAGE_TOKENS as usize).unwrap() + 1;
            let offset = u32::try_from(index % PAGE_TOKENS as usize).unwrap();
            placements.push(placement(
                token_id,
                page_id,
                offset,
                next(&mut seed).is_multiple_of(5),
            ));
        }
        let view = TokenView {
            class_id: 0,
            version: ViewVersion(next(&mut seed) | 1),
            page_tokens: PAGE_TOKENS,
            placements: placements.into_boxed_slice(),
        };
        let pages = (1..=u32::try_from(page_count).unwrap())
            .map(page_state)
            .collect::<Vec<_>>();
        let destinations = (100..108).map(destination).collect::<Vec<_>>();
        let Some(plan) =
            plan_token_relocation(&view, &pages, &destinations, RelocationPolicy::default())
                .unwrap()
        else {
            continue;
        };
        let next_view = apply_token_relocation(&view, &plan).unwrap();
        assert_eq!(retained_tokens(&view), retained_tokens(&next_view));
        assert!(plan.source_pages.len() > plan.destination_pages.len());
        assert_eq!(
            plan.projected_reclaimed_pages as usize,
            plan.source_pages.len() - plan.destination_pages.len()
        );
        validate_token_view(&next_view).unwrap();
    }
}

#[test]
fn canonical_manager_materializes_full_and_sliding_token_views() {
    let full = full_plan(CANONICAL_PAGE_TOKENS);
    let mut manager = manager_for_plan(&full, &[backend(0, 1, 16, 0)], 64, 16);
    let request = manager.acquire_request_leases_for_test(1).unwrap()[0];
    let completion = complete_initial_18(&mut manager, request);
    let query = TokenViewQuery {
        request,
        expected_snapshot: completion.publication.snapshot,
        class_id: 0,
    };
    let view = manager.token_views_batch(&[query]).unwrap()[0].clone();
    assert_eq!(view.version, completion.publication.view_version);
    assert_eq!(view.placements.len(), 18);
    assert!(
        view.placements
            .iter()
            .all(|item| item.disposition.retained())
    );
    assert_eq!(
        view.placements
            .iter()
            .map(|item| item.location.unwrap().offset)
            .collect::<Vec<_>>(),
        (0..16).chain(0..2).collect::<Vec<_>>()
    );

    let stale = TokenViewQuery {
        expected_snapshot: completion.detached_snapshot,
        ..query
    };
    assert_eq!(
        manager.token_views_batch(&[stale]),
        Err(KvManagerError::StaleTokenView)
    );

    let sliding = sliding_plan(18, CANONICAL_PAGE_TOKENS);
    let mut manager = manager_for_plan(&sliding, &[backend(0, 2, 16, 100)], 64, 16);
    let request = manager.acquire_request_leases_for_test(1).unwrap()[0];
    complete_initial_18(&mut manager, request);
    let completion = append_step(&mut manager, request, 37);
    let view = manager
        .token_views_batch(&[TokenViewQuery {
            request,
            expected_snapshot: completion.publication.snapshot,
            class_id: 0,
        }])
        .unwrap()[0]
        .clone();
    assert_eq!(view.placements.len(), 37);
    assert_eq!(
        view.placements
            .iter()
            .filter(|item| item.disposition.retained())
            .map(|item| item.token_id)
            .collect::<Vec<_>>(),
        (20..37).collect::<Vec<_>>()
    );
    assert!(view.placements[..20].iter().all(|item| {
        matches!(
            item.disposition.kind,
            TokenDispositionKind::SemanticallyDead
        ) && item.location.is_none()
    }));
}

#[test]
#[allow(clippy::too_many_lines)]
fn full_evacuation_relocates_exact_tokens_and_reclaims_source_pages() {
    let plan = full_plan(CANONICAL_PAGE_TOKENS);
    let mut manager = manager_for_plan(&plan, &[backend(0, 1, 8, 0)], 64, 8);
    let request = manager.acquire_request_leases_for_test(1).unwrap()[0];
    let initial = append_step(&mut manager, request, 48);
    let initial_view = manager
        .token_views_batch(&[TokenViewQuery {
            request,
            expected_snapshot: initial.publication.snapshot,
            class_id: 0,
        }])
        .unwrap()[0]
        .clone();
    let updates = (0..48_u64)
        .filter(|token_id| token_id % 16 >= 8)
        .map(|token_id| ClassTokenDispositionUpdate {
            class_id: 0,
            token_id,
            disposition: TokenDisposition::policy_evicted(71, 1, 99),
        })
        .collect::<Vec<_>>();
    let marked = manager
        .mark_token_dispositions_batch(&[TokenDispositionBatchItem {
            request,
            expected_snapshot: initial.publication.snapshot,
            updates: updates.into_boxed_slice(),
        }])
        .unwrap()[0];
    let marked_view = manager
        .token_views_batch(&[TokenViewQuery {
            request,
            expected_snapshot: marked.snapshot,
            class_id: 0,
        }])
        .unwrap()[0]
        .clone();
    assert_eq!(retained_tokens(&marked_view).len(), 24);

    let prepared = manager
        .prepare_relocation_batch(&[PrepareRelocationItem {
            request,
            expected_snapshot: marked.snapshot,
            class_id: 0,
            policy: RelocationPolicy {
                fragmentation_threshold_milli: 250,
                maximum_source_pages: 8,
                evacuation_headroom_pages: 2,
                full_evacuation: true,
            },
        }])
        .unwrap()[0]
        .clone();
    assert_eq!(prepared.plan.source_pages.len(), 3);
    assert_eq!(prepared.plan.destination_pages.len(), 2);
    assert_eq!(prepared.plan.projected_reclaimed_pages, 1);
    assert_eq!(retained_tokens(&initial_view).len(), 48);

    let image = state_image(&manager);
    assert_eq!(
        manager.complete_relocation_batch(
            BatchCompletionReceipt {
                engine_epoch: prepared.relocation.engine_epoch,
                completion_domain: 5,
                completion_value: 7,
                confirmed: 0,
                reserved: 0,
            },
            &[prepared.relocation],
        ),
        Err(KvManagerError::CompletionNotConfirmed)
    );
    assert_eq!(state_image(&manager), image);

    let receipts = prepared
        .plan
        .moves
        .iter()
        .map(|movement| RelocationCopyReceipt {
            relocation: prepared.relocation,
            token_id: movement.token_id,
            source: movement.source,
            destination: movement.destination,
            observed: 1,
            copied: 1,
            reserved16: 0,
            reserved32: 0,
        })
        .collect::<Vec<_>>();
    manager
        .submit_relocation_batch(&[prepared.relocation], &receipts)
        .unwrap();
    assert_eq!(
        manager.abort_relocations_batch(&[RelocationUnobservedReceipt {
            relocation: prepared.relocation,
            backend_unobserved: 1,
            reserved: 0,
        }]),
        Err(KvManagerError::StepAlreadySubmitted)
    );
    let output = manager
        .complete_relocation_batch(
            BatchCompletionReceipt {
                engine_epoch: prepared.relocation.engine_epoch,
                completion_domain: 5,
                completion_value: 7,
                confirmed: 1,
                reserved: 0,
            },
            &[prepared.relocation],
        )
        .unwrap();
    assert_eq!(output.publications.len(), 1);
    assert_eq!(output.retirements.len(), 3);
    let publication = output.publications[0];
    let packed = manager
        .token_views_batch(&[TokenViewQuery {
            request,
            expected_snapshot: publication.snapshot,
            class_id: 0,
        }])
        .unwrap()[0]
        .clone();
    assert_eq!(retained_tokens(&marked_view), retained_tokens(&packed));
    assert!(packed.placements.iter().all(|placement| {
        placement
            .location
            .is_none_or(|location| !prepared.plan.source_pages.contains(&location.page))
    }));
    assert!(matches!(
        manager.publish_prefix_batch(&[PrefixPublishItem {
            request,
            expected_head: publication.snapshot,
            key: prefix_key(7, 48),
        }]),
        Err(KvManagerError::UnsupportedProfile(_))
    ));
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&output.retirements))
        .unwrap();
    let appended = append_step(&mut manager, request, 49);
    assert_eq!(appended.publication.boundary, 49);
    let appended_view = manager
        .token_views_batch(&[TokenViewQuery {
            request,
            expected_snapshot: appended.publication.snapshot,
            class_id: 0,
        }])
        .unwrap()[0]
        .clone();
    assert_eq!(appended_view.placements.len(), 49);
    let newest = appended_view.placements[48];
    assert!(newest.disposition.retained());
    let newest_location = newest.location.unwrap();
    assert_eq!(newest_location.offset, 8);
    assert_eq!(newest_location.page, prepared.plan.destination_pages[1]);
    let appended = append_step(&mut manager, request, 65);
    let appended_view = manager
        .token_views_batch(&[TokenViewQuery {
            request,
            expected_snapshot: appended.publication.snapshot,
            class_id: 0,
        }])
        .unwrap()[0]
        .clone();
    assert_eq!(appended_view.placements.len(), 65);
    assert_eq!(
        appended_view
            .placements
            .iter()
            .filter(|placement| placement.disposition.retained())
            .count(),
        41
    );
    let newest_location = appended_view.placements[64].location.unwrap();
    assert_eq!(newest_location.offset, 8);
    assert!(
        !prepared
            .plan
            .destination_pages
            .contains(&newest_location.page)
    );
    let extra = manager.acquire_request_leases_for_test(1).unwrap()[0];
    assert!(matches!(
        manager.fork_requests_batch(&[RequestForkItem {
            source_request: request,
            expected_source_head: appended.publication.snapshot,
            target_empty_request: extra,
            expected_target_head: manager.request(extra).unwrap().head,
        }]),
        Err(KvManagerError::UnsupportedProfile(_))
    ));
    let stats = manager.stats();
    assert_eq!(stats.retiring_pages, 0);
    assert_eq!(stats.active_pages, 3);
    let release = manager
        .release_batch(&[ReleaseBatchItem {
            request,
            expected_head: appended.publication.snapshot,
        }])
        .unwrap();
    assert_eq!(release.retirements.len(), 3);
    assert_eq!(release.retirements[0].token_begin, 0);
    assert_eq!(release.retirements[0].token_end_exclusive, 16);
    assert_eq!(release.retirements[1].token_begin, 16);
    assert_eq!(release.retirements[1].token_end_exclusive, 32);
    assert_eq!(release.retirements[2].token_begin, 32);
    assert_eq!(release.retirements[2].token_end_exclusive, 41);
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&release.retirements))
        .unwrap();
    manager.recycle_requests_batch(&[request]).unwrap();
    manager.release_current_for_test(&[extra]).unwrap();
    manager.recycle_requests_batch(&[extra]).unwrap();
    let stats = manager.stats();
    assert_eq!(stats.free_pages, 8);
    assert_eq!(stats.active_pages, 0);
    assert_eq!(stats.retiring_pages, 0);
}

#[test]
#[allow(clippy::too_many_lines)]
fn hybrid_full_relocation_keeps_swa_class_specific_placement_and_appends() {
    let plan = hybrid_plan(128);
    let mut manager = manager_for_plan(
        &plan,
        &[backend(0, 1, 16, 0), backend(1, 2, 16, 100)],
        48,
        32,
    );
    let request = manager.acquire_request_leases_for_test(1).unwrap()[0];
    let initial = append_step(&mut manager, request, 48);
    let updates = (0..2_u16)
        .flat_map(|class_id| {
            (0..48_u64)
                .filter(|token_id| token_id % 16 >= 8)
                .map(move |token_id| ClassTokenDispositionUpdate {
                    class_id,
                    token_id,
                    disposition: TokenDisposition::policy_evicted(71, 1, 99),
                })
        })
        .collect::<Vec<_>>();
    let marked = manager
        .mark_token_dispositions_batch(&[TokenDispositionBatchItem {
            request,
            expected_snapshot: initial.publication.snapshot,
            updates: updates.into_boxed_slice(),
        }])
        .unwrap()[0];
    let views = manager
        .token_views_batch(&[
            TokenViewQuery {
                request,
                expected_snapshot: marked.snapshot,
                class_id: 0,
            },
            TokenViewQuery {
                request,
                expected_snapshot: marked.snapshot,
                class_id: 1,
            },
        ])
        .unwrap();
    assert_eq!(retained_tokens(&views[0]), retained_tokens(&views[1]));

    let prepared = manager
        .prepare_relocation_batch(&[PrepareRelocationItem {
            request,
            expected_snapshot: marked.snapshot,
            class_id: 0,
            policy: RelocationPolicy {
                fragmentation_threshold_milli: 250,
                maximum_source_pages: 3,
                evacuation_headroom_pages: 2,
                full_evacuation: true,
            },
        }])
        .unwrap()[0]
        .clone();
    let receipts = prepared
        .plan
        .moves
        .iter()
        .map(|movement| RelocationCopyReceipt {
            relocation: prepared.relocation,
            token_id: movement.token_id,
            source: movement.source,
            destination: movement.destination,
            observed: 1,
            copied: 1,
            reserved16: 0,
            reserved32: 0,
        })
        .collect::<Vec<_>>();
    manager
        .submit_relocation_batch(&[prepared.relocation], &receipts)
        .unwrap();
    let relocated = manager
        .complete_relocation_batch(
            BatchCompletionReceipt {
                engine_epoch: prepared.relocation.engine_epoch,
                completion_domain: 9,
                completion_value: 1,
                confirmed: 1,
                reserved: 0,
            },
            &[prepared.relocation],
        )
        .unwrap();
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&relocated.retirements))
        .unwrap();
    let publication = relocated.publications[0];
    let appended = append_step(&mut manager, request, 49);
    assert_eq!(appended.publication.boundary, 49);
    let views = manager
        .token_views_batch(&[
            TokenViewQuery {
                request,
                expected_snapshot: appended.publication.snapshot,
                class_id: 0,
            },
            TokenViewQuery {
                request,
                expected_snapshot: appended.publication.snapshot,
                class_id: 1,
            },
        ])
        .unwrap();
    assert_eq!(views[0].placements[48].location.unwrap().offset, 8);
    assert_eq!(views[1].placements[48].location.unwrap().offset, 0);
    assert_eq!(retained_tokens(&views[0]), retained_tokens(&views[1]));
    assert_ne!(publication.snapshot, appended.publication.snapshot);

    let released = manager
        .release_batch(&[ReleaseBatchItem {
            request,
            expected_head: appended.publication.snapshot,
        }])
        .unwrap();
    assert_eq!(released.retirements.len(), 6);
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&released.retirements))
        .unwrap();
    manager.recycle_requests_batch(&[request]).unwrap();
    let stats = manager.stats();
    assert_eq!(stats.free_pages, 32);
    assert_eq!(stats.active_pages, 0);
}

#[test]
fn relocation_prepare_abort_and_receipt_faults_are_failure_atomic() {
    let plan = full_plan(CANONICAL_PAGE_TOKENS);
    let mut manager = manager_for_plan(&plan, &[backend(0, 1, 8, 0)], 64, 8);
    let request = manager.acquire_request_leases_for_test(1).unwrap()[0];
    let initial = append_step(&mut manager, request, 48);
    let updates = (0..48_u64)
        .filter(|token_id| token_id % 16 >= 8)
        .map(|token_id| ClassTokenDispositionUpdate {
            class_id: 0,
            token_id,
            disposition: TokenDisposition::policy_evicted(81, 1, 101),
        })
        .collect::<Vec<_>>();
    let marked = manager
        .mark_token_dispositions_batch(&[TokenDispositionBatchItem {
            request,
            expected_snapshot: initial.publication.snapshot,
            updates: updates.into_boxed_slice(),
        }])
        .unwrap()[0];
    let policy = RelocationPolicy {
        fragmentation_threshold_milli: 250,
        maximum_source_pages: 8,
        evacuation_headroom_pages: 2,
        full_evacuation: true,
    };
    let before = state_image(&manager);
    assert_eq!(
        manager.prepare_relocation_batch(&[PrepareRelocationItem {
            request,
            expected_snapshot: marked.snapshot,
            class_id: 0,
            policy: RelocationPolicy {
                evacuation_headroom_pages: 1,
                ..policy
            },
        }]),
        Err(KvManagerError::InvalidRelocationPlan)
    );
    assert_eq!(state_image(&manager), before);

    let prepared = manager
        .prepare_relocation_batch(&[PrepareRelocationItem {
            request,
            expected_snapshot: marked.snapshot,
            class_id: 0,
            policy,
        }])
        .unwrap()[0]
        .clone();
    let reserved = manager.stats();
    assert_eq!(reserved.reserved_pages, 2);
    manager
        .abort_relocations_batch(&[RelocationUnobservedReceipt {
            relocation: prepared.relocation,
            backend_unobserved: 1,
            reserved: 0,
        }])
        .unwrap();
    assert_eq!(manager.stats().reserved_pages, 0);
    assert_eq!(manager.stats().free_pages, 5);

    let prepared = manager
        .prepare_relocation_batch(&[PrepareRelocationItem {
            request,
            expected_snapshot: marked.snapshot,
            class_id: 0,
            policy,
        }])
        .unwrap()[0]
        .clone();
    let mut receipts = relocation_receipts(&prepared);
    receipts[0].token_id += 1;
    assert_eq!(
        manager.submit_relocation_batch(&[prepared.relocation], &receipts),
        Err(KvManagerError::BatchQuarantined(Box::new(
            KvManagerError::CopyReceiptMismatch
        )))
    );
    assert!(manager.request(request).unwrap().quarantined);
    assert_eq!(manager.stats().quarantined_pages, 2);
}

fn relocation_receipts(prepared: &PreparedRelocation) -> Vec<RelocationCopyReceipt> {
    prepared
        .plan
        .moves
        .iter()
        .map(|movement| RelocationCopyReceipt {
            relocation: prepared.relocation,
            token_id: movement.token_id,
            source: movement.source,
            destination: movement.destination,
            observed: 1,
            copied: 1,
            reserved16: 0,
            reserved32: 0,
        })
        .collect()
}

fn append_step(
    manager: &mut CanonicalKvManager,
    request: RequestLease,
    target_boundary: u64,
) -> TestCompletion {
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).unwrap().head,
            target_boundary,
        }])
        .unwrap()[0]
        .clone();
    let submitted = submit(manager, &prepared);
    complete(manager, &submitted, 17, target_boundary)
}

fn retained_tokens(view: &TokenView) -> BTreeSet<u64> {
    view.placements
        .iter()
        .filter(|item| item.disposition.retained())
        .map(|item| item.token_id)
        .collect()
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}
