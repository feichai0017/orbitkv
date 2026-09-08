use super::*;

const CHUNK_TOKENS: u64 = 32;

fn append_to(
    manager: &mut CanonicalKvManager,
    request: RequestLease,
    target_boundary: u64,
    completion_value: u64,
) -> TestCompletion {
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("request head").head,
            target_boundary,
        }])
        .expect("chunked append prepares")[0]
        .clone();
    let submitted = submit(manager, &prepared);
    complete(manager, &submitted, 17, completion_value)
}

#[test]
fn chunked_exact_epoch_end_retires_absolute_root_and_publishes_empty_epoch() {
    let mut manager = chunked_manager(CHUNK_TOKENS, 4, 32, 4);
    let request = manager.acquire_request_leases_for_test(1).unwrap()[0];
    let plan = chunked_plan(CHUNK_TOKENS);
    let retirement = &plan.layout_program().unwrap().classes[0].retirement;
    assert_eq!(
        (0..2)
            .map(|ordinal| retirement.death_boundary(CANONICAL_PAGE_TOKENS, ordinal))
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
        vec![Some(CHUNK_TOKENS), Some(CHUNK_TOKENS)]
    );

    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).unwrap().head,
            target_boundary: 31,
        }])
        .unwrap()[0]
        .clone();
    assert_eq!(
        prepared.class_lowerings[0].flags,
        CLASS_LOWERING_RESETTABLE | CLASS_LOWERING_EPOCH_START
    );
    assert_eq!(prepared.class_lowerings[0].previous_layout_boundary, 0);
    assert_eq!(prepared.class_lowerings[0].target_layout_boundary, 31);
    let delta = match manager
        .operations
        .get(prepared.step.slot, prepared.step.generation)
        .unwrap()
    {
        OperationState::Prepared(state) => &state.delta.classes[0],
        OperationState::Submitted(_) => panic!("step must still be prepared"),
    };
    assert!(!delta.epoch_reset);
    assert_eq!(
        delta
            .writes
            .iter()
            .map(|entry| (
                entry.logical_ordinal,
                entry.temporal_cell_index,
                entry.temporal_cycle,
            ))
            .collect::<Vec<_>>(),
        vec![(0, 0, 0), (1, 1, 0)]
    );
    let submitted = submit(&mut manager, &prepared);
    assert!(
        complete(&mut manager, &submitted, 17, 1)
            .retirements
            .is_empty()
    );

    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).unwrap().head,
            target_boundary: CHUNK_TOKENS,
        }])
        .unwrap()[0]
        .clone();
    assert_eq!(prepared.class_lowerings[0].flags, CLASS_LOWERING_RESETTABLE);
    assert_eq!(prepared.class_lowerings[0].previous_layout_boundary, 31);
    assert_eq!(
        prepared.class_lowerings[0].target_layout_boundary,
        CHUNK_TOKENS
    );
    assert_eq!(prepared.tail_actions[0].kind, TailActionKind::InPlace);
    let submitted = submit(&mut manager, &prepared);
    let completion = complete(&mut manager, &submitted, 17, 2);
    assert_eq!(
        completion
            .retirements
            .iter()
            .map(|certificate| certificate.logical_ordinal)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(completion.publication.resident_count, 0);

    let snapshot = manager.request_snapshot(request).unwrap();
    assert_eq!(snapshot.boundary, CHUNK_TOKENS);
    assert_eq!(snapshot.roots[0].resident_tokens, 0);
    assert!(snapshot.roots[0].entries.is_empty());
    assert!(
        manager
            .materialize_request_views_batch(&[(request, completion.publication.snapshot)])
            .unwrap()[0]
            .pages
            .is_empty()
    );
}

#[test]
fn first_append_in_next_epoch_starts_fresh_with_absolute_ordinal() {
    let mut manager = chunked_manager(CHUNK_TOKENS, 5, 32, 5);
    let request = manager.acquire_request_leases_for_test(1).unwrap()[0];
    let epoch_end = append_to(&mut manager, request, CHUNK_TOKENS, 1);
    assert_eq!(epoch_end.retirements.len(), 2);
    assert!(snapshot_entries(&manager, request).is_empty());

    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).unwrap().head,
            target_boundary: CHUNK_TOKENS + 1,
        }])
        .unwrap()[0]
        .clone();
    assert_eq!(
        prepared.class_lowerings[0].flags,
        CLASS_LOWERING_RESETTABLE | CLASS_LOWERING_EPOCH_START
    );
    assert_eq!(
        prepared.class_lowerings[0].previous_layout_boundary,
        CHUNK_TOKENS
    );
    assert_eq!(
        prepared.class_lowerings[0].target_layout_boundary,
        CHUNK_TOKENS + 1
    );
    assert_eq!(prepared.tail_actions[0].kind, TailActionKind::None);
    assert!(prepared.copy_intents.is_empty());
    let reset_entry = match manager
        .operations
        .get(prepared.step.slot, prepared.step.generation)
        .unwrap()
    {
        OperationState::Prepared(state) => {
            let delta = &state.delta.classes[0];
            assert!(delta.epoch_reset);
            assert!(delta.tail_source.is_none());
            assert!(delta.tail_destination.is_none());
            assert!(delta.copy_intent.is_none());
            assert_eq!(delta.writes.len(), 1);
            delta.writes[0]
        }
        OperationState::Submitted(_) => panic!("step must still be prepared"),
    };
    assert_eq!(reset_entry.logical_ordinal, 2);
    assert_eq!(reset_entry.temporal_cell_index, 0);
    assert_eq!(reset_entry.temporal_cycle, 1);
    assert!(
        epoch_end
            .retirements
            .iter()
            .all(|certificate| certificate.page != reset_entry.page)
    );
    assert_eq!(
        manager.page(reset_entry.page.page_id).unwrap().phase,
        PagePhase::Reserved {
            step: prepared.step
        }
    );
    assert!(epoch_end.retirements.iter().all(|certificate| matches!(
        manager.page(certificate.page.page_id).unwrap().phase,
        PagePhase::Retiring { .. }
    )));

    let submitted = submit(&mut manager, &prepared);
    let completion = complete(&mut manager, &submitted, 17, 2);
    assert!(completion.retirements.is_empty());
    let snapshot = manager.request_snapshot(request).unwrap();
    assert_eq!(snapshot.boundary, CHUNK_TOKENS + 1);
    assert_eq!(snapshot.roots[0].resident_tokens, 1);
    assert_eq!(snapshot.roots[0].entries.len(), 1);
    assert_eq!(snapshot.roots[0].entries.front(), Some(&reset_entry));

    let materialized = manager
        .materialize_request_views_batch(&[(request, completion.publication.snapshot)])
        .unwrap();
    assert_eq!(materialized[0].pages.len(), 1);
    assert_eq!(materialized[0].pages[0].logical_ordinal, 2);
    assert_eq!(materialized[0].pages[0].temporal_cell_index, 0);
    assert_eq!(materialized[0].pages[0].temporal_cycle, 1);
    assert_eq!(materialized[0].pages[0].valid_token_count, 1);
    assert_eq!(materialized[0].pages[0].visible_token_count, 1);
}

#[test]
fn transaction_revalidation_rejects_a_forged_epoch_reset() {
    let mut manager = chunked_manager(CHUNK_TOKENS, 3, 32, 3);
    let request = manager.acquire_request_leases_for_test(1).unwrap()[0];
    append_to(&mut manager, request, CHUNK_TOKENS, 1);
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).unwrap().head,
            target_boundary: CHUNK_TOKENS + 1,
        }])
        .unwrap()[0]
        .clone();
    let state = match manager
        .operations
        .get(prepared.step.slot, prepared.step.generation)
        .unwrap()
    {
        OperationState::Prepared(state) => state.clone(),
        OperationState::Submitted(_) => panic!("step must still be prepared"),
    };
    let mut forged = state;
    let delta = Arc::make_mut(&mut forged.delta);
    let mut classes = delta.classes.to_vec();
    classes[0].epoch_reset = false;
    delta.classes = classes.into_boxed_slice();

    let before = state_image(&manager);
    assert_eq!(
        manager.preflight_prepared_delta(&forged, prepared.step),
        Err(KvManagerError::Invariant("delta epoch reset"))
    );
    assert_eq!(state_image(&manager), before);
}

#[test]
fn crossing_a_chunk_boundary_is_collectively_failure_atomic() {
    let mut manager = chunked_manager(CHUNK_TOKENS, 8, 32, 8);
    let requests = manager.acquire_request_leases_for_test(3).unwrap();
    append_to(&mut manager, requests[0], 31, 1);
    append_to(&mut manager, requests[1], 31, 2);

    let before = state_image(&manager);
    assert_eq!(
        manager.prepare_batch(&[PrepareBatchItem {
            request: requests[2],
            expected_head: manager.request(requests[2]).unwrap().head,
            target_boundary: 33,
        }]),
        Err(KvManagerError::ChunkBoundaryCrossed {
            previous: 0,
            target: 33,
            chunk_tokens: CHUNK_TOKENS,
        })
    );
    assert_eq!(state_image(&manager), before);

    let before = state_image(&manager);
    assert_eq!(
        manager.prepare_batch(&[PrepareBatchItem {
            request: requests[0],
            expected_head: manager.request(requests[0]).unwrap().head,
            target_boundary: 33,
        }]),
        Err(KvManagerError::ChunkBoundaryCrossed {
            previous: 31,
            target: 33,
            chunk_tokens: CHUNK_TOKENS,
        })
    );
    assert_eq!(state_image(&manager), before);

    let before = state_image(&manager);
    assert_eq!(
        manager.prepare_batch(&[
            PrepareBatchItem {
                request: requests[0],
                expected_head: manager.request(requests[0]).unwrap().head,
                target_boundary: 32,
            },
            PrepareBatchItem {
                request: requests[1],
                expected_head: manager.request(requests[1]).unwrap().head,
                target_boundary: 33,
            },
        ]),
        Err(KvManagerError::ChunkBoundaryCrossed {
            previous: 31,
            target: 33,
            chunk_tokens: CHUNK_TOKENS,
        })
    );
    assert_eq!(state_image(&manager), before);
}

#[test]
fn old_epoch_pages_are_completion_and_acknowledgement_gated() {
    let mut manager = chunked_manager(CHUNK_TOKENS, 2, 32, 2);
    let requests = manager.acquire_request_leases_for_test(2).unwrap();
    let request = requests[0];
    let competing = requests[1];
    append_to(&mut manager, request, CHUNK_TOKENS - 1, 1);
    let old_entries = snapshot_entries(&manager, request);
    let old_generations = old_entries
        .iter()
        .map(|entry| (entry.page.page_id, entry.page.generation))
        .collect::<BTreeMap<_, _>>();

    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).unwrap().head,
            target_boundary: CHUNK_TOKENS,
        }])
        .unwrap()[0]
        .clone();
    assert!(prepared.write_intents.is_empty());
    assert_eq!(prepared.tail_actions[0].kind, TailActionKind::InPlace);
    assert!(
        old_entries
            .iter()
            .all(|entry| manager.page(entry.page.page_id).unwrap().phase == PagePhase::Live)
    );
    let before = state_image(&manager);
    assert_eq!(
        manager.prepare_batch(&[PrepareBatchItem {
            request: competing,
            expected_head: manager.request(competing).unwrap().head,
            target_boundary: 1,
        }]),
        Err(KvManagerError::PageCapacityExhausted)
    );
    assert_eq!(state_image(&manager), before);
    let submitted = submit(&mut manager, &prepared);
    assert!(
        old_entries
            .iter()
            .all(|entry| manager.page(entry.page.page_id).unwrap().phase == PagePhase::Live)
    );
    let before = state_image(&manager);
    assert_eq!(
        manager.prepare_batch(&[PrepareBatchItem {
            request: competing,
            expected_head: manager.request(competing).unwrap().head,
            target_boundary: 1,
        }]),
        Err(KvManagerError::PageCapacityExhausted)
    );
    assert_eq!(state_image(&manager), before);
    let completion = complete(&mut manager, &submitted, 17, 2);
    assert_eq!(completion.retirements.len(), 2);
    assert!(snapshot_entries(&manager, request).is_empty());
    assert!(old_entries.iter().all(|entry| matches!(
        manager.page(entry.page.page_id).unwrap().phase,
        PagePhase::Retiring { .. }
    )));

    let before = state_image(&manager);
    assert_eq!(
        manager.prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).unwrap().head,
            target_boundary: CHUNK_TOKENS + 1,
        }]),
        Err(KvManagerError::PageCapacityExhausted)
    );
    assert_eq!(state_image(&manager), before);

    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&completion.retirements))
        .unwrap();
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).unwrap().head,
            target_boundary: CHUNK_TOKENS + 1,
        }])
        .unwrap();
    assert_eq!(prepared[0].write_intents.len(), 1);
    assert_eq!(prepared[0].tail_actions[0].kind, TailActionKind::None);
    let reused = prepared[0].write_intents[0];
    let old_generation = old_generations
        .get(&reused.page_id)
        .expect("acknowledged old epoch page is reused");
    assert_eq!(reused.page_generation, old_generation + 1);
}

#[test]
fn unconfirmed_epoch_end_preserves_old_epoch_and_submission() {
    let mut manager = chunked_manager(CHUNK_TOKENS, 2, 32, 2);
    let request = manager.acquire_request_leases_for_test(1).unwrap()[0];
    append_to(&mut manager, request, CHUNK_TOKENS - 1, 1);
    let old_entries = snapshot_entries(&manager, request);
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).unwrap().head,
            target_boundary: CHUNK_TOKENS,
        }])
        .unwrap()[0]
        .clone();
    let submitted = submit(&mut manager, &prepared);
    let before = state_image(&manager);
    assert_eq!(
        manager.complete_batch(
            BatchCompletionReceipt {
                engine_epoch: manager.engine_epoch,
                completion_domain: 17,
                completion_value: 2,
                confirmed: 0,
                reserved: 0,
            },
            &[submitted.submission],
        ),
        Err(KvManagerError::CompletionNotConfirmed)
    );
    assert_eq!(state_image(&manager), before);
    assert!(
        old_entries
            .iter()
            .all(|entry| manager.page(entry.page.page_id).unwrap().phase == PagePhase::Live)
    );
    assert_eq!(manager.stats().pending_reclamations, 0);

    let completion = complete(&mut manager, &submitted, 17, 2);
    assert_eq!(completion.retirements.len(), 2);
    assert!(snapshot_entries(&manager, request).is_empty());
}

#[test]
fn epoch_end_marks_previous_epoch_tokens_semantically_dead() {
    let mut manager = chunked_manager(CHUNK_TOKENS, 5, 32, 5);
    let request = manager.acquire_request_leases_for_test(1).unwrap()[0];
    let epoch_end = append_to(&mut manager, request, CHUNK_TOKENS, 1);
    let at_boundary = manager
        .token_views_batch(&[TokenViewQuery {
            request,
            expected_snapshot: epoch_end.publication.snapshot,
            class_id: 0,
        }])
        .unwrap()[0]
        .clone();
    assert!(at_boundary.placements.iter().all(|placement| {
        placement.disposition.kind == TokenDispositionKind::SemanticallyDead
            && placement.location.is_none()
    }));
    let completion = append_to(&mut manager, request, CHUNK_TOKENS + 1, 2);
    let view = manager
        .token_views_batch(&[TokenViewQuery {
            request,
            expected_snapshot: completion.publication.snapshot,
            class_id: 0,
        }])
        .unwrap()[0]
        .clone();
    assert_eq!(view.placements.len(), 33);
    assert!(view.placements[..32].iter().all(|placement| {
        placement.disposition.kind == TokenDispositionKind::SemanticallyDead
            && placement.location.is_none()
    }));
    let newest = view.placements[32];
    assert_eq!(newest.token_id, 32);
    assert!(newest.disposition.retained());
    assert!(newest.location.is_some());
}

#[test]
fn chunked_prefix_and_relocation_paths_fail_closed_without_mutation() {
    let mut manager = chunked_manager(CHUNK_TOKENS, 5, 32, 5);
    let request = manager.acquire_request_leases_for_test(1).unwrap()[0];
    let completion = append_to(&mut manager, request, CHUNK_TOKENS, 1);

    let before = state_image(&manager);
    assert!(matches!(
        manager.publish_prefix_batch(&[PrefixPublishItem {
            request,
            expected_head: completion.publication.snapshot,
            key: prefix_key(0xCC, CHUNK_TOKENS),
        }]),
        Err(KvManagerError::UnsupportedProfile(_))
    ));
    assert_eq!(state_image(&manager), before);

    let before = state_image(&manager);
    assert!(matches!(
        manager.publish_prefix_and_release_batch(&[PrefixPublishItem {
            request,
            expected_head: completion.publication.snapshot,
            key: prefix_key(0xCD, CHUNK_TOKENS),
        }]),
        Err(KvManagerError::UnsupportedProfile(_))
    ));
    assert_eq!(state_image(&manager), before);

    let before = state_image(&manager);
    assert!(matches!(
        manager.prepare_relocation_batch(&[PrepareRelocationItem {
            request,
            expected_snapshot: completion.publication.snapshot,
            class_id: 0,
            policy: RelocationPolicy::static_fragmentation(250, 8, 2, true),
        }]),
        Err(KvManagerError::UnsupportedProfile(_))
    ));
    assert_eq!(state_image(&manager), before);
}

#[test]
fn forked_chunked_requests_reset_without_premature_reclamation() {
    let mut manager = chunked_manager(CHUNK_TOKENS, 4, 32, 4);
    let requests = manager.acquire_request_leases_for_test(2).unwrap();
    let source = requests[0];
    let target = requests[1];
    append_to(&mut manager, source, CHUNK_TOKENS - 1, 1);
    let old_pages = snapshot_entries(&manager, source)
        .into_iter()
        .map(|entry| entry.page)
        .collect::<Vec<_>>();

    manager
        .fork_requests_batch(&fork_items(&manager, source, &[target]))
        .expect("chunked snapshot fork");
    assert!(
        old_pages
            .iter()
            .all(|page| manager.page(page.page_id).unwrap().request_refs == 2)
    );

    let source_epoch_end = append_to(&mut manager, source, CHUNK_TOKENS, 2);
    assert_eq!(source_epoch_end.retirements.len(), 1);
    assert!(
        source_epoch_end
            .retirements
            .iter()
            .all(|certificate| !old_pages.contains(&certificate.page))
    );
    assert!(snapshot_entries(&manager, source).is_empty());
    assert!(old_pages.iter().all(|page| {
        let state = manager.page(page.page_id).unwrap();
        state.request_refs == 1 && state.phase == PagePhase::Live
    }));

    let target_epoch_end = append_to(&mut manager, target, CHUNK_TOKENS, 3);
    assert_eq!(target_epoch_end.retirements.len(), 2);
    assert!(snapshot_entries(&manager, target).is_empty());
    assert!(old_pages.iter().all(|page| matches!(
        manager.page(page.page_id).unwrap().phase,
        PagePhase::Retiring { .. }
    )));

    let certificates = source_epoch_end
        .retirements
        .iter()
        .chain(target_epoch_end.retirements.iter())
        .cloned()
        .collect::<Vec<_>>();
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&certificates))
        .unwrap();
    let source_reset = append_to(&mut manager, source, CHUNK_TOKENS + 1, 4);
    let target_reset = append_to(&mut manager, target, CHUNK_TOKENS + 1, 5);
    assert!(source_reset.retirements.is_empty());
    assert!(target_reset.retirements.is_empty());
    assert_ne!(
        snapshot_entries(&manager, source)[0].page,
        snapshot_entries(&manager, target)[0].page
    );
}
