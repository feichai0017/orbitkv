use super::*;
use crate::kv_manager::{DetachedAction, DetachedReason};

fn hybrid_prefix_session(
    page_count: u32,
    maximum_requests: u32,
) -> (RuntimeSession, [BackendArenaRegistration; 2]) {
    let backends = [
        backend(0, 111, page_count, 50_000),
        backend(1, 112, page_count, 60_000),
    ];
    let session = session_with_config(
        &hybrid_plan(),
        &backends,
        ManagerConfig {
            maximum_requests,
            maximum_operations: 8,
            maximum_prefixes: 2,
            maximum_reclamations: page_count * 2,
            maximum_step_tokens: 64,
        },
    );
    (session, backends)
}

fn assert_hybrid_pages(pages: &[crate::kv_manager::SnapshotPage], boundary: u64) {
    let pages_per_class = usize::try_from(boundary.div_ceil(PAGE_TOKENS)).expect("page count");
    assert_eq!(pages.len(), pages_per_class * 2);
    assert_eq!(
        pages
            .iter()
            .map(|page| (page.class_id, page.logical_ordinal))
            .collect::<Vec<_>>(),
        (0_u16..2)
            .flat_map(|class_id| {
                (0..pages_per_class).map(move |ordinal| (class_id, ordinal as u64))
            })
            .collect::<Vec<_>>()
    );
    assert!(
        pages
            .iter()
            .all(|page| page.page.pool_id == 111 + u32::from(page.class_id))
    );
    assert!(
        pages
            .iter()
            .all(|page| page.backend_domain == page.class_id + 10)
    );
    assert!(pages.iter().all(|page| {
        page.valid_token_count
            == u32::try_from(
                boundary
                    .saturating_sub(page.logical_ordinal * PAGE_TOKENS)
                    .min(PAGE_TOKENS),
            )
            .expect("valid token count")
            && page.visible_token_count != 0
            && page.logical_ordinal * PAGE_TOKENS < boundary
    }));
}

#[test]
#[allow(clippy::too_many_lines)]
fn publish_attach_confirm_preserves_full_sliding_class_order() {
    let (mut session, backends) = hybrid_prefix_session(8, 2);
    let source = EngineRequestId(0);
    let target = EngineRequestId(u64::MAX);
    let key = prefix_key(60, 32);

    session.acquire_requests(&[source]).expect("acquire source");
    let (_, publication) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id: source,
            target_boundary: 32,
        }],
        61,
        1,
    );
    confirm_publication(&mut session, &publication);
    let published = session
        .publish_prefix_batch(&[(source, key)])
        .expect("publish hybrid prefix")[0];
    assert_eq!(published.resident_count, 4);
    let hint = session.lookup_prefix_batch(&[key]).expect("lookup prefix")[0];
    assert_eq!(hint.candidate, Some(published.prefix_id));

    session.acquire_requests(&[target]).expect("acquire target");
    let baseline = session.stats();
    let aborted = session
        .prepare_prefix_attach(&[(target, hint)])
        .expect("prepare attach");
    assert_eq!(session.stats(), baseline);
    session.abort_control(aborted).expect("abort attach");
    assert_eq!(session.stats(), baseline);

    let control_id = session
        .prepare_prefix_attach(&[(target, hint)])
        .expect("prepare attach retry");
    assert!(matches!(
        session.prepare_prefix_evict(&[published.prefix_id]),
        Err(RuntimeSessionError::PrefixNotReady {
            prefix_id,
            state: "attach control pending",
        }) if prefix_id == published.prefix_id
    ));
    let committed = session.commit_control(control_id).expect("commit attach");
    assert_eq!(session.commit_control(control_id), Ok(committed.clone()));
    let materialized = materialization(&committed);
    assert_eq!(materialized.requests.len(), 1);
    assert_eq!(materialized.requests[0].request_id, target);
    assert_eq!(materialized.requests[0].boundary, 32);
    assert_eq!(materialized.requests[0].resident_count, 4);
    assert_hybrid_pages(&materialized.requests[0].pages, 32);
    assert!(matches!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id: target,
            target_boundary: 33,
        }]),
        Err(RuntimeSessionError::RequestNotReady {
            request_id,
            state: "control materialization pending",
        }) if request_id == target
    ));
    assert_eq!(
        session.confirm_control(&EngineControlEvidence {
            control_id,
            mirror_updates_confirmed: false,
            reclamation_receipts: Box::new([]),
        }),
        Err(RuntimeSessionError::MirrorUpdatesNotConfirmed)
    );
    assert_eq!(
        session.confirm_control(&EngineControlEvidence {
            control_id,
            mirror_updates_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Ok(EngineControlOutcome::Materialized)
    );

    let continuation = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: target,
            target_boundary: 33,
        }])
        .expect("prepare attached append");
    let step = &continuation.steps[0];
    assert_eq!(
        step.class_lowerings
            .iter()
            .map(|lowering| lowering.class_id)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(step.tail_actions.len(), 2);
    assert!(step.copy_intents.is_empty());
    assert_eq!(step.write_intents.len(), 2);
    for lowering in &step.class_lowerings {
        assert_eq!(
            (
                lowering.tail_count,
                lowering.copy_count,
                lowering.write_count,
            ),
            (1, 0, 1)
        );
        let tail = step.tail_actions[lowering.tail_offset as usize];
        assert_eq!(tail.class_id, lowering.class_id);
        assert_eq!(tail.kind, TailActionKind::None);
        assert_eq!(tail.valid_token_count, 0);
    }
    session
        .abort_prepared_execution(
            continuation.batch_id,
            &[EngineStepAbortEvidence {
                request_id: target,
                backend_unobserved: true,
            }],
        )
        .expect("abort COW probe");
}

#[test]
#[allow(clippy::too_many_lines)]
fn atomic_prefix_cow_evict_exact_ack_reuses_full_sliding_generations_and_drains() {
    let (mut session, backends) = hybrid_prefix_session(3, 2);
    let source = EngineRequestId(71);
    let target = EngineRequestId(72);
    let sibling = EngineRequestId(74);
    let key = prefix_key(61, 16);
    session.acquire_requests(&[source]).expect("acquire source");
    let (_, initial) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id: source,
            target_boundary: 16,
        }],
        62,
        1,
    );
    confirm_publication(&mut session, &initial);

    let transfer = session
        .publish_prefix_and_release_batch(&[(source, key)])
        .expect("atomic publish-release");
    assert_eq!(transfer.items.len(), 1);
    let item = &transfer.items[0];
    assert_eq!(
        (item.request_id, item.key, item.resident_count),
        (source, key, 2)
    );
    assert_eq!(item.detached.len(), 2);
    assert_eq!(
        item.detached
            .iter()
            .map(|binding| (binding.class_id, binding.logical_ordinal))
            .collect::<Vec<_>>(),
        vec![(0, 0), (1, 0)]
    );
    assert!(item.detached.iter().all(|binding| {
        binding.action == DetachedAction::Clear
            && binding.reason == DetachedReason::PrefixTransfer
            && binding.replacement == PageLease::default()
    }));
    assert_eq!(
        session
            .lookup_prefix_batch(&[key])
            .expect("lookup transfer")[0]
            .candidate,
        Some(item.prefix_id)
    );
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: transfer.release_id,
            mirror_cleanup_confirmed: false,
            reclamation_receipts: Box::new([]),
        }),
        Err(RuntimeSessionError::MirrorCleanupNotConfirmed)
    );
    session.inject_test_fault(RuntimeSessionTestFault::ReleaseRecycleOnce);
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: transfer.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Ok(EngineReleaseOutcome::RecyclePending)
    );
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: transfer.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Err(RuntimeSessionError::ReleaseRetryNotIdOnly)
    );
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: transfer.release_id,
            mirror_cleanup_confirmed: false,
            reclamation_receipts: Box::new([]),
        }),
        Ok(EngineReleaseOutcome::Completed)
    );

    session.acquire_requests(&[target]).expect("acquire target");
    let hint = session.lookup_prefix_batch(&[key]).expect("lookup prefix")[0];
    let attach_id = session
        .prepare_prefix_attach(&[(target, hint)])
        .expect("prepare attach");
    let attached = session.commit_control(attach_id).expect("commit attach");
    assert_hybrid_pages(&materialization(&attached).requests[0].pages, 16);
    assert_eq!(
        session.confirm_control(&EngineControlEvidence {
            control_id: attach_id,
            mirror_updates_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Ok(EngineControlOutcome::Materialized)
    );

    let (_, partial_publication) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id: target,
            target_boundary: 18,
        }],
        62,
        2,
    );
    confirm_publication(&mut session, &partial_publication);
    session
        .acquire_requests(&[sibling])
        .expect("acquire sibling");
    let fork_id = session
        .prepare_request_fork(&[(target, sibling)])
        .expect("prepare shared-tail fork");
    let fork = session.commit_control(fork_id).expect("commit fork");
    assert_hybrid_pages(&materialization(&fork).requests[0].pages, 18);
    assert_eq!(
        session.confirm_control(&EngineControlEvidence {
            control_id: fork_id,
            mirror_updates_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Ok(EngineControlOutcome::Materialized)
    );

    let (cow, cow_publication) = append(
        &mut session,
        &backends,
        &[EngineAppendIntent {
            request_id: target,
            target_boundary: 19,
        }],
        62,
        3,
    );
    assert!(cow.steps[0].tail_actions.iter().all(|action| {
        action.kind == TailActionKind::CopyOnWrite && action.valid_token_count == 2
    }));
    assert_eq!(cow.steps[0].copy_intents.len(), 2);
    assert!(cow.steps[0].write_intents.is_empty());
    assert!(cow_publication.retirements.is_empty());
    assert_eq!(cow_publication.steps[0].detached.len(), 2);
    for copy in &cow.steps[0].copy_intents {
        let detached = cow_publication.steps[0]
            .detached
            .iter()
            .find(|binding| binding.class_id == copy.class_id)
            .expect("COW detach per class");
        assert_eq!(
            (detached.action, detached.reason),
            (DetachedAction::Replace, DetachedReason::CopyOnWrite)
        );
        assert_eq!(
            (detached.token_begin, detached.token_end_exclusive),
            (16, 18)
        );
        assert_eq!(
            (detached.old, detached.replacement),
            (copy.source, copy.destination)
        );
        assert_eq!(
            (
                detached.old_backend_index,
                detached.replacement_backend_index
            ),
            (copy.source_backend_index, copy.destination_backend_index)
        );
    }
    confirm_publication(&mut session, &cow_publication);
    let target_release = session
        .prepare_release_batch(&[target])
        .expect("release COW target");
    assert_eq!(
        target_release
            .retirements
            .iter()
            .map(|item| (item.class_id, item.logical_ordinal))
            .collect::<Vec<_>>(),
        vec![(0, 1), (1, 1)]
    );
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: target_release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&target_release.retirements),
        }),
        Ok(EngineReleaseOutcome::Completed)
    );
    let sibling_release = session
        .prepare_release_batch(&[sibling])
        .expect("release shared sibling");
    assert_eq!(
        sibling_release
            .retirements
            .iter()
            .map(|item| (item.class_id, item.logical_ordinal))
            .collect::<Vec<_>>(),
        vec![(0, 1), (1, 1)]
    );
    assert_eq!(
        session.confirm_release(&EngineReleaseEvidence {
            release_id: sibling_release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&sibling_release.retirements),
        }),
        Ok(EngineReleaseOutcome::Completed)
    );

    let evict_id = session
        .prepare_prefix_evict(&[item.prefix_id])
        .expect("prepare eviction");
    let evict_plan = session.commit_control(evict_id).expect("commit eviction");
    let retirements = eviction(&evict_plan).retirements.clone();
    assert_eq!(retirements.len(), 2);
    assert_eq!(
        retirements
            .iter()
            .map(|item| item.class_id)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        session.lookup_prefix_batch(&[key]).expect("evicted lookup")[0].candidate,
        None
    );
    let replacement = EngineRequestId(73);
    session
        .acquire_requests(&[replacement])
        .expect("replacement request");
    let before_bad_ack = session.stats();
    assert_eq!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id: replacement,
            target_boundary: 48,
        }]),
        Err(RuntimeSessionError::Manager(
            KvManagerError::PageCapacityExhausted
        ))
    );
    let mut forged = control_reclamation_evidence(&retirements);
    forged[1].backend_index += 1;
    assert_eq!(
        session.confirm_control(&EngineControlEvidence {
            control_id: evict_id,
            mirror_updates_confirmed: true,
            reclamation_receipts: forged,
        }),
        Err(RuntimeSessionError::ReclamationReceiptMismatch)
    );
    assert_eq!(session.stats(), before_bad_ack);
    assert_eq!(
        session.confirm_control(&EngineControlEvidence {
            control_id: evict_id,
            mirror_updates_confirmed: true,
            reclamation_receipts: control_reclamation_evidence(&retirements),
        }),
        Ok(EngineControlOutcome::Evicted)
    );
    assert_eq!(
        session.prepare_prefix_evict(&[item.prefix_id]),
        Err(RuntimeSessionError::StalePrefix(item.prefix_id))
    );

    let reused = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: replacement,
            target_boundary: 48,
        }])
        .expect("reuse evicted pages");
    let all_retirements = target_release
        .retirements
        .iter()
        .chain(sibling_release.retirements.iter())
        .chain(retirements.iter())
        .collect::<Vec<_>>();
    for lowering in &reused.steps[0].class_lowerings {
        let writes = &reused.steps[0].write_intents[lowering.write_offset as usize
            ..(lowering.write_offset + lowering.write_count) as usize];
        let retired = all_retirements
            .iter()
            .copied()
            .filter(|item| item.class_id == lowering.class_id)
            .collect::<Vec<_>>();
        assert_eq!(writes.len(), retired.len());
        for write in writes {
            let old = retired
                .iter()
                .find(|item| item.page.page_id == write.page_id)
                .expect("reused retired page");
            assert_eq!(write.page_generation, old.page.generation + 1);
        }
    }
    let evidence = execution_evidence(&session, &reused, &backends);
    let ticket = session.submit_execution(&evidence).expect("submit reuse");
    let publication = session
        .complete_execution_by_batch(
            ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: 62,
                completion_value: 4,
                confirmed: true,
            },
        )
        .expect("complete reuse");
    confirm_publication(&mut session, &publication);
    release_ready_request(&mut session, replacement);

    let stats = session.stats();
    assert_eq!(stats.free_pages, 6);
    assert_eq!(stats.active_requests, 0);
    assert_eq!(stats.active_snapshots, 0);
    assert_eq!(stats.active_prefixes, 0);
    assert_eq!(stats.evicted_prefixes, 0);
    assert_eq!(stats.prepared_steps, 0);
    assert_eq!(stats.submitted_steps, 0);
    assert_eq!(stats.reserved_pages, 0);
    assert_eq!(stats.writing_pages, 0);
    assert_eq!(stats.active_pages, 0);
    assert_eq!(stats.retiring_pages, 0);
    assert_eq!(stats.pending_reclamations, 0);
    assert_eq!(stats.total_request_page_refs, 0);
    assert_eq!(stats.total_prefix_page_refs, 0);
    assert_eq!(stats.total_reader_pins, 0);
    for arena in session.arena_stats() {
        assert_eq!(arena.free_pages, u64::from(arena.page_count));
        assert_eq!(
            (
                arena.reserved_pages,
                arena.writing_pages,
                arena.active_pages
            ),
            (0, 0, 0)
        );
        assert_eq!(
            (
                arena.retiring_pages,
                arena.quarantined_pages,
                arena.exhausted_pages
            ),
            (0, 0, 0)
        );
        assert_eq!(
            (
                arena.request_page_refs,
                arena.prefix_page_refs,
                arena.reader_pins
            ),
            (0, 0, 0)
        );
    }
}
