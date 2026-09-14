use super::*;

#[test]
#[allow(clippy::too_many_lines)]
fn randomized_prefix_request_reference_transactions_match_full_scan_oracle() {
    let plan = full_plan(CANONICAL_PAGE_TOKENS);
    let mut manager = CanonicalKvManager::new(
        &plan,
        ManagerConfig {
            maximum_requests: 5,
            maximum_operations: 4,
            maximum_prefixes: 4,
            maximum_reclamations: 64,
            maximum_step_tokens: 64,
        },
        &[backend(0, 234, 64, 180_000)],
    )
    .expect("random prefix manager");
    let mut seed = 0xcafe_f00d_1234_5678_u64;

    for cycle in 0..48_u64 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let attach_count = usize::try_from((seed >> 17) % 3 + 1).expect("attach count");
        let acquired = manager
            .acquire_requests_batch(attach_count + 1)
            .expect("random lifecycle acquire");
        let all_requests = acquired.iter().map(|view| view.request).collect::<Vec<_>>();
        let source = all_requests[0];
        let targets = &all_requests[1..];
        assert_reference_census_matches_full_scan(&manager);

        let boundary = ((seed >> 29) % 3 + 1) * CANONICAL_PAGE_TOKENS;
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request: source,
                expected_head: acquired[0].snapshot,
                target_boundary: boundary,
            }])
            .expect("random source prepare");
        let submitted = submit(&mut manager, &prepared[0]);
        complete(&mut manager, &submitted, 41, cycle + 1);
        assert_reference_census_matches_full_scan(&manager);
        assert_incremental_census_matches_full_scan(&manager);

        let key = prefix_key(u8::try_from(cycle + 1).expect("unique key tag"), boundary);
        let transfer_source = seed & 1 != 0;
        let prefix = if transfer_source {
            manager
                .publish_prefix_and_release_batch(&[PrefixPublishItem {
                    request: source,
                    expected_head: manager.request(source).expect("source head").head,
                    key,
                }])
                .expect("random publish release")[0]
                .publication
                .prefix
        } else {
            manager
                .publish_prefix_batch(&[PrefixPublishItem {
                    request: source,
                    expected_head: manager.request(source).expect("source head").head,
                    key,
                }])
                .expect("random publish")[0]
                .prefix
        };
        assert_reference_census_matches_full_scan(&manager);
        assert_incremental_census_matches_full_scan(&manager);

        let lookup_keys = vec![key; attach_count];
        let hints = manager
            .lookup_prefix_batch(&lookup_keys)
            .expect("random prefix lookup");
        let items = attach_items(&manager, targets, &hints);
        let attached = manager
            .attach_prefix_batch(&items)
            .expect("random prefix attach");
        assert_eq!(attached.len(), attach_count);
        assert_reference_census_matches_full_scan(&manager);
        assert_incremental_census_matches_full_scan(&manager);

        if seed & 2 != 0 {
            let extension = (seed >> 37) % (CANONICAL_PAGE_TOKENS - 1) + 1;
            let target = targets[0];
            let prepared = manager
                .prepare_batch(&[PrepareBatchItem {
                    request: target,
                    expected_head: manager.request(target).expect("target head").head,
                    target_boundary: boundary + extension,
                }])
                .expect("extend one attached request");
            let submitted = submit(&mut manager, &prepared[0]);
            complete(&mut manager, &submitted, 42, cycle + 1);
            assert_reference_census_matches_full_scan(&manager);
            assert_incremental_census_matches_full_scan(&manager);
        }

        let evict_first = seed & 4 != 0;
        if evict_first {
            let eviction = manager
                .evict_prefix_batch(&[prefix])
                .expect("early random eviction");
            if !eviction.retirements.is_empty() {
                manager
                    .acknowledge_reclamations_batch(&reclamation_receipts(&eviction.retirements))
                    .expect("early eviction acknowledgement");
            }
            assert_reference_census_matches_full_scan(&manager);
            assert_incremental_census_matches_full_scan(&manager);
        }

        let mut live_requests = targets.to_vec();
        if !transfer_source {
            live_requests.push(source);
        }
        if seed & 8 != 0 {
            live_requests.reverse();
        } else if live_requests.len() > 1 {
            let rotate = usize::try_from((seed >> 43) % live_requests.len() as u64)
                .expect("release rotation");
            live_requests.rotate_left(rotate);
        }
        let mut cursor = 0_usize;
        while cursor < live_requests.len() {
            let width =
                if seed.rotate_left(u32::try_from(cursor).expect("release cursor")) & 1 == 0 {
                    1
                } else {
                    2
                }
                .min(live_requests.len() - cursor);
            let release = manager
                .release_current_for_test(&live_requests[cursor..cursor + width])
                .expect("random grouped release");
            if !release.retirements.is_empty() {
                manager
                    .acknowledge_reclamations_batch(&reclamation_receipts(&release.retirements))
                    .expect("random release acknowledgement");
            }
            cursor += width;
            assert_reference_census_matches_full_scan(&manager);
            assert_incremental_census_matches_full_scan(&manager);
        }

        if !evict_first {
            let eviction = manager
                .evict_prefix_batch(&[prefix])
                .expect("late random eviction");
            if !eviction.retirements.is_empty() {
                manager
                    .acknowledge_reclamations_batch(&reclamation_receipts(&eviction.retirements))
                    .expect("late eviction acknowledgement");
            }
            assert_reference_census_matches_full_scan(&manager);
            assert_incremental_census_matches_full_scan(&manager);
        }
        manager
            .recycle_prefixes_batch(&[prefix])
            .expect("random prefix recycle");
        assert!(matches!(
            manager.prefixes.get(prefix.slot, prefix.generation),
            Err(KvManagerError::StaleLease("prefix"))
        ));
        manager
            .recycle_requests_batch(&all_requests)
            .expect("random request recycle");
        assert_reference_census_matches_full_scan(&manager);
        assert_incremental_census_matches_full_scan(&manager);
        assert_eq!(manager.requests.active_len(), 0);
        assert_eq!(manager.snapshots.active_len(), 0);
        assert_eq!(manager.prefixes.active_len(), 0);
        assert_eq!(manager.stats().free_pages, 64);
        assert_eq!(manager.stats().pending_reclamations, 0);
    }
}

#[test]
fn hybrid_large_step_retires_only_the_exact_prefix() {
    const PAGE_COUNT: u32 = 4_096;
    const TARGET: u64 = PAGE_COUNT as u64 * CANONICAL_PAGE_TOKENS;
    let plan = hybrid_plan(18);
    let mut manager = manager_for_plan(
        &plan,
        &[
            backend(0, 94, PAGE_COUNT, 40_000),
            backend(1, 95, PAGE_COUNT, 50_000),
        ],
        u32::try_from(TARGET).expect("maximum step"),
        PAGE_COUNT * 2,
    );
    let request = manager.acquire_request_leases_for_test(1).expect("request")[0];
    let initial = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 18,
        }])
        .expect("initial prepare");
    let submitted = submit(&mut manager, &initial[0]);
    complete(&mut manager, &submitted, 1, 1);

    manager.hot_path = HotPathInstrumentation::default();
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: TARGET,
        }])
        .expect("large hybrid prepare");
    assert_eq!(
        prepared[0].write_intents.len(),
        (usize::try_from(PAGE_COUNT).expect("page count") - 2) * 2
    );
    let submitted = submit(&mut manager, &prepared[0]);
    manager.hot_path = HotPathInstrumentation::default();
    let completion = complete(&mut manager, &submitted, 2, 2);
    assert_eq!(completion.publication.resident_count, PAGE_COUNT + 2);
    assert_eq!(completion.retirements.len(), PAGE_COUNT as usize - 2);
    assert!(completion.retirements.iter().all(|item| item.class_id == 1));
    assert_eq!(
        completion
            .retirements
            .first()
            .expect("first")
            .logical_ordinal,
        0
    );
    assert_eq!(
        completion.retirements.last().expect("last").logical_ordinal,
        u64::from(PAGE_COUNT) - 3
    );
    assert_eq!(manager.hot_path.hot_root_entries_visited, 2);
    assert_eq!(
        manager.hot_path.retirement_entries_touched,
        u64::from(PAGE_COUNT - 2)
    );
    assert_eq!(manager.hot_path.device_view_entries_materialized, 0);
    assert_eq!(manager.hot_path.snapshot_entries_cloned, 0);
}

#[test]
fn hybrid_lifecycle_property_holds_across_irregular_boundaries() {
    let plan = hybrid_plan(18);
    let mut manager = manager_for_plan(
        &plan,
        &[backend(0, 61, 8, 10_000), backend(1, 62, 4, 20_000)],
        64,
        12,
    );
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    let mut full_pages = BTreeMap::new();
    for boundary in [
        1_u64, 2, 15, 16, 17, 18, 31, 32, 33, 35, 48, 49, 64, 79, 80, 81, 95,
    ] {
        let published_before = snapshot_entries(&manager, request);
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                expected_head: manager.request(request).expect("expected head").head,
                target_boundary: boundary,
            }])
            .map(|items| items[0].clone())
            .expect("prepare property step");
        assert_eq!(snapshot_entries(&manager, request), published_before);
        let submitted = submit(&mut manager, &prepared);
        assert_eq!(snapshot_entries(&manager, request), published_before);
        let completion = complete(&mut manager, &submitted, 9, boundary);
        if !completion.retirements.is_empty() {
            manager
                .acknowledge_reclamations_batch(&reclamation_receipts(&completion.retirements))
                .expect("ack property retirements");
        }

        let entries = published_entries(&manager, request);
        let full = entries
            .iter()
            .filter(|entry| entry.class_id == 0)
            .collect::<Vec<_>>();
        let sliding = entries
            .iter()
            .filter(|entry| entry.class_id == 1)
            .collect::<Vec<_>>();
        assert_eq!(full.len() as u64, boundary.div_ceil(16));
        let retain_start = boundary.saturating_sub(17);
        let last = (boundary - 1) / 16;
        let expected_sliding = (0..=last)
            .filter(|ordinal| {
                let token_begin = ordinal * 16;
                let token_end = boundary.min(token_begin + 16);
                token_end > retain_start || (!boundary.is_multiple_of(16) && *ordinal == last)
            })
            .count();
        assert_eq!(sliding.len(), expected_sliding);
        for entry in full {
            assert_eq!(entry.pool_id, 61);
            assert_eq!(entry.temporal_cell_index, entry.logical_ordinal);
            assert_eq!(entry.temporal_cycle, 0);
            if let Some(page_id) = full_pages.insert(entry.logical_ordinal, entry.page_id) {
                assert_eq!(entry.page_id, page_id);
            }
        }
        for entry in sliding {
            assert_eq!(entry.pool_id, 62);
            assert_eq!(entry.temporal_cell_index, entry.logical_ordinal % 3);
            assert_eq!(entry.temporal_cycle, entry.logical_ordinal / 3);
        }
        let unique_pages = entries
            .iter()
            .map(|entry| entry.page_id)
            .collect::<BTreeSet<_>>();
        assert_eq!(unique_pages.len(), entries.len());
        assert_eq!(manager.stats().active_pages, entries.len() as u64);
        assert_eq!(manager.stats().retiring_pages, 0);
        assert_eq!(manager.stats().pending_reclamations, 0);
    }
    let release = manager
        .release_current_for_test(&[request])
        .expect("release property request");
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&release.retirements))
        .expect("ack property release");
    manager
        .recycle_requests_batch(&[request])
        .expect("recycle property request");
    assert_eq!(manager.stats().active_requests, 0);
    assert_eq!(manager.stats().free_pages, 12);
}

#[test]
#[allow(clippy::too_many_lines)]
fn randomized_hybrid_snapshot_delta_matches_reference_model() {
    let plan = hybrid_plan(18);
    let mut manager = manager_for_plan(
        &plan,
        &[backend(0, 63, 300, 30_000), backend(1, 64, 32, 40_000)],
        64,
        332,
    );
    let request = manager.acquire_request_leases_for_test(1).expect("request")[0];
    let mut reference = BTreeMap::<(u16, u64), (u32, u64)>::new();
    let mut boundary = 0_u64;
    let mut seed = 0x5eed_cafe_f00d_beef_u64;

    for step_index in 0..128_u64 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let increment = ((seed >> 32) % 64) + 1;
        let target = boundary + increment;
        let prepared = manager
            .prepare_batch(&[PrepareBatchItem {
                request,
                expected_head: manager.request(request).expect("expected head").head,
                target_boundary: target,
            }])
            .expect("random prepare");
        let expected_new = usize::try_from(
            target.div_ceil(CANONICAL_PAGE_TOKENS) - boundary.div_ceil(CANONICAL_PAGE_TOKENS),
        )
        .expect("new pages");
        assert!(
            prepared[0]
                .class_lowerings
                .iter()
                .all(|lowering| lowering.write_count as usize == expected_new)
        );
        assert_eq!(prepared[0].write_intents.len(), expected_new * 2);
        assert!(prepared[0].class_lowerings.iter().all(|lowering| {
            lowering.flags == 0
                && lowering.tail_count == 1
                && prepared[0].tail_actions[lowering.tail_offset as usize].kind
                    == if boundary.is_multiple_of(CANONICAL_PAGE_TOKENS) {
                        TailActionKind::None
                    } else {
                        TailActionKind::InPlace
                    }
        }));

        let candidate =
            operation_entries(&manager, prepared[0].step.slot, prepared[0].step.generation);
        let mut candidate_model = BTreeMap::new();
        for entry in &candidate {
            let key = (entry.class_id, entry.logical_ordinal);
            let identity = (entry.page_id, entry.page_generation);
            assert!(candidate_model.insert(key, identity).is_none());
            if let Some(previous) = reference.get(&key) {
                assert_eq!(
                    *previous, identity,
                    "stable page identity at step {step_index}"
                );
            }
        }
        for class_id in [0_u16, 1] {
            let first = if class_id == 0 {
                0
            } else {
                boundary.saturating_sub(17) / CANONICAL_PAGE_TOKENS
            };
            let end = target.div_ceil(CANONICAL_PAGE_TOKENS);
            let ordinals = candidate_model
                .keys()
                .filter(|(candidate_class, _)| *candidate_class == class_id)
                .map(|(_, ordinal)| *ordinal)
                .collect::<Vec<_>>();
            assert_eq!(ordinals, (first..end).collect::<Vec<_>>());
        }

        let submitted = submit(&mut manager, &prepared[0]);
        let completion = complete(&mut manager, &submitted, 9, step_index + 1);
        let sliding_retain_first = target.saturating_sub(17) / CANONICAL_PAGE_TOKENS;
        let expected_retirements = candidate_model
            .iter()
            .filter(|((class_id, ordinal), _)| *class_id == 1 && *ordinal < sliding_retain_first)
            .map(|(&(class_id, ordinal), &(page_id, generation))| {
                (class_id, ordinal, page_id, generation)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            completion
                .retirements
                .iter()
                .map(|item| {
                    (
                        item.class_id,
                        item.logical_ordinal,
                        item.page.page_id,
                        item.page.generation,
                    )
                })
                .collect::<Vec<_>>(),
            expected_retirements
        );
        reference = candidate_model
            .into_iter()
            .filter(|((class_id, ordinal), _)| *class_id == 0 || *ordinal >= sliding_retain_first)
            .collect();
        assert_eq!(
            published_entries(&manager, request)
                .iter()
                .map(|entry| {
                    (
                        (entry.class_id, entry.logical_ordinal),
                        (entry.page_id, entry.page_generation),
                    )
                })
                .collect::<BTreeMap<_, _>>(),
            reference
        );
        if !completion.retirements.is_empty() {
            manager
                .acknowledge_reclamations_batch(&reclamation_receipts(&completion.retirements))
                .expect("random retirement acknowledgement");
        }
        assert_incremental_census_matches_full_scan(&manager);
        assert_reference_census_matches_full_scan(&manager);
        boundary = target;
    }
}

#[test]
fn window_one_exact_boundaries_can_publish_an_empty_snapshot() {
    let mut manager = manager_with(1, 4, 32, 4);
    let request = manager.acquire_request_leases_for_test(1).expect("request")[0];
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 16,
        }])
        .expect("aligned prepare");
    let submitted = submit(&mut manager, &prepared[0]);
    let completion = complete(&mut manager, &submitted, 1, 1);
    assert_eq!(completion.publication.resident_count, 0);
    assert_eq!(completion.retirements.len(), 1);
    manager
        .acknowledge_reclamations_batch(&reclamation_receipts(&completion.retirements))
        .expect("aligned retirement");
    assert!(
        manager
            .request_snapshot(request)
            .expect("request snapshot")
            .is_empty()
    );

    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 17,
        }])
        .expect("post-empty prepare");
    assert_eq!(prepared[0].write_intents.len(), 1);
    let lowering = prepared[0].class_lowerings[0];
    assert_eq!(lowering.flags, 0);
    assert_eq!(lowering.tail_count, 1);
    assert_eq!(
        prepared[0].tail_actions[lowering.tail_offset as usize].kind,
        TailActionKind::None
    );
    let submitted = submit(&mut manager, &prepared[0]);
    let completion = complete(&mut manager, &submitted, 1, 2);
    assert_eq!(completion.publication.resident_count, 1);
    assert!(completion.retirements.is_empty());
    assert_eq!(published_entries(&manager, request)[0].logical_ordinal, 1);
}

#[test]
fn w18_spans_are_exact_across_wrap() {
    let mut manager = manager_with(18, 5, 64, 8);
    let request = manager
        .acquire_request_leases_for_test(1)
        .map(|requests| requests[0])
        .expect("request");
    assert!(
        complete_initial_18(&mut manager, request)
            .retirements
            .is_empty()
    );

    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 35,
        }])
        .map(|items| items[0].clone())
        .expect("prepare wrap");
    let entries = operation_entries(&manager, prepared.step.slot, prepared.step.generation);
    assert_eq!(entries.len(), 3);
    assert_eq!(
        entries
            .iter()
            .map(|entry| (
                entry.logical_ordinal,
                entry.visible_token_offset,
                entry.visible_token_count,
                entry.access_flags,
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, 1, 15, DEVICE_KV_ACCESS_READ),
            (1, 0, 16, DEVICE_KV_ACCESS_READ | DEVICE_KV_ACCESS_WRITE),
            (
                2,
                0,
                3,
                DEVICE_KV_ACCESS_READ | DEVICE_KV_ACCESS_WRITE | DEVICE_KV_NEEDS_BINDING,
            ),
        ]
    );
    let submitted = submit(&mut manager, &prepared);
    let completion = complete(&mut manager, &submitted, 11, 22);
    assert_eq!(completion.retirements.len(), 1);
    assert_eq!(completion.retirements[0].logical_ordinal, 0);
    assert_eq!(completion.retirements[0].completion_domain, 11);
    assert_eq!(completion.retirements[0].completion_value, 22);
    assert_eq!(
        published_entries(&manager, request)
            .iter()
            .map(|entry| (
                entry.logical_ordinal,
                entry.visible_token_offset,
                entry.visible_token_count,
            ))
            .collect::<Vec<_>>(),
        vec![(1, 2, 14), (2, 0, 3)]
    );
}
