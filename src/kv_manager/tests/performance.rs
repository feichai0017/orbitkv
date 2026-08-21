use super::*;

#[test]
fn prefix_append_path_copies_only_persistent_tree_spine() {
    const ROOT_PAGES: u32 = 1_024;
    let plan = full_plan(CANONICAL_PAGE_TOKENS);
    let mut manager = manager_for_plan(
        &plan,
        &[backend(0, 231, ROOT_PAGES + 2, 150_000)],
        (ROOT_PAGES + 1) * 16,
        ROOT_PAGES + 2,
    );
    let source = manager.acquire_request_leases_for_test(1).expect("source")[0];
    let boundary = u64::from(ROOT_PAGES) * CANONICAL_PAGE_TOKENS;
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request: source,
            expected_head: manager.request(source).expect("expected head").head,
            target_boundary: boundary,
        }])
        .expect("long source prepare");
    let submitted = submit(&mut manager, &prepared[0]);
    complete(&mut manager, &submitted, 31, 1);
    let key = prefix_key(31, boundary);
    manager
        .publish_prefix_batch(&[PrefixPublishItem {
            request: source,
            expected_head: manager.request(source).expect("expected head").head,
            key,
        }])
        .expect("publish long prefix");
    let attached = manager
        .acquire_request_leases_for_test(1)
        .expect("attached request")[0];
    let lookup_instrumentation_before = root_instrumentation();
    let hints = manager
        .lookup_prefix_batch(&[key; 8])
        .expect("large-prefix B8 lookup");
    assert_eq!(root_instrumentation(), lookup_instrumentation_before);
    assert!(hints.iter().all(|hint| hint.resident_count == ROOT_PAGES));
    let hint = hints[0];
    manager
        .attach_prefix_batch(&[PrefixAttachItem {
            request: attached,
            expected_empty_head: manager.request(attached).expect("expected empty head").head,
            hint,
        }])
        .expect("attach long prefix");

    let old_root = manager
        .request_snapshot(source)
        .expect("source snapshot")
        .roots[0]
        .entries
        .root
        .clone();
    let mut old_addresses = BTreeSet::new();
    root_node_addresses(old_root.as_ref(), &mut old_addresses);
    assert_eq!(old_addresses.len(), ROOT_PAGES as usize);

    manager.hot_path = HotPathInstrumentation::default();
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request: attached,
            expected_head: manager.request(attached).expect("expected head").head,
            target_boundary: boundary + CANONICAL_PAGE_TOKENS,
        }])
        .expect("persistent append prepare");
    let submitted = submit(&mut manager, &prepared[0]);
    complete(&mut manager, &submitted, 31, 2);
    let new_root = manager
        .request_snapshot(attached)
        .expect("extended snapshot")
        .roots[0]
        .entries
        .root
        .clone();
    let mut new_addresses = BTreeSet::new();
    root_node_addresses(new_root.as_ref(), &mut new_addresses);
    let shared_nodes = old_addresses.intersection(&new_addresses).count();
    assert!(
        shared_nodes >= ROOT_PAGES as usize - 64,
        "append copied more than one logarithmic AVL spine: {shared_nodes} shared"
    );
    assert_eq!(
        manager
            .request_snapshot(source)
            .expect("unchanged source")
            .resident_count(),
        ROOT_PAGES as usize
    );
    assert_eq!(
        manager
            .request_snapshot(attached)
            .expect("extended request")
            .resident_count(),
        ROOT_PAGES as usize + 1
    );
    assert_eq!(manager.hot_path.snapshot_entries_cloned, 0);
    assert_eq!(manager.hot_path.hot_root_entries_visited, 0);
    assert_reference_census_matches_full_scan(&manager);
}

#[test]
fn large_sliding_completion_retirement_preflight_is_linear_and_exact() {
    const PAGE_COUNT: u32 = 4_096;
    const TARGET: u64 = 65_536;

    let mut manager = manager_with(
        18,
        PAGE_COUNT,
        u32::try_from(TARGET).expect("target fits maximum step"),
        PAGE_COUNT,
    );
    let request = manager
        .acquire_request_leases_for_test(1)
        .expect("request batch")[0];
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: TARGET,
        }])
        .expect("large sliding prepare");
    assert_eq!(prepared[0].write_intents.len(), PAGE_COUNT as usize);
    let (items, receipts, copies) = batch_submit_items(&manager, &prepared);
    let submitted = manager
        .submit_batch(&items, &receipts, &copies)
        .expect("large sliding submit");
    let completed = manager
        .complete_batch(
            BatchCompletionReceipt {
                engine_epoch: manager.engine_epoch,
                completion_domain: 1,
                completion_value: 1,
                confirmed: 1,
                reserved: 0,
            },
            &[submitted[0].submission],
        )
        .expect("large sliding completion");
    assert_eq!(completed.completions[0].publication.resident_count, 2);
    assert_eq!(completed.retirements.len(), PAGE_COUNT as usize - 2);
    assert_eq!(
        completed.completions[0].detached.len(),
        PAGE_COUNT as usize - 2
    );
    assert!(completed.completions[0].detached.iter().all(|binding| {
        binding.action == DetachedAction::Clear
            && binding.reason == DetachedReason::Retention
            && binding.replacement == PageLease::default()
    }));
    assert_eq!(
        completed.completions[0]
            .detached
            .iter()
            .map(|binding| binding.old)
            .collect::<BTreeSet<_>>(),
        completed
            .retirements
            .iter()
            .map(|certificate| certificate.page)
            .collect::<BTreeSet<_>>()
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn full_8192_steady_hot_path_is_snapshot_delta_only() {
    const CONTEXT_PAGES: u32 = 8_192;
    const CONTEXT_TOKENS: u64 = CONTEXT_PAGES as u64 * CANONICAL_PAGE_TOKENS;
    let plan = full_plan(CANONICAL_PAGE_TOKENS);
    let mut manager = manager_for_plan(
        &plan,
        &[backend(0, 93, CONTEXT_PAGES + 2, 30_000)],
        u32::try_from(CONTEXT_TOKENS + 2).expect("maximum step"),
        CONTEXT_PAGES + 2,
    );
    let request = manager.acquire_request_leases_for_test(1).expect("request")[0];
    let initial = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: CONTEXT_TOKENS,
        }])
        .expect("initial prepare");
    let submitted = submit(&mut manager, &initial[0]);
    complete(&mut manager, &submitted, 1, 1);
    assert_eq!(
        manager
            .request_snapshot(request)
            .expect("request snapshot")
            .resident_count(),
        CONTEXT_PAGES as usize
    );
    let lookup_key = prefix_key(0x7f, CONTEXT_TOKENS);
    manager
        .publish_prefix_batch(&[PrefixPublishItem {
            request,
            expected_head: manager.request(request).expect("lookup source head").head,
            key: lookup_key,
        }])
        .expect("publish 8192-page lookup fixture");
    let lookup_instrumentation_before = root_instrumentation();
    let hints = manager
        .lookup_prefix_batch(&[lookup_key; 4])
        .expect("8192-page B4 lookup");
    assert_eq!(root_instrumentation(), lookup_instrumentation_before);
    assert!(
        hints
            .iter()
            .all(|hint| hint.resident_count == CONTEXT_PAGES)
    );

    manager.hot_path = HotPathInstrumentation::default();
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: CONTEXT_TOKENS + 1,
        }])
        .expect("steady prepare");
    assert_eq!(prepared[0].write_intents.len(), 1);
    assert_eq!(
        manager.hot_path,
        HotPathInstrumentation {
            delta_entries_touched: 1,
            ..HotPathInstrumentation::default()
        }
    );

    manager.hot_path = HotPathInstrumentation::default();
    let submitted = submit(&mut manager, &prepared[0]);
    assert_eq!(
        manager.hot_path,
        HotPathInstrumentation {
            delta_entries_touched: 1,
            ..HotPathInstrumentation::default()
        }
    );

    manager.hot_path = HotPathInstrumentation::default();
    let completion = complete(&mut manager, &submitted, 1, 2);
    assert!(completion.retirements.is_empty());
    assert_eq!(completion.publication.resident_count, CONTEXT_PAGES + 1);
    assert_eq!(manager.hot_path.root_node_visits, 0);
    assert_eq!(manager.hot_path.root_iterator_allocs, 0);
    assert!(manager.hot_path.path_nodes_cloned > 0);
    assert!(manager.hot_path.path_nodes_cloned <= 32);
    assert_eq!(manager.hot_path.delta_entries_touched, 1);
    assert_eq!(manager.hot_path.retirement_entries_touched, 0);
    assert_eq!(manager.hot_path.device_view_entries_materialized, 0);
    assert_eq!(manager.hot_path.snapshot_entries_cloned, 0);

    manager.hot_path = HotPathInstrumentation::default();
    let tail = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: CONTEXT_TOKENS + 2,
        }])
        .expect("tail prepare");
    assert!(tail[0].write_intents.is_empty());
    let lowering = tail[0].class_lowerings[0];
    assert_eq!(lowering.flags, 0);
    assert_eq!(lowering.tail_count, 1);
    assert_eq!(lowering.copy_count, 0);
    assert_eq!(
        tail[0].tail_actions[lowering.tail_offset as usize].kind,
        TailActionKind::InPlace
    );
    assert_eq!(manager.hot_path, HotPathInstrumentation::default());
    let submitted = submit(&mut manager, &tail[0]);
    complete(&mut manager, &submitted, 1, 3);
    assert_eq!(manager.hot_path.hot_root_entries_visited, 0);
    assert_eq!(manager.hot_path.root_node_visits, 0);
    assert_eq!(manager.hot_path.root_iterator_allocs, 0);
    assert_eq!(manager.hot_path.path_nodes_cloned, 0);
    assert_eq!(manager.hot_path.device_view_entries_materialized, 0);
    assert_eq!(manager.hot_path.snapshot_entries_cloned, 0);
}

#[test]
#[ignore = "CPU microbenchmark; run explicitly with --ignored --nocapture"]
#[allow(clippy::too_many_lines)]
fn cpu_microbench_large_pool_b1_b4_control_paths() {
    const PAGE_COUNT: u32 = 65_536;
    const LIFECYCLE_ITERATIONS: u128 = 2_000;
    const CENSUS_ITERATIONS: u128 = 10_000;
    const STEADY_ITERATIONS: u64 = 64;
    for batch_size in [1_usize, 4] {
        let plan = full_plan(CANONICAL_PAGE_TOKENS);
        let mut manager =
            manager_for_plan(&plan, &[backend(0, 91, PAGE_COUNT, 10_000)], 16, PAGE_COUNT);
        let requests = manager
            .acquire_request_leases_for_test(batch_size)
            .expect("benchmark request batch");
        let prepare_items = requests
            .iter()
            .copied()
            .map(|request| PrepareBatchItem {
                request,
                expected_head: manager.request(request).expect("expected head").head,
                target_boundary: 1,
            })
            .collect::<Vec<_>>();

        let lifecycle_start = Instant::now();
        for _ in 0..LIFECYCLE_ITERATIONS {
            let prepared = manager
                .prepare_batch(&prepare_items)
                .expect("benchmark prepare");
            let aborts = prepared
                .iter()
                .map(|item| BackendUnobservedReceipt {
                    step: item.step,
                    backend_unobserved: 1,
                    reserved: 0,
                })
                .collect::<Vec<_>>();
            manager.abort_steps_batch(&aborts).expect("benchmark abort");
        }
        let lifecycle_elapsed = lifecycle_start.elapsed();

        let census_start = Instant::now();
        for _ in 0..CENSUS_ITERATIONS {
            std::hint::black_box(manager.stats());
            std::hint::black_box(manager.arena_stats());
        }
        let census_elapsed = census_start.elapsed();
        assert_incremental_census_matches_full_scan(&manager);
        eprintln!(
            "orbitkv_cpu_microbench pool_pages={PAGE_COUNT} batch={batch_size} lifecycle_ns_per_iter={} census_ns_per_iter={}",
            lifecycle_elapsed.as_nanos() / LIFECYCLE_ITERATIONS,
            census_elapsed.as_nanos() / CENSUS_ITERATIONS,
        );
    }

    for context_pages in [512_u32, 8_192] {
        for batch_size in [1_usize, 4] {
            let batch_u32 = u32::try_from(batch_size).expect("benchmark batch size");
            let extra_pages = u32::try_from(STEADY_ITERATIONS)
                .expect("steady iterations")
                .div_ceil(u32::try_from(CANONICAL_PAGE_TOKENS).expect("page tokens"));
            let page_count = context_pages
                .checked_add(extra_pages)
                .and_then(|pages| pages.checked_mul(batch_u32))
                .expect("benchmark page capacity");
            let context_tokens = u64::from(context_pages) * CANONICAL_PAGE_TOKENS;
            let maximum_step_tokens = u32::try_from(context_tokens).expect("benchmark step tokens");
            let plan = full_plan(CANONICAL_PAGE_TOKENS);
            let mut manager = manager_for_plan(
                &plan,
                &[backend(0, 92, page_count, 20_000)],
                maximum_step_tokens,
                page_count,
            );
            let requests = manager
                .acquire_request_leases_for_test(batch_size)
                .expect("steady benchmark request batch");
            let initial_items = requests
                .iter()
                .copied()
                .map(|request| PrepareBatchItem {
                    request,
                    expected_head: manager.request(request).expect("expected head").head,
                    target_boundary: context_tokens,
                })
                .collect::<Vec<_>>();
            let prepared = manager
                .prepare_batch(&initial_items)
                .expect("steady benchmark initial prepare");
            let (submit_items, receipts, copies) = batch_submit_items(&manager, &prepared);
            let submitted = manager
                .submit_batch(&submit_items, &receipts, &copies)
                .expect("steady benchmark initial submit");
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
                .expect("steady benchmark initial complete");

            manager.hot_path = HotPathInstrumentation::default();
            let mut prepare_ns = 0_u128;
            let mut submit_ns = 0_u128;
            let mut complete_ns = 0_u128;
            let steady_start = Instant::now();
            for iteration in 0..STEADY_ITERATIONS {
                let extend_items = requests
                    .iter()
                    .copied()
                    .map(|request| PrepareBatchItem {
                        request,
                        expected_head: manager.request(request).expect("expected head").head,
                        target_boundary: context_tokens + iteration + 1,
                    })
                    .collect::<Vec<_>>();
                let phase_start = Instant::now();
                let prepared = manager
                    .prepare_batch(&extend_items)
                    .expect("steady benchmark extend prepare");
                prepare_ns += phase_start.elapsed().as_nanos();
                std::hint::black_box(&prepared);
                let (submit_items, receipts, copies) = batch_submit_items(&manager, &prepared);
                let phase_start = Instant::now();
                let submitted = manager
                    .submit_batch(&submit_items, &receipts, &copies)
                    .expect("steady benchmark submit");
                submit_ns += phase_start.elapsed().as_nanos();
                let submissions = submitted
                    .iter()
                    .map(|item| item.submission)
                    .collect::<Vec<_>>();
                let phase_start = Instant::now();
                manager
                    .complete_batch(
                        BatchCompletionReceipt {
                            engine_epoch: manager.engine_epoch,
                            completion_domain: 1,
                            completion_value: iteration + 2,
                            confirmed: 1,
                            reserved: 0,
                        },
                        &submissions,
                    )
                    .expect("steady benchmark complete");
                complete_ns += phase_start.elapsed().as_nanos();
            }
            let steady_elapsed = steady_start.elapsed();
            let iterations = u128::from(STEADY_ITERATIONS);
            eprintln!(
                "orbitkv_steady_full context_pages={context_pages} batch={batch_size} iterations={STEADY_ITERATIONS} prepare_ns_per_iter={} submit_ns_per_iter={} complete_ns_per_iter={} phase_total_ns_per_iter={} wall_ns_per_iter={} hot_root_entries_visited={} device_view_entries_materialized={} snapshot_entries_cloned={} delta_entries_touched={} retirement_entries_touched={}",
                prepare_ns / iterations,
                submit_ns / iterations,
                complete_ns / iterations,
                (prepare_ns + submit_ns + complete_ns) / iterations,
                steady_elapsed.as_nanos() / iterations,
                manager.hot_path.hot_root_entries_visited,
                manager.hot_path.device_view_entries_materialized,
                manager.hot_path.snapshot_entries_cloned,
                manager.hot_path.delta_entries_touched,
                manager.hot_path.retirement_entries_touched,
            );
            assert_eq!(manager.hot_path.hot_root_entries_visited, 0);
            assert_eq!(manager.hot_path.device_view_entries_materialized, 0);
            assert_eq!(manager.hot_path.snapshot_entries_cloned, 0);
            assert_eq!(manager.hot_path.retirement_entries_touched, 0);
            assert_incremental_census_matches_full_scan(&manager);
        }
    }
}
