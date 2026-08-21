use super::*;
use crate::plan::{KvClassSpec, KvPlanInput, compile_plan};
use std::collections::BTreeMap;
use std::time::Instant;

type ArenaCounts = (u16, u32, u32, u64, u64, u64, u64, u64, u64, u64);

fn sliding_plan(window_tokens: u64, page_tokens: u64) -> CompiledKvPlan {
    compile_plan(KvPlanInput {
        page_tokens,
        classes: vec![KvClassSpec {
            name: "swa".into(),
            layers: vec![0],
            retention: RetentionKind::Sliding,
            bytes_per_token_per_layer: 128,
            window_tokens: Some(window_tokens),
        }],
    })
    .expect("test plan compiles")
}

fn full_plan(page_tokens: u64) -> CompiledKvPlan {
    compile_plan(KvPlanInput {
        page_tokens,
        classes: vec![KvClassSpec {
            name: "full".into(),
            layers: vec![0],
            retention: RetentionKind::Full,
            bytes_per_token_per_layer: 128,
            window_tokens: None,
        }],
    })
    .expect("test full plan compiles")
}

fn hybrid_plan(window_tokens: u64) -> CompiledKvPlan {
    compile_plan(KvPlanInput {
        page_tokens: CANONICAL_PAGE_TOKENS,
        classes: vec![
            KvClassSpec {
                name: "full".into(),
                layers: vec![0],
                retention: RetentionKind::Full,
                bytes_per_token_per_layer: 128,
                window_tokens: None,
            },
            KvClassSpec {
                name: "swa".into(),
                layers: vec![1],
                retention: RetentionKind::Sliding,
                bytes_per_token_per_layer: 128,
                window_tokens: Some(window_tokens),
            },
        ],
    })
    .expect("test hybrid plan compiles")
}

const fn backend(
    class_id: u16,
    pool_id: u32,
    page_count: u32,
    backend_base_index: u64,
) -> BackendArenaRegistration {
    BackendArenaRegistration {
        pool_id,
        class_id,
        backend_domain: class_id + 10,
        page_count,
        reserved: 0,
        backend_base_index,
    }
}

fn manager_for_plan(
    plan: &CompiledKvPlan,
    backends: &[BackendArenaRegistration],
    maximum_step_tokens: u32,
    maximum_reclamations: u32,
) -> CanonicalKvManager {
    CanonicalKvManager::new(
        plan,
        ManagerConfig {
            maximum_requests: 4,
            maximum_operations: 4,
            maximum_prefixes: 16,
            maximum_reclamations,
            maximum_step_tokens,
        },
        backends,
    )
    .expect("test manager constructs")
}

fn manager_with(
    window_tokens: u64,
    page_count: u32,
    maximum_step_tokens: u32,
    maximum_reclamations: u32,
) -> CanonicalKvManager {
    manager_for_plan(
        &sliding_plan(window_tokens, CANONICAL_PAGE_TOKENS),
        &[BackendArenaRegistration {
            pool_id: 7,
            class_id: 0,
            backend_domain: 3,
            page_count,
            reserved: 0,
            backend_base_index: 100,
        }],
        maximum_step_tokens,
        maximum_reclamations,
    )
}

fn operation_entries(
    manager: &CanonicalKvManager,
    slot: u32,
    generation: u32,
) -> Vec<DeviceKvEntry> {
    let delta = match manager
        .operations
        .get(slot, generation)
        .expect("operation exists")
    {
        OperationState::Prepared(prepared) => &prepared.delta,
        OperationState::Submitted(submitted) => &submitted.delta,
    };
    let snapshot = manager
        .request_snapshot(delta.request)
        .expect("operation request snapshot");
    let first_new = delta.previous_boundary.div_ceil(manager.page_tokens);
    let write_first = delta.previous_boundary / manager.page_tokens;
    manager
        .classes
        .iter()
        .copied()
        .zip(snapshot.roots.iter())
        .zip(delta.classes.iter())
        .flat_map(|((class, root), class_delta)| {
            root.entries
                .iter()
                .chain(class_delta.writes.iter())
                .map(move |entry| {
                    let token_begin = entry.logical_ordinal * manager.page_tokens;
                    let page_end = token_begin.saturating_add(manager.page_tokens);
                    let valid_end = delta.target_boundary.min(page_end);
                    let visible_begin = class
                        .candidate_start(delta.previous_boundary)
                        .max(token_begin);
                    let visible_end = delta.target_boundary.min(page_end);
                    let mut access_flags = DEVICE_KV_ACCESS_READ;
                    if entry.logical_ordinal >= write_first {
                        access_flags |= DEVICE_KV_ACCESS_WRITE;
                    }
                    if entry.logical_ordinal >= first_new {
                        access_flags |= DEVICE_KV_NEEDS_BINDING;
                    }
                    manager
                        .device_entry(
                            *entry,
                            access_flags,
                            valid_end - token_begin,
                            visible_begin - token_begin,
                            visible_end - visible_begin,
                        )
                        .expect("operation entry")
                })
        })
        .collect()
}

fn snapshot_entries(manager: &CanonicalKvManager, request: RequestLease) -> Vec<RootEntry> {
    manager
        .request_snapshot(request)
        .expect("request snapshot")
        .roots
        .iter()
        .flat_map(|root| root.entries.iter().copied())
        .collect()
}

/// Test-only model of a future snapshot fork: both requests keep distinct
/// generation-checked snapshot leases while sharing one immutable root
/// bundle and contributing independent request references.
fn share_snapshot_for_cow(
    manager: &mut CanonicalKvManager,
    source: RequestLease,
    target: RequestLease,
) {
    let source_snapshot = manager
        .request_snapshot(source)
        .expect("fork source snapshot")
        .clone();
    let target_head = manager.request(target).expect("fork target").head;
    let target_snapshot = manager
        .snapshots
        .get(target_head.slot, target_head.generation)
        .expect("fork target snapshot");
    assert_eq!(target_snapshot.boundary, 0);
    assert!(target_snapshot.is_empty());
    for entry in CanonicalKvManager::root_entries(&source_snapshot.roots) {
        let page = manager.page_mut(entry.page.page_id).expect("fork page");
        page.request_refs = page.request_refs.checked_add(1).expect("fork ref count");
    }
    *manager
        .snapshots
        .get_mut(target_head.slot, target_head.generation)
        .expect("fork target snapshot") = RequestSnapshot {
        boundary: source_snapshot.boundary,
        view_version: source_snapshot.view_version,
        roots: Arc::clone(&source_snapshot.roots),
    };
}

fn root_node_addresses(root: Option<&Arc<RootTreeNode>>, addresses: &mut BTreeSet<usize>) {
    if let Some(node) = root {
        addresses.insert(Arc::as_ptr(node) as usize);
        root_node_addresses(node.left.as_ref(), addresses);
        root_node_addresses(node.right.as_ref(), addresses);
    }
}

fn published_entries(manager: &CanonicalKvManager, request: RequestLease) -> Vec<DeviceKvEntry> {
    let snapshot = manager
        .request_snapshot(request)
        .expect("request snapshot exists");
    snapshot
        .roots
        .iter()
        .flat_map(|root| root.entries.iter().copied())
        .map(|root| {
            let class = manager
                .runtime_class(root.class_id)
                .expect("published class exists");
            let token_begin = root.logical_ordinal * manager.page_tokens;
            let token_end = token_begin
                .saturating_add(manager.page_tokens)
                .min(snapshot.boundary);
            let visible_begin = class
                .retained_start(snapshot.boundary)
                .max(token_begin)
                .min(token_end);
            let visible_end = snapshot
                .boundary
                .min(token_begin.saturating_add(manager.page_tokens))
                .max(visible_begin);
            manager
                .device_entry(
                    root,
                    u32::from(visible_end > visible_begin) * DEVICE_KV_ACCESS_READ,
                    token_end - token_begin,
                    visible_begin - token_begin,
                    visible_end - visible_begin,
                )
                .expect("published entry")
        })
        .collect()
}

fn assert_materialization_matches_snapshot(
    manager: &CanonicalKvManager,
    materialized: &MaterializedRequestView,
) {
    assert_eq!(
        materialized.view,
        manager
            .request_view(materialized.view.request)
            .expect("materialized request view")
    );
    let expected = published_entries(manager, materialized.view.request);
    assert_eq!(materialized.pages.len(), expected.len());
    for (page, entry) in materialized.pages.iter().zip(expected.iter()) {
        assert_eq!(page.class_id, entry.class_id);
        assert_eq!(page.backend_domain, entry.backend_domain);
        assert_eq!(page.logical_ordinal, entry.logical_ordinal);
        assert_eq!(page.temporal_cell_index, entry.temporal_cell_index);
        assert_eq!(page.temporal_cycle, entry.temporal_cycle);
        assert_eq!(page.page.engine_epoch, manager.engine_epoch);
        assert_eq!(page.page.pool_epoch, entry.pool_epoch);
        assert_eq!(page.page.generation, entry.page_generation);
        assert_eq!(page.page.page_id, entry.page_id);
        assert_eq!(page.page.pool_id, entry.pool_id);
        assert_eq!(page.backend_index, entry.backend_index);
        assert_eq!(page.valid_token_count, entry.valid_token_count);
        assert_eq!(page.visible_token_offset, entry.visible_token_offset);
        assert_eq!(page.visible_token_count, entry.visible_token_count);
    }
}

fn assert_canonical_prepared_spans(prepared: &PreparedStep) {
    let mut expected_tail_offset = 0_u32;
    let mut expected_copy_offset = 0_u32;
    let mut expected_write_offset = 0_u32;
    for (class_index, lowering) in prepared.class_lowerings.iter().enumerate() {
        assert_eq!(usize::from(lowering.class_id), class_index);
        assert_eq!(lowering.flags, 0);
        assert_eq!(lowering.reserved, 0);
        assert_eq!(lowering.tail_offset, expected_tail_offset);
        assert_eq!(lowering.tail_count, 1);
        assert_eq!(lowering.copy_offset, expected_copy_offset);
        assert!(lowering.copy_count <= 1);
        assert_eq!(lowering.write_offset, expected_write_offset);
        let tail_end = lowering
            .tail_offset
            .checked_add(lowering.tail_count)
            .expect("tail span");
        let copy_end = lowering
            .copy_offset
            .checked_add(lowering.copy_count)
            .expect("copy span");
        let write_end = lowering
            .write_offset
            .checked_add(lowering.write_count)
            .expect("write span");
        let actions = &prepared.tail_actions[lowering.tail_offset as usize..tail_end as usize];
        assert_eq!(actions[0].class_id, lowering.class_id);
        assert!(
            prepared.copy_intents[lowering.copy_offset as usize..copy_end as usize]
                .iter()
                .all(|intent| intent.class_id == lowering.class_id)
        );
        expected_tail_offset = tail_end;
        expected_copy_offset = copy_end;
        expected_write_offset = write_end;
    }
    assert_eq!(expected_tail_offset as usize, prepared.tail_actions.len());
    assert_eq!(expected_copy_offset as usize, prepared.copy_intents.len());
    assert_eq!(expected_write_offset as usize, prepared.write_intents.len());
}

fn binding_receipts(
    manager: &CanonicalKvManager,
    prepared: &PreparedStep,
) -> Vec<BackendBindReceipt> {
    assert_canonical_prepared_spans(prepared);
    let mut receipts = Vec::new();
    for lowering in &prepared.class_lowerings {
        let tail_begin = lowering.tail_offset as usize;
        let tail_end = tail_begin + lowering.tail_count as usize;
        for action in prepared.tail_actions[tail_begin..tail_end]
            .iter()
            .filter(|action| {
                matches!(
                    action.kind,
                    TailActionKind::CopyOnWrite | TailActionKind::Fresh
                )
            })
        {
            let class = manager
                .runtime_class(action.class_id)
                .expect("prepared tail class exists");
            receipts.push(BackendBindReceipt {
                step: prepared.step,
                page: action.destination,
                backend_domain: class.backend.backend_domain,
                mapped: 1,
                writable: 1,
                reserved: 0,
                backend_index: class
                    .backend_index(action.destination.page_id)
                    .expect("prepared tail page belongs to class"),
            });
        }
        let class = manager
            .runtime_class(lowering.class_id)
            .expect("prepared class exists");
        let begin = usize::try_from(lowering.write_offset).expect("write offset");
        let end = begin + usize::try_from(lowering.write_count).expect("write count");
        for intent in &prepared.write_intents[begin..end] {
            receipts.push(BackendBindReceipt {
                step: prepared.step,
                page: PageLease {
                    engine_epoch: prepared.request.engine_epoch,
                    pool_epoch: manager.pool_epoch,
                    generation: intent.page_generation,
                    page_id: intent.page_id,
                    pool_id: class.backend.pool_id,
                },
                backend_domain: class.backend.backend_domain,
                mapped: 1,
                writable: 1,
                reserved: 0,
                backend_index: class
                    .backend_index(intent.page_id)
                    .expect("prepared page belongs to class"),
            });
        }
    }
    receipts
}

fn copy_receipts(prepared: &PreparedStep) -> Vec<BackendCopyReceipt> {
    assert_canonical_prepared_spans(prepared);
    prepared
        .class_lowerings
        .iter()
        .flat_map(|lowering| {
            let begin = lowering.copy_offset as usize;
            let end = begin + lowering.copy_count as usize;
            prepared.copy_intents[begin..end]
                .iter()
                .map(|intent| BackendCopyReceipt {
                    step: prepared.step,
                    class_id: intent.class_id,
                    backend_domain: intent.backend_domain,
                    token_count: intent.token_count,
                    source_token_offset: intent.source_token_offset,
                    destination_token_offset: intent.destination_token_offset,
                    observed: 1,
                    copied: 1,
                    ordered_before_writes: 1,
                    reserved8: 0,
                    reserved32: 0,
                    source: intent.source,
                    destination: intent.destination,
                    source_backend_index: intent.source_backend_index,
                    destination_backend_index: intent.destination_backend_index,
                })
        })
        .collect()
}

fn submit(manager: &mut CanonicalKvManager, prepared: &PreparedStep) -> SubmittedStep {
    let receipts = binding_receipts(manager, prepared);
    let copies = copy_receipts(prepared);
    manager
        .submit_batch(
            &[SubmitBatchItem {
                step: prepared.step,
                receipt_offset: 0,
                receipt_count: u32::try_from(receipts.len()).expect("receipt count"),
                copy_receipt_offset: 0,
                copy_receipt_count: u32::try_from(copies.len()).expect("copy receipt count"),
            }],
            &receipts,
            &copies,
        )
        .expect("test submit succeeds")[0]
        .clone()
}

#[derive(Clone, Debug)]
struct TestCompletion {
    step: StepCompletion,
    retirements: Box<[ReclamationCertificate]>,
}

impl std::ops::Deref for TestCompletion {
    type Target = StepCompletion;

    fn deref(&self) -> &Self::Target {
        &self.step
    }
}

fn complete(
    manager: &mut CanonicalKvManager,
    submitted: &SubmittedStep,
    domain: u64,
    value: u64,
) -> TestCompletion {
    let batch = manager
        .complete_batch(
            BatchCompletionReceipt {
                engine_epoch: submitted.submission.engine_epoch,
                completion_domain: domain,
                completion_value: value,
                confirmed: 1,
                reserved: 0,
            },
            &[submitted.submission],
        )
        .expect("test completion succeeds");
    TestCompletion {
        step: batch.completions[0].clone(),
        retirements: batch.retirements,
    }
}

fn assert_snapshot_transition_identities(
    manager: &CanonicalKvManager,
    prepared: &PreparedStep,
    submitted: &SubmittedStep,
    completion: &StepCompletion,
) {
    assert_eq!(submitted.request, prepared.request);
    assert_eq!(submitted.target_snapshot, prepared.target_snapshot);
    assert_eq!(completion.request, prepared.request);
    assert_eq!(completion.detached_snapshot, prepared.base_snapshot);
    assert_eq!(completion.publication.snapshot, prepared.target_snapshot);
    assert_eq!(completion.publication.snapshot, submitted.target_snapshot);
    assert_eq!(
        manager.snapshots.get(
            completion.detached_snapshot.slot,
            completion.detached_snapshot.generation,
        ),
        Err(KvManagerError::StaleLease("snapshot"))
    );
}

fn reclamation_receipts(certificates: &[ReclamationCertificate]) -> Vec<ReclamationReceipt> {
    certificates
        .iter()
        .map(|certificate| ReclamationReceipt {
            reclamation: certificate.reclamation,
            page: certificate.page,
            backend_domain: certificate.backend_domain,
            acknowledged: 1,
            reserved8: 0,
            reserved32: 0,
            backend_index: certificate.backend_index,
        })
        .collect()
}

fn arena_counts(manager: &CanonicalKvManager) -> Vec<ArenaCounts> {
    manager
        .arena_stats()
        .iter()
        .map(|stats| {
            (
                stats.class_id,
                stats.pool_id,
                stats.page_count,
                stats.free_pages,
                stats.reserved_pages,
                stats.writing_pages,
                stats.active_pages,
                stats.retiring_pages,
                stats.quarantined_pages,
                stats.exhausted_pages,
            )
        })
        .collect()
}

fn assert_incremental_census_matches_full_scan(manager: &CanonicalKvManager) {
    let mut scanned = vec![PageCounts::default(); manager.classes.len()];
    for page in &manager.pages {
        scanned[usize::from(page.class_id)].increment(page.phase);
    }
    assert_eq!(manager.page_counts, scanned);

    let mut prepared = 0_u64;
    let mut submitted = 0_u64;
    for slot in &manager.operations.slots {
        match slot.value {
            Some(OperationState::Prepared(_)) => prepared += 1,
            Some(OperationState::Submitted(_)) => submitted += 1,
            None => {}
        }
    }
    assert_eq!(manager.prepared_steps, prepared);
    assert_eq!(manager.submitted_steps, submitted);
    let stats = manager.stats();
    assert_eq!(
        stats.pending_reclamations,
        manager.reclamations.active_len() as u64
    );
    assert_eq!(
        stats.active_snapshots,
        manager.snapshots.active_len() as u64
    );
    let (active_prefixes, evicted_prefixes) = manager
        .prefixes
        .slots
        .iter()
        .filter_map(|slot| slot.value.as_ref())
        .fold((0_u64, 0_u64), |(active, evicted), prefix| {
            if prefix.evicted {
                (active, evicted + 1)
            } else {
                (active + 1, evicted)
            }
        });
    assert_eq!(stats.active_prefixes, active_prefixes);
    assert_eq!(stats.evicted_prefixes, evicted_prefixes);
    assert_eq!(
        stats.total_request_page_refs,
        manager
            .pages
            .iter()
            .map(|page| u64::from(page.request_refs))
            .sum::<u64>()
    );
    assert_eq!(
        stats.total_prefix_page_refs,
        manager
            .pages
            .iter()
            .map(|page| u64::from(page.prefix_refs))
            .sum::<u64>()
    );
    assert_eq!(
        stats.total_reader_pins,
        manager
            .pages
            .iter()
            .map(|page| u64::from(page.reader_pins))
            .sum::<u64>()
    );
    for arena in &manager.arena_stats() {
        let pages = manager
            .pages
            .iter()
            .filter(|page| page.class_id == arena.class_id)
            .collect::<Vec<_>>();
        assert_eq!(
            arena.request_page_refs,
            pages
                .iter()
                .map(|page| u64::from(page.request_refs))
                .sum::<u64>()
        );
        assert_eq!(
            arena.prefix_page_refs,
            pages
                .iter()
                .map(|page| u64::from(page.prefix_refs))
                .sum::<u64>()
        );
        assert_eq!(
            arena.reader_pins,
            pages
                .iter()
                .map(|page| u64::from(page.reader_pins))
                .sum::<u64>()
        );
    }
}

fn assert_reference_census_matches_full_scan(manager: &CanonicalKvManager) {
    let mut request_refs = BTreeMap::<PageLease, u32>::new();
    let mut heads = BTreeSet::new();
    for request in manager
        .requests
        .slots
        .iter()
        .filter_map(|slot| slot.value.as_ref())
    {
        if request.released {
            assert_eq!(
                manager
                    .snapshots
                    .get(request.head.slot, request.head.generation),
                Err(KvManagerError::StaleLease("snapshot")),
                "released request must not retain a resolvable snapshot"
            );
            continue;
        }
        assert!(
            heads.insert(request.head),
            "request heads must be independent"
        );
        let snapshot = manager
            .snapshots
            .get(request.head.slot, request.head.generation)
            .expect("live request head");
        for entry in CanonicalKvManager::root_entries(&snapshot.roots) {
            *request_refs.entry(entry.page).or_default() += 1;
        }
    }
    let mut prefix_refs = BTreeMap::<PageLease, u32>::new();
    for prefix in manager
        .prefixes
        .slots
        .iter()
        .filter_map(|slot| slot.value.as_ref())
        .filter(|prefix| !prefix.evicted)
    {
        for entry in CanonicalKvManager::root_entries(&prefix.roots) {
            *prefix_refs.entry(entry.page).or_default() += 1;
        }
    }
    for (index, page) in manager.pages.iter().enumerate() {
        let page_id = u32::try_from(index + 1).expect("test page id");
        let class = manager.runtime_class(page.class_id).expect("page class");
        let lease = PageLease {
            engine_epoch: manager.engine_epoch,
            pool_epoch: manager.pool_epoch,
            generation: page.generation,
            page_id,
            pool_id: class.backend.pool_id,
        };
        assert_eq!(
            page.request_refs,
            request_refs.get(&lease).copied().unwrap_or(0),
            "request ref mismatch for page {page_id}"
        );
        assert_eq!(
            page.prefix_refs,
            prefix_refs.get(&lease).copied().unwrap_or(0),
            "prefix ref mismatch for page {page_id}"
        );
    }
}

fn prefix_key(tag: u8, boundary: u64) -> PrefixSemanticKey {
    PrefixSemanticKey {
        namespace: [0xA5; 32],
        digest: [tag; 32],
        boundary,
    }
}

fn staged_prefix_attach_fixture() -> (
    CanonicalKvManager,
    Box<[RequestLease]>,
    Box<[PrefixLookupHint]>,
) {
    let plan = full_plan(CANONICAL_PAGE_TOKENS);
    let mut manager = CanonicalKvManager::new(
        &plan,
        ManagerConfig {
            maximum_requests: 4,
            maximum_operations: 1,
            maximum_prefixes: 2,
            maximum_reclamations: 8,
            maximum_step_tokens: 32,
        },
        &[backend(0, 191, 8, 9_000)],
    )
    .expect("staged attach manager");
    let source = manager.acquire_requests_batch(1).expect("source view")[0].request;
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request: source,
            expected_head: manager.request(source).expect("source head").head,
            target_boundary: CANONICAL_PAGE_TOKENS,
        }])
        .expect("source prepare");
    let submitted = submit(&mut manager, &prepared[0]);
    complete(&mut manager, &submitted, 3, 5);
    let key = prefix_key(0x44, CANONICAL_PAGE_TOKENS);
    manager
        .publish_prefix_batch(&[PrefixPublishItem {
            request: source,
            expected_head: manager.request(source).expect("published head").head,
            key,
        }])
        .expect("publish fixture prefix");
    let requests = manager
        .acquire_requests_batch(3)
        .expect("three empty request views")
        .iter()
        .map(|view| view.request)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let hints = manager
        .lookup_prefix_batch(&[key; 3])
        .expect("fixture lookup");
    (manager, requests, hints)
}

fn attach_items(
    manager: &CanonicalKvManager,
    requests: &[RequestLease],
    hints: &[PrefixLookupHint],
) -> Vec<PrefixAttachItem> {
    requests
        .iter()
        .copied()
        .zip(hints.iter().copied())
        .map(|(request, hint)| PrefixAttachItem {
            request,
            expected_empty_head: manager.request(request).expect("empty request head").head,
            hint,
        })
        .collect()
}

fn partial_fork_fixture(
    maximum_operations: u32,
) -> (CanonicalKvManager, RequestLease, Box<[RequestLease]>) {
    let plan = hybrid_plan(18);
    let mut manager = CanonicalKvManager::new(
        &plan,
        ManagerConfig {
            maximum_requests: 5,
            maximum_operations,
            maximum_prefixes: 2,
            maximum_reclamations: 64,
            maximum_step_tokens: 32,
        },
        &[backend(0, 193, 32, 11_000), backend(1, 194, 32, 12_000)],
    )
    .expect("partial fork manager");
    let acquired = manager
        .acquire_requests_batch(5)
        .expect("source plus B4 targets");
    let source = acquired[0].request;
    let targets = acquired[1..]
        .iter()
        .map(|view| view.request)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request: source,
            expected_head: acquired[0].snapshot,
            target_boundary: 18,
        }])
        .expect("partial source prepare");
    let submitted = submit(&mut manager, &prepared[0]);
    complete(&mut manager, &submitted, 13, 17);
    (manager, source, targets)
}

fn fork_items(
    manager: &CanonicalKvManager,
    source: RequestLease,
    targets: &[RequestLease],
) -> Vec<RequestForkItem> {
    let expected_source_head = manager.request(source).expect("source head").head;
    targets
        .iter()
        .copied()
        .map(|target| RequestForkItem {
            source_request: source,
            expected_source_head,
            target_empty_request: target,
            expected_target_head: manager.request(target).expect("target head").head,
        })
        .collect()
}

fn state_image(manager: &CanonicalKvManager) -> String {
    format!("{manager:#?}")
}

fn complete_initial_18(manager: &mut CanonicalKvManager, request: RequestLease) -> TestCompletion {
    let prepared = manager
        .prepare_batch(&[PrepareBatchItem {
            request,
            expected_head: manager.request(request).expect("expected head").head,
            target_boundary: 18,
        }])
        .expect("prepare 18")[0]
        .clone();
    let submitted = submit(manager, &prepared);
    complete(manager, &submitted, 7, 9)
}

fn batch_submit_items(
    manager: &CanonicalKvManager,
    prepared: &[PreparedStep],
) -> (
    Vec<SubmitBatchItem>,
    Vec<BackendBindReceipt>,
    Vec<BackendCopyReceipt>,
) {
    let mut items = Vec::with_capacity(prepared.len());
    let mut receipts = Vec::new();
    let mut copies = Vec::new();
    for item in prepared {
        let item_receipts = binding_receipts(manager, item);
        let item_copies = copy_receipts(item);
        items.push(SubmitBatchItem {
            step: item.step,
            receipt_offset: u32::try_from(receipts.len()).expect("receipt offset"),
            receipt_count: u32::try_from(item_receipts.len()).expect("receipt count"),
            copy_receipt_offset: u32::try_from(copies.len()).expect("copy receipt offset"),
            copy_receipt_count: u32::try_from(item_copies.len()).expect("copy receipt count"),
        });
        receipts.extend(item_receipts);
        copies.extend(item_copies);
    }
    (items, receipts, copies)
}
