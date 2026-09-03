use super::*;

const FULL_SLIDING_PLAN: &[u8] = br#"{
  "page_tokens": 16,
  "classes": [
    {"name":"full","layers":[0],"retention":"full","bytes_per_token_per_layer":128},
    {"name":"swa","layers":[1],"retention":"sliding","bytes_per_token_per_layer":128,"window_tokens":18}
  ]
}"#;

fn hybrid_backends() -> [OrbitKvBackendArenaRegistration; 2] {
    [
        OrbitKvBackendArenaRegistration {
            pool_id: 9,
            class_id: 1,
            backend_domain: 5,
            page_count: 8,
            reserved: 0,
            backend_base_index: 1_000,
        },
        OrbitKvBackendArenaRegistration {
            pool_id: 7,
            class_id: 0,
            backend_domain: 3,
            page_count: 8,
            reserved: 0,
            backend_base_index: 100,
        },
    ]
}

struct HybridSession {
    session: Session,
    arenas: Vec<OrbitKvArenaIdentity>,
}

impl HybridSession {
    fn new() -> Self {
        let mut handle = std::ptr::null_mut();
        let mut error = [0; 256];
        let backends = hybrid_backends();
        let config = OrbitKvSessionCreateConfig {
            manager: OrbitKvManagerConfig {
                maximum_requests: 4,
                maximum_operations: 4,
                maximum_prefixes: 2,
                maximum_reclamations: 16,
                maximum_step_tokens: 64,
                plan_format: 1,
                reserved: 0,
            },
            cache_sharing_policy: ORBITKV_CACHE_SHARING_POLICY_SHARED_PREFIX,
            reserved: 0,
        };
        assert_eq!(
            unsafe {
                orbitkv_session_create(
                    FULL_SLIDING_PLAN.as_ptr(),
                    FULL_SLIDING_PLAN.len(),
                    &config,
                    backends.as_ptr(),
                    backends.len() as u32,
                    &mut handle,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert!(!handle.is_null());

        let sentinel = OrbitKvArenaIdentity {
            engine_epoch: u64::MAX,
            pool_epoch: u64::MAX,
            backend_base_index: u64::MAX,
            pool_id: u32::MAX,
            page_count: u32::MAX,
            page_tokens: u32::MAX,
            class_id: u16::MAX,
            backend_domain: u16::MAX,
            first_page_id: u32::MAX,
            reserved: u32::MAX,
        };
        let mut short = [sentinel];
        let mut count = u32::MAX;
        assert_eq!(
            unsafe {
                orbitkv_session_arena_identities(
                    handle,
                    short.as_mut_ptr(),
                    1,
                    &mut count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_BUFFER_TOO_SMALL
        );
        assert_eq!(count, 2);
        assert_eq!(short, [sentinel]);
        let mut arenas = vec![OrbitKvArenaIdentity::default(); count as usize];
        assert_eq!(
            unsafe {
                orbitkv_session_arena_identities(
                    handle,
                    arenas.as_mut_ptr(),
                    arenas.len() as u32,
                    &mut count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert_eq!(
            arenas
                .iter()
                .map(|arena| (arena.class_id, arena.pool_id, arena.backend_domain))
                .collect::<Vec<_>>(),
            vec![(0, 7, 3), (1, 9, 5)]
        );
        Self {
            session: Session(handle),
            arenas,
        }
    }

    const fn as_ptr(&self) -> *mut OrbitKvSessionHandle {
        self.session.as_ptr()
    }

    fn arena(&self, class_id: u16) -> OrbitKvArenaIdentity {
        self.arenas
            .iter()
            .copied()
            .find(|arena| arena.class_id == class_id)
            .expect("class arena")
    }
}

struct HybridPrepared {
    batch: OrbitKvSessionBatchId,
    step: OrbitKvSessionPreparedStep,
    classes: Vec<OrbitKvClassLowering>,
    tails: Vec<OrbitKvTailAction>,
    copies: Vec<OrbitKvCopyIntent>,
    writes: Vec<OrbitKvWriteIntent>,
}

fn acquire_hybrid(session: &HybridSession, request_id: u64) {
    let mut output = OrbitKvSessionRequestView::default();
    let mut count = 0;
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_acquire_requests(
                session.as_ptr(),
                &request_id,
                1,
                &mut output,
                1,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!((output.request_id, count), (request_id, 1));
}

fn prepare_hybrid(
    session: &HybridSession,
    request_id: u64,
    target_boundary: u64,
) -> HybridPrepared {
    let intent = OrbitKvSessionAppendIntent {
        request_id,
        target_boundary,
    };
    let mut batch = OrbitKvSessionBatchId::default();
    let mut step = OrbitKvSessionPreparedStep::default();
    let mut classes = vec![OrbitKvClassLowering::default(); 2];
    let mut tails = vec![OrbitKvTailAction::default(); 2];
    let mut copies = vec![OrbitKvCopyIntent::default(); 2];
    let mut writes = vec![OrbitKvWriteIntent::default(); 8];
    let (mut step_count, mut class_count, mut tail_count, mut copy_count, mut write_count) =
        (0, 0, 0, 0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_append(
                session.as_ptr(),
                &intent,
                1,
                &mut batch,
                &mut step,
                1,
                &mut step_count,
                classes.as_mut_ptr(),
                classes.len() as u32,
                &mut class_count,
                tails.as_mut_ptr(),
                tails.len() as u32,
                &mut tail_count,
                copies.as_mut_ptr(),
                copies.len() as u32,
                &mut copy_count,
                writes.as_mut_ptr(),
                writes.len() as u32,
                &mut write_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(step_count, 1);
    classes.truncate(class_count as usize);
    tails.truncate(tail_count as usize);
    copies.truncate(copy_count as usize);
    writes.truncate(write_count as usize);
    HybridPrepared {
        batch,
        step,
        classes,
        tails,
        copies,
        writes,
    }
}

fn submit_hybrid(session: &HybridSession, prepared: &HybridPrepared) {
    let mut binds = Vec::new();
    for class in &prepared.classes {
        let arena = session.arena(class.class_id);
        let backend_index =
            |page_id: u32| arena.backend_base_index + u64::from(page_id - arena.first_page_id);
        for tail in &prepared.tails
            [class.tail_offset as usize..(class.tail_offset + class.tail_count) as usize]
        {
            if tail.kind == ORBITKV_TAIL_COPY_ON_WRITE || tail.kind == ORBITKV_TAIL_FRESH {
                binds.push(OrbitKvSessionBindEvidence {
                    page: tail.destination,
                    backend_domain: arena.backend_domain,
                    mapped: 1,
                    writable: 1,
                    reserved: 0,
                    backend_index: backend_index(tail.destination.page_id),
                });
            }
        }
        for write in &prepared.writes
            [class.write_offset as usize..(class.write_offset + class.write_count) as usize]
        {
            binds.push(OrbitKvSessionBindEvidence {
                page: OrbitKvPageLease {
                    engine_epoch: arena.engine_epoch,
                    pool_epoch: arena.pool_epoch,
                    generation: write.page_generation,
                    page_id: write.page_id,
                    pool_id: arena.pool_id,
                },
                backend_domain: arena.backend_domain,
                mapped: 1,
                writable: 1,
                reserved: 0,
                backend_index: backend_index(write.page_id),
            });
        }
    }
    let copies = prepared
        .copies
        .iter()
        .map(|copy| OrbitKvSessionCopyEvidence {
            class_id: copy.class_id,
            backend_domain: copy.backend_domain,
            token_count: copy.token_count,
            source_token_offset: copy.source_token_offset,
            destination_token_offset: copy.destination_token_offset,
            observed: 1,
            copied: 1,
            ordered_before_writes: 1,
            reserved8: 0,
            reserved32: 0,
            source: copy.source,
            destination: copy.destination,
            source_backend_index: copy.source_backend_index,
            destination_backend_index: copy.destination_backend_index,
        })
        .collect::<Vec<_>>();
    let evidence = OrbitKvSessionStepExecutionEvidence {
        request_id: prepared.step.request_id,
        bind_offset: 0,
        bind_count: binds.len() as u32,
        copy_offset: 0,
        copy_count: copies.len() as u32,
        reserved: 0,
    };
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_submit_execution(
                session.as_ptr(),
                prepared.batch,
                &evidence,
                1,
                binds.as_ptr(),
                binds.len() as u32,
                copies.as_ptr(),
                copies.len() as u32,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
}

fn complete_hybrid(session: &HybridSession, prepared: &HybridPrepared) {
    let mut publication_id = OrbitKvSessionPublicationId::default();
    let mut publication = OrbitKvSessionStepPublication::default();
    let mut detached = vec![OrbitKvDetachedBinding::default(); 16];
    let mut retirements = vec![OrbitKvSessionRetirement::default(); 16];
    let (mut step_count, mut detached_count, mut retirement_count) = (0, 0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_complete_execution(
                session.as_ptr(),
                prepared.batch,
                OrbitKvSessionCompletionEvidence {
                    completion_domain: 73,
                    completion_value: 1,
                    confirmed: 1,
                    reserved: 0,
                },
                &mut publication_id,
                &mut publication,
                1,
                &mut step_count,
                detached.as_mut_ptr(),
                detached.len() as u32,
                &mut detached_count,
                retirements.as_mut_ptr(),
                retirements.len() as u32,
                &mut retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(
        (publication.request_id, publication.boundary),
        (prepared.step.request_id, 32)
    );
    retirements.truncate(retirement_count as usize);
    let evidence = retirement_evidence(&retirements);
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_publication(
                session.as_ptr(),
                OrbitKvSessionPublicationEvidence {
                    publication_id,
                    mirror_cleanup_confirmed: 1,
                    reserved: 0,
                },
                evidence.as_ptr(),
                evidence.len() as u32,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
}

#[test]
fn two_arena_prefix_roundtrip_orders_classes_and_preserves_short_outputs() {
    let session = HybridSession::new();
    let source = 0;
    let target = u64::MAX;
    acquire_hybrid(&session, source);
    let prepared = prepare_hybrid(&session, source, 32);
    assert_eq!(
        prepared
            .classes
            .iter()
            .map(|class| class.class_id)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    submit_hybrid(&session, &prepared);
    complete_hybrid(&session, &prepared);

    let semantic_key = key(70, 32);
    let published = publish_prefix(&session.session, source, semantic_key);
    assert_eq!(published.resident_count, 4);
    let lookup = lookup_prefix(&session.session, semantic_key);
    assert_eq!(lookup.candidate_present, 1);
    assert_eq!(lookup.candidate, published.prefix_id);
    acquire_hybrid(&session, target);

    let mut foreign = OrbitKvSessionPrefixAttachItem {
        target_request_id: target,
        prefix_id: lookup.candidate,
        key: lookup.key,
        resident_count: lookup.resident_count,
        reserved: 0,
    };
    foreign.prefix_id.session_epoch = foreign.prefix_id.session_epoch.wrapping_add(1);
    let mut control_id = OrbitKvSessionControlId {
        session_epoch: u64::MAX,
        sequence: u64::MAX,
    };
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_prefix_attach(
                session.as_ptr(),
                &foreign,
                1,
                &mut control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    assert_eq!(control_id, OrbitKvSessionControlId::default());

    let item = OrbitKvSessionPrefixAttachItem {
        prefix_id: lookup.candidate,
        ..foreign
    };
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_prefix_attach(
                session.as_ptr(),
                &item,
                1,
                &mut control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    let info = commit(&session.session, control_id);
    assert_eq!(info.kind, ORBITKV_SESSION_CONTROL_KIND_MATERIALIZATION);
    assert_eq!((info.request_count, info.page_count), (1, 4));

    let sentinel = OrbitKvSessionMaterializedRequest {
        request_id: 17,
        view_version: 18,
        boundary: 19,
        resident_count: 20,
        page_offset: 21,
        page_count: 22,
        reserved: 23,
    };
    let page_sentinel = OrbitKvSnapshotPage {
        page: OrbitKvPageLease {
            engine_epoch: u64::MAX,
            pool_epoch: u64::MAX,
            generation: u64::MAX,
            page_id: u32::MAX,
            pool_id: u32::MAX,
        },
        logical_ordinal: u64::MAX,
        temporal_cell_index: u64::MAX,
        temporal_cycle: u64::MAX,
        backend_index: u64::MAX,
        class_id: u16::MAX,
        backend_domain: u16::MAX,
        valid_token_count: u32::MAX,
        visible_token_offset: u32::MAX,
        visible_token_count: u32::MAX,
        reserved: u32::MAX,
    };
    let mut request_short = [sentinel];
    let mut page_short = [page_sentinel; 3];
    let (mut request_count, mut page_count, mut prefix_count, mut retirement_count) =
        (u32::MAX, u32::MAX, u32::MAX, u32::MAX);
    assert_eq!(
        unsafe {
            orbitkv_session_read_control_plan(
                session.as_ptr(),
                control_id,
                request_short.as_mut_ptr(),
                1,
                &mut request_count,
                page_short.as_mut_ptr(),
                3,
                &mut page_count,
                std::ptr::null_mut(),
                0,
                &mut prefix_count,
                std::ptr::null_mut(),
                0,
                &mut retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!(
        (request_count, page_count, prefix_count, retirement_count),
        (1, 4, 0, 0)
    );
    assert_eq!(request_short, [sentinel]);
    assert_eq!(page_short, [page_sentinel; 3]);

    let mut requests = [OrbitKvSessionMaterializedRequest::default(); 1];
    let mut pages = [OrbitKvSnapshotPage::default(); 4];
    assert_eq!(
        unsafe {
            orbitkv_session_read_control_plan(
                session.as_ptr(),
                control_id,
                requests.as_mut_ptr(),
                1,
                &mut request_count,
                pages.as_mut_ptr(),
                4,
                &mut page_count,
                std::ptr::null_mut(),
                0,
                &mut prefix_count,
                std::ptr::null_mut(),
                0,
                &mut retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!((requests[0].request_id, requests[0].boundary), (target, 32));
    assert_eq!((requests[0].resident_count, requests[0].page_count), (4, 4));
    assert_eq!(
        pages
            .iter()
            .map(|page| (page.class_id, page.page.pool_id, page.backend_domain))
            .collect::<Vec<_>>(),
        vec![(0, 7, 3), (0, 7, 3), (1, 9, 5), (1, 9, 5)]
    );
    for page in pages {
        let arena = session.arena(page.class_id);
        assert_eq!(
            page.backend_index,
            arena.backend_base_index + u64::from(page.page.page_id - arena.first_page_id)
        );
    }
    confirm_materialization(&session.session, control_id);

    let continuation = prepare_hybrid(&session, target, 33);
    assert_eq!(
        continuation
            .classes
            .iter()
            .map(|class| class.class_id)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    let abort = OrbitKvSessionStepAbortEvidence {
        request_id: target,
        backend_unobserved: 1,
        reserved: 0,
    };
    assert_eq!(
        unsafe {
            orbitkv_session_abort_prepared(
                session.as_ptr(),
                continuation.batch,
                &abort,
                1,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
}
