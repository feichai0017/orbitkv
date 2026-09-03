#![allow(clippy::borrow_as_ptr, clippy::cast_possible_truncation)]

use super::*;
use crate::{
    ORBITKV_PLAN_FORMAT_KV_PLAN, ORBITKV_STATUS_INVALID_ARGUMENT, ORBITKV_STATUS_MANAGER_ERROR,
    OrbitKvClassTokenDispositionUpdate, OrbitKvPageLease, OrbitKvRelocationPolicy,
    OrbitKvTokenDisposition, OrbitKvTokenMove, OrbitKvTokenPlacement,
};

const PLAN: &[u8] = br#"{
  "page_tokens": 16,
  "classes": [{
    "name": "full",
    "layers": [0],
    "retention": "full",
    "bytes_per_token_per_layer": 128
  }]
}"#;
const POOL_ID: u32 = 501;
const BACKEND_DOMAIN: u16 = 41;
const BACKEND_BASE_INDEX: u64 = 50_000;
const PAGE_CAPACITY: u32 = 16;
const REQUEST_ID: u64 = 7_001;

struct Session(*mut OrbitKvSessionHandle);

impl Session {
    fn new() -> Self {
        let config = OrbitKvSessionCreateConfig {
            manager: OrbitKvManagerConfig {
                maximum_requests: 2,
                maximum_operations: 4,
                maximum_prefixes: 1,
                maximum_reclamations: PAGE_CAPACITY,
                maximum_step_tokens: 64,
                plan_format: ORBITKV_PLAN_FORMAT_KV_PLAN,
                reserved: 0,
            },
            cache_sharing_policy: ORBITKV_CACHE_SHARING_POLICY_REQUEST_PRIVATE,
            reserved: 0,
        };
        let backend = OrbitKvBackendArenaRegistration {
            pool_id: POOL_ID,
            class_id: 0,
            backend_domain: BACKEND_DOMAIN,
            page_count: PAGE_CAPACITY,
            reserved: 0,
            backend_base_index: BACKEND_BASE_INDEX,
        };
        let mut handle = std::ptr::null_mut();
        let mut error = [0; 256];
        assert_eq!(
            unsafe {
                orbitkv_session_create(
                    PLAN.as_ptr(),
                    PLAN.len(),
                    &config,
                    &backend,
                    1,
                    &mut handle,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert!(!handle.is_null());
        Self(handle)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        assert_eq!(
            unsafe { orbitkv_session_destroy(self.0, std::ptr::null_mut(), 0) },
            ORBITKV_STATUS_OK
        );
    }
}

#[derive(Debug)]
struct PreparedAppend {
    id: OrbitKvSessionBatchId,
    step: OrbitKvSessionPreparedStep,
    classes: Vec<OrbitKvClassLowering>,
    tails: Vec<OrbitKvTailAction>,
    copies: Vec<OrbitKvCopyIntent>,
    writes: Vec<OrbitKvWriteIntent>,
}

fn acquire(session: &Session, request_id: u64) {
    let mut output = OrbitKvSessionRequestView::default();
    let mut count = 0;
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_acquire_requests(
                session.0,
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
    assert_eq!(count, 1);
}

fn prepare_append(session: &Session, request_id: u64, boundary: u64) -> PreparedAppend {
    let intent = OrbitKvSessionAppendIntent {
        request_id,
        target_boundary: boundary,
    };
    let mut id = OrbitKvSessionBatchId::default();
    let mut steps = [OrbitKvSessionPreparedStep::default(); 1];
    let mut classes = [OrbitKvClassLowering::default(); 1];
    let mut tails = [OrbitKvTailAction::default(); 1];
    let mut copies = [OrbitKvCopyIntent::default(); 1];
    let mut writes = [OrbitKvWriteIntent::default(); 4];
    let (mut step_count, mut class_count, mut tail_count, mut copy_count, mut write_count) =
        (0, 0, 0, 0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_append(
                session.0,
                &intent,
                1,
                &mut id,
                steps.as_mut_ptr(),
                1,
                &mut step_count,
                classes.as_mut_ptr(),
                1,
                &mut class_count,
                tails.as_mut_ptr(),
                1,
                &mut tail_count,
                copies.as_mut_ptr(),
                1,
                &mut copy_count,
                writes.as_mut_ptr(),
                4,
                &mut write_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(step_count, 1);
    PreparedAppend {
        id,
        step: steps[0],
        classes: classes[..class_count as usize].to_vec(),
        tails: tails[..tail_count as usize].to_vec(),
        copies: copies[..copy_count as usize].to_vec(),
        writes: writes[..write_count as usize].to_vec(),
    }
}

fn submit_append(session: &Session, prepared: &PreparedAppend) {
    let mut binds = Vec::new();
    for class in &prepared.classes {
        let tail_begin = class.tail_offset as usize;
        let tail_end = tail_begin + class.tail_count as usize;
        for tail in &prepared.tails[tail_begin..tail_end] {
            if tail.kind == crate::ORBITKV_TAIL_COPY_ON_WRITE
                || tail.kind == crate::ORBITKV_TAIL_FRESH
            {
                binds.push(OrbitKvSessionBindEvidence {
                    page: tail.destination,
                    backend_domain: BACKEND_DOMAIN,
                    mapped: 1,
                    writable: 1,
                    reserved: 0,
                    backend_index: BACKEND_BASE_INDEX + u64::from(tail.destination.page_id - 1),
                });
            }
        }
        let write_begin = class.write_offset as usize;
        let write_end = write_begin + class.write_count as usize;
        for write in &prepared.writes[write_begin..write_end] {
            binds.push(OrbitKvSessionBindEvidence {
                page: OrbitKvPageLease {
                    engine_epoch: prepared.id.session_epoch,
                    pool_epoch: prepared.id.session_epoch + 1,
                    generation: write.page_generation,
                    page_id: write.page_id,
                    pool_id: POOL_ID,
                },
                backend_domain: BACKEND_DOMAIN,
                mapped: 1,
                writable: 1,
                reserved: 0,
                backend_index: BACKEND_BASE_INDEX + u64::from(write.page_id - 1),
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
                session.0,
                prepared.id,
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

fn retirement_evidence(
    retirements: &[OrbitKvSessionRetirement],
) -> Vec<OrbitKvSessionRetirementEvidence> {
    retirements
        .iter()
        .map(|retirement| OrbitKvSessionRetirementEvidence {
            page: retirement.page,
            backend_domain: retirement.backend_domain,
            acknowledged: 1,
            reserved8: 0,
            reserved32: 0,
            backend_index: retirement.backend_index,
        })
        .collect()
}

fn append_and_confirm(session: &Session, request_id: u64, boundary: u64) {
    let prepared = prepare_append(session, request_id, boundary);
    submit_append(session, &prepared);
    let mut publication_id = OrbitKvSessionPublicationId::default();
    let mut publications = [OrbitKvSessionStepPublication::default(); 1];
    let mut detached = [OrbitKvDetachedBinding::default(); PAGE_CAPACITY as usize];
    let mut retirements = [OrbitKvSessionRetirement::default(); PAGE_CAPACITY as usize];
    let (mut publication_count, mut detached_count, mut retirement_count) = (0, 0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_complete_execution(
                session.0,
                prepared.id,
                OrbitKvSessionCompletionEvidence {
                    completion_domain: 91,
                    completion_value: 1,
                    confirmed: 1,
                    reserved: 0,
                },
                &mut publication_id,
                publications.as_mut_ptr(),
                1,
                &mut publication_count,
                detached.as_mut_ptr(),
                PAGE_CAPACITY,
                &mut detached_count,
                retirements.as_mut_ptr(),
                PAGE_CAPACITY,
                &mut retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    let evidence = retirement_evidence(&retirements[..retirement_count as usize]);
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_publication(
                session.0,
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

fn read_tokens(
    session: &Session,
    request_id: u64,
    expected_boundary: u64,
) -> (OrbitKvSessionTokenView, Vec<OrbitKvTokenPlacement>) {
    let query = OrbitKvSessionTokenViewQuery {
        request_id,
        expected_boundary,
        class_id: 0,
        reserved16: 0,
        reserved32: 0,
    };
    let (mut view_count, mut placement_count) = (0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_token_views_batch(
                session.0,
                &query,
                1,
                std::ptr::null_mut(),
                0,
                &mut view_count,
                std::ptr::null_mut(),
                0,
                &mut placement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!((view_count, placement_count), (1, expected_boundary as u32));
    let mut view = OrbitKvSessionTokenView::default();
    let mut placements = vec![OrbitKvTokenPlacement::default(); placement_count as usize];
    assert_eq!(
        unsafe {
            orbitkv_session_token_views_batch(
                session.0,
                &query,
                1,
                &mut view,
                1,
                &mut view_count,
                placements.as_mut_ptr(),
                placement_count,
                &mut placement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(view.placement_count, expected_boundary as u32);
    (view, placements)
}

fn mark_victims(session: &Session, request_id: u64) -> OrbitKvSessionRequestView {
    let updates = (8..16_u64)
        .chain(24..32)
        .chain(40..48)
        .map(|token_id| OrbitKvClassTokenDispositionUpdate {
            token_id,
            disposition: OrbitKvTokenDisposition {
                policy_or_proof_id: 17,
                version: 1,
                quality_contract: 99,
                kind: 2,
                reserved16: 0,
                reserved32: 0,
            },
            class_id: 0,
            reserved16: 0,
            reserved32: 0,
        })
        .collect::<Vec<_>>();
    let item = OrbitKvSessionTokenDispositionBatchItem {
        request_id,
        update_offset: 0,
        update_count: updates.len() as u32,
    };
    let mut output = OrbitKvSessionRequestView::default();
    let mut output_count = 0;
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_mark_token_dispositions_batch(
                session.0,
                &item,
                1,
                updates.as_ptr(),
                updates.len() as u32,
                &mut output,
                1,
                &mut output_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(output_count, 1);
    output
}

#[derive(Debug)]
struct PreparedRelocation {
    id: OrbitKvSessionRelocationId,
    plan: OrbitKvSessionRelocationPlan,
    sources: Vec<OrbitKvPageLease>,
    moves: Vec<OrbitKvTokenMove>,
}

fn prepare_relocation(session: &Session, request_id: u64) -> PreparedRelocation {
    let item = OrbitKvSessionPrepareRelocationItem {
        request_id,
        policy: OrbitKvRelocationPolicy {
            maximum_source_pages: 8,
            evacuation_headroom_pages: 2,
            fragmentation_threshold_milli: 250,
            full_evacuation: 1,
            reserved8: 0,
            reserved32: 0,
        },
        class_id: 0,
        reserved16: 0,
        reserved32: 0,
    };
    let mut id = OrbitKvSessionRelocationId::default();
    let mut plans = [OrbitKvSessionRelocationPlan::default(); 1];
    let mut sources = [OrbitKvPageLease::default(); PAGE_CAPACITY as usize];
    let mut destinations = [OrbitKvPageLease::default(); PAGE_CAPACITY as usize];
    let mut moves = vec![OrbitKvTokenMove::default(); (PAGE_CAPACITY * 16) as usize];
    let (mut plan_count, mut source_count, mut destination_count, mut move_count) = (0, 0, 0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_relocation_batch(
                session.0,
                &item,
                1,
                &mut id,
                plans.as_mut_ptr(),
                1,
                &mut plan_count,
                sources.as_mut_ptr(),
                PAGE_CAPACITY,
                &mut source_count,
                destinations.as_mut_ptr(),
                PAGE_CAPACITY,
                &mut destination_count,
                moves.as_mut_ptr(),
                PAGE_CAPACITY * 16,
                &mut move_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(plan_count, 1);
    assert_eq!(plans[0].request_id, request_id);
    assert_eq!(plans[0].move_count, move_count);
    assert_eq!(plans[0].source_count, source_count);
    assert_eq!(plans[0].destination_count, destination_count);
    PreparedRelocation {
        id,
        plan: plans[0],
        sources: sources[..source_count as usize].to_vec(),
        moves: moves[..move_count as usize].to_vec(),
    }
}

fn submit_relocation(session: &Session, prepared: &PreparedRelocation, copied_flag: u8) -> i32 {
    let copies = prepared
        .moves
        .iter()
        .map(|movement| OrbitKvSessionRelocationCopyEvidence {
            token_id: movement.token_id,
            source: movement.source,
            destination: movement.destination,
            observed: 1,
            copied: copied_flag,
            reserved16: 0,
            reserved32: 0,
        })
        .collect::<Vec<_>>();
    let request = OrbitKvSessionRelocationRequestEvidence {
        request_id: prepared.plan.request_id,
        copy_offset: 0,
        copy_count: copies.len() as u32,
    };
    let mut error = [0; 256];
    unsafe {
        orbitkv_session_submit_relocation(
            session.0,
            prepared.id,
            &request,
            1,
            copies.as_ptr(),
            copies.len() as u32,
            error.as_mut_ptr(),
            error.len(),
        )
    }
}

fn prepare_relocatable(session: &Session) -> PreparedRelocation {
    acquire(session, REQUEST_ID);
    append_and_confirm(session, REQUEST_ID, 48);
    let (_, before) = read_tokens(session, REQUEST_ID, 48);
    assert_eq!(before.len(), 48);
    mark_victims(session, REQUEST_ID);
    prepare_relocation(session, REQUEST_ID)
}

#[test]
#[allow(clippy::too_many_lines)]
fn session_relocation_wire_is_bounded_and_ack_gated() {
    assert_eq!(crate::ORBITKV_WIRE_VERSION, 14);
    let session = Session::new();
    acquire(&session, REQUEST_ID);
    append_and_confirm(&session, REQUEST_ID, 48);

    let wrong_query = OrbitKvSessionTokenViewQuery {
        request_id: REQUEST_ID,
        expected_boundary: 47,
        class_id: 0,
        reserved16: 0,
        reserved32: 0,
    };
    let (mut view_count, mut placement_count) = (0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_token_views_batch(
                session.0,
                &wrong_query,
                1,
                std::ptr::null_mut(),
                0,
                &mut view_count,
                std::ptr::null_mut(),
                0,
                &mut placement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    read_tokens(&session, REQUEST_ID, 48);
    mark_victims(&session, REQUEST_ID);

    let item = OrbitKvSessionPrepareRelocationItem {
        request_id: REQUEST_ID,
        policy: OrbitKvRelocationPolicy {
            maximum_source_pages: 8,
            evacuation_headroom_pages: 2,
            fragmentation_threshold_milli: 250,
            full_evacuation: 1,
            reserved8: 0,
            reserved32: 0,
        },
        class_id: 0,
        reserved16: 0,
        reserved32: 0,
    };
    let mut id = OrbitKvSessionRelocationId {
        session_epoch: u64::MAX,
        sequence: u64::MAX,
    };
    let (mut plan_count, mut source_count, mut destination_count, mut move_count) = (0, 0, 0, 0);
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_relocation_batch(
                session.0,
                &item,
                1,
                &mut id,
                std::ptr::null_mut(),
                0,
                &mut plan_count,
                std::ptr::null_mut(),
                0,
                &mut source_count,
                std::ptr::null_mut(),
                0,
                &mut destination_count,
                std::ptr::null_mut(),
                0,
                &mut move_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!(id, OrbitKvSessionRelocationId::default());
    assert_eq!(
        (plan_count, source_count, destination_count, move_count),
        (1, PAGE_CAPACITY, PAGE_CAPACITY, PAGE_CAPACITY * 16)
    );

    let prepared = prepare_relocation(&session, REQUEST_ID);
    assert_eq!(
        (prepared.plan.source_count, prepared.plan.destination_count),
        (3, 2)
    );
    assert_eq!(prepared.plan.move_count, 24);
    assert_eq!(submit_relocation(&session, &prepared, 1), ORBITKV_STATUS_OK);

    let mut publications = [OrbitKvSessionRelocationRequestPublication::default(); 1];
    let mut retirements = [OrbitKvSessionRetirement::default(); PAGE_CAPACITY as usize];
    let (mut publication_count, mut retirement_count) = (0, 0);
    let completion = OrbitKvSessionCompletionEvidence {
        completion_domain: 91,
        completion_value: 2,
        confirmed: 1,
        reserved: 0,
    };
    assert_eq!(
        unsafe {
            orbitkv_session_complete_relocation(
                session.0,
                prepared.id,
                completion,
                std::ptr::null_mut(),
                0,
                &mut publication_count,
                std::ptr::null_mut(),
                0,
                &mut retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!((publication_count, retirement_count), (1, PAGE_CAPACITY));
    assert_eq!(
        unsafe {
            orbitkv_session_complete_relocation(
                session.0,
                prepared.id,
                OrbitKvSessionCompletionEvidence {
                    confirmed: 0,
                    ..completion
                },
                publications.as_mut_ptr(),
                1,
                &mut publication_count,
                retirements.as_mut_ptr(),
                PAGE_CAPACITY,
                &mut retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_MANAGER_ERROR
    );
    assert_eq!(
        unsafe {
            orbitkv_session_complete_relocation(
                session.0,
                prepared.id,
                completion,
                publications.as_mut_ptr(),
                1,
                &mut publication_count,
                retirements.as_mut_ptr(),
                PAGE_CAPACITY,
                &mut retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(publication_count, 1);
    assert_eq!(retirement_count, 3);
    let retirements = &retirements[..retirement_count as usize];
    let exact = retirement_evidence(retirements);
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_relocation_publication(
                session.0,
                OrbitKvSessionRelocationPublicationEvidence {
                    relocation_id: prepared.id,
                    mirror_cleanup_confirmed: 0,
                    reserved: 0,
                },
                exact.as_ptr(),
                exact.len() as u32,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    let mut wrong = exact.clone();
    wrong[0].backend_index += 1;
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_relocation_publication(
                session.0,
                OrbitKvSessionRelocationPublicationEvidence {
                    relocation_id: prepared.id,
                    mirror_cleanup_confirmed: 1,
                    reserved: 0,
                },
                wrong.as_ptr(),
                wrong.len() as u32,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_relocation_publication(
                session.0,
                OrbitKvSessionRelocationPublicationEvidence {
                    relocation_id: prepared.id,
                    mirror_cleanup_confirmed: 1,
                    reserved: 0,
                },
                exact.as_ptr(),
                exact.len() as u32,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    let (_, after) = read_tokens(&session, REQUEST_ID, 48);
    assert!(after.iter().all(|placement| {
        placement.location_present == 0 || !prepared.sources.contains(&placement.location.page)
    }));

    let next = prepare_append(&session, REQUEST_ID, 49);
    let abort = OrbitKvSessionStepAbortEvidence {
        request_id: REQUEST_ID,
        backend_unobserved: 1,
        reserved: 0,
    };
    assert_eq!(
        unsafe {
            orbitkv_session_abort_prepared(
                session.0,
                next.id,
                &abort,
                1,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
}

#[test]
fn relocation_abort_and_explicit_quarantine_are_fail_closed() {
    let session = Session::new();
    let prepared = prepare_relocatable(&session);
    let mut error = [0; 256];
    let mut evidence = OrbitKvSessionRelocationAbortEvidence {
        request_id: REQUEST_ID,
        backend_unobserved: 0,
        reserved: 0,
    };
    assert_eq!(
        unsafe {
            orbitkv_session_abort_prepared_relocation(
                session.0,
                prepared.id,
                &evidence,
                1,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_MANAGER_ERROR
    );
    evidence.backend_unobserved = 1;
    assert_eq!(
        unsafe {
            orbitkv_session_abort_prepared_relocation(
                session.0,
                prepared.id,
                &evidence,
                1,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    let prepared = prepare_relocation(&session, REQUEST_ID);
    assert_eq!(
        unsafe {
            orbitkv_session_quarantine_relocation(
                session.0,
                prepared.id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_FAIL_STOPPED
    );
    let request_id = REQUEST_ID + 1;
    let mut view = OrbitKvSessionRequestView::default();
    let mut count = 0;
    assert_eq!(
        unsafe {
            orbitkv_session_acquire_requests(
                session.0,
                &request_id,
                1,
                &mut view,
                1,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_FAIL_STOPPED
    );
    let mut stats = OrbitKvManagerStats::default();
    assert_eq!(
        unsafe { orbitkv_session_stats(session.0, &mut stats, error.as_mut_ptr(), error.len()) },
        ORBITKV_STATUS_OK
    );
}

#[test]
fn rejected_prepare_does_not_consume_an_opaque_relocation_id() {
    let session = Session::new();
    acquire(&session, REQUEST_ID);
    append_and_confirm(&session, REQUEST_ID, 48);
    mark_victims(&session, REQUEST_ID);
    let mut item = OrbitKvSessionPrepareRelocationItem {
        request_id: REQUEST_ID,
        policy: OrbitKvRelocationPolicy {
            maximum_source_pages: 0,
            evacuation_headroom_pages: 2,
            fragmentation_threshold_milli: 250,
            full_evacuation: 1,
            reserved8: 0,
            reserved32: 0,
        },
        class_id: 0,
        reserved16: 0,
        reserved32: 0,
    };
    let mut id = OrbitKvSessionRelocationId::default();
    let mut plans = [OrbitKvSessionRelocationPlan::default(); 1];
    let mut sources = [OrbitKvPageLease::default(); PAGE_CAPACITY as usize];
    let mut destinations = [OrbitKvPageLease::default(); PAGE_CAPACITY as usize];
    let mut moves = vec![OrbitKvTokenMove::default(); (PAGE_CAPACITY * 16) as usize];
    let (mut plan_count, mut source_count, mut destination_count, mut move_count) = (0, 0, 0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_relocation_batch(
                session.0,
                &item,
                1,
                &mut id,
                plans.as_mut_ptr(),
                1,
                &mut plan_count,
                sources.as_mut_ptr(),
                PAGE_CAPACITY,
                &mut source_count,
                destinations.as_mut_ptr(),
                PAGE_CAPACITY,
                &mut destination_count,
                moves.as_mut_ptr(),
                PAGE_CAPACITY * 16,
                &mut move_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_MANAGER_ERROR
    );
    assert_eq!(id, OrbitKvSessionRelocationId::default());

    item.policy.maximum_source_pages = 8;
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_relocation_batch(
                session.0,
                &item,
                1,
                &mut id,
                plans.as_mut_ptr(),
                1,
                &mut plan_count,
                sources.as_mut_ptr(),
                PAGE_CAPACITY,
                &mut source_count,
                destinations.as_mut_ptr(),
                PAGE_CAPACITY,
                &mut destination_count,
                moves.as_mut_ptr(),
                PAGE_CAPACITY * 16,
                &mut move_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(id.sequence, 1);
}

#[test]
fn semantic_copy_mismatch_sticky_fail_stops_the_wire_session() {
    let session = Session::new();
    let prepared = prepare_relocatable(&session);
    assert_eq!(
        submit_relocation(&session, &prepared, 0),
        ORBITKV_STATUS_FAIL_STOPPED
    );
    let query = OrbitKvSessionTokenViewQuery {
        request_id: REQUEST_ID,
        expected_boundary: 48,
        class_id: 0,
        reserved16: 0,
        reserved32: 0,
    };
    let (mut view_count, mut placement_count) = (0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_token_views_batch(
                session.0,
                &query,
                1,
                std::ptr::null_mut(),
                0,
                &mut view_count,
                std::ptr::null_mut(),
                0,
                &mut placement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_FAIL_STOPPED
    );
}
