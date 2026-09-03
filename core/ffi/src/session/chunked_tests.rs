#![allow(clippy::borrow_as_ptr, clippy::cast_possible_truncation)]

use super::*;
use crate::{
    ORBITKV_CLASS_LOWERING_EPOCH_START, ORBITKV_CLASS_LOWERING_RESETTABLE, ORBITKV_TAIL_IN_PLACE,
    ORBITKV_TAIL_NONE, OrbitKvPageLease, OrbitKvPrefixSemanticKey,
};

const CHUNKED_PLAN: &[u8] = br#"{
  "schema": "orbitkv.retention-ir.v1",
  "page_tokens": 16,
  "states": [{
    "name": "chunked",
    "layers": [0],
    "bytes_per_token_per_layer": 128,
    "may_read": {
      "op": "equal",
      "lhs": {"op": "floor_div", "value": {"op": "query_position"}, "divisor": 32},
      "rhs": {"op": "floor_div", "value": {"op": "key_position"}, "divisor": 32}
    }
  }]
}"#;

const POOL_ID: u32 = 131;
const BACKEND_DOMAIN: u16 = 13;
const BACKEND_BASE_INDEX: u64 = 80_000;

fn chunked_config(page_count: u32, maximum_requests: u32) -> OrbitKvSessionCreateConfig {
    OrbitKvSessionCreateConfig {
        manager: OrbitKvManagerConfig {
            maximum_requests,
            maximum_operations: 8,
            maximum_prefixes: 2,
            maximum_reclamations: page_count,
            maximum_step_tokens: 32,
            plan_format: 2,
            reserved: 0,
        },
        cache_sharing_policy: ORBITKV_CACHE_SHARING_POLICY_REQUEST_PRIVATE,
        reserved: 0,
    }
}

fn chunked_backend(page_count: u32) -> OrbitKvBackendArenaRegistration {
    OrbitKvBackendArenaRegistration {
        pool_id: POOL_ID,
        class_id: 0,
        backend_domain: BACKEND_DOMAIN,
        page_count,
        reserved: 0,
        backend_base_index: BACKEND_BASE_INDEX,
    }
}

struct ChunkedSession {
    handle: *mut OrbitKvSessionHandle,
    arena: OrbitKvArenaIdentity,
}

impl ChunkedSession {
    fn new(page_count: u32, maximum_requests: u32) -> Self {
        let config = chunked_config(page_count, maximum_requests);
        let backend = chunked_backend(page_count);
        let mut handle = std::ptr::null_mut();
        let mut error = [0; 256];
        assert_eq!(
            unsafe {
                orbitkv_session_create(
                    CHUNKED_PLAN.as_ptr(),
                    CHUNKED_PLAN.len(),
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
        let mut arena = OrbitKvArenaIdentity::default();
        let mut count = 0;
        assert_eq!(
            unsafe {
                orbitkv_session_arena_identities(
                    handle,
                    &mut arena,
                    1,
                    &mut count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_OK
        );
        assert_eq!(count, 1);
        assert_eq!(
            (arena.pool_id, arena.backend_domain, arena.page_count),
            (POOL_ID, BACKEND_DOMAIN, page_count)
        );
        Self { handle, arena }
    }

    const fn as_ptr(&self) -> *mut OrbitKvSessionHandle {
        self.handle
    }
}

impl Drop for ChunkedSession {
    fn drop(&mut self) {
        unsafe {
            orbitkv_session_destroy(self.handle, std::ptr::null_mut(), 0);
        }
    }
}

#[derive(Debug)]
struct Prepared {
    batch: OrbitKvSessionBatchId,
    step: OrbitKvSessionPreparedStep,
    class: OrbitKvClassLowering,
    tails: Vec<OrbitKvTailAction>,
    copies: Vec<OrbitKvCopyIntent>,
    writes: Vec<OrbitKvWriteIntent>,
}

#[derive(Debug)]
struct Publication {
    id: OrbitKvSessionPublicationId,
    step: OrbitKvSessionStepPublication,
    detached: Vec<OrbitKvDetachedBinding>,
    retirements: Vec<OrbitKvSessionRetirement>,
}

fn acquire(session: &ChunkedSession, request_ids: &[u64]) {
    let mut views = vec![OrbitKvSessionRequestView::default(); request_ids.len()];
    let mut count = 0;
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_acquire_requests(
                session.as_ptr(),
                request_ids.as_ptr(),
                request_ids.len() as u32,
                views.as_mut_ptr(),
                views.len() as u32,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(count as usize, request_ids.len());
}

fn prepare(session: &ChunkedSession, request_id: u64, target_boundary: u64) -> Prepared {
    let intent = OrbitKvSessionAppendIntent {
        request_id,
        target_boundary,
    };
    let mut batch = OrbitKvSessionBatchId::default();
    let mut step = OrbitKvSessionPreparedStep::default();
    let mut class = OrbitKvClassLowering::default();
    let mut tails = [OrbitKvTailAction::default(); 1];
    let mut copies = [OrbitKvCopyIntent::default(); 1];
    let mut writes = [OrbitKvWriteIntent::default(); 2];
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
                &mut class,
                1,
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
    assert_eq!((step_count, class_count), (1, 1));
    Prepared {
        batch,
        step,
        class,
        tails: tails[..tail_count as usize].to_vec(),
        copies: copies[..copy_count as usize].to_vec(),
        writes: writes[..write_count as usize].to_vec(),
    }
}

fn submit(session: &ChunkedSession, prepared: &Prepared) {
    let backend_index = |page_id: u32| {
        session.arena.backend_base_index + u64::from(page_id - session.arena.first_page_id)
    };
    let mut binds = Vec::new();
    for tail in &prepared.tails {
        if matches!(tail.kind, 2 | 3) {
            binds.push(OrbitKvSessionBindEvidence {
                page: tail.destination,
                backend_domain: session.arena.backend_domain,
                mapped: 1,
                writable: 1,
                reserved: 0,
                backend_index: backend_index(tail.destination.page_id),
            });
        }
    }
    for write in &prepared.writes {
        binds.push(OrbitKvSessionBindEvidence {
            page: OrbitKvPageLease {
                engine_epoch: session.arena.engine_epoch,
                pool_epoch: session.arena.pool_epoch,
                generation: write.page_generation,
                page_id: write.page_id,
                pool_id: session.arena.pool_id,
            },
            backend_domain: session.arena.backend_domain,
            mapped: 1,
            writable: 1,
            reserved: 0,
            backend_index: backend_index(write.page_id),
        });
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

fn complete(session: &ChunkedSession, prepared: &Prepared, completion_value: u64) -> Publication {
    let mut id = OrbitKvSessionPublicationId::default();
    let mut step = OrbitKvSessionStepPublication::default();
    let mut detached = [OrbitKvDetachedBinding::default(); 4];
    let mut retirements = [OrbitKvSessionRetirement::default(); 4];
    let (mut step_count, mut detached_count, mut retirement_count) = (0, 0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_complete_execution(
                session.as_ptr(),
                prepared.batch,
                OrbitKvSessionCompletionEvidence {
                    completion_domain: 83,
                    completion_value,
                    confirmed: 1,
                    reserved: 0,
                },
                &mut id,
                &mut step,
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
    assert_eq!(step_count, 1);
    Publication {
        id,
        step,
        detached: detached[..detached_count as usize].to_vec(),
        retirements: retirements[..retirement_count as usize].to_vec(),
    }
}

fn evidence(retirements: &[OrbitKvSessionRetirement]) -> Vec<OrbitKvSessionRetirementEvidence> {
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

fn confirm(session: &ChunkedSession, publication: &Publication) -> i32 {
    let evidence = evidence(&publication.retirements);
    let mut error = [0; 256];
    unsafe {
        orbitkv_session_confirm_publication(
            session.as_ptr(),
            OrbitKvSessionPublicationEvidence {
                publication_id: publication.id,
                mirror_cleanup_confirmed: 1,
                reserved: 0,
            },
            evidence.as_ptr(),
            evidence.len() as u32,
            error.as_mut_ptr(),
            error.len(),
        )
    }
}

fn stats(session: &ChunkedSession) -> OrbitKvManagerStats {
    let mut value = OrbitKvManagerStats::default();
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_stats(
                session.as_ptr(),
                &mut value,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    value
}

fn prepare_status(
    session: &ChunkedSession,
    intents: &[OrbitKvSessionAppendIntent],
) -> (i32, OrbitKvSessionBatchId) {
    let mut batch = OrbitKvSessionBatchId {
        session_epoch: u64::MAX,
        sequence: u64::MAX,
    };
    let mut steps = vec![OrbitKvSessionPreparedStep::default(); intents.len()];
    let mut classes = vec![OrbitKvClassLowering::default(); intents.len()];
    let mut tails = vec![OrbitKvTailAction::default(); intents.len()];
    let mut copies = vec![OrbitKvCopyIntent::default(); intents.len()];
    let mut writes = vec![OrbitKvWriteIntent::default(); intents.len() * 2];
    let (mut step_count, mut class_count, mut tail_count, mut copy_count, mut write_count) =
        (0, 0, 0, 0, 0);
    let mut error = [0; 256];
    let status = unsafe {
        orbitkv_session_prepare_append(
            session.as_ptr(),
            intents.as_ptr(),
            intents.len() as u32,
            &mut batch,
            steps.as_mut_ptr(),
            steps.len() as u32,
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
    };
    (status, batch)
}

#[test]
#[allow(clippy::too_many_lines)]
fn exact_chunked_wire_epoch_lifecycle_ack_and_reuse() {
    let session = ChunkedSession::new(2, 2);
    let request = 81;
    let competing = 82;
    acquire(&session, &[request, competing]);

    let to_31 = prepare(&session, request, 31);
    assert_eq!(
        to_31.class.flags,
        ORBITKV_CLASS_LOWERING_RESETTABLE | ORBITKV_CLASS_LOWERING_EPOCH_START
    );
    assert_eq!(
        (
            to_31.class.previous_layout_boundary,
            to_31.class.target_layout_boundary,
            to_31.class.write_count,
        ),
        (0, 31, 2)
    );
    submit(&session, &to_31);
    let at_31 = complete(&session, &to_31, 1);
    assert_eq!((at_31.step.boundary, at_31.step.resident_count), (31, 2));
    assert!(at_31.detached.is_empty());
    assert!(at_31.retirements.is_empty());
    assert_eq!(confirm(&session, &at_31), ORBITKV_STATUS_OK);

    let to_32 = prepare(&session, request, 32);
    assert_eq!(to_32.class.flags, ORBITKV_CLASS_LOWERING_RESETTABLE);
    assert_eq!(
        (
            to_32.class.previous_layout_boundary,
            to_32.class.target_layout_boundary,
        ),
        (31, 32)
    );
    assert_eq!(to_32.tails[0].kind, ORBITKV_TAIL_IN_PLACE);
    assert!(to_32.writes.is_empty());
    submit(&session, &to_32);
    let at_32 = complete(&session, &to_32, 2);
    assert_eq!((at_32.step.boundary, at_32.step.resident_count), (32, 0));
    assert_eq!(at_32.detached.len(), 2);
    assert_eq!(at_32.retirements.len(), 2);
    for (ordinal, (detached, retirement)) in at_32
        .detached
        .iter()
        .zip(at_32.retirements.iter())
        .enumerate()
    {
        let ordinal = u64::try_from(ordinal).expect("retirement ordinal fits u64");
        assert_eq!(detached.action, 1);
        assert_eq!(detached.reason, 1);
        assert_eq!((detached.class_id, detached.logical_ordinal), (0, ordinal));
        assert_eq!(detached.old, retirement.page);
        assert_eq!(detached.old_backend_index, retirement.backend_index);
        assert_eq!(
            (detached.token_begin, detached.token_end_exclusive),
            (ordinal * 16, (ordinal + 1) * 16)
        );
        assert_eq!(
            (retirement.class_id, retirement.logical_ordinal),
            (0, ordinal)
        );
        assert_eq!(
            (retirement.completion_domain, retirement.completion_value),
            (83, 2)
        );
    }

    let pending = stats(&session);
    assert_eq!((pending.free_pages, pending.retiring_pages), (0, 2));
    let mut error = [0; 256];
    let intent = OrbitKvSessionAppendIntent {
        request_id: competing,
        target_boundary: 1,
    };
    let mut batch = OrbitKvSessionBatchId::default();
    let mut step = OrbitKvSessionPreparedStep::default();
    let mut class = OrbitKvClassLowering::default();
    let mut tail = OrbitKvTailAction::default();
    let mut copy = OrbitKvCopyIntent::default();
    let mut writes = [OrbitKvWriteIntent::default(); 2];
    let (mut step_count, mut class_count, mut tail_count, mut copy_count, mut write_count) =
        (0, 0, 0, 0, 0);
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
                &mut class,
                1,
                &mut class_count,
                &mut tail,
                1,
                &mut tail_count,
                &mut copy,
                1,
                &mut copy_count,
                writes.as_mut_ptr(),
                2,
                &mut write_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_MANAGER_ERROR
    );
    assert_eq!(batch, OrbitKvSessionBatchId::default());
    assert_eq!(stats(&session), pending);

    let exact = evidence(&at_32.retirements);
    let mut forged = exact.clone();
    forged[1].backend_index += 1;
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_publication(
                session.as_ptr(),
                OrbitKvSessionPublicationEvidence {
                    publication_id: at_32.id,
                    mirror_cleanup_confirmed: 1,
                    reserved: 0,
                },
                forged.as_ptr(),
                forged.len() as u32,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(stats(&session), pending);
    assert_eq!(confirm(&session, &at_32), ORBITKV_STATUS_OK);

    let to_33 = prepare(&session, request, 33);
    assert_eq!(
        to_33.class.flags,
        ORBITKV_CLASS_LOWERING_RESETTABLE | ORBITKV_CLASS_LOWERING_EPOCH_START
    );
    assert_eq!(
        (
            to_33.class.previous_layout_boundary,
            to_33.class.target_layout_boundary,
        ),
        (32, 33)
    );
    assert_eq!(to_33.tails[0].kind, ORBITKV_TAIL_NONE);
    assert_eq!(to_33.writes.len(), 1);
    let reused = to_33.writes[0];
    let retired = at_32
        .retirements
        .iter()
        .find(|retirement| retirement.page.page_id == reused.page_id)
        .expect("retired page reused");
    assert_eq!(reused.page_generation, retired.page.generation + 1);
    submit(&session, &to_33);
    let at_33 = complete(&session, &to_33, 3);
    assert_eq!((at_33.step.boundary, at_33.step.resident_count), (33, 1));
    assert!(at_33.detached.is_empty());
    assert!(at_33.retirements.is_empty());
    assert_eq!(confirm(&session, &at_33), ORBITKV_STATUS_OK);
}

#[test]
fn chunked_wire_cross_boundary_is_atomic() {
    let session = ChunkedSession::new(8, 1);
    let source = 91;
    acquire(&session, &[source]);
    let before = stats(&session);
    let crossing = OrbitKvSessionAppendIntent {
        request_id: source,
        target_boundary: 33,
    };
    let (status, batch) = prepare_status(&session, &[crossing]);
    assert_eq!(status, ORBITKV_STATUS_MANAGER_ERROR);
    assert_eq!(batch, OrbitKvSessionBatchId::default());
    assert_eq!(stats(&session), before);
}

#[test]
fn chunked_wire_prefix_lookup_and_publish_fail_closed() {
    let session = ChunkedSession::new(8, 2);
    let source = 91;
    acquire(&session, &[source]);
    let mut error = [0; 256];
    let before = stats(&session);
    let key = OrbitKvPrefixSemanticKey {
        namespace: [0xCE; 32],
        digest: [0xCF; 32],
        boundary: 32,
    };
    let mut lookup = OrbitKvSessionPrefixLookup::default();
    let mut output_count = u32::MAX;
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_lookup_batch(
                session.as_ptr(),
                &key,
                1,
                &mut lookup,
                1,
                &mut output_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(stats(&session), before);

    let publish = OrbitKvSessionPrefixPublishItem {
        request_id: source,
        key,
    };
    let mut published = OrbitKvSessionPublishedPrefix::default();
    output_count = u32::MAX;
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_publish_batch(
                session.as_ptr(),
                &publish,
                1,
                &mut published,
                1,
                &mut output_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(stats(&session), before);
}

#[test]
fn chunked_wire_prefix_control_and_transfer_fail_closed() {
    let session = ChunkedSession::new(8, 2);
    let source = 91;
    let target = 92;
    acquire(&session, &[source, target]);
    let mut error = [0; 256];
    let before = stats(&session);
    let key = OrbitKvPrefixSemanticKey {
        namespace: [0xCE; 32],
        digest: [0xCF; 32],
        boundary: 32,
    };
    let publish = OrbitKvSessionPrefixPublishItem {
        request_id: source,
        key,
    };
    let fake_prefix = OrbitKvSessionPrefixId {
        session_epoch: session.arena.engine_epoch,
        sequence: 1,
    };
    let attach = OrbitKvSessionPrefixAttachItem {
        target_request_id: target,
        prefix_id: fake_prefix,
        key,
        resident_count: 0,
        reserved: 0,
    };
    let mut control_id = OrbitKvSessionControlId::default();
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_prefix_attach(
                session.as_ptr(),
                &attach,
                1,
                &mut control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(control_id, OrbitKvSessionControlId::default());
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_prefix_evict(
                session.as_ptr(),
                &fake_prefix,
                1,
                &mut control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );

    let mut release_id = OrbitKvSessionReleaseId::default();
    let mut transferred = OrbitKvSessionPublishedPrefixRelease::default();
    let mut detached = [OrbitKvDetachedBinding::default(); 8];
    let mut detached_count = 0;
    let mut output_count = 0;
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_publish_release_batch(
                session.as_ptr(),
                &publish,
                1,
                &mut release_id,
                &mut transferred,
                1,
                &mut output_count,
                detached.as_mut_ptr(),
                detached.len() as u32,
                &mut detached_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(release_id, OrbitKvSessionReleaseId::default());
    assert_eq!(stats(&session), before);
}
