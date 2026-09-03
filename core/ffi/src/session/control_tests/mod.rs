#![allow(
    clippy::borrow_as_ptr,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

use super::control::*;
use super::*;
use crate::{
    ORBITKV_TAIL_COPY_ON_WRITE, ORBITKV_TAIL_FRESH, OrbitKvPageLease, OrbitKvPrefixSemanticKey,
    OrbitKvSnapshotPage,
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
const POOL_ID: u32 = 41;
const BACKEND_DOMAIN: u16 = 7;
const BACKEND_BASE_INDEX: u64 = 100;
const PAGE_CAPACITY: u32 = 32;
const MAXIMUM_REQUESTS: u32 = 8;
const MAXIMUM_OPERATIONS: u32 = 8;
const MAXIMUM_PREFIXES: u32 = 4;

fn config() -> OrbitKvSessionCreateConfig {
    OrbitKvSessionCreateConfig {
        manager: OrbitKvManagerConfig {
            maximum_requests: MAXIMUM_REQUESTS,
            maximum_operations: MAXIMUM_OPERATIONS,
            maximum_prefixes: MAXIMUM_PREFIXES,
            maximum_reclamations: PAGE_CAPACITY,
            maximum_step_tokens: 16,
            plan_format: 1,
            reserved: 0,
        },
        cache_sharing_policy: ORBITKV_CACHE_SHARING_POLICY_SHARED_PREFIX,
        reserved: 0,
    }
}

fn backend() -> OrbitKvBackendArenaRegistration {
    OrbitKvBackendArenaRegistration {
        pool_id: POOL_ID,
        class_id: 0,
        backend_domain: BACKEND_DOMAIN,
        page_count: PAGE_CAPACITY,
        reserved: 0,
        backend_base_index: BACKEND_BASE_INDEX,
    }
}

struct Session(*mut OrbitKvSessionHandle);

impl Session {
    fn new() -> Self {
        let mut handle = std::ptr::null_mut();
        let mut error = [0; 256];
        let backend = backend();
        assert_eq!(
            unsafe {
                orbitkv_session_create(
                    PLAN.as_ptr(),
                    PLAN.len(),
                    &config(),
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

    const fn as_ptr(&self) -> *mut OrbitKvSessionHandle {
        self.0
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            orbitkv_session_destroy(self.0, std::ptr::null_mut(), 0);
        }
    }
}

fn key(tag: u8, boundary: u64) -> OrbitKvPrefixSemanticKey {
    OrbitKvPrefixSemanticKey {
        namespace: [0xA5; 32],
        digest: [tag; 32],
        boundary,
    }
}

fn acquire(session: &Session, request_ids: &[u64]) -> Vec<OrbitKvSessionRequestView> {
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
    views
}

struct PreparedAppend {
    batch: OrbitKvSessionBatchId,
    step: OrbitKvSessionPreparedStep,
    classes: Vec<OrbitKvClassLowering>,
    tails: Vec<OrbitKvTailAction>,
    copies: Vec<OrbitKvCopyIntent>,
    writes: Vec<OrbitKvWriteIntent>,
}

fn prepare_append(session: &Session, request_id: u64, target_boundary: u64) -> PreparedAppend {
    let intent = OrbitKvSessionAppendIntent {
        request_id,
        target_boundary,
    };
    let mut batch = OrbitKvSessionBatchId::default();
    let mut step = OrbitKvSessionPreparedStep::default();
    let mut classes = vec![OrbitKvClassLowering::default(); 1];
    let mut tails = vec![OrbitKvTailAction::default(); 1];
    let mut copies = vec![OrbitKvCopyIntent::default(); 1];
    let mut writes = vec![OrbitKvWriteIntent::default(); 1];
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
    PreparedAppend {
        batch,
        step,
        classes,
        tails,
        copies,
        writes,
    }
}

fn submit_append(session: &Session, prepared: &PreparedAppend) {
    let mut binds = Vec::new();
    for class in &prepared.classes {
        for tail in &prepared.tails
            [class.tail_offset as usize..(class.tail_offset + class.tail_count) as usize]
        {
            if tail.kind == ORBITKV_TAIL_COPY_ON_WRITE || tail.kind == ORBITKV_TAIL_FRESH {
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
        for write in &prepared.writes
            [class.write_offset as usize..(class.write_offset + class.write_count) as usize]
        {
            binds.push(OrbitKvSessionBindEvidence {
                page: OrbitKvPageLease {
                    engine_epoch: prepared.batch.session_epoch,
                    pool_epoch: prepared.batch.session_epoch + 1,
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
    let step = OrbitKvSessionStepExecutionEvidence {
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
                &step,
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

fn complete_append(session: &Session, prepared: &PreparedAppend) {
    let mut publication_id = OrbitKvSessionPublicationId::default();
    let mut step = OrbitKvSessionStepPublication::default();
    let mut detached = vec![OrbitKvDetachedBinding::default(); PAGE_CAPACITY as usize];
    let mut retirements = vec![OrbitKvSessionRetirement::default(); PAGE_CAPACITY as usize];
    let (mut step_count, mut detached_count, mut retirement_count) = (0, 0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_complete_execution(
                session.as_ptr(),
                prepared.batch,
                OrbitKvSessionCompletionEvidence {
                    completion_domain: 91,
                    completion_value: 1,
                    confirmed: 1,
                    reserved: 0,
                },
                &mut publication_id,
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

fn make_ready(session: &Session, request_id: u64) {
    acquire(session, &[request_id]);
    let prepared = prepare_append(session, request_id, 16);
    submit_append(session, &prepared);
    complete_append(session, &prepared);
}

fn lookup_prefix(
    session: &Session,
    semantic_key: OrbitKvPrefixSemanticKey,
) -> OrbitKvSessionPrefixLookup {
    let mut lookup = OrbitKvSessionPrefixLookup::default();
    let mut count = 0;
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_lookup_batch(
                session.as_ptr(),
                &semantic_key,
                1,
                &mut lookup,
                1,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(count, 1);
    lookup
}

fn publish_prefix(
    session: &Session,
    request_id: u64,
    semantic_key: OrbitKvPrefixSemanticKey,
) -> OrbitKvSessionPublishedPrefix {
    let item = OrbitKvSessionPrefixPublishItem {
        request_id,
        key: semantic_key,
    };
    let mut published = OrbitKvSessionPublishedPrefix::default();
    let mut count = 0;
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_publish_batch(
                session.as_ptr(),
                &item,
                1,
                &mut published,
                1,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(count, 1);
    published
}

fn release_request(session: &Session, request_id: u64) {
    let mut release_id = OrbitKvSessionReleaseId::default();
    let mut release = OrbitKvSessionReleasedRequest::default();
    let mut detached = [OrbitKvDetachedBinding::default(); 1];
    let mut retirements = [OrbitKvSessionRetirement::default(); 1];
    let (mut release_count, mut detached_count, mut retirement_count) = (0, 0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_release(
                session.as_ptr(),
                &request_id,
                1,
                &mut release_id,
                &mut release,
                1,
                &mut release_count,
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
    assert_eq!(release_count, 1);
    let evidence = retirement_evidence(&retirements[..retirement_count as usize]);
    let mut outcome = OrbitKvSessionReleaseOutcome::default();
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_release(
                session.as_ptr(),
                OrbitKvSessionReleaseEvidence {
                    release_id,
                    mirror_cleanup_confirmed: 1,
                    reserved: 0,
                },
                evidence.as_ptr(),
                evidence.len() as u32,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(outcome.disposition, ORBITKV_SESSION_RELEASE_COMPLETED);
}

fn stats(session: &Session) -> OrbitKvManagerStats {
    let mut stats = OrbitKvManagerStats::default();
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_stats(
                session.as_ptr(),
                &mut stats,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    stats
}

fn prepare_fork(
    session: &Session,
    source_request_id: u64,
    target_request_id: u64,
) -> OrbitKvSessionControlId {
    let item = OrbitKvSessionRequestForkItem {
        source_request_id,
        target_request_id,
    };
    let mut control_id = OrbitKvSessionControlId::default();
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_request_fork(
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
    control_id
}

fn prepare_evict(session: &Session, prefix_id: OrbitKvSessionPrefixId) -> OrbitKvSessionControlId {
    let mut control_id = OrbitKvSessionControlId::default();
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_prefix_evict(
                session.as_ptr(),
                &prefix_id,
                1,
                &mut control_id,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    control_id
}

fn commit(session: &Session, control_id: OrbitKvSessionControlId) -> OrbitKvSessionControlPlanInfo {
    let mut info = OrbitKvSessionControlPlanInfo::default();
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_commit_control(
                session.as_ptr(),
                control_id,
                &mut info,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    info
}

fn read_materialization_plan(
    session: &Session,
    control_id: OrbitKvSessionControlId,
) -> (
    Vec<OrbitKvSessionMaterializedRequest>,
    Vec<OrbitKvSnapshotPage>,
) {
    let (mut request_count, mut page_count, mut prefix_count, mut retirement_count) = (0, 0, 0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_read_control_plan(
                session.as_ptr(),
                control_id,
                std::ptr::null_mut(),
                0,
                &mut request_count,
                std::ptr::null_mut(),
                0,
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
    assert_eq!((prefix_count, retirement_count), (0, 0));
    let mut requests = vec![OrbitKvSessionMaterializedRequest::default(); request_count as usize];
    let mut pages = vec![OrbitKvSnapshotPage::default(); page_count as usize];
    assert_eq!(
        unsafe {
            orbitkv_session_read_control_plan(
                session.as_ptr(),
                control_id,
                requests.as_mut_ptr(),
                requests.len() as u32,
                &mut request_count,
                pages.as_mut_ptr(),
                pages.len() as u32,
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
    requests.truncate(request_count as usize);
    pages.truncate(page_count as usize);
    (requests, pages)
}

fn read_eviction_plan(
    session: &Session,
    control_id: OrbitKvSessionControlId,
) -> (Vec<OrbitKvSessionPrefixId>, Vec<OrbitKvSessionRetirement>) {
    let (mut request_count, mut page_count, mut prefix_count, mut retirement_count) = (0, 0, 0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_read_control_plan(
                session.as_ptr(),
                control_id,
                std::ptr::null_mut(),
                0,
                &mut request_count,
                std::ptr::null_mut(),
                0,
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
    assert_eq!((request_count, page_count), (0, 0));
    let mut prefixes = vec![OrbitKvSessionPrefixId::default(); prefix_count as usize];
    let mut retirements = vec![OrbitKvSessionRetirement::default(); retirement_count as usize];
    assert_eq!(
        unsafe {
            orbitkv_session_read_control_plan(
                session.as_ptr(),
                control_id,
                std::ptr::null_mut(),
                0,
                &mut request_count,
                std::ptr::null_mut(),
                0,
                &mut page_count,
                prefixes.as_mut_ptr(),
                prefixes.len() as u32,
                &mut prefix_count,
                retirements.as_mut_ptr(),
                retirements.len() as u32,
                &mut retirement_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    (prefixes, retirements)
}

fn confirm_materialization(session: &Session, control_id: OrbitKvSessionControlId) {
    let mut outcome = OrbitKvSessionControlOutcome::default();
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_control(
                session.as_ptr(),
                OrbitKvSessionControlEvidence {
                    id: control_id,
                    mirror_updates_confirmed: 1,
                    reserved: 0,
                },
                std::ptr::null(),
                0,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(outcome.id, control_id);
    assert_eq!(
        outcome.disposition,
        ORBITKV_SESSION_CONTROL_OUTCOME_MATERIALIZED
    );
    assert_eq!(outcome.reserved, 0);
}

mod full_sliding_prefix;
mod lifecycle;
mod validation;
