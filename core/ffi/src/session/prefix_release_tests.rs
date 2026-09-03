#![allow(
    clippy::borrow_as_ptr,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

use super::*;
use orbitkv::kv_manager::{DetachedAction, DetachedReason};

use crate::{
    ORBITKV_TAIL_COPY_ON_WRITE, ORBITKV_TAIL_FRESH, OrbitKvPageLease, OrbitKvPrefixSemanticKey,
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
const POOL_ID: u32 = 71;
const BACKEND_DOMAIN: u16 = 17;
const BACKEND_BASE_INDEX: u64 = 10_000;
const PAGE_CAPACITY: u32 = 32;
const MAXIMUM_REQUESTS: u32 = 8;
const MAXIMUM_PREFIXES: u32 = 8;

fn config() -> OrbitKvSessionCreateConfig {
    OrbitKvSessionCreateConfig {
        manager: OrbitKvManagerConfig {
            maximum_requests: MAXIMUM_REQUESTS,
            maximum_operations: 8,
            maximum_prefixes: MAXIMUM_PREFIXES,
            maximum_reclamations: PAGE_CAPACITY,
            maximum_step_tokens: 32,
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

    fn into_raw(self) -> *mut OrbitKvSessionHandle {
        let raw = self.0;
        std::mem::forget(self);
        raw
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

fn key(tag: u8, boundary: u64) -> OrbitKvPrefixSemanticKey {
    OrbitKvPrefixSemanticKey {
        namespace: [0xC3; 32],
        digest: [tag; 32],
        boundary,
    }
}

fn stats(session: &Session) -> OrbitKvManagerStats {
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
    let mut writes = vec![OrbitKvWriteIntent::default(); 2];
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

fn complete_append(session: &Session, prepared: &PreparedAppend, completion_value: u64) {
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
                    completion_value,
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
    let evidence = retirements[..retirement_count as usize]
        .iter()
        .map(|retirement| OrbitKvSessionRetirementEvidence {
            page: retirement.page,
            backend_domain: retirement.backend_domain,
            acknowledged: 1,
            reserved8: 0,
            reserved32: 0,
            backend_index: retirement.backend_index,
        })
        .collect::<Vec<_>>();
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

fn make_ready(session: &Session, request_id: u64, boundary: u64, completion_value: u64) {
    acquire(session, &[request_id]);
    let prepared = prepare_append(session, request_id, boundary);
    submit_append(session, &prepared);
    complete_append(session, &prepared, completion_value);
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

fn publish_release(
    session: &Session,
    items: &[OrbitKvSessionPrefixPublishItem],
) -> (
    OrbitKvSessionReleaseId,
    Vec<OrbitKvSessionPublishedPrefixRelease>,
    Vec<OrbitKvDetachedBinding>,
) {
    let mut release_id = OrbitKvSessionReleaseId::default();
    let mut outputs = vec![OrbitKvSessionPublishedPrefixRelease::default(); items.len()];
    let mut detached = vec![OrbitKvDetachedBinding::default(); PAGE_CAPACITY as usize];
    let (mut output_count, mut detached_count) = (0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_publish_release_batch(
                session.as_ptr(),
                items.as_ptr(),
                items.len() as u32,
                &mut release_id,
                outputs.as_mut_ptr(),
                outputs.len() as u32,
                &mut output_count,
                detached.as_mut_ptr(),
                detached.len() as u32,
                &mut detached_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    outputs.truncate(output_count as usize);
    detached.truncate(detached_count as usize);
    (release_id, outputs, detached)
}

fn confirm_release(session: &Session, release_id: OrbitKvSessionReleaseId) {
    let mut outcome = OrbitKvSessionReleaseOutcome::default();
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_release(
                session.as_ptr(),
                OrbitKvSessionReleaseEvidence {
                    release_id,
                    mirror_cleanup_confirmed: 1,
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
    assert_eq!(outcome.release_id, release_id);
    assert_eq!(outcome.disposition, ORBITKV_SESSION_RELEASE_COMPLETED);
    assert_eq!(outcome.reserved, 0);
}

fn fork_ready_request(session: &Session, source_request_id: u64, target_request_id: u64) {
    acquire(session, &[target_request_id]);
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
    let mut info = OrbitKvSessionControlPlanInfo::default();
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
    assert_eq!(info.id, control_id);
    assert_eq!(info.kind, ORBITKV_SESSION_CONTROL_KIND_MATERIALIZATION);
    assert_eq!(info.request_count, 1);
    let mut outcome = OrbitKvSessionControlOutcome::default();
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
}

fn published_sentinel() -> OrbitKvSessionPublishedPrefixRelease {
    OrbitKvSessionPublishedPrefixRelease {
        request_id: u64::MAX,
        prefix_id: OrbitKvSessionPrefixId {
            session_epoch: u64::MAX,
            sequence: u64::MAX,
        },
        key: key(0xEE, u64::MAX),
        resident_count: u32::MAX,
        detached_offset: u32::MAX,
        detached_count: u32::MAX,
        reserved: u32::MAX,
    }
}

fn detached_sentinel() -> OrbitKvDetachedBinding {
    OrbitKvDetachedBinding {
        old: OrbitKvPageLease {
            engine_epoch: u64::MAX,
            pool_epoch: u64::MAX,
            generation: u64::MAX,
            page_id: u32::MAX,
            pool_id: u32::MAX,
        },
        replacement: OrbitKvPageLease::default(),
        logical_ordinal: u64::MAX,
        old_backend_index: u64::MAX,
        replacement_backend_index: u64::MAX,
        token_begin: u64::MAX,
        token_end_exclusive: u64::MAX,
        class_id: u16::MAX,
        backend_domain: u16::MAX,
        action: u16::MAX,
        reason: u16::MAX,
        reserved: u64::MAX,
    }
}

#[test]
fn prefix_publish_release_b1_b2_lifecycle_has_exact_canonical_spans() {
    for (case, boundaries) in [(1_u8, vec![16_u64]), (2, vec![16, 32])] {
        let session = Session::new();
        let request_ids = (0..boundaries.len())
            .map(|index| 100 + u64::from(case) * 10 + index as u64)
            .collect::<Vec<_>>();
        for (index, (&request_id, &boundary)) in request_ids.iter().zip(&boundaries).enumerate() {
            make_ready(&session, request_id, boundary, index as u64 + 1);
        }
        let items = request_ids
            .iter()
            .zip(&boundaries)
            .enumerate()
            .map(
                |(index, (&request_id, &boundary))| OrbitKvSessionPrefixPublishItem {
                    request_id,
                    key: key(case * 16 + index as u8, boundary),
                },
            )
            .collect::<Vec<_>>();
        let before = stats(&session);
        assert_eq!(before.active_requests, items.len() as u64);
        assert_eq!(before.active_prefixes, 0);
        assert_eq!(
            before.total_request_page_refs,
            boundaries.iter().sum::<u64>() / 16
        );
        assert_eq!(before.total_prefix_page_refs, 0);

        let (release_id, outputs, detached) = publish_release(&session, &items);
        assert_ne!(release_id, OrbitKvSessionReleaseId::default());
        assert_eq!(outputs.len(), items.len());
        assert_eq!(detached.len() as u64, boundaries.iter().sum::<u64>() / 16);
        let mut cursor = 0_u32;
        let mut prefix_ids = std::collections::BTreeSet::new();
        for ((input, output), boundary) in items.iter().zip(&outputs).zip(&boundaries) {
            assert_eq!(output.request_id, input.request_id);
            assert_eq!(output.key, input.key);
            assert_eq!(output.prefix_id.session_epoch, release_id.session_epoch);
            assert_ne!(output.prefix_id.sequence, 0);
            assert!(prefix_ids.insert(output.prefix_id));
            assert_eq!(output.resident_count, (*boundary / 16) as u32);
            assert_eq!(output.detached_offset, cursor);
            assert_eq!(output.detached_count, output.resident_count);
            assert_eq!(output.reserved, 0);
            let end = cursor + output.detached_count;
            for (ordinal, binding) in detached[cursor as usize..end as usize].iter().enumerate() {
                assert_eq!(binding.logical_ordinal, ordinal as u64);
                assert_eq!(binding.token_begin, ordinal as u64 * 16);
                assert_eq!(binding.token_end_exclusive, (ordinal as u64 + 1) * 16);
                assert_eq!(binding.old.pool_id, POOL_ID);
                assert_eq!(
                    binding.old_backend_index,
                    BACKEND_BASE_INDEX + u64::from(binding.old.page_id - 1)
                );
                assert_eq!(binding.replacement, OrbitKvPageLease::default());
                assert_eq!(binding.replacement_backend_index, 0);
                assert_eq!(binding.class_id, 0);
                assert_eq!(binding.backend_domain, BACKEND_DOMAIN);
                assert_eq!(binding.action, DetachedAction::Clear as u16);
                assert_eq!(binding.reason, DetachedReason::PrefixTransfer as u16);
                assert_eq!(binding.reserved, 0);
            }
            cursor = end;
            let lookup = lookup_prefix(&session, input.key);
            assert_eq!(lookup.candidate_present, 1);
            assert_eq!(lookup.candidate, output.prefix_id);
            assert_eq!(lookup.resident_count, output.resident_count);
        }
        assert_eq!(cursor as usize, detached.len());

        let transferred = stats(&session);
        assert_eq!(transferred.active_requests, items.len() as u64);
        assert_eq!(transferred.active_prefixes, items.len() as u64);
        assert_eq!(transferred.total_request_page_refs, 0);
        assert_eq!(transferred.total_prefix_page_refs, detached.len() as u64);
        let intent = OrbitKvSessionAppendIntent {
            request_id: request_ids[0],
            target_boundary: boundaries[0] + 16,
        };
        let mut batch = OrbitKvSessionBatchId::default();
        let mut step = OrbitKvSessionPreparedStep::default();
        let mut classes = [OrbitKvClassLowering::default(); 1];
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
                    2,
                    &mut write_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_RETRYABLE_CONFLICT
        );
        assert_eq!(batch, OrbitKvSessionBatchId::default());

        let mut outcome = OrbitKvSessionReleaseOutcome {
            release_id,
            disposition: u32::MAX,
            reserved: u32::MAX,
        };
        assert_eq!(
            unsafe {
                orbitkv_session_confirm_release(
                    session.as_ptr(),
                    OrbitKvSessionReleaseEvidence {
                        release_id,
                        mirror_cleanup_confirmed: 0,
                        reserved: 0,
                    },
                    std::ptr::null(),
                    0,
                    &mut outcome,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_INVALID_ARGUMENT
        );
        assert_eq!(outcome, OrbitKvSessionReleaseOutcome::default());
        confirm_release(&session, release_id);
        let confirmed = stats(&session);
        assert_eq!(confirmed.active_requests, 0);
        assert_eq!(confirmed.active_prefixes, items.len() as u64);
        assert_eq!(confirmed.total_request_page_refs, 0);
        assert_eq!(confirmed.total_prefix_page_refs, detached.len() as u64);
        let recycled = acquire(&session, &request_ids);
        assert!(recycled.iter().all(|view| view.boundary == 0));
        for (input, output) in items.iter().zip(outputs) {
            assert_eq!(
                lookup_prefix(&session, input.key).candidate,
                output.prefix_id
            );
        }
    }
}

#[test]
fn prefix_publish_release_short_outputs_are_non_mutating_and_retryable() {
    let session = Session::new();
    make_ready(&session, 201, 32, 1);
    let semantic_key = key(0x41, 32);
    let item = OrbitKvSessionPrefixPublishItem {
        request_id: 201,
        key: semantic_key,
    };
    let baseline = stats(&session);
    let output_sentinel = published_sentinel();
    let detached_sentinel = detached_sentinel();
    let mut error = [0; 256];

    for (output_capacity, detached_capacity) in [(0, 2), (1, 0), (0, 0)] {
        let mut release_id = OrbitKvSessionReleaseId {
            session_epoch: u64::MAX,
            sequence: u64::MAX,
        };
        let mut output = output_sentinel;
        let mut detached = [detached_sentinel; 2];
        let (mut output_count, mut detached_count) = (u32::MAX, u32::MAX);
        assert_eq!(
            unsafe {
                orbitkv_session_prefix_publish_release_batch(
                    session.as_ptr(),
                    &item,
                    1,
                    &mut release_id,
                    &mut output,
                    output_capacity,
                    &mut output_count,
                    detached.as_mut_ptr(),
                    detached_capacity,
                    &mut detached_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_BUFFER_TOO_SMALL
        );
        assert_eq!(release_id, OrbitKvSessionReleaseId::default());
        assert_eq!((output_count, detached_count), (1, 2));
        assert_eq!(output, output_sentinel);
        assert_eq!(detached, [detached_sentinel; 2]);
        assert_eq!(stats(&session), baseline);
        assert_eq!(lookup_prefix(&session, semantic_key).candidate_present, 0);
    }

    let (release_id, outputs, detached) = publish_release(&session, &[item]);
    assert_eq!(release_id.sequence, 1);
    assert_eq!(outputs[0].prefix_id.sequence, 1);
    assert_eq!(outputs[0].detached_offset, 0);
    assert_eq!(outputs[0].detached_count, 2);
    assert_eq!(detached.len(), 2);
    confirm_release(&session, release_id);
}

#[test]
fn prefix_publish_release_shared_page_transfers_each_reference_exactly() {
    let session = Session::new();
    make_ready(&session, 251, 16, 1);
    fork_ready_request(&session, 251, 252);
    let before = stats(&session);
    assert_eq!(before.active_requests, 2);
    assert_eq!(before.active_prefixes, 0);
    assert_eq!(before.active_pages, 1);
    assert_eq!(before.total_request_page_refs, 2);
    assert_eq!(before.total_prefix_page_refs, 0);

    let items = [
        OrbitKvSessionPrefixPublishItem {
            request_id: 251,
            key: key(0x49, 16),
        },
        OrbitKvSessionPrefixPublishItem {
            request_id: 252,
            key: key(0x4A, 16),
        },
    ];
    let (release_id, outputs, detached) = publish_release(&session, &items);
    assert_eq!(outputs.len(), 2);
    assert_eq!(detached.len(), 2);
    assert_eq!(outputs[0].detached_offset, 0);
    assert_eq!(outputs[1].detached_offset, 1);
    assert_eq!(outputs[0].detached_count, 1);
    assert_eq!(outputs[1].detached_count, 1);
    assert_eq!(detached[0], detached[1]);
    let transferred = stats(&session);
    assert_eq!(transferred.active_pages, 1);
    assert_eq!(transferred.total_request_page_refs, 0);
    assert_eq!(transferred.total_prefix_page_refs, 2);

    confirm_release(&session, release_id);
    let confirmed = stats(&session);
    assert_eq!(confirmed.active_requests, 0);
    assert_eq!(confirmed.active_prefixes, 2);
    assert_eq!(confirmed.active_pages, 1);
    assert_eq!(confirmed.total_request_page_refs, 0);
    assert_eq!(confirmed.total_prefix_page_refs, 2);
}

#[test]
#[allow(clippy::too_many_lines)]
fn prefix_publish_release_rejects_invalid_null_count_and_core_errors_atomically() {
    let session = Session::new();
    make_ready(&session, 301, 16, 1);
    make_ready(&session, 302, 16, 2);
    acquire(&session, &[303]);
    let _pending = prepare_append(&session, 303, 16);
    let good = OrbitKvSessionPrefixPublishItem {
        request_id: 301,
        key: key(0x51, 16),
    };
    let second = OrbitKvSessionPrefixPublishItem {
        request_id: 302,
        key: key(0x52, 16),
    };
    let output_sentinel = published_sentinel();
    let detached_sentinel = detached_sentinel();
    let baseline = stats(&session);
    let mut error = [0; 256];

    let mut release_id = OrbitKvSessionReleaseId {
        session_epoch: u64::MAX,
        sequence: u64::MAX,
    };
    let mut output = output_sentinel;
    let mut detached = [detached_sentinel; 2];
    let (mut output_count, mut detached_count) = (u32::MAX, u32::MAX);
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_publish_release_batch(
                session.as_ptr(),
                &good,
                1,
                std::ptr::null_mut(),
                &mut output,
                1,
                &mut output_count,
                detached.as_mut_ptr(),
                2,
                &mut detached_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(output, output_sentinel);
    assert_eq!(detached, [detached_sentinel; 2]);

    release_id.sequence = u64::MAX;
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_publish_release_batch(
                std::ptr::null_mut(),
                &good,
                1,
                &mut release_id,
                &mut output,
                1,
                &mut output_count,
                detached.as_mut_ptr(),
                2,
                &mut detached_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(release_id, OrbitKvSessionReleaseId::default());
    assert_eq!(output, output_sentinel);
    assert_eq!(detached, [detached_sentinel; 2]);

    for (items, item_count) in [
        (std::ptr::null(), 0),
        (std::ptr::null(), 1),
        (std::ptr::null(), MAXIMUM_PREFIXES + 1),
    ] {
        release_id.sequence = u64::MAX;
        assert_eq!(
            unsafe {
                orbitkv_session_prefix_publish_release_batch(
                    session.as_ptr(),
                    items,
                    item_count,
                    &mut release_id,
                    &mut output,
                    1,
                    &mut output_count,
                    detached.as_mut_ptr(),
                    2,
                    &mut detached_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            ORBITKV_STATUS_INVALID_ARGUMENT
        );
        assert_eq!(release_id, OrbitKvSessionReleaseId::default());
        assert_eq!(output, output_sentinel);
        assert_eq!(detached, [detached_sentinel; 2]);
    }

    for missing in 0..4 {
        release_id.sequence = u64::MAX;
        output_count = u32::MAX;
        detached_count = u32::MAX;
        let status = unsafe {
            orbitkv_session_prefix_publish_release_batch(
                session.as_ptr(),
                &good,
                1,
                &mut release_id,
                if missing == 0 {
                    std::ptr::null_mut()
                } else {
                    &mut output
                },
                1,
                if missing == 1 {
                    std::ptr::null_mut()
                } else {
                    &mut output_count
                },
                if missing == 2 {
                    std::ptr::null_mut()
                } else {
                    detached.as_mut_ptr()
                },
                2,
                if missing == 3 {
                    std::ptr::null_mut()
                } else {
                    &mut detached_count
                },
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert_eq!(status, ORBITKV_STATUS_INVALID_ARGUMENT);
        assert_eq!(release_id, OrbitKvSessionReleaseId::default());
        assert_eq!(output, output_sentinel);
        assert_eq!(detached, [detached_sentinel; 2]);
    }

    let cases = [
        (
            vec![
                good,
                OrbitKvSessionPrefixPublishItem {
                    request_id: good.request_id,
                    key: key(0x53, 16),
                },
            ],
            ORBITKV_STATUS_INVALID_ARGUMENT,
        ),
        (
            vec![
                good,
                OrbitKvSessionPrefixPublishItem {
                    key: good.key,
                    ..second
                },
            ],
            ORBITKV_STATUS_RETRYABLE_CONFLICT,
        ),
        (
            vec![
                good,
                OrbitKvSessionPrefixPublishItem {
                    key: key(0x54, 32),
                    ..second
                },
            ],
            ORBITKV_STATUS_MANAGER_ERROR,
        ),
        (
            vec![OrbitKvSessionPrefixPublishItem {
                request_id: 303,
                key: key(0x55, 0),
            }],
            ORBITKV_STATUS_RETRYABLE_CONFLICT,
        ),
        (
            vec![OrbitKvSessionPrefixPublishItem {
                request_id: 999,
                key: key(0x56, 16),
            }],
            ORBITKV_STATUS_RETRYABLE_CONFLICT,
        ),
    ];
    for (items, expected) in cases {
        let mut release_id = OrbitKvSessionReleaseId {
            session_epoch: u64::MAX,
            sequence: u64::MAX,
        };
        let mut outputs = [output_sentinel; 2];
        let mut detached = [detached_sentinel; 2];
        let (mut output_count, mut detached_count) = (u32::MAX, u32::MAX);
        assert_eq!(
            unsafe {
                orbitkv_session_prefix_publish_release_batch(
                    session.as_ptr(),
                    items.as_ptr(),
                    items.len() as u32,
                    &mut release_id,
                    outputs.as_mut_ptr(),
                    2,
                    &mut output_count,
                    detached.as_mut_ptr(),
                    2,
                    &mut detached_count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            expected
        );
        assert_eq!(release_id, OrbitKvSessionReleaseId::default());
        assert_eq!(outputs, [output_sentinel; 2]);
        assert_eq!(detached, [detached_sentinel; 2]);
        assert_eq!(stats(&session), baseline);
    }
    assert_eq!(lookup_prefix(&session, good.key).candidate_present, 0);
    assert_eq!(lookup_prefix(&session, second.key).candidate_present, 0);

    let (release_id, outputs, detached) = publish_release(&session, &[good, second]);
    assert_eq!(release_id.sequence, 1);
    assert_eq!(
        outputs
            .iter()
            .map(|item| item.prefix_id.sequence)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(detached.len(), 2);
    confirm_release(&session, release_id);
}

#[test]
fn prefix_publish_release_output_is_capability_free_and_pending_destroy_is_safe() {
    let source = include_str!("layouts.rs");
    let declaration = source
        .split("pub struct OrbitKvSessionPublishedPrefixRelease")
        .nth(1)
        .expect("published prefix-release wire DTO")
        .split('}')
        .next()
        .expect("wire DTO body");
    for forbidden in [
        "RequestLease",
        "SnapshotLease",
        "PrefixLease",
        "ReclamationLease",
        "StepLease",
        "SubmissionLease",
    ] {
        assert!(
            !declaration.contains(forbidden),
            "session prefix-release DTO leaked {forbidden}"
        );
    }

    let session = Session::new();
    make_ready(&session, 401, 16, 1);
    let item = OrbitKvSessionPrefixPublishItem {
        request_id: 401,
        key: key(0x61, 16),
    };
    let (release_id, outputs, detached) = publish_release(&session, &[item]);
    assert_ne!(release_id, OrbitKvSessionReleaseId::default());
    assert_eq!(outputs.len(), 1);
    assert_eq!(detached.len(), 1);
    let raw = session.into_raw();
    let mut error = [0; 256];
    assert_eq!(
        unsafe { orbitkv_session_destroy(raw, error.as_mut_ptr(), error.len()) },
        ORBITKV_STATUS_OK
    );
    assert_eq!(
        unsafe { orbitkv_session_destroy(std::ptr::null_mut(), error.as_mut_ptr(), error.len()) },
        ORBITKV_STATUS_OK
    );
}
