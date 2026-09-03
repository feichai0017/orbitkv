#![allow(clippy::borrow_as_ptr, clippy::cast_possible_truncation)]

use super::*;
use crate::{OrbitKvPageLease, OrbitKvPrefixSemanticKey};

const PLAN: &[u8] = br#"{
  "page_tokens": 16,
  "classes": [{
    "name": "full",
    "layers": [0],
    "retention": "full",
    "bytes_per_token_per_layer": 128
  }]
}"#;

const LATENT_PLAN: &[u8] = br#"{
  "page_tokens": 16,
  "classes": [{
    "name": "mla",
    "layers": [0],
    "retention": "full",
    "bytes_per_token_per_layer": 1152,
    "storage": "latent_kv",
    "components": [
      {"name": "latent", "bytes_per_token_per_layer": 1024},
      {"name": "rope", "bytes_per_token_per_layer": 128}
    ]
  }]
}"#;

fn config() -> OrbitKvSessionCreateConfig {
    OrbitKvSessionCreateConfig {
        manager: OrbitKvManagerConfig {
            maximum_requests: 2,
            maximum_operations: 2,
            maximum_prefixes: 1,
            maximum_reclamations: 8,
            maximum_step_tokens: 32,
            plan_format: 1,
            reserved: 0,
        },
        cache_sharing_policy: ORBITKV_CACHE_SHARING_POLICY_SHARED_PREFIX,
        reserved: 0,
    }
}

fn backend(pool_id: u32) -> OrbitKvBackendArenaRegistration {
    OrbitKvBackendArenaRegistration {
        pool_id,
        class_id: 0,
        backend_domain: 7,
        page_count: 8,
        reserved: 0,
        backend_base_index: 100,
    }
}

fn create(pool_id: u32) -> *mut OrbitKvSessionHandle {
    let mut handle = std::ptr::null_mut();
    let mut error = [0; 256];
    let backend = backend(pool_id);
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
    handle
}

fn create_status(
    plan: &[u8],
    config: &OrbitKvSessionCreateConfig,
    backend: &OrbitKvBackendArenaRegistration,
) -> (i32, *mut OrbitKvSessionHandle, String) {
    let mut handle = std::ptr::null_mut();
    let mut error = [0; 256];
    let status = unsafe {
        orbitkv_session_create(
            plan.as_ptr(),
            plan.len(),
            config,
            backend,
            1,
            &mut handle,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    let message = unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    (status, handle, message)
}

fn acquire(handle: *mut OrbitKvSessionHandle, request_id: u64) -> OrbitKvSessionRequestView {
    let mut error = [0; 256];
    let mut view = OrbitKvSessionRequestView::default();
    let mut count = 0;
    assert_eq!(
        unsafe {
            orbitkv_session_acquire_requests(
                handle,
                &request_id,
                1,
                &mut view,
                1,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(count, 1);
    view
}

struct Prepared {
    batch: OrbitKvSessionBatchId,
    step: OrbitKvSessionPreparedStep,
    classes: Vec<OrbitKvClassLowering>,
    tails: Vec<OrbitKvTailAction>,
    copies: Vec<OrbitKvCopyIntent>,
    writes: Vec<OrbitKvWriteIntent>,
}

fn prepare(handle: *mut OrbitKvSessionHandle, request_id: u64, boundary: u64) -> Prepared {
    let intent = OrbitKvSessionAppendIntent {
        request_id,
        target_boundary: boundary,
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
                handle,
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
    Prepared {
        batch,
        step,
        classes,
        tails,
        copies,
        writes,
    }
}

fn submit(handle: *mut OrbitKvSessionHandle, prepared: &Prepared) {
    let mut binds = Vec::new();
    for class in &prepared.classes {
        for tail in &prepared.tails
            [class.tail_offset as usize..(class.tail_offset + class.tail_count) as usize]
        {
            if tail.kind == 2 || tail.kind == 3 {
                binds.push(OrbitKvSessionBindEvidence {
                    page: tail.destination,
                    backend_domain: 7,
                    mapped: 1,
                    writable: 1,
                    reserved: 0,
                    backend_index: 100 + u64::from(tail.destination.page_id - 1),
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
                    pool_id: 41,
                },
                backend_domain: 7,
                mapped: 1,
                writable: 1,
                reserved: 0,
                backend_index: 100 + u64::from(write.page_id - 1),
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
                handle,
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

fn complete_and_confirm(handle: *mut OrbitKvSessionHandle, prepared: &Prepared) {
    let mut publication_id = OrbitKvSessionPublicationId::default();
    let mut step = OrbitKvSessionStepPublication::default();
    let mut detached = vec![OrbitKvDetachedBinding::default(); 8];
    let mut retirements = vec![OrbitKvSessionRetirement::default(); 8];
    let (mut step_count, mut detached_count, mut retirement_count) = (0, 0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_complete_execution(
                handle,
                prepared.batch,
                OrbitKvSessionCompletionEvidence {
                    completion_domain: 41,
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
                handle,
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

fn prepare_release(
    handle: *mut OrbitKvSessionHandle,
    request_id: u64,
) -> (
    OrbitKvSessionReleaseId,
    Vec<OrbitKvSessionRetirementEvidence>,
) {
    let mut release_id = OrbitKvSessionReleaseId::default();
    let mut release = OrbitKvSessionReleasedRequest::default();
    let mut detached = vec![OrbitKvDetachedBinding::default(); 8];
    let mut retirements = vec![OrbitKvSessionRetirement::default(); 8];
    let (mut release_count, mut detached_count, mut retirement_count) = (0, 0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_release(
                handle,
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
    assert_eq!(release.request_id, request_id);
    retirements.truncate(retirement_count as usize);
    (release_id, retirement_evidence(&retirements))
}

#[test]
fn session_structs_do_not_contain_internal_transaction_leases() {
    let source = include_str!("layouts.rs");
    for forbidden in [
        "OrbitKvRequestLease",
        "OrbitKvSnapshotLease",
        "OrbitKvStepLease",
        "OrbitKvSubmissionLease",
        "OrbitKvReclamationLease",
    ] {
        assert!(!source.contains(forbidden), "wire leaked {forbidden}");
    }
}

#[test]
fn session_create_rejects_reserved_unknown_policy_and_shared_latent() {
    let backend = backend(91);

    let mut create_config = config();
    create_config.reserved = 1;
    let (status, handle, message) = create_status(PLAN, &create_config, &backend);
    assert_eq!(status, ORBITKV_STATUS_INVALID_ARGUMENT);
    assert!(handle.is_null());
    assert!(message.contains("session create config reserved"));

    create_config = config();
    create_config.cache_sharing_policy = 0;
    let (status, handle, message) = create_status(PLAN, &create_config, &backend);
    assert_eq!(status, ORBITKV_STATUS_INVALID_ARGUMENT);
    assert!(handle.is_null());
    assert!(message.contains("cache sharing policy"));

    create_config = config();
    let (status, handle, message) = create_status(LATENT_PLAN, &create_config, &backend);
    assert_eq!(status, ORBITKV_STATUS_INVALID_ARGUMENT);
    assert!(handle.is_null());
    assert!(message.contains("shared-prefix sessions require"));
}

#[test]
#[allow(clippy::too_many_lines)]
fn session_create_accepts_full_latent_request_private() {
    let backend = backend(92);
    let mut create_config = config();
    create_config.cache_sharing_policy = ORBITKV_CACHE_SHARING_POLICY_REQUEST_PRIVATE;
    let (status, handle, message) = create_status(LATENT_PLAN, &create_config, &backend);
    assert_eq!(status, ORBITKV_STATUS_OK, "{message}");
    assert!(!handle.is_null());

    let mut error = [0; 256];
    let key = OrbitKvPrefixSemanticKey::default();
    let mut lookup = OrbitKvSessionPrefixLookup::default();
    let mut output_count = u32::MAX;
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_lookup_batch(
                handle,
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
    let publish = OrbitKvSessionPrefixPublishItem { request_id: 1, key };
    let mut published = OrbitKvSessionPublishedPrefix::default();
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_publish_batch(
                handle,
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
    let control_id = OrbitKvSessionControlId {
        session_epoch: u64::MAX,
        sequence: u64::MAX,
    };
    let prefix_id = OrbitKvSessionPrefixId {
        session_epoch: u64::MAX,
        sequence: u64::MAX,
    };
    let mut prepared_control = OrbitKvSessionControlId::default();
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_prefix_attach(
                handle,
                &OrbitKvSessionPrefixAttachItem {
                    target_request_id: 2,
                    prefix_id,
                    key,
                    resident_count: 1,
                    reserved: 0,
                },
                1,
                &mut prepared_control,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_request_fork(
                handle,
                &OrbitKvSessionRequestForkItem {
                    source_request_id: 1,
                    target_request_id: 2,
                },
                1,
                &mut prepared_control,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_prefix_evict(
                handle,
                &prefix_id,
                1,
                &mut prepared_control,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    let mut release_id = OrbitKvSessionReleaseId::default();
    let mut transferred = OrbitKvSessionPublishedPrefixRelease::default();
    let mut detached_count = u32::MAX;
    assert_eq!(
        unsafe {
            orbitkv_session_prefix_publish_release_batch(
                handle,
                &publish,
                1,
                &mut release_id,
                &mut transferred,
                1,
                &mut output_count,
                std::ptr::null_mut(),
                0,
                &mut detached_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    let mut control_plan = OrbitKvSessionControlPlanInfo::default();
    assert_eq!(
        unsafe {
            orbitkv_session_commit_control(
                handle,
                control_id,
                &mut control_plan,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    let policy_error = unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    assert!(policy_error.contains("cache-sharing policy"));

    assert_eq!(
        unsafe {
            orbitkv_session_abort_control(handle, control_id, error.as_mut_ptr(), error.len())
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            orbitkv_session_read_control_plan(
                handle,
                control_id,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    let expected = OrbitKvSessionPendingAttachCancel {
        control_id,
        request_id: u64::MAX,
        prefix_id,
        view_version: u64::MAX,
        boundary: u64::MAX,
        resident_count: u32::MAX,
    };
    let mut cancel_outcome = OrbitKvSessionPendingAttachCancelOutcome::default();
    assert_eq!(
        unsafe {
            orbitkv_session_cancel_pending_attach(
                handle,
                expected,
                &mut cancel_outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            orbitkv_session_finalize_pending_attach_cancel(
                handle,
                expected,
                &mut cancel_outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    let mut control_outcome = OrbitKvSessionControlOutcome::default();
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_control(
                handle,
                OrbitKvSessionControlEvidence {
                    id: control_id,
                    mirror_updates_confirmed: u32::MAX,
                    reserved: u32::MAX,
                },
                std::ptr::null(),
                0,
                &mut control_outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            orbitkv_session_quarantine_control(handle, control_id, error.as_mut_ptr(), error.len())
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    unsafe { orbitkv_session_destroy(handle, std::ptr::null_mut(), 0) };
}

#[test]
fn short_prepare_is_non_mutating_and_foreign_ids_are_rejected() {
    let first = create(41);
    let second = create(42);
    acquire(first, 7);
    acquire(second, 7);
    let intent = OrbitKvSessionAppendIntent {
        request_id: 7,
        target_boundary: 16,
    };
    let mut batch = OrbitKvSessionBatchId::default();
    let (mut steps, mut classes, mut tails, mut copies, mut writes) = (0, 0, 0, 0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_append(
                first,
                &intent,
                1,
                &mut batch,
                std::ptr::null_mut(),
                0,
                &mut steps,
                std::ptr::null_mut(),
                0,
                &mut classes,
                std::ptr::null_mut(),
                0,
                &mut tails,
                std::ptr::null_mut(),
                0,
                &mut copies,
                std::ptr::null_mut(),
                0,
                &mut writes,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_BUFFER_TOO_SMALL
    );
    assert_eq!(batch, OrbitKvSessionBatchId::default());
    let prepared = prepare(first, 7, 16);
    let evidence = [OrbitKvSessionStepAbortEvidence {
        request_id: 7,
        backend_unobserved: 1,
        reserved: 0,
    }];
    assert_eq!(
        unsafe {
            orbitkv_session_abort_prepared(
                second,
                prepared.batch,
                evidence.as_ptr(),
                1,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    assert_eq!(
        unsafe {
            orbitkv_session_abort_prepared(
                first,
                prepared.batch,
                evidence.as_ptr(),
                1,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    unsafe {
        orbitkv_session_destroy(first, std::ptr::null_mut(), 0);
        orbitkv_session_destroy(second, std::ptr::null_mut(), 0);
    }
}

#[test]
fn publication_ack_gates_request_reuse() {
    let handle = create(41);
    acquire(handle, 9);
    let prepared = prepare(handle, 9, 16);
    submit(handle, &prepared);
    let mut publication_id = OrbitKvSessionPublicationId::default();
    let mut step = OrbitKvSessionStepPublication::default();
    let mut detached = vec![OrbitKvDetachedBinding::default(); 6];
    let mut retirements = vec![OrbitKvSessionRetirement::default(); 6];
    let (mut step_count, mut detached_count, mut retirement_count) = (0, 0, 0);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_complete_execution(
                handle,
                prepared.batch,
                OrbitKvSessionCompletionEvidence {
                    completion_domain: 3,
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
    let intent = OrbitKvSessionAppendIntent {
        request_id: 9,
        target_boundary: 32,
    };
    let mut rejected_batch = OrbitKvSessionBatchId::default();
    let (mut s, mut c, mut t, mut cp, mut w) = (0, 0, 0, 0, 0);
    let mut rejected_step = OrbitKvSessionPreparedStep::default();
    let mut rejected_class = OrbitKvClassLowering::default();
    let mut rejected_tail = OrbitKvTailAction::default();
    let mut rejected_copy = OrbitKvCopyIntent::default();
    let mut rejected_writes = [OrbitKvWriteIntent::default(); 2];
    assert_eq!(
        unsafe {
            orbitkv_session_prepare_append(
                handle,
                &intent,
                1,
                &mut rejected_batch,
                &mut rejected_step,
                1,
                &mut s,
                &mut rejected_class,
                1,
                &mut c,
                &mut rejected_tail,
                1,
                &mut t,
                &mut rejected_copy,
                1,
                &mut cp,
                rejected_writes.as_mut_ptr(),
                rejected_writes.len() as u32,
                &mut w,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_RETRYABLE_CONFLICT
    );
    assert_eq!(rejected_batch, OrbitKvSessionBatchId::default());
    assert_eq!(retirement_count, 0);
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_publication(
                handle,
                OrbitKvSessionPublicationEvidence {
                    publication_id,
                    mirror_cleanup_confirmed: 1,
                    reserved: 0,
                },
                std::ptr::null(),
                0,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    let retry = prepare(handle, 9, 32);
    assert_ne!(retry.batch.sequence, 0);
    unsafe { orbitkv_session_destroy(handle, std::ptr::null_mut(), 0) };
}

#[test]
fn release_confirmation_requires_and_initializes_typed_outcome() {
    let handle = create(41);
    let request_id = 13;
    acquire(handle, request_id);
    let prepared = prepare(handle, request_id, 16);
    submit(handle, &prepared);
    complete_and_confirm(handle, &prepared);
    let (release_id, retirements) = prepare_release(handle, request_id);
    let evidence = OrbitKvSessionReleaseEvidence {
        release_id,
        mirror_cleanup_confirmed: 1,
        reserved: 0,
    };
    let mut error = [0; 256];

    assert_eq!(
        unsafe {
            orbitkv_session_confirm_release(
                handle,
                evidence,
                retirements.as_ptr(),
                retirements.len() as u32,
                std::ptr::null_mut(),
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );

    let mut outcome = OrbitKvSessionReleaseOutcome {
        release_id,
        disposition: u32::MAX,
        reserved: u32::MAX,
    };
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_release(
                handle,
                OrbitKvSessionReleaseEvidence {
                    release_id,
                    mirror_cleanup_confirmed: 0,
                    reserved: 0,
                },
                retirements.as_ptr(),
                retirements.len() as u32,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(outcome, OrbitKvSessionReleaseOutcome::default());
    outcome.disposition = u32::MAX;
    outcome.reserved = u32::MAX;
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_release(
                handle,
                evidence,
                retirements.as_ptr(),
                retirements.len() as u32,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(
        outcome,
        OrbitKvSessionReleaseOutcome {
            release_id,
            disposition: ORBITKV_SESSION_RELEASE_COMPLETED,
            reserved: 0,
        }
    );
    acquire(handle, request_id);
    unsafe { orbitkv_session_destroy(handle, std::ptr::null_mut(), 0) };
}

#[cfg(feature = "test-support")]
#[test]
fn post_ack_recycle_fault_returns_pending_and_id_only_retry_completes() {
    let handle = create(41);
    let request_id = 15;
    acquire(handle, request_id);
    let prepared = prepare(handle, request_id, 16);
    submit(handle, &prepared);
    complete_and_confirm(handle, &prepared);
    let (release_id, retirements) = prepare_release(handle, request_id);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_test_inject_release_recycle_once(
                handle,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );

    let mut outcome = OrbitKvSessionReleaseOutcome::default();
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_release(
                handle,
                OrbitKvSessionReleaseEvidence {
                    release_id,
                    mirror_cleanup_confirmed: 1,
                    reserved: 0,
                },
                retirements.as_ptr(),
                retirements.len() as u32,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_OK
    );
    assert_eq!(outcome.release_id, release_id);
    assert_eq!(outcome.disposition, ORBITKV_SESSION_RELEASE_RECYCLE_PENDING);
    let mut stats = OrbitKvManagerStats::default();
    assert_eq!(
        unsafe { orbitkv_session_stats(handle, &mut stats, error.as_mut_ptr(), error.len()) },
        ORBITKV_STATUS_OK
    );
    assert_eq!(stats.pending_reclamations, 0);
    assert_eq!(stats.active_requests, 1);

    outcome = OrbitKvSessionReleaseOutcome {
        release_id,
        disposition: u32::MAX,
        reserved: u32::MAX,
    };
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_release(
                handle,
                OrbitKvSessionReleaseEvidence {
                    release_id,
                    mirror_cleanup_confirmed: 1,
                    reserved: 0,
                },
                retirements.as_ptr(),
                retirements.len() as u32,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(outcome, OrbitKvSessionReleaseOutcome::default());

    assert_eq!(
        unsafe {
            orbitkv_session_confirm_release(
                handle,
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
        ORBITKV_STATUS_OK
    );
    assert_eq!(outcome.release_id, release_id);
    assert_eq!(outcome.disposition, ORBITKV_SESSION_RELEASE_COMPLETED);
    acquire(handle, request_id);
    unsafe { orbitkv_session_destroy(handle, std::ptr::null_mut(), 0) };
}

#[cfg(feature = "test-support")]
#[test]
fn unexpected_post_ack_recycle_failure_fail_stops_the_ffi_session() {
    let handle = create(41);
    let request_id = 16;
    acquire(handle, request_id);
    let prepared = prepare(handle, request_id, 16);
    submit(handle, &prepared);
    complete_and_confirm(handle, &prepared);
    let (release_id, retirements) = prepare_release(handle, request_id);
    lock_state(unsafe { &*handle })
        .expect("lock session")
        .runtime
        .inject_test_fault(RuntimeSessionTestFault::ReleaseRecycleFatalOnce);

    let mut outcome = OrbitKvSessionReleaseOutcome {
        release_id,
        disposition: u32::MAX,
        reserved: u32::MAX,
    };
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_confirm_release(
                handle,
                OrbitKvSessionReleaseEvidence {
                    release_id,
                    mirror_cleanup_confirmed: 1,
                    reserved: 0,
                },
                retirements.as_ptr(),
                retirements.len() as u32,
                &mut outcome,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_FAIL_STOPPED
    );
    assert_eq!(outcome, OrbitKvSessionReleaseOutcome::default());

    let next_request = request_id + 1;
    let mut view = OrbitKvSessionRequestView::default();
    let mut view_count = 0;
    assert_eq!(
        unsafe {
            orbitkv_session_acquire_requests(
                handle,
                &next_request,
                1,
                &mut view,
                1,
                &mut view_count,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_FAIL_STOPPED
    );
    unsafe { orbitkv_session_destroy(handle, std::ptr::null_mut(), 0) };
}

#[test]
fn explicit_quarantine_fail_stops_every_mutation_but_keeps_stats_available() {
    let handle = create(41);
    acquire(handle, 17);
    let prepared = prepare(handle, 17, 16);
    let mut error = [0; 256];
    assert_eq!(
        unsafe {
            orbitkv_session_quarantine_prepared(
                handle,
                prepared.batch,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        ORBITKV_STATUS_FAIL_STOPPED
    );

    let mut view = OrbitKvSessionRequestView::default();
    let mut count = 0;
    let request_id = 18_u64;
    assert_eq!(
        unsafe {
            orbitkv_session_acquire_requests(
                handle,
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
        unsafe { orbitkv_session_stats(handle, &mut stats, error.as_mut_ptr(), error.len()) },
        ORBITKV_STATUS_OK
    );
    assert_eq!(stats.quarantined_pages, 1);
    unsafe { orbitkv_session_destroy(handle, std::ptr::null_mut(), 0) };
}
