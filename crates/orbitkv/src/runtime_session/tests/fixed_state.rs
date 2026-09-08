use super::*;
use crate::{StateCheckpointPool, StatePoolStats};

fn fixed_session() -> (RuntimeSession, [BackendArenaRegistration; 1]) {
    let backends = [backend(0, 90, 8, 4_000)];
    let manager = CanonicalKvManager::new(
        &full_plan(),
        ManagerConfig {
            maximum_requests: 1,
            maximum_operations: 4,
            maximum_prefixes: 1,
            maximum_reclamations: 8,
            maximum_step_tokens: 64,
        },
        &backends,
    )
    .unwrap();
    let engine_epoch = manager.arena_stats()[0].engine_epoch;
    let session = RuntimeSession::with_fixed_states(
        manager,
        CacheSharingPolicy::RequestPrivate,
        [
            (
                1,
                StateCheckpointPool::new(engine_epoch, engine_epoch + 10, 91, 64, 2).unwrap(),
            ),
            (
                2,
                StateCheckpointPool::new(engine_epoch, engine_epoch + 11, 92, 32, 2).unwrap(),
            ),
        ],
    )
    .unwrap();
    (session, backends)
}

fn fixed_evidence(
    session: &RuntimeSession,
    plan: &EngineBatchPlan,
    backends: &[BackendArenaRegistration],
) -> ExecutionEvidence {
    let mut evidence = execution_evidence(session, plan, backends);
    for (step, source) in evidence.steps.iter_mut().zip(&plan.steps) {
        step.fixed_states = source
            .fixed_states
            .iter()
            .map(|state| EngineFixedStateEvidence {
                state_id: state.state_id,
                source: state.source,
                destination: state.destination,
                byte_count: state.byte_count,
                observed: true,
                written: true,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
    }
    evidence
}

fn state_stats(session: &RuntimeSession, state_id: u16) -> StatePoolStats {
    session
        .fixed_state_stats()
        .iter()
        .find(|(id, _)| *id == state_id)
        .map(|(_, stats)| *stats)
        .unwrap()
}

fn assert_released_slots_reuse_with_new_generations(
    session: &mut RuntimeSession,
    previous: &EngineBatchPublication,
) {
    let request_id = EngineRequestId(8);
    session.acquire_requests(&[request_id]).unwrap();
    let reused = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 1,
        }])
        .unwrap();
    for state in &reused.steps[0].fixed_states {
        let old = previous.steps[0]
            .fixed_states
            .iter()
            .find(|old| old.state_id == state.state_id)
            .unwrap();
        assert_eq!(state.destination.slot_id, old.slot.slot_id);
        assert!(state.destination.generation > old.slot.generation);
    }
    session
        .abort_prepared_execution(
            reused.batch_id,
            &[EngineStepAbortEvidence {
                request_id,
                backend_unobserved: true,
            }],
        )
        .unwrap();
}

#[test]
fn token_and_fixed_state_share_prepare_submit_complete_and_release() {
    let (mut session, backends) = fixed_session();
    let request = EngineRequestId(7);
    session.acquire_requests(&[request]).unwrap();

    let first = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: request,
            target_boundary: 16,
        }])
        .unwrap();
    assert_eq!(first.steps[0].fixed_states.len(), 2);
    assert!(
        first.steps[0]
            .fixed_states
            .iter()
            .all(|state| state.source.is_none())
    );
    assert_eq!(state_stats(&session, 1).reserved_slots, 1);

    let evidence = fixed_evidence(&session, &first, &backends);
    let ticket = session.submit_execution(&evidence).unwrap();
    assert_eq!(state_stats(&session, 1).copying_slots, 1);
    let first_publication = session
        .complete_execution_by_batch(
            ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: 8,
                completion_value: 1,
                confirmed: true,
            },
        )
        .unwrap();
    assert_eq!(first_publication.steps[0].fixed_states.len(), 2);
    let first_slots = first_publication.steps[0]
        .fixed_states
        .iter()
        .map(|state| (state.state_id, state.slot))
        .collect::<BTreeMap<_, _>>();
    confirm_publication(&mut session, &first_publication);
    assert_eq!(state_stats(&session, 1).live_slots, 1);

    let second = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: request,
            target_boundary: 17,
        }])
        .unwrap();
    assert!(second.steps[0].fixed_states.iter().all(|state| {
        state.source == first_slots.get(&state.state_id).copied()
            && state.destination != state.source.unwrap()
    }));
    let evidence = fixed_evidence(&session, &second, &backends);
    let ticket = session.submit_execution(&evidence).unwrap();
    assert!(matches!(
        session.complete_execution_by_batch(
            ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: 8,
                completion_value: 1,
                confirmed: true,
            },
        ),
        Err(RuntimeSessionError::FixedState(
            crate::StateCheckpointError::CompletionNotConfirmed
        ))
    ));
    let second_publication = session
        .complete_execution_by_batch(
            ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: 8,
                completion_value: 2,
                confirmed: true,
            },
        )
        .unwrap();
    confirm_publication(&mut session, &second_publication);
    assert_eq!(state_stats(&session, 1).free_slots, 1);
    assert_eq!(state_stats(&session, 1).live_slots, 1);

    release_ready_request(&mut session, request);
    assert_eq!(state_stats(&session, 1).free_slots, 2);
    assert_eq!(state_stats(&session, 2).free_slots, 2);

    assert_released_slots_reuse_with_new_generations(&mut session, &second_publication);
}

#[test]
fn fixed_state_abort_and_bad_evidence_preserve_collective_safety() {
    let (mut session, backends) = fixed_session();
    let request = EngineRequestId(9);
    session.acquire_requests(&[request]).unwrap();

    let aborted = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: request,
            target_boundary: 1,
        }])
        .unwrap();
    session
        .abort_prepared_execution(
            aborted.batch_id,
            &[EngineStepAbortEvidence {
                request_id: request,
                backend_unobserved: true,
            }],
        )
        .unwrap();
    assert_eq!(state_stats(&session, 1).free_slots, 2);

    let forged = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: request,
            target_boundary: 1,
        }])
        .unwrap();
    let mut evidence = fixed_evidence(&session, &forged, &backends);
    evidence.steps[0].fixed_states[0].byte_count += 1;
    assert_eq!(
        session.submit_execution(&evidence),
        Err(RuntimeSessionError::FixedStateEvidenceMismatch)
    );
    assert_eq!(state_stats(&session, 1).quarantined_slots, 1);
    assert!(matches!(
        session.prepare_append_batch(&[EngineAppendIntent {
            request_id: request,
            target_boundary: 1,
        }]),
        Err(RuntimeSessionError::RequestNotReady {
            state: "quarantined",
            ..
        })
    ));
}

#[test]
fn fixed_state_sessions_reject_shared_prefix_authority() {
    let backends = [backend(0, 93, 8, 5_000)];
    let manager = CanonicalKvManager::new(
        &full_plan(),
        ManagerConfig {
            maximum_requests: 1,
            maximum_operations: 2,
            maximum_prefixes: 1,
            maximum_reclamations: 8,
            maximum_step_tokens: 16,
        },
        &backends,
    )
    .unwrap();
    let epoch = manager.arena_stats()[0].engine_epoch;
    assert!(matches!(
        RuntimeSession::with_fixed_states(
            manager,
            CacheSharingPolicy::SharedPrefix,
            [(
                1,
                StateCheckpointPool::new(epoch, epoch + 1, 94, 16, 2).unwrap()
            )],
        ),
        Err(RuntimeSessionError::FixedStateSharingUnsupported)
    ));
}
