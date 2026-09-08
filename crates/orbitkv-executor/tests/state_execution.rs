use orbitkv::{
    AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage, CacheSharingPolicy,
    EngineAppendIntent, EngineFixedStateEvidence, EngineRequestId, RecurrentFamily, RuntimeSession,
    StateCheckpointPool, compile_attention_state_plan, compile_plan, compile_runtime_manifest,
    kv_manager::{BackendArenaRegistration, CanonicalKvManager, ManagerConfig},
    plan::RetentionKind,
};
use orbitkv_executor::{ExecutorArena, ExecutorError, ExecutorPlan, FixedStateExecutionEvidence};

#[test]
fn token_executor_does_not_forge_fixed_state_evidence() {
    let input = AttentionStatePlanInput {
        page_tokens: 16,
        states: vec![
            AttentionStateSpec {
                name: "full".into(),
                layers: vec![1],
                storage: AttentionStateStorage::TokenKv {
                    key_bytes_per_token_per_layer: 128,
                    value_bytes_per_token_per_layer: 128,
                    retention: RetentionKind::Full,
                    window_tokens: None,
                },
            },
            AttentionStateSpec {
                name: "recurrent".into(),
                layers: vec![0],
                storage: AttentionStateStorage::Recurrent {
                    family: RecurrentFamily::Gdn,
                    state_bytes_per_layer: 64,
                    checkpoint_slots_per_request: 2,
                },
            },
        ],
    };
    let manifest = compile_runtime_manifest(input.clone()).unwrap();
    let manager_plan = compile_plan(
        compile_attention_state_plan(input)
            .unwrap()
            .token_manager_plan()
            .unwrap(),
    )
    .unwrap();
    let registration = BackendArenaRegistration {
        pool_id: 7,
        class_id: 0,
        backend_domain: 11,
        page_count: 8,
        reserved: 0,
        backend_base_index: 4,
    };
    let manager = CanonicalKvManager::new(
        &manager_plan,
        ManagerConfig {
            maximum_requests: 1,
            maximum_operations: 2,
            maximum_prefixes: 1,
            maximum_reclamations: 8,
            maximum_step_tokens: 16,
        },
        &[registration],
    )
    .unwrap();
    let arena_stats = manager.arena_stats()[0];
    let pool = StateCheckpointPool::new(
        arena_stats.engine_epoch,
        arena_stats.pool_epoch + 10,
        8,
        64,
        2,
    )
    .unwrap();
    let mut session =
        RuntimeSession::with_fixed_states(manager, CacheSharingPolicy::RequestPrivate, [(1, pool)])
            .unwrap();
    let request_id = EngineRequestId(9);
    session.acquire_requests(&[request_id]).unwrap();
    let source = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 1,
        }])
        .unwrap();
    let fixed_plan = source.steps[0].fixed_states[0];
    let arena = ExecutorArena::bind(arena_stats, registration).unwrap();
    let lowered = ExecutorPlan::compile(&manifest)
        .unwrap()
        .lower_prepared(source, &[arena])
        .unwrap();
    assert!(matches!(
        lowered.execution_evidence_after_success(&[arena]),
        Err(ExecutorError::FixedStateExecutionMissing)
    ));
    let mismatched = FixedStateExecutionEvidence {
        request_id: request_id.0,
        states: vec![EngineFixedStateEvidence {
            state_id: fixed_plan.state_id,
            source: fixed_plan.source,
            destination: fixed_plan.destination,
            byte_count: fixed_plan.byte_count,
            observed: true,
            written: false,
        }]
        .into_boxed_slice(),
    };
    assert!(matches!(
        lowered.execution_evidence_after_state_success(&[arena], &[mismatched]),
        Err(ExecutorError::FixedStateExecutionMissing)
    ));
}
