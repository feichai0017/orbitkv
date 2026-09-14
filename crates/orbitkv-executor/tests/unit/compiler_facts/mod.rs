use orbitkv::{
    AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage, compile_runtime_manifest,
    plan::RetentionKind,
};

use super::*;

fn plan_and_arenas() -> (ExecutorPlan, [ExecutorArena; 2]) {
    let manifest = compile_runtime_manifest(AttentionStatePlanInput {
        page_tokens: 16,
        states: vec![
            AttentionStateSpec {
                name: "global".into(),
                layers: vec![0, 2],
                storage: AttentionStateStorage::TokenKv {
                    key_bytes_per_token_per_layer: 128,
                    value_bytes_per_token_per_layer: 128,
                    retention: RetentionKind::Full,
                    window_tokens: None,
                },
            },
            AttentionStateSpec {
                name: "local".into(),
                layers: vec![1, 3],
                storage: AttentionStateStorage::TokenKv {
                    key_bytes_per_token_per_layer: 128,
                    value_bytes_per_token_per_layer: 128,
                    retention: RetentionKind::Sliding,
                    window_tokens: Some(64),
                },
            },
        ],
    })
    .unwrap();
    let plan = ExecutorPlan::compile(&manifest).unwrap();
    let arena = |class_id, pool_id, backend_base_index| ExecutorArena {
        engine_epoch: 1,
        pool_epoch: 1,
        pool_id,
        class_id,
        backend_domain: class_id + 1,
        first_page_id: u32::from(class_id) * 8 + 1,
        page_count: 8,
        backend_base_index,
    };
    (plan, [arena(0, 1, 0), arena(1, 2, 8)])
}

#[test]
fn lowers_manifest_and_arena_contract_to_deterministic_facts() {
    let (plan, arenas) = plan_and_arenas();
    let facts = plan.compiler_facts(&arenas).unwrap();
    let repeated = plan.compiler_facts(&arenas).unwrap();
    assert_eq!(facts.digest(), repeated.digest());
    assert_eq!(facts.manifest_fingerprint(), plan.manifest_fingerprint);
    assert_eq!(facts.classes().len(), 2);
    assert!(
        facts
            .egglog()
            .contains("(persistent-state-retention-full 0)")
    );
    assert!(
        facts
            .egglog()
            .contains("(persistent-state-retention-sliding 1 64)")
    );
    assert!(facts.egglog().contains("(persistent-state-arena 1 8 8)"));
}

#[test]
fn rejects_reordered_or_overflowing_arena_bindings() {
    let (plan, mut arenas) = plan_and_arenas();
    arenas.swap(0, 1);
    assert!(matches!(
        plan.compiler_facts(&arenas),
        Err(ExecutorError::PreparedGeometryMismatch)
    ));

    let (plan, mut arenas) = plan_and_arenas();
    arenas[0].backend_base_index = u64::MAX;
    assert!(matches!(
        plan.compiler_facts(&arenas),
        Err(ExecutorError::CompilerFactsMismatch)
    ));
}

#[test]
fn lowers_fixed_state_geometry_alongside_token_arenas() {
    let manifest = compile_runtime_manifest(AttentionStatePlanInput {
        page_tokens: 16,
        states: vec![
            AttentionStateSpec {
                name: "global".into(),
                layers: vec![3],
                storage: AttentionStateStorage::TokenKv {
                    key_bytes_per_token_per_layer: 128,
                    value_bytes_per_token_per_layer: 128,
                    retention: RetentionKind::Full,
                    window_tokens: None,
                },
            },
            AttentionStateSpec {
                name: "recurrent".into(),
                layers: vec![0, 1, 2],
                storage: AttentionStateStorage::Recurrent {
                    family: orbitkv::RecurrentFamily::Gdn,
                    state_bytes_per_layer: 256,
                    checkpoint_slots_per_request: 2,
                },
            },
        ],
    })
    .unwrap();
    let plan = ExecutorPlan::compile(&manifest).unwrap();
    let arenas = [ExecutorArena {
        engine_epoch: 1,
        pool_epoch: 1,
        pool_id: 1,
        class_id: 0,
        backend_domain: 1,
        first_page_id: 1,
        page_count: 8,
        backend_base_index: 0,
    }];
    let facts = plan.compiler_facts(&arenas).unwrap();
    assert_eq!(facts.classes().len(), 1);
    assert_eq!(facts.fixed_states().len(), 1);
    assert_eq!(facts.fixed_states()[0].state_id, 1);
    assert!(
        facts
            .egglog()
            .contains("(persistent-fixed-state-recurrent 1 \"gdn\" 256 2 1536)")
    );
}
