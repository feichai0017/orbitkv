use orbitkv_compiler::compiler::{IslandKind, partition_baseline};
use orbitkv_compiler::ir::{EffectKind, Operation, StateScope};
use orbitkv_compiler::model::{LayerKind, Qwen38Contract};

#[test]
fn pinned_layer_schedule_is_three_gdn_then_full_attention() {
    let kinds: Vec<_> = (0..Qwen38Contract::LAYERS).map(Qwen38Contract::layer_kind).collect();
    assert_eq!(kinds.iter().filter(|kind| **kind == LayerKind::GatedDelta).count(), 48);
    assert_eq!(kinds.iter().filter(|kind| **kind == LayerKind::FullAttention).count(), 16);
    for group in kinds.chunks_exact(4) {
        assert_eq!(
            group,
            &[LayerKind::GatedDelta, LayerKind::GatedDelta, LayerKind::GatedDelta, LayerKind::FullAttention,]
        );
    }
}

#[test]
fn target_decode_graph_has_explicit_state_effects() {
    let graph = Qwen38Contract::target_decode_graph();
    graph.validate().unwrap();

    assert_eq!(graph.tasks.len(), 67);
    assert_eq!(graph.states.len(), 3);
    assert_eq!(graph.states[0].scope, StateScope::PerSequence);
    assert_eq!(graph.states[0].bytes, 150_994_944);
    assert_eq!(graph.states[1].bytes, 2_949_120);
    assert_eq!(graph.states[2].scope, StateScope::PerToken);
    assert_eq!(graph.states[2].bytes, 65_536);

    let gdn =
        graph.tasks.iter().find(|task| matches!(task.operation, Operation::GatedDeltaBlock { layer: 0, .. })).unwrap();
    assert_eq!(gdn.state_effects.len(), 6);
    assert!(gdn.state_effects.iter().any(|effect| effect.kind == EffectKind::Read));
    assert!(gdn.state_effects.iter().any(|effect| effect.kind == EffectKind::TentativeWrite { version: 0 }));
    assert!(gdn.state_effects.iter().any(|effect| effect.kind == EffectKind::Commit { version: 0 }));
}

#[test]
fn baseline_partition_forms_sixteen_three_gdn_islands() {
    let plan = partition_baseline(&Qwen38Contract::target_decode_graph()).unwrap();
    assert_eq!(plan.islands.len(), 35);

    let gdn_islands: Vec<_> =
        plan.islands.iter().filter(|island| matches!(island.kind, IslandKind::StatefulGatedDelta { .. })).collect();
    assert_eq!(gdn_islands.len(), 16);
    assert!(gdn_islands.iter().all(|island| island.tasks.len() == 3));
}

#[test]
fn commit_without_a_tentative_write_is_rejected() {
    let mut graph = Qwen38Contract::target_decode_graph();
    let gdn = graph
        .tasks
        .iter_mut()
        .find(|task| matches!(task.operation, Operation::GatedDeltaBlock { layer: 0, .. }))
        .unwrap();
    gdn.state_effects.retain(|effect| effect.kind != EffectKind::TentativeWrite { version: 0 });

    let error = graph.validate().unwrap_err();
    assert!(error.to_string().contains("without a prior tentative write"));
}
