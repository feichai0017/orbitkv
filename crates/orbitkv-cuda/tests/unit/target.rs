use super::*;
use orbitkv_compiler::{
    egglog_utils::{hlir_to_egglog, run_egglog},
    hlir::HLIROps,
    op::IntoEgglogOp,
    prelude::Graph,
};

#[test]
fn conflicting_execution_targets_are_rejected_during_saturation() {
    let mut graph = Graph::default();
    graph.tensor(1).output();
    let (program, root) = hlir_to_egglog(&graph);
    let ops = <(CudaTargetFacts, HLIROps)>::into_vec();
    let first = CudaTarget { major: 8, minor: 0 }.compiler_facts();
    let second = CudaTarget { major: 9, minor: 0 }.compiler_facts();
    assert!(run_egglog(&format!("{first}\n{first}\n{program}"), &root, &ops, false).is_ok());
    assert!(run_egglog(&format!("{first}\n{second}\n{program}"), &root, &ops, false).is_err());
}
