use super::*;
use crate::hlir::{Input, LoopEnd, LoopStart, Output, ReferenceOp, Sin};
use crate::op::LLIROp;
use crate::prelude::{DType, Expression};
use crate::search::profile::ChoiceSite;
use std::sync::Arc;

#[test]
fn loop_copies_keep_the_rolled_origin_without_changing_the_program() {
    let ops = vec![
        LLIROp::new::<Input>(Box::new(Input::default())),
        LLIROp::new::<LoopStart>(Box::new(LoopStart {
            loop_id: 0,
            slot_idx: 0,
            iters: Expression::from(3),
            dtype: DType::F32,
        })),
        LLIROp::new::<dyn ReferenceOp>(Box::new(Sin::default())),
        LLIROp::new::<LoopEnd>(Box::new(LoopEnd {
            loop_id: 0,
            slot_idx: 0,
            dtype: DType::F32,
        })),
        LLIROp::new::<Output>(Box::new(Output {
            node: 9,
            persist_only: false,
        })),
    ];
    let mut packed = PackedLLIRGraph {
        origins: (0..ops.len())
            .map(|i| Arc::from(vec![ChoiceSite(i as u32)]))
            .collect(),
        ops,
        incoming_offsets: vec![0, 0, 1, 2, 3, 4],
        incoming_sources: vec![0, 1, 2, 3],
        outgoing_offsets: vec![0, 1, 2, 3, 4, 4],
        outgoing_targets: vec![1, 2, 3, 4],
    };
    let identity = packed.fingerprint();
    packed.origins[2] = vec![ChoiceSite(42)].into();
    assert_eq!(
        identity,
        packed.fingerprint(),
        "provenance changed a semantic identity"
    );
    let expected = materialize_unrolled_llir(&packed.to_stable()).unwrap();
    let (actual, origins) = unroll_with_origins(packed);
    assert_eq!(
        origins.iter().map(|s| s[0].0).collect::<Vec<_>>(),
        [0, 42, 42, 42, 4]
    );
    assert_eq!(
        crate::graph::llir_program_identity(&actual),
        crate::graph::llir_program_identity(&expected)
    );
    assert_eq!(actual.edge_count(), 4);
}
