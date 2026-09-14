use super::*;
use crate::egglog_utils::hash_egglog_normalized;
use crate::hlir::{Input, LoopEnd, LoopInput, LoopStart, Output, ReferenceOp, Sin};
use crate::search::unroll::materialize_unrolled_llir;

// A rolling candidate is only collapsible if every non-state boundary input
// is fed from OUTSIDE the candidate's occurrences. A non-state input produced
// by another occurrence (e.g. occ `i` consuming occ `i-2`'s output) is a
// non-adjacent cross-occurrence dependency that, once the bodies are folded
// into one, becomes a directed cycle — which makes the egglog Kahn toposort
// drop nodes and panic. `candidate_is_rollable` must reject those.
#[test]
fn candidate_is_rollable_rejects_cross_occurrence_dep() {
    let n = NodeIndex::new;
    let occ = |body: usize, input: usize| RollingOccurrence {
        nodes: vec![n(body)],
        boundary_inputs: vec![n(input)],
        output_nodes: vec![n(body)],
    };

    // Both occurrences' inputs come from outside the candidate (100, 101) →
    // rollable.
    let rollable = vec![occ(0, 100), occ(1, 101)];
    assert!(candidate_is_rollable(&rollable, &[]));

    // Occurrence 1's input is node 0, which lives INSIDE occurrence 0, and
    // param 0 is not a state param → reject.
    let cyclic = vec![occ(0, 100), occ(1, 0)];
    assert!(!candidate_is_rollable(&cyclic, &[]));

    // Same shape, but param 0 is declared a loop-carried state param → the
    // adjacent loop-carry is allowed.
    assert!(candidate_is_rollable(&cyclic, &[0]));

    // Fewer than two occurrences is never rollable.
    assert!(!candidate_is_rollable(&[occ(0, 100)], &[]));
}
use crate::tests::{assert_close, random_vec};

#[test]
fn materialize_many_disjoint_loops_without_a_global_cartesian_product() {
    const N_LOOPS: usize = 64;
    let mut rolled = LLIRGraph::default();
    for loop_id in 0..N_LOOPS {
        let input = rolled.add_node(LLIROp::new::<Input>(Box::new(Input {
            node: loop_id,
            label: String::new(),
            dtype: DType::F32,
        })));
        let start = rolled.add_node(LLIROp::new::<LoopStart>(Box::new(LoopStart {
            loop_id,
            slot_idx: 0,
            iters: Expression::from(2),
            dtype: DType::F32,
        })));
        let body = rolled.add_node(LLIROp::new::<dyn ReferenceOp>(
            Box::new(Sin::default()) as Box<dyn ReferenceOp>
        ));
        let end = rolled.add_node(LLIROp::new::<LoopEnd>(Box::new(LoopEnd {
            loop_id,
            slot_idx: 0,
            dtype: DType::F32,
        })));
        let output = rolled.add_node(LLIROp::new::<Output>(Box::new(Output {
            node: loop_id,
            persist_only: false,
        })));
        rolled.add_edge(input, start, ());
        rolled.add_edge(start, body, ());
        rolled.add_edge(body, end, ());
        rolled.add_edge(end, output, ());
    }

    let materialized = materialize_unrolled_llir(&rolled)
        .expect("independent loop regions must not multiply one another's contexts");

    // Per region: one input + two body instances + one output, connected
    // as a three-edge chain. A global product would overflow at 2^64.
    assert_eq!(materialized.node_count(), N_LOOPS * 4);
    assert_eq!(materialized.edge_count(), N_LOOPS * 3);
    assert!(
        materialized
            .node_weights()
            .all(|op| { op.to_op::<LoopStart>().is_none() && op.to_op::<LoopEnd>().is_none() })
    );
}

#[test]
fn test_hash_egglog_normalized_same_structure() {
    // Two egglog texts differing only in Input node indices and labels
    let text_a = r#"(let t0 (Input 42 "boundary" (F32)))
(let t1 (Input 100 "layers.0.wq.weight" (F32)))
(let t2 (Add (ECons 128 (ECons 4096 (ENil))) t1 (ECons 1 (ECons 128 (ENil))) t0 (ECons 1 (ECons 1 (ENil))) (ECons 1 (ECons 128 (ENil)))))
(let t3 (Output t2 42 false))
"#;
    let text_b = r#"(let t0 (Input 84 "boundary" (F32)))
(let t1 (Input 200 "layers.1.wq.weight" (F32)))
(let t2 (Add (ECons 128 (ECons 4096 (ENil))) t1 (ECons 1 (ECons 128 (ENil))) t0 (ECons 1 (ECons 1 (ENil))) (ECons 1 (ECons 128 (ENil)))))
(let t3 (Output t2 84 false))
"#;
    assert_eq!(
        hash_egglog_normalized(text_a),
        hash_egglog_normalized(text_b),
        "Structurally identical chunks should hash the same"
    );
}

#[test]
fn test_hash_egglog_normalized_different_structure() {
    let text_a = r#"(let t0 (Input 42 "boundary" (F32)))
(let t1 (Add (ECons 128 (ENil)) t0 (ECons 1 (ENil)) t0 (ECons 1 (ENil)) (ECons 1 (ENil))))
"#;
    let text_b = r#"(let t0 (Input 42 "boundary" (F32)))
(let t1 (Mul (ECons 128 (ENil)) t0 (ECons 1 (ENil)) t0 (ECons 1 (ENil)) (ECons 1 (ENil))))
"#;
    assert_ne!(
        hash_egglog_normalized(text_a),
        hash_egglog_normalized(text_b),
        "Different op types should produce different hashes"
    );
}

#[test]
fn test_hash_egglog_normalized_different_dtypes() {
    let text_a = "(let t0 (Input 42 \"boundary\" (F32)))\n";
    let text_b = "(let t0 (Input 42 \"boundary\" (F16)))\n";
    assert_ne!(
        hash_egglog_normalized(text_a),
        hash_egglog_normalized(text_b),
        "Different dtypes should produce different hashes"
    );
}

#[test]
fn test_hash_egglog_normalized_output_join_not_normalized() {
    // OutputJoin lines should be hashed verbatim, not treated as Output
    let text_a = "(let t0 (OutputJoin t1 t2))\n";
    let text_b = "(let t0 (OutputJoin t3 t4))\n";
    assert_ne!(
        hash_egglog_normalized(text_a),
        hash_egglog_normalized(text_b),
        "OutputJoin lines should be hashed verbatim"
    );
}

#[test]
fn test_hash_egglog_normalized_distinguishes_persist_only_output() {
    let observed = "(let t1 (Output t0 42 false))\n";
    let persist_only = "(let t1 (Output t0 42 true))\n";
    assert_ne!(
        hash_egglog_normalized(observed),
        hash_egglog_normalized(persist_only),
        "persistence and observed-output semantics must not share a cached egraph"
    );
}

#[test]
fn test_hash_egglog_normalized_custom_op_id() {
    // CustomOpKind lines differ only in the integer ID (layer index)
    let text_a = r#"(let t0 (Input 441 "boundary" (F32)))
(let t1 (Op (CustomOpKind 1 (F32)) (ICons t74 (ICons t120 (ICons t28 (INil))))))
(let t2 (Output t1 585 false))
"#;
    let text_b = r#"(let t0 (Input 585 "boundary" (F32)))
(let t1 (Op (CustomOpKind 2 (F32)) (ICons t74 (ICons t120 (ICons t28 (INil))))))
(let t2 (Output t1 729 false))
"#;
    assert_eq!(
        hash_egglog_normalized(text_a),
        hash_egglog_normalized(text_b),
        "CustomOpKind with different IDs should hash the same"
    );
}

#[test]
fn test_hash_egglog_normalized_custom_op_different_structure() {
    // CustomOpKind lines with different input lists should hash differently
    let text_a = "(let t1 (Op (CustomOpKind 1 (F32)) (ICons t74 (ICons t120 (INil)))))\n";
    let text_b = "(let t1 (Op (CustomOpKind 1 (F32)) (ICons t74 (ICons t99 (INil)))))\n";
    assert_ne!(
        hash_egglog_normalized(text_a),
        hash_egglog_normalized(text_b),
        "CustomOpKind with different input lists should hash differently"
    );
}

#[test]
fn test_rolling_op_signature_custom_op_content() {
    #[derive(Debug)]
    struct TestCustomOp {
        #[allow(dead_code)]
        name: &'static str,
    }
    impl CustomOp for TestCustomOp {
        fn to_llir_op(&self) -> LLIROp {
            unimplemented!()
        }
    }

    // The signature cache is keyed by NodeIndex and only cleared by
    // best_rolling_candidate; drop entries another test on this thread
    // may have left behind for the same indices.
    clear_rolling_sig_cache();

    let mut cx = Graph::new();
    cx.custom_ops.push(Box::new(TestCustomOp { name: "rope" }));
    cx.custom_ops.push(Box::new(TestCustomOp { name: "rope" }));
    cx.custom_ops.push(Box::new(TestCustomOp { name: "topk" }));
    let ids: Vec<_> = (0..3)
        .map(|id| {
            cx.add_op(
                CustomOpKind {
                    id,
                    dtype: DType::F32,
                },
                &[],
            )
        })
        .collect();

    assert_eq!(
        rolling_op_signature(&cx.graph, ids[0], &cx.custom_ops),
        rolling_op_signature(&cx.graph, ids[1], &cx.custom_ops),
        "separate instances of the same custom op should sign the same"
    );
    assert_ne!(
        rolling_op_signature(&cx.graph, ids[0], &cx.custom_ops),
        rolling_op_signature(&cx.graph, ids[2], &cx.custom_ops),
        "different custom ops should sign differently"
    );
}

#[test]
fn test_auto_roll_loops_prepass_creates_regions_for_chain_recurrence() {
    let mut cx = Graph::new();
    let x = cx.tensor(8);
    let out = x.exp2().sin().exp2().sin().exp2().sin().output();

    let inserted = cx.auto_roll_loops_prepass_with_log(true);
    assert!(
        inserted >= 2,
        "expected at least two loop boundaries for 3 repeated bodies, got {inserted}"
    );

    let vals = random_vec(8);
    let mut rt = ReferenceRuntime::default();
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
    rt.set_data(x.id, vals.clone());
    rt.execute(&cx.dyn_map);

    let expected = vals
        .into_iter()
        .map(|v| v.exp2().sin().exp2().sin().exp2().sin())
        .collect::<Vec<f32>>();
    assert_close(rt.get_f32(out.id), &expected);
}

#[test]
fn caller_compiler_facts_are_visible_during_search_space_build() {
    let mut cx = Graph::new();
    let input = cx.tensor(1);
    input.output();
    cx.build_search_space::<ReferenceRuntime>(
        CompileOptions::default()
            .compiler_facts("(relation external-layout-class (i64))\n(external-layout-class 7)"),
    );
    assert!(
        cx.egraph()
            .unwrap()
            .enodes
            .values()
            .any(|(op, _)| { op == "external-layout-class" })
    );
}

#[test]
fn runtime_facts_join_caller_constraints_before_saturation() {
    struct DeviceRuntime(i64);
    impl Runtime for DeviceRuntime {
        type Ops = ();
        type CompileArg = i64;
        type ExecReturn = ();

        fn initialize(device: i64) -> Self {
            Self(device)
        }
        fn load_llir(&mut self, _: &LLIRGraph) {
            unreachable!("this test observes the saturation-to-runtime boundary")
        }
        fn compilation_facts(&self) -> String {
            format!(
                "(relation backend-device (i64))\n(backend-device {})",
                self.0
            )
        }
        fn compile(
            &mut self,
            space: &crate::search::SearchSpace,
            _: &DynMap,
            options: &CompileOptions,
            _: &mut dyn rand::RngCore,
        ) {
            for relation in ["backend-device", "external-layout-class"] {
                assert!(
                    space.buckets[0]
                        .egraph
                        .enodes
                        .values()
                        .any(|(op, _)| op == relation)
                );
            }
            assert!(
                options
                    .compiler_facts
                    .contains(&format!("(backend-device {})", self.0))
            );
        }
        fn execute(&mut self, _: &DynMap) {}
    }

    for device in [2, 5] {
        let mut graph = Graph::new();
        graph.tensor(1).output();
        graph.compile(
            DeviceRuntime::initialize(device),
            CompileOptions::default().compiler_facts(
                "(relation external-layout-class (i64))\n(external-layout-class 7)",
            ),
        );
    }
}

#[derive(Debug)]
struct CompilerFactCustomOp;

impl CustomOp for CompilerFactCustomOp {
    fn to_llir_op(&self) -> LLIROp {
        use crate::hlir::{ReferenceOp, Sin};
        LLIROp::new::<dyn ReferenceOp>(Box::new(Sin::default()) as Box<dyn ReferenceOp>)
    }

    fn compiler_declarations(&self) -> &'static str {
        "(relation test-custom-fact (i64))"
    }

    fn compiler_facts(&self, custom_op_id: usize) -> String {
        format!("(test-custom-fact {custom_op_id})")
    }
}

#[test]
fn custom_op_compiler_facts_are_visible_during_search_space_build() {
    let mut cx = Graph::new();
    let input = cx.tensor(1);
    cx.custom_op(CompilerFactCustomOp, input, 1, DType::F32)
        .output();
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    assert!(
        cx.egraph()
            .unwrap()
            .enodes
            .values()
            .any(|(op, _)| op == "test-custom-fact")
    );
}

#[test]
#[should_panic(expected = "UnboundFunction(\"test-custom-fact\"")]
fn custom_op_facts_are_rejected_without_their_declaration() {
    #[derive(Debug)]
    struct MissingDeclaration;

    impl CustomOp for MissingDeclaration {
        fn to_llir_op(&self) -> LLIROp {
            use crate::hlir::{ReferenceOp, Sin};
            LLIROp::new::<dyn ReferenceOp>(Box::new(Sin::default()) as Box<dyn ReferenceOp>)
        }

        fn compiler_facts(&self, custom_op_id: usize) -> String {
            format!("(test-custom-fact {custom_op_id})")
        }
    }

    let mut cx = Graph::new();
    let input = cx.tensor(1);
    cx.custom_op(MissingDeclaration, input, 1, DType::F32)
        .output();
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
}

#[test]
fn test_auto_roll_loops_prepass_rolls_recurrence_with_interleaved_outputs() {
    let mut cx = Graph::new();
    let x = cx.tensor(8);
    let mut y = x;
    for _ in 0..10 {
        y.exp2().output();
        y = y.sin();
    }
    let y = y.output();

    let before = cx.graph.node_count();
    let inserted = cx.auto_roll_loops_prepass_with_log(true);
    let after = cx.graph.node_count();
    assert!(
        inserted >= 2,
        "expected loop markers for recurrence split by Output nodes, got {inserted}"
    );
    assert!(
        after < before,
        "expected rolling to reduce nodes for recurrence split by Output nodes ({before} -> {after})"
    );

    let vals = random_vec(8);
    let mut rt = ReferenceRuntime::default();
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
    rt.set_data(x.id, vals.clone());
    rt.execute(&cx.dyn_map);

    let expected = vals
        .into_iter()
        .map(|mut v| {
            for _ in 0..10 {
                v = v.sin();
            }
            v
        })
        .collect::<Vec<f32>>();
    assert_close(rt.get_f32(y.id), &expected);
}

#[test]
fn test_auto_roll_loops_prepass_skips_non_recurrent_branches() {
    let mut cx = Graph::new();
    let x = cx.tensor(8);
    let y = cx.tensor(8);
    let _out = (x.exp().sin() + y.exp().sin()).output();

    let inserted = cx.auto_roll_loops_prepass_with_log(true);
    assert_eq!(inserted, 0, "branch-only reuse should not roll into loops");
}

#[test]
fn test_auto_roll_loops_prepass_runs_when_logging_is_disabled() {
    let mut cx = Graph::new();
    let x = cx.tensor(8);
    let out = x.exp2().sin().exp2().sin().exp2().sin().output();

    let before = cx.graph.node_count();
    let inserted = cx.auto_roll_loops_prepass();
    let after = cx.graph.node_count();

    assert!(
        inserted >= 2,
        "expected loop rolling to run without ROLLING_LOG, got {inserted}"
    );
    assert!(
        after < before,
        "expected loop rolling to reduce nodes ({before} -> {after})"
    );
    assert!(
        cx.graph
            .neighbors_directed(out.id, Direction::Outgoing)
            .next()
            .is_none(),
        "output should remain a graph root"
    );
}

#[test]
fn test_nested_loop_rolling_rolls_periodic_layer_pattern() {
    // Periodic pattern like alternating-attention transformers: blocks
    // of 3 identical "layers" (sin) closed by a distinct one (exp2),
    // repeated 4 times. The first pass rolls the 4 blocks; the second
    // rolls the 3 identical layers inside the surviving body.
    let mut cx = Graph::new();
    let x = cx.tensor(8);
    let mut y = x;
    for _ in 0..4 {
        for _ in 0..3 {
            y = y.sin();
        }
        y = y.exp2();
    }
    let out = y.output();

    let first = cx.auto_roll_loops_prepass_with_log(true);
    assert!(first > 0, "expected the block pattern to roll");
    let second = cx.auto_roll_loops_prepass_with_log(true);
    assert!(
        second > 0,
        "expected the repeated layers inside the rolled body to roll"
    );

    let loop_ids: FxHashSet<usize> = cx
        .graph
        .node_indices()
        .filter_map(|n| {
            cx.try_get_op::<crate::hlir::LoopStart>(n)
                .map(|ls| ls.loop_id)
        })
        .collect();
    assert_eq!(loop_ids.len(), 2, "expected two distinct loop regions");

    let vals = random_vec(8);
    let mut rt = ReferenceRuntime::default();
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
    rt.set_data(x.id, vals.clone());
    rt.execute(&cx.dyn_map);

    let expected = vals
        .into_iter()
        .map(|mut v| {
            for _ in 0..4 {
                for _ in 0..3 {
                    v = v.sin();
                }
                v = v.exp2();
            }
            v
        })
        .collect::<Vec<f32>>();
    assert_close(rt.get_f32(out.id), &expected);
}

#[test]
fn test_nested_loop_rolling_chained_sibling_inner_regions() {
    // Mirror of gemma's rolled topology: an outer periodic block whose
    // body contains multiple distinct repeated runs, chained through
    // non-repeating ops. The runs roll into sibling regions nested
    // inside the outer region; unroll must find each sibling innermost.
    let mut cx = Graph::new();
    let x = cx.tensor(8);
    let mut y = x;
    for _ in 0..4 {
        for _ in 0..3 {
            y = y.sin();
        }
        y = y.exp2();
        for _ in 0..4 {
            y = y.sin();
        }
        y = y.reciprocal();
    }
    let out = y.output();

    let mut passes = 0;
    while cx.auto_roll_loops_prepass_with_log(true) > 0 {
        passes += 1;
    }
    assert!(
        passes >= 3,
        "expected outer + two sibling inner rolls, got {passes}"
    );

    let vals = random_vec(8);
    let mut rt = ReferenceRuntime::default();
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
    rt.set_data(x.id, vals.clone());
    rt.execute(&cx.dyn_map);

    let expected = vals
        .into_iter()
        .map(|mut v| {
            for _ in 0..4 {
                for _ in 0..3 {
                    v = v.sin();
                }
                v = v.exp2();
                for _ in 0..4 {
                    v = v.sin();
                }
                v = v.recip();
            }
            v
        })
        .collect::<Vec<f32>>();
    assert_close(rt.get_f32(out.id), &expected);
}

#[test]
fn test_nested_loop_rolling_with_varying_weights() {
    // Gemma-shaped: an outer periodic block of 3 identical weighted
    // layers plus a distinct closer, repeated 4 times, with a DISTINCT
    // weight tensor per layer. The inner region's per-iteration inputs
    // are then varying streams fed by the outer region's own LoopInput
    // markers — the exact structure of per-layer weights in a rolled
    // transformer.
    let mut cx = Graph::new();
    let x = cx.tensor(8);
    let weights: Vec<GraphTensor> = (0..12).map(|_| cx.tensor(8)).collect();
    let mut y = x;
    for block in 0..4 {
        for layer in 0..3 {
            y = (y * weights[block * 3 + layer]).sin();
        }
        y = y.exp2();
    }
    let out = y.output();

    let mut passes = 0;
    while cx.auto_roll_loops_prepass_with_log(true) > 0 {
        passes += 1;
    }
    assert!(passes >= 2, "expected nested rolls, got {passes}");

    let xv = random_vec(8);
    let wvs: Vec<Vec<f32>> = (0..12).map(|_| random_vec(8)).collect();
    let mut rt = ReferenceRuntime::default();
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
    rt.set_data(x.id, xv.clone());
    for (w, wv) in weights.iter().zip(&wvs) {
        rt.set_data(w.id, wv.clone());
    }
    rt.execute(&cx.dyn_map);

    let expected: Vec<f32> = (0..8)
        .map(|j| {
            let mut v = xv[j];
            for block in 0..4 {
                for layer in 0..3 {
                    v = (v * wvs[block * 3 + layer][j]).sin();
                }
                v = v.exp2();
            }
            v
        })
        .collect();
    assert_close(rt.get_f32(out.id), &expected);
}

#[test]
fn loop_rolling_stamps_concrete_varying_stream_dtypes() {
    // A transformer-style recurrence carries F32 activations while every
    // repeated layer receives a distinct BF16 weight. The weight casts are
    // part of the repeated body, so the rolled boundary itself must retain
    // BF16 rather than a placeholder chosen independently of its sources.
    let mut cx = Graph::new();
    let x = cx.tensor(8);
    let weights: Vec<GraphTensor> = (0..4).map(|_| cx.tensor(8).as_dtype(DType::Bf16)).collect();
    let mut y = x;
    for weight in weights {
        y = (y * weight.cast(DType::F32)).sin();
    }
    let _ = y.output();

    assert!(
        cx.auto_roll_loops_prepass_with_log(true) > 0,
        "expected the repeated weighted recurrence to roll"
    );

    let loop_inputs: Vec<_> = cx
        .graph
        .node_indices()
        .filter_map(|node| cx.try_get_op::<LoopInput>(node))
        .collect();
    assert!(!loop_inputs.is_empty(), "expected a varying weight stream");
    assert!(
        loop_inputs.iter().any(|input| input.dtype == DType::Bf16),
        "expected a concrete BF16 LoopInput, got {loop_inputs:?}"
    );
    assert!(
        cx.graph.node_indices().all(|node| {
            cx.try_get_op::<LoopStart>(node)
                .is_none_or(|start| start.dtype == DType::F32)
        }),
        "the carried F32 activation must remain concretely F32"
    );

    // The concrete field remains the source of truth through egglog
    // construction; this used to turn every marker field into F32.
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
}

#[test]
fn loop_rolling_preserves_integer_gather_stream_dtypes() {
    // Regression for mixed-type repeated regions: Gather consumes Int
    // indexes but produces the dtype of its F32 data input. Both facts
    // must survive rolling without one eclass overwriting the other.
    let mut cx = Graph::new();
    let x = cx.tensor(8);
    let indexes: Vec<GraphTensor> = (0..4)
        .map(|layer| {
            cx.named_tensor(format!("indexes.{layer}"), 8)
                .as_dtype(DType::Int)
        })
        .collect();
    let mut y = x;
    for index in indexes {
        y = y.gather(index).sin();
    }
    let _ = y.output();

    assert!(
        cx.auto_roll_loops_prepass_with_log(true) > 0,
        "expected the repeated gather recurrence to roll"
    );

    let loop_inputs: Vec<_> = cx
        .graph
        .node_indices()
        .filter_map(|node| cx.try_get_op::<LoopInput>(node))
        .collect();
    assert!(
        loop_inputs.iter().any(|input| input.dtype == DType::Int),
        "expected a concrete Int LoopInput, got {loop_inputs:?}"
    );
    assert!(
        cx.graph.node_indices().all(|node| {
            cx.try_get_op::<LoopStart>(node)
                .is_none_or(|start| start.dtype == DType::F32)
        }),
        "Gather must preserve the carried F32 activation dtype"
    );

    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
}

#[test]
fn test_nested_loop_rolling_per_layer_outputs_not_permuted() {
    // Cache-analog regression test: every layer persists a side output
    // (like per-layer KV caches). Nested rolling + unroll must route each
    // side output to ITS OWN layer's value — a permutation across layers
    // corrupts state promotion even when the final output is correct.
    let mut cx = Graph::new();
    let x = cx.tensor(8);
    let weights: Vec<GraphTensor> = (0..24).map(|_| cx.tensor(8)).collect();
    let mut y = x;
    let mut side = Vec::new();
    for block in 0..4 {
        for layer in 0..5 {
            y = (y * weights[block * 6 + layer]).sin();
            side.push((y * 2.0_f32).output());
        }
        y = (y * weights[block * 6 + 5]).exp2();
        side.push((y * 2.0_f32).output());
    }
    let out = y.output();

    let mut passes = 0;
    while cx.auto_roll_loops_prepass_with_log(true) > 0 {
        passes += 1;
    }
    assert!(passes >= 2, "expected nested rolls, got {passes}");

    let xv = random_vec(8);
    let wvs: Vec<Vec<f32>> = (0..24).map(|_| random_vec(8)).collect();
    let mut rt = ReferenceRuntime::default();
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
    rt.set_data(x.id, xv.clone());
    for (w, wv) in weights.iter().zip(&wvs) {
        rt.set_data(w.id, wv.clone());
    }
    rt.execute(&cx.dyn_map);

    let mut refs: Vec<Vec<f32>> = Vec::new();
    let mut v: Vec<f32> = xv.clone();
    for block in 0..4 {
        for layer in 0..5 {
            v = v
                .iter()
                .zip(&wvs[block * 6 + layer])
                .map(|(a, b)| (a * b).sin())
                .collect();
            refs.push(v.iter().map(|a| a * 2.0).collect());
        }
        v = v
            .iter()
            .zip(&wvs[block * 6 + 5])
            .map(|(a, b)| (a * b).exp2())
            .collect();
        refs.push(v.iter().map(|a| a * 2.0).collect());
    }
    let mut permutation = Vec::new();
    for (idx, t) in side.iter().enumerate() {
        let got = rt.get_f32(t.id);
        let matches: Vec<usize> = refs
            .iter()
            .enumerate()
            .filter(|(_, r)| got.iter().zip(*r).all(|(a, b)| (a - b).abs() < 1e-5))
            .map(|(j, _)| j)
            .collect();
        permutation.push((idx, matches));
    }
    let bad: Vec<_> = permutation.iter().filter(|(i, m)| !m.contains(i)).collect();
    assert!(
        bad.is_empty(),
        "side outputs carry other layers' values: {permutation:?}"
    );
    let final_expected: Vec<f32> = refs.last().unwrap().iter().map(|a| a / 2.0).collect();
    assert!(
        rt.get_f32(out.id)
            .iter()
            .zip(&final_expected)
            .all(|(a, b)| (a - b).abs() < 1e-5),
        "final output mismatch"
    );
}

#[test]
fn test_nested_loop_rolling_handles_disjoint_regions() {
    // Two independent recurrence chains roll into two disjoint regions
    // across successive passes; unroll must handle both.
    let mut cx = Graph::new();
    let x = cx.tensor(8);
    let y = cx.tensor(8);
    let a = x.sin().sin().sin().sin();
    let b = y.exp2().exp2().exp2().exp2();
    let out = (a + b).output();

    let mut passes = 0;
    while cx.auto_roll_loops_prepass_with_log(true) > 0 {
        passes += 1;
    }
    assert!(
        passes >= 2,
        "expected both chains to roll, got {passes} passes"
    );

    let xv = random_vec(8);
    let yv = random_vec(8);
    let mut rt = ReferenceRuntime::default();
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
    rt.set_data(x.id, xv.clone());
    rt.set_data(y.id, yv.clone());
    rt.execute(&cx.dyn_map);

    let expected = xv
        .into_iter()
        .zip(yv)
        .map(|(mut a, mut b)| {
            for _ in 0..4 {
                a = a.sin();
                b = b.exp2();
            }
            a + b
        })
        .collect::<Vec<f32>>();
    assert_close(rt.get_f32(out.id), &expected);
}
