use super::*;
use crate::{
    graph::CompileOptions,
    hlir::{Recip, ReferenceOp, ReferenceRuntime, Sin},
    op::{CustomOp, LLIROp},
};

#[derive(Debug)]
struct ReferenceSin;

impl CustomOp for ReferenceSin {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn ReferenceOp>(Box::new(Sin::default()) as Box<dyn ReferenceOp>)
    }
}

#[derive(Debug)]
struct ReferenceRecip;

impl CustomOp for ReferenceRecip {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn ReferenceOp>(Box::new(Recip::default()) as Box<dyn ReferenceOp>)
    }
}

fn renumber_llir(llir: &LLIRGraph) -> LLIRGraph {
    let nodes = llir.node_indices().collect::<Vec<_>>();
    let mut rebuilt = LLIRGraph::default();
    let mapping = nodes
        .iter()
        .rev()
        .map(|&node| (node, rebuilt.add_node(llir[node].clone())))
        .collect::<FxHashMap<_, _>>();
    for &target in nodes.iter().rev() {
        let mut incoming = llir
            .edges_directed(target, petgraph::Direction::Incoming)
            .map(|edge| (edge.id().index(), edge.source()))
            .collect::<Vec<_>>();
        incoming.sort_unstable_by_key(|(edge, _)| *edge);
        for (_, source) in incoming {
            rebuilt.add_edge(mapping[&source], mapping[&target], ());
        }
    }
    rebuilt
}

fn selected_schedule() -> (Graph, SelectedSchedule) {
    let mut graph = Graph::new();
    let _ = graph.tensor(4).sin().output();
    graph.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let space = graph.search_space().unwrap();
    let ctx = &space.bucket_contexts(&graph.dyn_map)[0];
    let mut extractor = LlirExtractor::new(ctx.egraph(), &space.ops);
    let genome = extractor.random_indexed_choice(&mut rand::rng());
    let llir = unroll_packed_llir(extractor.extract_indexed_packed(&genome, &[]));
    let selected = SelectedProgram {
        bucket_indices: ctx.bucket_indices().clone(),
        representative_dyn_map: ctx.representative_dyn_map.clone(),
        genome,
        llir,
    };
    let schedule = SelectedSchedule::from_search(space, &[selected]).unwrap();
    (graph, schedule)
}

#[test]
fn selected_schedule_round_trip_skips_search() {
    let (graph, schedule) = selected_schedule();
    let bytes = serde_json::to_vec(&schedule).unwrap();
    let schedule = serde_json::from_slice(&bytes).unwrap();
    let loaded =
        Graph::from_selected_schedule(graph.dyn_map.clone(), graph.input_meta.clone(), schedule);
    loaded
        .load_selected_schedule(&mut ReferenceRuntime::initialize(()))
        .unwrap();
}

#[test]
fn selected_schedule_round_trip_resolves_current_custom_ops() {
    let build = || {
        let mut graph = Graph::new();
        let input = graph.tensor(4);
        let output = graph.custom_op(ReferenceSin, input, 4, DType::F32);
        output.output();
        graph
    };
    let mut searched = build();
    searched.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let space = searched.search_space().unwrap();
    let ctx = &space.bucket_contexts(&searched.dyn_map)[0];
    let mut extractor = LlirExtractor::new(ctx.egraph(), &space.ops);
    let genome = extractor.random_indexed_choice(&mut rand::rng());
    let llir = unroll_packed_llir(extractor.extract_indexed_packed(&genome, &space.custom_ops));
    let selected = SelectedProgram {
        bucket_indices: ctx.bucket_indices().clone(),
        representative_dyn_map: ctx.representative_dyn_map.clone(),
        genome,
        llir,
    };
    let bytes =
        serde_json::to_vec(&SelectedSchedule::from_search(space, &[selected]).unwrap()).unwrap();

    let mut loaded = build();
    loaded.install_selected_schedule(serde_json::from_slice(&bytes).unwrap());
    loaded
        .load_selected_schedule(&mut ReferenceRuntime::initialize(()))
        .unwrap();
}

#[test]
fn selected_schedule_rejects_changed_custom_op() {
    let mut searched = Graph::new();
    let input = searched.tensor(4);
    searched
        .custom_op(ReferenceSin, input, 4, DType::F32)
        .output();
    searched.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let space = searched.search_space().unwrap();
    let ctx = &space.bucket_contexts(&searched.dyn_map)[0];
    let mut extractor = LlirExtractor::new(ctx.egraph(), &space.ops);
    let genome = extractor.random_indexed_choice(&mut rand::rng());
    let selected = SelectedProgram {
        bucket_indices: ctx.bucket_indices().clone(),
        representative_dyn_map: ctx.representative_dyn_map.clone(),
        llir: unroll_packed_llir(extractor.extract_indexed_packed(&genome, &space.custom_ops)),
        genome,
    };
    let schedule = SelectedSchedule::from_search(space, &[selected]).unwrap();

    let mut changed = Graph::new();
    let input = changed.tensor(4);
    changed
        .custom_op(ReferenceRecip, input, 4, DType::F32)
        .output();
    changed.install_selected_schedule(schedule);
    let error = changed
        .load_selected_schedule(&mut ReferenceRuntime::initialize(()))
        .unwrap_err();
    assert!(error.contains("fingerprint mismatch"), "{error}");
}

#[test]
fn selected_schedule_rejects_changed_llir() {
    let (graph, mut schedule) = selected_schedule();
    schedule.buckets[0].unrolled_llir_fingerprint.0 ^= 1;
    let loaded = Graph::from_selected_schedule(graph.dyn_map, graph.input_meta, schedule);
    let error = loaded
        .load_selected_schedule(&mut ReferenceRuntime::initialize(()))
        .unwrap_err();
    assert!(error.contains("fingerprint mismatch"), "{error}");
}

#[test]
fn reference_compile_retains_selected_schedule() {
    let mut graph = Graph::new();
    let _ = graph.tensor(4).sin().output();
    let _runtime = graph.compile(ReferenceRuntime::initialize(()), CompileOptions::default());

    assert!(graph.selected_schedule().is_some());
}

#[test]
fn llir_fingerprint_ignores_graph_allocation_order() {
    let (graph, _) = selected_schedule();
    let space = graph.search_space().unwrap();
    let ctx = &space.bucket_contexts(&graph.dyn_map)[0];
    let mut extractor = LlirExtractor::new(ctx.egraph(), &space.ops);
    let genome = extractor.random_indexed_choice(&mut rand::rng());
    let llir = unroll_packed_llir(extractor.extract_indexed_packed(&genome, &[]));

    assert_eq!(
        fingerprint_llir(&llir),
        fingerprint_llir(&renumber_llir(&llir))
    );
}

#[test]
fn llir_fingerprint_retains_operation_parameters() {
    let graph_with_constant = |value| {
        let mut llir = LLIRGraph::default();
        llir.add_node(LLIROp::new::<dyn ReferenceOp>(
            Box::new(crate::hlir::Constant(value)) as Box<dyn ReferenceOp>,
        ));
        llir
    };

    assert_ne!(
        fingerprint_llir(&graph_with_constant(1.0)),
        fingerprint_llir(&graph_with_constant(2.0))
    );
}
