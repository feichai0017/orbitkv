//! Semantic identity must survive resource planning, device preparation and replay.

use super::*;
use orbitkv_compiler::graph::{LLIRGraph, llir_program_identity};
use rand::SeedableRng;

fn matmul(rows: Expression) -> CuBlasLt {
    CuBlasLt {
        m: rows,
        n: 4.into(),
        k: 3.into(),
        a_layout: cublasOperation_t::CUBLAS_OP_N,
        b_layout: cublasOperation_t::CUBLAS_OP_N,
        a_order: cublasLtOrder_t::CUBLASLT_ORDER_ROW,
        b_order: cublasLtOrder_t::CUBLASLT_ORDER_ROW,
        c_order: cublasLtOrder_t::CUBLASLT_ORDER_ROW,
        d_order: cublasLtOrder_t::CUBLASLT_ORDER_ROW,
        lda: 3.into(),
        ldb: 4.into(),
        ldc: 4.into(),
        ldd: 4.into(),
        ..Default::default()
    }
}

fn graph_with_op(op: CuBlasLt) -> (LLIRGraph, NodeIndex) {
    let mut graph = LLIRGraph::default();
    let node = graph.add_node(LLIROp::new::<dyn HostOp>(Box::new(op) as Box<dyn HostOp>));
    (graph, node)
}

fn operation(graph: &LLIRGraph, node: NodeIndex) -> &CuBlasLt {
    let host: &dyn HostOp = &***graph[node].to_dialect::<dyn HostOp>().unwrap();
    host.as_any().downcast_ref::<CuBlasLt>().unwrap()
}

#[test]
fn program_identity_ignores_prepared_shapes_but_retains_matmul_semantics() {
    let (graph, node) = graph_with_op(matmul('m'.into()));
    let identity = llir_program_identity(&graph);
    let op = operation(&graph, node);
    assert!(op.resource_prepare_cache.lock().unwrap().is_none());
    for rows in [2, 5, 1] {
        let dims = [(Symbol::from('m'), rows)].into_iter().collect();
        let prepared = op.prepare_key_for_resources(&dims).unwrap();
        assert_eq!(prepared.spec.problem.m, rows as u64);
        assert!(op.resource_prepare_cache.lock().unwrap().is_some());
        assert_eq!(llir_program_identity(&graph), identity);
    }
    let changes: [fn(&mut CuBlasLt); 4] = [
        |op: &mut CuBlasLt| op.alpha = 2.0,
        |op: &mut CuBlasLt| op.n = 5.into(),
        |op: &mut CuBlasLt| op.a_order = cublasLtOrder_t::CUBLASLT_ORDER_COL,
        |op: &mut CuBlasLt| op.a_scale_input = true,
    ];
    for change in changes {
        let mut changed = matmul('m'.into());
        change(&mut changed);
        assert_ne!(llir_program_identity(&graph_with_op(changed).0), identity);
    }
}

#[test]
#[ignore = "requires CUDA and cuBLASLt; verifies identity across real handle state changes"]
fn program_identity_ignores_cublas_handle_and_context_stream_counts() {
    let context = crate::cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let (graph, node) = graph_with_op(matmul('m'.into()));
    let identity = llir_program_identity(&graph);
    let op = operation(&graph, node);
    op.get_cublaslt(&stream).unwrap();
    assert!(op.cublaslt.get().is_some());
    assert_eq!(llir_program_identity(&graph), identity);
    let other_stream = context.new_stream().unwrap();
    op.prepare_key_for_resources(&[(Symbol::from('m'), 2)].into_iter().collect())
        .unwrap();
    assert_eq!(llir_program_identity(&graph), identity);
    drop(other_stream);
    assert_eq!(llir_program_identity(&graph), identity);
}

#[test]
#[ignore = "requires CUDA and cuBLASLt; checks profiled trace and strict schedule replay"]
fn executed_matmul_trace_and_replayed_schedule_share_program_identity() {
    use crate::runtime::CudaRuntime;
    use orbitkv_compiler::op::CustomOp;
    use serde_json::Value;

    #[derive(Debug)]
    struct Projection;
    impl CustomOp for Projection {
        fn to_llir_op(&self) -> LLIROp {
            LLIROp::new::<dyn HostOp>(Box::new(matmul(2.into())) as Box<dyn HostOp>)
        }
    }
    let build = || {
        let mut graph = Graph::default();
        let input = graph.tensor((2, 3));
        let weight = graph.tensor((3, 4));
        let output = graph
            .custom_op(Projection, vec![input, weight], (2, 4), DType::F32)
            .output();
        (graph, input, weight, output)
    };
    let path = std::env::temp_dir().join(format!(
        "orbitkv-cublaslt-identity-{}.jsonl",
        uuid::Uuid::new_v4()
    ));
    let context = crate::cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let (mut graph, input, weight, output) = build();
    let mut runtime = CudaRuntime::initialize(stream.clone());
    let input_values = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    runtime.set_data(input, input_values.clone());
    runtime.set_data(weight, vec![1.0_f32; 12]);
    let options = CompileOptions::default()
        .search_graph_limit(1)
        .search_log(false)
        .search_trace(&path);
    runtime = graph.compile_with_rng(runtime, options, &mut rand::rngs::StdRng::seed_from_u64(17));
    runtime.set_data(input, input_values.clone());
    runtime.set_data(weight, vec![1.0_f32; 12]);
    runtime.execute(&graph.dyn_map);
    assert_eq!(
        runtime.get_f32(output),
        [6.0; 4].into_iter().chain([15.0; 4]).collect::<Vec<_>>()
    );

    let records = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    let selected = records
        .iter()
        .find(|row| row["event"] == "selected")
        .unwrap();
    for event in ["direct", "deployment", "finalist_validation", "program"] {
        assert!(
            records
                .iter()
                .any(|row| row["event"] == event && row["program"] == selected["program"]),
            "missing {event} for selected program"
        );
    }
    let schedule = graph.selected_schedule().unwrap();
    let serialized = serde_json::to_value(schedule).unwrap();
    for bucket in serialized["buckets"].as_array().unwrap() {
        let pair = bucket["unrolled_llir_fingerprint"].as_array().unwrap();
        let identity = format!(
            "{:016x}{:016x}",
            pair[0].as_u64().unwrap(),
            pair[1].as_u64().unwrap()
        );
        assert_eq!(identity, selected["program"].as_str().unwrap());
    }

    let (mut loaded, input, weight, output) = build();
    loaded.install_selected_schedule(schedule.clone());
    let mut replay = CudaRuntime::initialize(stream);
    replay.set_data(input, input_values);
    replay.set_data(weight, vec![1.0_f32; 12]);
    loaded.load_selected_schedule(&mut replay).unwrap();
    replay.execute(&loaded.dyn_map);
    assert_eq!(
        replay.get_f32(output),
        [6.0; 4].into_iter().chain([15.0; 4]).collect::<Vec<_>>()
    );
    std::fs::remove_file(path).unwrap();
}
