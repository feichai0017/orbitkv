use crate::runtime::CudaRuntime;
use cudarc::driver::{CudaContext, CudaSlice};
use orbitkv_compiler::prelude::*;
use rand::{SeedableRng, rngs::SmallRng};

const ROW_CAPACITY: usize = 8;

/// In-place state writes make accidental model execution during preparation
/// observable. The query interval also admits shapes other than representatives.
struct StateGraph {
    graph: Graph,
    runtime: CudaRuntime,
    state: CudaSlice<u8>,
    input: GraphTensor,
    slots: GraphTensor,
    output: GraphTensor,
}

impl StateGraph {
    fn new(capacity: usize) -> Self {
        let stream = CudaContext::new(0).unwrap().default_stream();
        let mut graph = Graph::default();
        graph.set_dim('s', 2);
        let input = graph.tensor(('s', 4));
        let slots = graph.tensor(('s',)).as_dtype(DType::Int);
        let cache = graph.tensor((8, 4)).persist();
        let output = orbitkv_ops::scatter_rows(input, slots, cache, 4).output();
        let mut runtime = CudaRuntime::initialize(stream.clone());
        runtime.set_data_with_capacity(input, vec![7_f32; 8], 8 * 4 * size_of::<f32>());
        runtime.set_data_with_capacity(slots, vec![0_i32, 1], ROW_CAPACITY * size_of::<i32>());
        let mut state = runtime.alias_state_required(cache, output, 8 * 4 * size_of::<f32>());
        runtime = graph.compile_with_rng(
            runtime,
            CompileOptions::default().search_graph_limit(1).dim_buckets(
                's',
                &[
                    DimBucket::new(1, 4).representative(2),
                    DimBucket::new(5, 8).representative(6),
                ],
            ),
            &mut SmallRng::seed_from_u64(47),
        );
        stream.memset_zeros(&mut state).unwrap();
        runtime.set_max_materialized_buckets(Some(capacity));
        runtime.begin_cuda_graph_warmup(&[]);
        Self {
            graph,
            runtime,
            state,
            input,
            slots,
            output,
        }
    }

    fn bind(&mut self, rows: usize) {
        self.graph.set_dim('s', rows);
        self.runtime.set_data(self.input, vec![7_f32; rows * 4]);
        self.runtime
            .set_data(self.slots, (0..rows as i32).collect::<Vec<_>>());
    }

    fn prepare(&mut self, rows: usize) {
        self.bind(rows);
        self.runtime
            .prepare_cuda_graphs(&self.graph.dyn_map)
            .unwrap();
        assert!(
            self.runtime
                .cuda_stream
                .clone_dtoh(&self.state)
                .unwrap()
                .iter()
                .all(|&byte| byte == 0),
            "preparation executed a persistent state write"
        );
    }
}

#[test]
#[ignore = "requires CUDA; verifies preparation, state preservation and dynamic replay"]
fn preparation_preserves_state_and_reuses_retained_buckets() {
    let mut fixture = StateGraph::new(2);
    assert_eq!(fixture.runtime.max_materialized_buckets(), Some(2));
    let allocation = fixture.runtime.input_allocation(fixture.input);
    fixture.prepare(2);
    fixture.prepare(6);
    assert_eq!(fixture.runtime.debug_materialized_bucket_indices(), [0, 1]);
    let prepared = fixture.runtime.cuda_graph_residency_stats();
    fixture.prepare(2);
    fixture.prepare(6);
    assert_eq!(fixture.runtime.cuda_graph_residency_stats(), prepared);
    assert_eq!(fixture.runtime.input_allocation(fixture.input), allocation);
    fixture.bind(3);
    fixture.runtime.execute(&fixture.graph.dyn_map);
    let expected = [vec![7_f32; 3 * 4], vec![0_f32; 5 * 4]].concat();
    assert_eq!(fixture.runtime.get_f32(fixture.output), expected);
    assert_eq!(fixture.runtime.cuda_graph_residency_stats(), prepared);
}

#[test]
#[ignore = "requires CUDA; verifies preparation obeys the same eviction budget as execution"]
fn preparation_obeys_capacity_and_rebuilds_evicted_buckets() {
    let mut fixture = StateGraph::new(1);
    assert_eq!(fixture.runtime.max_materialized_buckets(), Some(1));
    let mut previous = fixture.runtime.cuda_graph_residency_stats().0;
    for (rows, bucket) in [(2, 0), (6, 1), (3, 0), (8, 1)] {
        fixture.prepare(rows);
        assert_eq!(
            fixture.runtime.debug_materialized_bucket_indices(),
            [bucket]
        );
        let current = fixture.runtime.cuda_graph_residency_stats().0;
        assert!(current > previous);
        previous = current;
    }
}

#[test]
#[ignore = "requires CUDA; a frozen deployment must reject further preparation"]
fn preparation_rejects_frozen_execution() {
    let mut fixture = StateGraph::new(2);
    fixture.runtime.begin_cuda_graph_preparation(&[]);
    fixture.prepare(2);
    fixture.prepare(6);
    fixture.runtime.finish_cuda_graph_preparation().unwrap();
    let before = fixture.runtime.cuda_graph_residency_stats();
    assert!(
        fixture
            .runtime
            .prepare_cuda_graphs(&fixture.graph.dyn_map)
            .is_err()
    );
    assert_eq!(fixture.runtime.cuda_graph_residency_stats(), before);
}
