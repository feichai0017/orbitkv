//! A retained graph must observe CSR contents as well as tensor shapes.

use super::*;
use crate::runtime::CudaRuntimeImpl;
use half::bf16;
use orbitkv_compiler::op::Runtime as _;
use orbitkv_ops::ops::attention::*;
use rand::{SeedableRng, rngs::SmallRng};

type Runtime = CudaRuntimeImpl<(crate::kernel::Ops, AttentionSemantics, FlashInferAttention)>;
const REQUESTS: usize = 2;
const QUERY_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const PAGE_SIZE: usize = 16;
const PAGES: usize = 18;
const QUERY_CAPACITY: usize = 132;
const VALUE_SCALE: f32 = 1.0 / 256.0;

struct Fixture {
    graph: Graph,
    runtime: Runtime,
    query: GraphTensor,
    query_indptr: GraphTensor,
    page_indptr: GraphTensor,
    pages: [usize; REQUESTS],
    output: GraphTensor,
}

impl Fixture {
    fn new() -> Self {
        let mut graph = Graph::default();
        let query = graph
            .tensor(('q', QUERY_HEADS, HEAD_DIM))
            .as_dtype(DType::Bf16)
            .persist();
        let key = graph
            .tensor((PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM))
            .as_dtype(DType::Bf16)
            .persist();
        let value = graph
            .tensor((PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM))
            .as_dtype(DType::Bf16)
            .persist();
        let indices = graph.tensor(PAGES).as_dtype(DType::Int).persist();
        let query_indptr = graph.tensor(REQUESTS + 1).as_dtype(DType::Int).persist();
        let page_indptr = graph.tensor(REQUESTS + 1).as_dtype(DType::Int).persist();
        let last = graph.tensor(REQUESTS).as_dtype(DType::Int).persist();
        let output = attention(
            AttentionInputs {
                query,
                query_indptr,
                kv: KvView::Paged(PagedKvView {
                    state_class_id: 0,
                    key,
                    value,
                    page_size: PAGE_SIZE,
                    layout: PagedKvLayout::TokenMajor,
                    page_indices: indices,
                    page_indptr,
                    last_page_len: last,
                }),
            },
            AttentionSpec {
                query_heads: QUERY_HEADS,
                kv_heads: KV_HEADS,
                query_key_dim: HEAD_DIM,
                value_dim: HEAD_DIM,
                dtype: DType::Bf16,
                scale: 1.0 / (HEAD_DIM as f64).sqrt(),
                mask: AttentionMask::Causal,
            },
        )
        .unwrap()
        .output();
        let context = cudarc::driver::CudaContext::new(0).unwrap();
        // Match serving's nonblocking stream; the legacy default stream can
        // implicitly hide missing ordering with the private capture stream.
        let mut runtime = Runtime::initialize(context.new_stream().unwrap());
        runtime.set_data(
            query,
            vec![bf16::ZERO; QUERY_CAPACITY * QUERY_HEADS * HEAD_DIM],
        );
        runtime.set_data(
            key,
            vec![bf16::ZERO; PAGES * PAGE_SIZE * KV_HEADS * HEAD_DIM],
        );
        // The reference averages these BF16-rounded values independently.
        let values = (0..PAGES * PAGE_SIZE)
            .flat_map(|token| {
                std::iter::repeat_n(
                    bf16::from_f32(token as f32 * VALUE_SCALE),
                    KV_HEADS * HEAD_DIM,
                )
            })
            .collect::<Vec<_>>();
        runtime.set_data(value, values);
        runtime.set_data(indices, (0..PAGES as i32).collect::<Vec<_>>());
        runtime.set_data(
            page_indptr,
            vec![0_i32, (PAGES / REQUESTS) as i32, PAGES as i32],
        );
        runtime.set_data(last, vec![PAGE_SIZE as i32; REQUESTS]);
        runtime.set_data(
            query_indptr,
            vec![0_i32, (QUERY_CAPACITY - 1) as i32, QUERY_CAPACITY as i32],
        );
        runtime.register_profile_input_generator(move |dims| {
            let rows = dims[&Symbol::from('q')];
            vec![(query_indptr.id, vec![0, (rows - 1) as i32, rows as i32])]
        });
        graph.set_dim('q', QUERY_CAPACITY);
        runtime = graph.compile_with_rng(
            runtime,
            CompileOptions::default().search_graph_limit(1).dim_buckets(
                'q',
                &[
                    DimBucket::new(REQUESTS, REQUESTS),
                    DimBucket::new(REQUESTS + 1, QUERY_CAPACITY).representative(QUERY_CAPACITY),
                ],
            ),
            &mut SmallRng::seed_from_u64(0xC58),
        );
        runtime.set_max_materialized_buckets(Some(2));
        Self {
            graph,
            runtime,
            query,
            query_indptr,
            page_indptr,
            pages: [PAGES / REQUESTS; REQUESTS],
            output,
        }
    }

    fn bind(&mut self, queries: [usize; REQUESTS]) {
        let rows = queries.iter().sum::<usize>();
        self.graph.set_dim('q', rows);
        self.runtime
            .set_data(self.query, vec![bf16::ZERO; rows * QUERY_HEADS * HEAD_DIM]);
        self.runtime.set_data(
            self.query_indptr,
            vec![0_i32, queries[0] as i32, rows as i32],
        );
        self.runtime.set_data(
            self.page_indptr,
            vec![0_i32, self.pages[0] as i32, PAGES as i32],
        );
    }

    fn check(&mut self, queries: [usize; REQUESTS]) {
        self.bind(queries);
        self.runtime.execute(&self.graph.dyn_map);
        let actual = self.runtime.get_bf16(self.output);
        let mut row = 0;
        for (request, query_count) in queries.into_iter().enumerate() {
            for query in 0..query_count {
                let visible = self.pages[request] * PAGE_SIZE - query_count + query + 1;
                // Uniform causal softmax: average values 0..visible for this request.
                let first_page = self.pages[..request].iter().sum::<usize>();
                let expected = (0..visible)
                    .map(|token| {
                        bf16::from_f32((first_page * PAGE_SIZE + token) as f32 * VALUE_SCALE)
                            .to_f32()
                    })
                    .sum::<f32>()
                    / visible as f32;
                for head in 0..QUERY_HEADS {
                    let start = (head * queries.iter().sum::<usize>() + row) * HEAD_DIM;
                    for value in &actual[start..start + HEAD_DIM] {
                        assert!(
                            (value.to_f32() - expected).abs() < 0.016,
                            "queries={queries:?}, row={row}, actual={value}, expected={expected}"
                        );
                    }
                }
                row += 1;
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA and FlashInfer; retained plans must observe changed CSR contents"]
fn prepared_buckets_refresh_csr_without_shape_or_pointer_changes() {
    let mut fixture = Fixture::new();
    let skewed = [QUERY_CAPACITY - 1, 1];
    let balanced = [QUERY_CAPACITY / REQUESTS; REQUESTS];
    for queries in [skewed, [1, 1]] {
        fixture.bind(queries);
        fixture
            .runtime
            .prepare_cuda_graphs(&fixture.graph.dyn_map)
            .unwrap();
    }
    for queries in [balanced, [1, 1], balanced, [1, QUERY_CAPACITY - 1], skewed] {
        fixture.check(queries);
    }
    for pages in [[7, 11], [11, 7], [PAGES / REQUESTS; REQUESTS]] {
        fixture.pages = pages;
        fixture.check(balanced);
    }
    let recaptures = || {
        fixture
            .runtime
            .debug_cuda_graph_summaries()
            .iter()
            .flat_map(|summary| &summary.flashinfer_recapture_counts)
            .sum::<usize>()
    };
    let before = recaptures();
    fixture.check(balanced);
    let after = fixture
        .runtime
        .debug_cuda_graph_summaries()
        .iter()
        .flat_map(|summary| &summary.flashinfer_recapture_counts)
        .sum::<usize>();
    assert_eq!(before, after, "identical CSR must reuse the prepared plan");
}
