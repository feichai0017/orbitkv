//! Dynamic mixed-library capture regression with an independent constant oracle.

use super::*;
use crate::{providers::flashinfer::FlashInferAttention, runtime::CudaRuntime};
use orbitkv_compiler::{graph::DimBucket, op::CustomOp};

const WIDTH: usize = 64;
const PAGE_TOKENS: usize = 16;
const MAX_REQUESTS: usize = 4;
const MAX_ROWS: usize = 12;

#[derive(Debug)]
struct RowProjection;

impl CustomOp for RowProjection {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn HostOp>(Box::new(CuBlasLt {
            m: 's'.into(),
            n: WIDTH.into(),
            k: WIDTH.into(),
            a_layout: cublasOperation_t::CUBLAS_OP_N,
            b_layout: cublasOperation_t::CUBLAS_OP_N,
            a_order: cublasLtOrder_t::CUBLASLT_ORDER_ROW,
            b_order: cublasLtOrder_t::CUBLASLT_ORDER_ROW,
            c_order: cublasLtOrder_t::CUBLASLT_ORDER_ROW,
            d_order: cublasLtOrder_t::CUBLASLT_ORDER_ROW,
            lda: WIDTH.into(),
            ldb: WIDTH.into(),
            ldc: WIDTH.into(),
            ldd: WIDTH.into(),
            a_dtype: DType::Bf16,
            b_dtype: DType::Bf16,
            c_dtype: DType::Bf16,
            d_dtype: DType::Bf16,
            ..Default::default()
        }) as Box<dyn HostOp>)
    }
}

#[test]
#[ignore = "requires an SM80+ GPU and local FlashInfer provider; run separately with optional ORBITKV_CUDA_PROFILE_GRAPH_STEPS"]
fn flashinfer_then_cublaslt_recaptures_dynamic_rows_with_stable_inputs() {
    let context = crate::cudarc::driver::CudaContext::new(0).expect("CUDA GPU required");
    assert!(context.compute_capability().unwrap().0 >= 8);
    let mut runtime = CudaRuntime::initialize(context.default_stream());
    let mut graph = Graph::default();
    let rows = Expression::from('s');
    let batch = Expression::from('b');
    let pages = Expression::from('c');
    let q = graph.tensor((rows, WIDTH)).as_dtype(DType::Bf16).persist();
    let k = graph
        .tensor((MAX_REQUESTS, PAGE_TOKENS, WIDTH))
        .as_dtype(DType::Bf16)
        .persist();
    let v = graph
        .tensor((MAX_REQUESTS, PAGE_TOKENS, WIDTH))
        .as_dtype(DType::Bf16)
        .persist();
    let page_indices = graph.tensor(pages).as_dtype(DType::Int).persist();
    let query_indptr = graph.tensor(batch + 1).as_dtype(DType::Int).persist();
    let page_indptr = graph.tensor(batch + 1).as_dtype(DType::Int).persist();
    let last_page_len = graph.tensor(batch).as_dtype(DType::Int).persist();
    let weight = graph.tensor((WIDTH, WIDTH)).as_dtype(DType::Bf16).persist();
    let attention = graph.custom_op(
        FlashInferAttention::paged(
            crate::providers::flashinfer::FlashInferAlgorithm::TensorCore,
            1,
            1,
            WIDTH,
            PAGE_TOKENS,
            rows,
            pages,
            batch,
            DType::Bf16,
            0.0,
            None,
        ),
        vec![
            q,
            k,
            v,
            page_indices,
            query_indptr,
            page_indptr,
            last_page_len,
        ],
        (rows, WIDTH),
        DType::Bf16,
    );
    let output = graph
        .custom_op(
            RowProjection,
            vec![attention, weight],
            (rows, WIDTH),
            DType::Bf16,
        )
        .output();

    runtime.set_data(k, vec![bf16::ZERO; MAX_REQUESTS * PAGE_TOKENS * WIDTH]);
    runtime.set_data(v, vec![bf16::ONE; MAX_REQUESTS * PAGE_TOKENS * WIDTH]);
    runtime.set_data(weight, vec![bf16::ONE; WIDTH * WIDTH]);
    runtime.set_data_with_capacity(q, vec![bf16::ZERO; 4 * WIDTH], MAX_ROWS * WIDTH * 2);
    runtime.set_data_with_capacity(page_indices, vec![0_i32, 1], MAX_REQUESTS * 4);
    runtime.set_data_with_capacity(query_indptr, vec![0_i32, 2, 4], (MAX_REQUESTS + 1) * 4);
    runtime.set_data_with_capacity(page_indptr, vec![0_i32, 1, 2], (MAX_REQUESTS + 1) * 4);
    runtime.set_data_with_capacity(last_page_len, vec![2_i32, 2], MAX_REQUESTS * 4);
    let stable_inputs = [
        q,
        k,
        v,
        weight,
        page_indices,
        query_indptr,
        page_indptr,
        last_page_len,
    ]
    .map(|input| (input, runtime.input_allocation(input).unwrap()));
    graph.set_dim('s', 4);
    graph.set_dim('b', 2);
    graph.set_dim('c', 2);
    runtime = graph.compile(
        runtime,
        CompileOptions::default()
            .search_graph_limit(1)
            .dim_buckets('s', &[DimBucket::new(1, MAX_ROWS).representative(4)])
            .dim_buckets('b', &[DimBucket::new(1, MAX_REQUESTS).representative(2)])
            .dim_buckets('c', &[DimBucket::new(1, MAX_REQUESTS).representative(2)]),
    );

    // Explicit custom operations make this a capture-runtime regression, with
    // no random provider-selection prerequisite. All shapes share one bucket.
    for (batch, row_indptr, last_lengths) in [
        (2, vec![0_i32, 2, 4], vec![2_i32, 2]),
        (4, vec![0_i32, 2, 4, 8, 12], vec![4_i32; 4]),
        (2, vec![0_i32, 2, 4], vec![2_i32, 2]),
    ] {
        let rows = usize::try_from(*row_indptr.last().unwrap()).unwrap();
        graph.set_dim('s', rows);
        graph.set_dim('b', batch);
        graph.set_dim('c', batch);
        runtime.set_data(q, vec![bf16::ZERO; rows * WIDTH]);
        runtime.set_data(query_indptr, row_indptr);
        runtime.set_data(
            page_indptr,
            (0..=batch)
                .map(|i| i32::try_from(i).unwrap())
                .collect::<Vec<_>>(),
        );
        runtime.set_data(
            page_indices,
            (0..batch)
                .map(|i| i32::try_from(i).unwrap())
                .collect::<Vec<_>>(),
        );
        runtime.set_data(last_page_len, last_lengths);
        for (input, allocation) in stable_inputs {
            assert_eq!(runtime.input_allocation(input), Some(allocation));
        }
        runtime.execute(&graph.dyn_map);
        let values = runtime.get_bf16(output);
        assert_eq!(values.len(), rows * WIDTH);
        // Attention over constant V=1 gives one in every channel for every
        // valid causal row, and multiplication by an all-one matrix sums 64.
        assert!(values.iter().all(|value| value.to_f32() == 64.0));
        let summaries = runtime.debug_cuda_graph_summaries();
        assert!(
            summaries
                .iter()
                .any(|summary| summary.n_flashinfer == 1 && summary.n_cublaslt == 1),
            "regression must recapture FlashInfer and cuBLASLt in the same CUDA graph: {summaries:?}"
        );
        for summary in &summaries {
            eprintln!(
                "mixed-library capture b={batch} s={rows}: builds={} variants={} cublas_captures={:?} cublas_cache_hits={:?} flashinfer_recaptures={:?}",
                summary.graph_builds,
                summary.resident_variants,
                summary.cublaslt_capture_counts,
                summary.cublaslt_capture_cache_hits,
                summary.flashinfer_recapture_counts,
            );
        }
        eprintln!("mixed-library recapture b={batch} s={rows}: constant oracle passed");
    }
}
