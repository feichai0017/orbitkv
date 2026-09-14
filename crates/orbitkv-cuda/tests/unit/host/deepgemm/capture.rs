use super::*;
use orbitkv_compiler::graph::DimBucket;

#[test]
#[ignore = "requires SM90 and DeepGEMM; exercises graph retirement and private scratch growth"]
fn block_scaled_graph_rebuild_retires_scratch_after_dynamic_row_growth() {
    let context = cudarc::driver::CudaContext::new(0).expect("SM90 GPU required");
    assert_eq!(context.compute_capability().unwrap(), (9, 0));
    let stream = context.default_stream();
    let width = 128usize;
    let max_rows = 24usize;
    for (reference, resident) in [(false, false), (true, false), (false, true), (true, true)] {
        let shared_scratch = Arc::new(Mutex::new(None));
        let mut graph = Graph::default();
        graph.set_dim('s', 4);
        let input = graph.tensor(('s', width)).as_dtype(DType::Bf16).persist();
        let weights = graph
            .tensor((width, width))
            .as_dtype(DType::F8E4M3)
            .persist();
        let scales = graph.tensor((1, 1)).persist();
        let spec = BlockScaledLinearSpec {
            rows: 's'.into(),
            output_features: width,
            input_features: width,
            weight_block_rows: BLOCK,
            weight_block_columns: BLOCK,
        };
        let output = if reference {
            graph.custom_op(
                DirectReference(BlockScaledLinearReference::new(spec)),
                vec![input, weights, scales],
                ('s', width),
                DType::Bf16,
            )
        } else {
            graph.custom_op(
                DeepGemm {
                    rows: 's'.into(),
                    output_features: width,
                    input_features: width,
                    variant: 0,
                    provider: jit::provider_identity().unwrap(),
                    scratch: shared_scratch.clone(),
                },
                vec![input, weights, scales],
                ('s', width),
                DType::Bf16,
            )
        }
        .output();
        let mut runtime = crate::runtime::CudaRuntime::initialize(stream.clone());
        // Resource validation evaluates the full bucket capacity before any
        // execution; provide that complete input allocation for compilation.
        runtime.set_data_with_capacity(
            input,
            vec![bf16::ONE; max_rows * width],
            max_rows * width * 2,
        );
        runtime.set_data(weights, vec![0x38_u8; width * width]);
        runtime.set_data(scales, vec![1.0_f32]);
        runtime = graph.compile(
            runtime,
            CompileOptions::default()
                .search_graph_limit(1)
                .dim_buckets('s', &[DimBucket::new(1, max_rows).representative(4)]),
        );
        if resident {
            runtime.begin_cuda_graph_preparation(&['s'.into()]);
        }
        let allocation = runtime.input_allocation(input);
        let mut seen = std::collections::HashSet::new();
        let mut first_owner: Option<std::sync::Weak<Scratch>> = None;
        for rows in [4, 12, 4, 24, 1, 4] {
            graph.set_dim('s', rows);
            // Keep the registered byte span at the bucket capacity; `s`
            // controls the logical rows consumed by the operator.
            runtime.set_data(input, vec![bf16::ONE; max_rows * width]);
            assert_eq!(runtime.input_allocation(input), allocation);
            let builds_before: usize = runtime
                .debug_cuda_graph_summaries()
                .iter()
                .map(|summary| summary.graph_builds)
                .sum();
            runtime.execute(&graph.dyn_map);
            let builds_after: usize = runtime
                .debug_cuda_graph_summaries()
                .iter()
                .map(|summary| summary.graph_builds)
                .sum();
            if resident && !seen.insert(rows) {
                assert_eq!(
                    builds_before, builds_after,
                    "cached shape must reuse its graph"
                );
            }
            if !reference {
                if let Some(owner) = &first_owner {
                    if rows >= 12 {
                        assert_eq!(
                            owner.upgrade().is_some(),
                            resident,
                            "old scratch lives exactly while its resident graph retains ownership"
                        );
                    }
                } else {
                    first_owner = Some(Arc::downgrade(
                        shared_scratch.lock().unwrap().as_ref().unwrap(),
                    ));
                }
            }
            let values = runtime.get_bf16(output);
            assert_eq!(values.len(), rows * width);
            // Quantized activation 1, FP8 weight 1 and weight scale 1:
            // the independent dot-product oracle is the reduction width.
            assert!(values.iter().all(|value| value.to_f32() == width as f32));
            context.check_err().unwrap();
            eprintln!(
                "block-scaled dynamic rebuild reference={reference} resident={resident} rows={rows}: oracle passed"
            );
        }
        if resident {
            runtime.finish_cuda_graph_preparation().unwrap();
            runtime.execute(&graph.dyn_map);
            assert!(
                runtime
                    .get_bf16(output)
                    .iter()
                    .all(|value| value.to_f32() == width as f32)
            );
        }
        drop(runtime);
        assert!(
            first_owner.is_none_or(|owner| owner.upgrade().is_none()),
            "retired resident graph must release its old scratch"
        );
        context.check_err().unwrap();
    }
}
