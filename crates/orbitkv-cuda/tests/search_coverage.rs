//! Search the same snapshot twice, while independently checking selected GEMMs.

use cudarc::driver::CudaContext;
use half::bf16;
use orbitkv_compiler::op::Runtime as _;
use orbitkv_compiler::prelude::*;
use orbitkv_cuda::runtime::CudaRuntime as Runtime;
use rand::{SeedableRng, rngs::StdRng};
use serde_json::Value;

#[test]
#[ignore = "requires an NVIDIA GPU; measures generated and cuBLASLt candidates"]
fn repeated_snapshot_search_covers_gemm_candidates_and_matches_reference() {
    const INPUT_WIDTH: usize = 128;
    const OUTPUT_WIDTH: usize = 256;
    const SEARCH_BUDGET: usize = 8;
    let context = CudaContext::new(0).unwrap();
    let directory = tempfile::tempdir().unwrap();
    for rows in [1, 4, 8, 32] {
        let mut graph = Graph::default();
        let input = graph.tensor((rows, INPUT_WIDTH)).as_dtype(DType::Bf16);
        let weight = graph
            .tensor((OUTPUT_WIDTH, INPUT_WIDTH))
            .as_dtype(DType::Bf16);
        let output = input.matmul(weight.t()).output();
        let input_values = (0..rows * INPUT_WIDTH)
            .map(|i| bf16::from_f32(((i % 7) as f32 - 3.0) * 0.25))
            .collect::<Vec<_>>();
        let weight_values = (0..OUTPUT_WIDTH * INPUT_WIDTH)
            .map(|i| bf16::from_f32(((i % 5) as f32 - 2.0) * 0.125))
            .collect::<Vec<_>>();
        let options = CompileOptions::default()
            .search_graph_limit(SEARCH_BUDGET)
            .initial_population(SEARCH_BUDGET)
            .keep_best(2)
            .search_log(false);
        graph.build_search_space::<Runtime>(options.clone());
        let mut sequences = Vec::new();
        for repetition in 0..2 {
            let path = directory
                .path()
                .join(format!("rows-{rows}-{repetition}.jsonl"));
            let mut runtime = Runtime::initialize(context.default_stream());
            runtime.set_data(input, input_values.clone());
            runtime.set_data(weight, weight_values.clone());
            runtime = graph.search_with_rng(
                runtime,
                options.clone().search_trace(&path),
                &mut StdRng::seed_from_u64(73),
            );
            runtime.execute(&graph.dyn_map);
            let actual = runtime.get_bf16(output);
            assert_eq!(actual.len(), rows * OUTPUT_WIDTH);
            for row in 0..rows {
                for column in 0..OUTPUT_WIDTH {
                    let expected = (0..INPUT_WIDTH)
                        .map(|inner| {
                            input_values[row * INPUT_WIDTH + inner].to_f32()
                                * weight_values[column * INPUT_WIDTH + inner].to_f32()
                        })
                        .sum::<f32>();
                    assert!((actual[row * OUTPUT_WIDTH + column].to_f32() - expected).abs() < 0.01);
                }
            }
            let records = std::fs::read_to_string(path)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str::<Value>(line).unwrap())
                .collect::<Vec<_>>();
            let direct = records
                .iter()
                .filter(|row| row["event"] == "direct")
                .collect::<Vec<_>>();
            assert!(
                direct
                    .iter()
                    .filter(|row| row["status"] == "measured")
                    .count()
                    <= SEARCH_BUDGET
            );
            for implementation in ["generated", "CuBlasLt"] {
                assert!(
                    direct.iter().any(|candidate| {
                        candidate["status"] == "measured"
                            && records.iter().any(|record| {
                                record["event"] == "program"
                                    && record["program"] == candidate["program"]
                                    && record["operations"].as_array().unwrap().iter().any(|op| {
                                        if implementation == "generated" {
                                            matches!(
                                                op["kernel"].as_str(),
                                                Some("GenericMatmul" | "Gemv")
                                            )
                                        } else {
                                            op["host_provider"] == implementation
                                        }
                                    })
                            })
                    }),
                    "rows={rows}: {implementation} was not measured: {records:?}"
                );
            }
            sequences.push(
                direct
                    .iter()
                    .map(|row| {
                        (
                            row["program"].clone(),
                            row["sampling"].clone(),
                            row["status"].clone(),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(
            sequences[0], sequences[1],
            "rows={rows}: candidate order changed"
        );
    }
}
