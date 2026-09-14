//! A GEMM hotspot beside persistent scatter state. All admitted implementations
//! must still compute the independent reference and preserve untouched rows.
use cudarc::driver::CudaContext;
use half::bf16;
use orbitkv_compiler::op::Runtime as _;
use orbitkv_compiler::prelude::*;
use orbitkv_cuda::runtime::CudaRuntime;
use rand::{SeedableRng, rngs::StdRng};
use serde_json::Value;

#[test]
#[ignore = "requires CUDA; profiles exact local neighbors and checks arithmetic/state"]
fn local_search_measures_gemm_alternatives_without_losing_persistent_state() {
    const ROWS: usize = 8;
    const INNER: usize = 128;
    const COLUMNS: usize = 256;
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("search.jsonl");
    let mut graph = Graph::default();
    let input = graph.tensor((ROWS, INNER)).as_dtype(DType::Bf16);
    let weights = graph.tensor((COLUMNS, INNER)).as_dtype(DType::Bf16);
    let output = input.matmul(weights.t()).output();
    let update = graph.tensor((1, 4));
    let slots = graph.tensor(1).as_dtype(DType::Int);
    let state = graph.tensor((4, 4)).persist();
    let updated = orbitkv_ops::scatter_rows(update, slots, state, 4).output();
    let mut runtime = CudaRuntime::initialize(stream.clone());
    let a = (0..ROWS * INNER)
        .map(|i| bf16::from_f32((i % 7) as f32 * 0.25 - 0.5))
        .collect::<Vec<_>>();
    let b = (0..COLUMNS * INNER)
        .map(|i| bf16::from_f32((i % 5) as f32 * 0.125 - 0.25))
        .collect::<Vec<_>>();
    runtime.set_data(input, a.clone());
    runtime.set_data(weights, b.clone());
    runtime.set_data(update, vec![2.0f32, 3.0, 5.0, 7.0]);
    runtime.set_data(slots, vec![2_i32]);
    let mut allocation = runtime.alias_state_required(state, updated, 16 * size_of::<f32>());
    runtime = graph.compile_with_rng(
        runtime,
        CompileOptions::default()
            .search_graph_limit(8)
            .hotspot_candidates(32)
            .keep_best(2)
            .search_trace(&path)
            .search_log(false),
        &mut StdRng::seed_from_u64(73),
    );
    assert!(runtime.output_aliases_input_in_all_buckets(updated, state));
    // Search runs may update persistent state. Reset it before each independent
    // execution so stale search output cannot satisfy the state oracle.
    for row in [2, 1] {
        stream.memset_zeros(&mut allocation).unwrap();
        // Ordinary input tensors are consumed by execute; each request supplies
        // fresh bindings. Only the explicit state allocation persists.
        runtime.set_data(input, a.clone());
        runtime.set_data(weights, b.clone());
        runtime.set_data(update, vec![2.0f32, 3.0, 5.0, 7.0]);
        runtime.set_data(slots, vec![row as i32]);
        runtime.execute(&graph.dyn_map);
        let actual = runtime.get_bf16(output);
        for i in 0..ROWS {
            for j in 0..COLUMNS {
                let expected = (0..INNER)
                    .map(|k| a[i * INNER + k].to_f32() * b[j * INNER + k].to_f32())
                    .sum::<f32>();
                assert!((actual[i * COLUMNS + j].to_f32() - expected).abs() < 0.01);
            }
        }
        let actual = stream.clone_dtoh(&allocation).unwrap();
        let actual = actual
            .as_chunks()
            .0
            .iter()
            .copied()
            .map(f32::from_ne_bytes)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0f32; 16];
        expected[row * 4..row * 4 + 4].copy_from_slice(&[2.0, 3.0, 5.0, 7.0]);
        assert_eq!(actual, expected);
    }
    let records = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    let direct = records
        .iter()
        .filter(|r| r["event"] == "direct")
        .collect::<Vec<_>>();
    assert!(
        direct
            .iter()
            .any(|r| r["sampling"] == "Hotspot" && r["status"] == "measured")
    );
    for family in ["generated", "library"] {
        assert!(
            direct
                .iter()
                .any(|candidate| candidate["status"] == "measured"
                    && records.iter().any(|program| {
                        program["event"] == "program"
                            && program["program"] == candidate["program"]
                            && program["operations"].as_array().unwrap().iter().any(|op| {
                                if family == "generated" {
                                    matches!(op["kernel"].as_str(), Some("GenericMatmul" | "Gemv"))
                                } else {
                                    op["host_provider"] == "CuBlasLt"
                                }
                            })
                    })),
            "missing measured {family}: {direct:?}"
        );
    }
    // Provider coverage must come from a local step of this exact parent,
    // rather than two unrelated graphs drawn later by the random fallback.
    let family = |candidate: &Value| {
        let program = records
            .iter()
            .find(|r| r["event"] == "program" && r["program"] == candidate["program"])
            .unwrap();
        program["operations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|op| op["host_provider"] == "CuBlasLt")
    };
    assert!(
        direct.iter().any(|candidate| {
            candidate["sampling"] == "Hotspot"
                && candidate["status"] == "measured"
                && direct.iter().any(|parent| {
                    parent["candidate"] == candidate["targeted_choice"]["parent"]
                        && parent["status"] == "measured"
                        && family(parent) != family(candidate)
                })
        }),
        "no measured local provider transition: {direct:?}"
    );
    for candidate in direct.iter().filter(|r| r["status"] == "measured") {
        assert!(!candidate["profile_regions"].as_array().unwrap().is_empty());
        let program = records
            .iter()
            .find(|r| r["event"] == "program" && r["program"] == candidate["program"])
            .unwrap();
        for region in candidate["profile_regions"].as_array().unwrap() {
            for node in region["nodes"].as_array().unwrap() {
                assert!(
                    program["operations"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|op| op["node"] == *node),
                    "execution provenance escaped the candidate LLIR"
                );
            }
        }
        if candidate["sampling"] == "Hotspot" {
            assert!(
                candidate["targeted_choice"]["cost_seconds"]
                    .as_f64()
                    .unwrap()
                    > 0.0
            );
            assert_ne!(
                candidate["targeted_choice"]["from"],
                candidate["targeted_choice"]["to"]
            );
        }
    }
}
