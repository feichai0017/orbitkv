use super::*;
use orbitkv_compiler::{
    hlir::ReferenceRuntime,
    prelude::{
        Graph,
        rand::{SeedableRng, rngs::StdRng},
    },
    search::{Finalists, GeneticSearch},
};
use std::time::Instant;

#[test]
fn trace_connects_direct_deployment_and_selection_without_model_assumptions() {
    let mut graph = Graph::default();
    let input = graph.tensor(5);
    (input.sin() + input.cos()).output();
    let options = CompileOptions::default()
        .search_graph_limit(1)
        .search_log(false);
    graph.build_search_space::<ReferenceRuntime>(options.clone());
    let space = graph.search_space().unwrap();
    let contexts = space.bucket_contexts(&graph.dyn_map);
    let ctx = &contexts[0];
    let started = Instant::now();
    let mut search = GeneticSearch::new(space, ctx, &options, started);
    let mut rng = StdRng::seed_from_u64(17);
    let candidate = search.next_candidate(&mut rng).unwrap();
    let program = llir_program_identity(&candidate.llir);
    let mut trace = SearchTrace::new(Vec::new());
    trace.bucket(ctx);
    search.report_with_observer(
        candidate,
        Outcome::Measured(Duration::from_micros(7), "direct".into()),
        |candidate, outcome, timed_out| {
            trace.direct(
                candidate,
                outcome,
                timed_out,
                ctx,
                Duration::from_micros(12),
            );
        },
    );
    let mut finalists = Finalists::new(search.into_ranked(), space, ctx, &options, started);
    let pending = finalists.extract_next().unwrap();
    trace.deployment(
        &pending,
        ctx,
        1,
        &Ok(Duration::from_micros(3)),
        Duration::from_micros(9),
    );
    trace.validation(&pending, ctx, &Err("fixture workspace budget".into()));
    trace.selected(&[SelectedProgram {
        bucket_indices: ctx.bucket_indices().clone(),
        representative_dyn_map: pending.dyn_map,
        genome: pending.genome,
        llir: pending.llir,
    }]);
    let records = String::from_utf8(trace.writer)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    let manifests = records
        .iter()
        .filter(|row| row["event"] == "program")
        .collect::<Vec<_>>();
    assert_eq!(manifests.len(), 1);
    assert!(
        manifests[0]["operations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|op| op["inputs"].as_array().unwrap().len() == 2)
    );
    let direct = records.iter().find(|row| row["event"] == "direct").unwrap();
    let bucket = records
        .iter()
        .find(|row| row["event"] == "bucket_started")
        .unwrap();
    assert_eq!(bucket["snapshot_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(direct["sampling"], "Coverage");
    let deployment = records
        .iter()
        .find(|row| row["event"] == "deployment")
        .unwrap();
    assert_eq!(direct["program"], program);
    assert_eq!(deployment["program"], program);
    assert_eq!(direct["device_duration_ns"], 7_000);
    assert_eq!(deployment["direct_duration_ns"], 7_000);
    assert_eq!(deployment["cuda_graph_duration_ns"], 3_000);
    assert!(
        records
            .iter()
            .any(|row| row["event"] == "finalist_validation"
                && row["reason"] == "fixture workspace budget")
    );
    assert!(
        records
            .iter()
            .any(|row| row["event"] == "selected" && row["program"] == program)
    );
    assert_eq!(records.last().unwrap()["event"], "search_completed");
}

#[test]
fn trace_preserves_write_failures() {
    struct BrokenWriter;
    impl Write for BrokenWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("fixture disk failure"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let error = SearchTrace::new(BrokenWriter)
        .try_write(&json!({"event": "test"}))
        .unwrap_err();
    assert!(error.to_string().contains("fixture disk failure"));
}

#[test]
fn requested_trace_never_overwrites_existing_evidence() {
    let path = std::env::temp_dir().join(format!(
        "orbitkv-search-trace-{}.jsonl",
        uuid::Uuid::new_v4()
    ));
    std::fs::write(&path, b"previous evidence\n").unwrap();
    let options = CompileOptions::default().search_trace(&path);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        SearchTrace::configured(&options)
    }));
    assert!(result.is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"previous evidence\n");
    std::fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "requires a CUDA device"]
fn cuda_search_trace_records_the_executed_selected_program() {
    use crate::{cudarc::driver::CudaContext, runtime::CudaRuntime};
    use orbitkv_compiler::op::Runtime;

    let path = std::env::temp_dir().join(format!(
        "orbitkv-search-device-{}.jsonl",
        uuid::Uuid::new_v4()
    ));
    let mut graph = Graph::default();
    let input = graph.tensor(8);
    let output = (input + input).output();
    let options = CompileOptions::default()
        .search_graph_limit(2)
        .keep_best(2)
        .search_log(false)
        .search_trace(&path);
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    let mut runtime = CudaRuntime::initialize(stream);
    let values = (0..8).map(|value| value as f32).collect::<Vec<_>>();
    runtime.set_data(input, values.clone());
    runtime = graph.compile_with_rng(runtime, options, &mut StdRng::seed_from_u64(17));
    runtime.set_data(input, values.clone());
    runtime.execute(&graph.dyn_map);
    assert_eq!(
        runtime.get_f32(output),
        values.iter().map(|value| value * 2.0).collect::<Vec<_>>()
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
    assert_eq!(records.last().unwrap()["event"], "search_completed");
    std::fs::remove_file(path).unwrap();
}
