use super::*;
use crate::runtime::CudaRuntime;
use orbitkv_compiler::{graph::DimBucket, prelude::*};
use rand::SeedableRng;
use tracing_subscriber::prelude::*;

fn graph() -> (Graph, GraphTensor, GraphTensor) {
    let mut graph = Graph::default();
    graph.set_dim('s', 3);
    let input = graph.tensor(('s', 17));
    let output = ((input + 1.0) * (input + 2.0)).output();
    (graph, input, output)
}

#[test]
#[ignore = "requires CUDA; covers modules reused from process-wide provider caches"]
fn cached_provider_modules_participate_in_capture_and_strict_replay() {
    let context = CudaContext::new(0).expect("CUDA device required");
    let source = r#"extern "C" __global__ void artifact_probe(float* out) { out[0] = 1.0f; }"#;
    let _previous_image = crate::compile_module_image_for_current_device(&context, source).unwrap();
    let captured = module_artifact_session(&context, None).unwrap();
    captured.lock().unwrap().capturing = true;
    with_module_artifact_session(captured.clone(), || {
        observe_cached_module(&context, source).unwrap();
    });
    {
        let mut artifact = captured.lock().unwrap();
        assert_eq!(artifact.data.image_count(), 1);
        artifact.capturing = false;
        artifact.loading = true;
    }
    with_module_artifact_session(captured.clone(), || {
        observe_cached_module(&context, source).unwrap();
        captured.lock().unwrap().data.images.clear();
        let error = observe_cached_module(&context, source).unwrap_err();
        assert!(matches!(
            error.failure,
            crate::CudaModuleImageCompileFailure::ArtifactMiss { .. }
        ));
    });
}

#[test]
#[ignore = "requires CUDA; captures selected modules, replays two buckets and rejects missing images"]
fn selected_modules_replay_without_nvrtc_and_reject_warm_cache_misses() {
    let context = CudaContext::new(0).expect("CUDA device required");
    let stream = context.default_stream();
    let (mut searched, input, _) = graph();
    let options = CompileOptions::default()
        .dim_buckets(
            's',
            &[DimBucket::new(1, 1), DimBucket::new(2, 4).representative(3)],
        )
        .search_graph_limit(2);
    let mut runtime = CudaRuntime::initialize(stream.clone());
    runtime.set_data(input, vec![0.0_f32; 4 * 17]);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
    runtime = searched.compile_with_rng(runtime, options.clone(), &mut rng);
    let artifact = runtime.capture_module_artifact(&searched).unwrap();
    assert!(artifact.image_count() > 0);
    assert!(current_module_artifact_session().is_none());
    let bytes = serde_json::to_vec(&artifact).unwrap();
    let artifact: CudaModuleArtifact = serde_json::from_slice(&bytes).unwrap();
    let schedule = searched.selected_schedule().unwrap().clone();
    drop(runtime);
    drop(searched);

    let trace_path = std::env::temp_dir().join(format!(
        "orbitkv-module-artifact-{}.jsonl",
        uuid::Uuid::new_v4()
    ));
    let (layer, trace) = orbitkv_tracing::stage_trace_layer(&trace_path).unwrap();
    let subscriber = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(subscriber, || {
        let (mut loaded, input, output) = graph();
        loaded.prepare_selected_schedule(&options);
        loaded.install_selected_schedule(schedule);
        let mut replay = CudaRuntime::initialize(stream);
        replay.set_data(input, vec![0.0_f32; 4 * 17]);
        replay
            .load_selected_schedule_with_modules(&loaded, &artifact)
            .unwrap();
        replay.set_max_materialized_buckets(Some(1));
        for rows in [3, 1, 4, 1] {
            let values = (0..rows * 17).map(|i| i as f32 / 8.0).collect::<Vec<_>>();
            let expected = values
                .iter()
                .map(|x| (x + 1.0) * (x + 2.0))
                .collect::<Vec<_>>();
            loaded.set_dim('s', rows);
            replay.set_data(input, values);
            replay.execute(&loaded.dyn_map);
            assert_eq!(replay.get_f32(output), expected);
        }
        // A warm runtime must still validate every needed image. The retained
        // CUDA-function cache cannot bypass strict artifact completeness.
        let mut incomplete = artifact.clone();
        incomplete.images.clear();
        let error = replay
            .load_selected_schedule_with_modules(&loaded, &incomplete)
            .unwrap_err();
        assert!(error.contains("ArtifactMiss"), "{error}");
        assert!(current_module_artifact_session().is_none());
    });
    trace.finish().unwrap();
    let rows = std::fs::read_to_string(&trace_path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert!(!rows.iter().any(|row| row["name"] == "cuda.nvrtc.compile"));
    assert!(
        rows.iter()
            .any(|row| row["name"] == "cuda.module_image.hit")
    );
    std::fs::remove_file(trace_path).unwrap();
}
