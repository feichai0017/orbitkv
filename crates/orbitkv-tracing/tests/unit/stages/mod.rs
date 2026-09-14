use super::*;

fn records(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn nested_stages_keep_parent_fields_and_inclusive_time_without_eager_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stages.jsonl");
    let (layer, guard) = stage_trace_layer(&path).unwrap();
    let subscriber = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(subscriber, || {
        let _parent =
            tracing::info_span!(target: "orbitkv::stage", "compile", bucket = 2_u64).entered();
        {
            let child = tracing::info_span!(target: "orbitkv::stage", "prepare", status = tracing::field::Empty);
            let _entered = child.enter();
            child.record("status", "rejected");
        }
        let _ignored = tracing::info_span!("not_a_stage").entered();
        assert_eq!(
            records(&path).len(),
            1,
            "measurement must not flush records"
        );
    });
    guard.finish().unwrap();
    let rows = records(&path);
    let child = rows.iter().find(|r| r["name"] == "prepare").unwrap();
    let parent = rows.iter().find(|r| r["name"] == "compile").unwrap();
    assert_eq!(child["parent"], parent["id"]);
    assert_eq!(child["fields"]["status"], "rejected");
    assert_eq!(parent["fields"]["bucket"], 2);
    assert!(
        parent["wall_duration_ns"].as_u64().unwrap() >= child["wall_duration_ns"].as_u64().unwrap()
    );
    assert_eq!(rows.len(), 4);
    assert_eq!(rows.last().unwrap()["event"], "trace_completed");
}

#[test]
fn existing_files_are_preserved_and_unclosed_spans_cannot_claim_completion() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stages.jsonl");
    let (layer, guard) = stage_trace_layer(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    assert!(stage_trace_layer(&path).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let subscriber = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(subscriber, || {
        let _live = tracing::info_span!(target: "orbitkv::stage", "live").entered();
        assert!(guard.finish().is_err());
    });
    let rows = records(&path);
    assert_eq!(rows.last().unwrap()["event"], "trace_incomplete");
    assert_eq!(rows.last().unwrap()["open_spans"], 1);
}

#[test]
fn dropped_trace_is_incomplete_and_thread_roots_have_distinct_ids() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stages.jsonl");
    let (layer, guard) = stage_trace_layer(&path).unwrap();
    let dispatch = tracing::Dispatch::new(tracing_subscriber::registry().with(layer));
    let worker = dispatch.clone();
    std::thread::spawn(move || {
        tracing::dispatcher::with_default(&worker, || {
            let _span = tracing::info_span!(target: "orbitkv::stage", "worker").entered();
        })
    })
    .join()
    .unwrap();
    tracing::dispatcher::with_default(&dispatch, || {
        let _span = tracing::info_span!(target: "orbitkv::stage", "caller").entered();
    });
    drop(guard);
    let rows = records(&path);
    let stages = rows
        .iter()
        .filter(|r| r["event"] == "stage")
        .collect::<Vec<_>>();
    assert_eq!(stages.len(), 2);
    assert_ne!(stages[0]["id"], stages[1]["id"]);
    assert_ne!(stages[0]["thread"], stages[1]["thread"]);
    assert!(stages.iter().all(|r| r["parent"].is_null()));
    assert_eq!(rows.last().unwrap()["event"], "trace_incomplete");
}

#[test]
fn measurements_keep_explicit_parent_and_numeric_types_without_becoming_spans() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("metrics.jsonl");
    let (layer, guard) = stage_trace_layer(&path).unwrap();
    let subscriber = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(subscriber, || {
        let parent = tracing::info_span!(target: "orbitkv::stage", "schedule");
        let _entered = parent.enter();
        let _unrelated = tracing::info_span!(target: "orbitkv::stage", "unrelated").entered();
        tracing::event!(name: "rule", target: "orbitkv::stage", parent: &parent,
            tracing::Level::INFO, rule = "full rule name", matches = 0_u64,
            search_apply_ns = 123_u128, device_ms = 0.25_f64);
        tracing::info!("ignored event");
        assert_eq!(records(&path).len(), 1);
    });
    guard.finish().unwrap();
    let rows = records(&path);
    assert_eq!(rows[0]["schema"], TRACE_SCHEMA_VERSION);
    let parent = rows.iter().find(|row| row["name"] == "schedule").unwrap();
    let measurements = rows
        .iter()
        .filter(|row| row["event"] == "metric")
        .collect::<Vec<_>>();
    assert_eq!(measurements.len(), 1);
    let measured = measurements[0];
    assert_eq!(measured["parent"], parent["id"]);
    assert_eq!(measured["fields"]["search_apply_ns"], 123);
    assert_eq!(measured["fields"]["matches"], 0);
    assert_eq!(measured["fields"]["device_ms"], 0.25);
    assert!(measured.get("wall_duration_ns").is_none());
    let at = measured["at_ns"].as_u64().unwrap();
    let start = parent["start_ns"].as_u64().unwrap();
    assert!(at >= start && at <= start + parent["wall_duration_ns"].as_u64().unwrap());
}

#[test]
fn contextual_measurements_cross_unrecorded_scopes_and_allow_explicit_roots() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("metrics.jsonl");
    let (layer, guard) = stage_trace_layer(&path).unwrap();
    tracing::subscriber::with_default(tracing_subscriber::registry().with(layer), || {
        let _stage = tracing::info_span!(target: "orbitkv::stage", "outer").entered();
        let _ordinary = tracing::info_span!("ordinary").entered();
        tracing::event!(name: "nested", target: "orbitkv::stage", tracing::Level::INFO, count = 1_u64);
        tracing::event!(name: "root", target: "orbitkv::stage", parent: None, tracing::Level::INFO, count = 2_u64);
    });
    guard.finish().unwrap();
    let rows = records(&path);
    let outer = rows.iter().find(|row| row["name"] == "outer").unwrap();
    let nested = rows.iter().find(|row| row["name"] == "nested").unwrap();
    let root = rows.iter().find(|row| row["name"] == "root").unwrap();
    assert_eq!(nested["parent"], outer["id"]);
    assert!(root["parent"].is_null());
}
