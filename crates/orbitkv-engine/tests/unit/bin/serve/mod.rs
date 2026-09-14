use super::*;

#[test]
fn startup_preparation_is_explicit_and_does_not_change_compilation() {
    let base = [
        "orbitkv-serve",
        "--model",
        "/models/checkpoint",
        "--page-counts",
        "64",
    ];
    let default = ServeConfig::from_args(base.map(str::to_owned)).unwrap();
    assert!(default.engine.prepare_execution);
    let lazy = ServeConfig::from_args(
        base.into_iter()
            .chain(["--prepare-execution", "false"])
            .map(str::to_owned),
    )
    .unwrap();
    assert!(!lazy.engine.prepare_execution);
    assert_eq!(
        default.engine.graph_cache_capacity,
        lazy.engine.graph_cache_capacity
    );
    assert_eq!(default.engine.search_graphs, lazy.engine.search_graphs);
    assert_eq!(default.tuning, lazy.tuning);
}

#[test]
fn graph_cache_capacity_is_positive_and_independent_of_search() {
    let base = [
        "orbitkv-serve",
        "--model",
        "/models/checkpoint",
        "--page-counts",
        "64",
        "--graph-cache-capacity",
    ];
    let config = ServeConfig::from_args(base.into_iter().chain(["3"]).map(str::to_owned)).unwrap();
    assert_eq!(config.engine.graph_cache_capacity.get(), 3);
    assert_eq!(config.engine.search_graphs, 2);
    assert!(ServeConfig::from_args(base.into_iter().chain(["0"]).map(str::to_owned)).is_err());
}

#[test]
fn reads_explicit_tuning_profile_without_changing_engine_capacities() {
    let profile = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(profile.path(), br#"{"batch_sizes":[1,2],"prefill_tokens":[2,16],"keep_best":3,"search_time_limit_ms":5000}"#).unwrap();
    let config = ServeConfig::from_args([
        "orbitkv-serve".to_owned(),
        "--model".to_owned(),
        "/models/checkpoint".to_owned(),
        "--page-counts".to_owned(),
        "128,66".to_owned(),
        "--search-graphs".to_owned(),
        "4".to_owned(),
        "--tuning-profile".to_owned(),
        profile.path().to_string_lossy().into_owned(),
    ])
    .unwrap();
    assert_eq!(config.tuning.batch_sizes, [1, 2]);
    assert_eq!(config.tuning.keep_best, 3);
    assert_eq!(config.tuning.search_time_limit_ms, Some(5000));
    assert!(!config.tuning.enable_shared_fp8_quantization);
    assert_eq!(config.engine.maximum_active_requests, 2);
    assert_eq!(config.engine.maximum_batch_tokens, 1024);
}

#[test]
fn parses_complete_server_configuration() {
    let config = ServeConfig::from_args(
        [
            "orbitkv-serve",
            "--model",
            "/models/checkpoint",
            "--page-counts",
            "128,66",
            "--max-model-tokens",
            "1024",
            "--max-active-requests",
            "2",
        ]
        .map(str::to_string),
    )
    .unwrap();
    assert_eq!(config.engine.page_counts, [128, 66]);
    assert_eq!(config.engine.decoder_artifact, None);
    assert_eq!(config.frontend.logical_kv_page_count, 128);
    assert_eq!(config.frontend.served_model_names, ["checkpoint"]);
    assert_eq!(config.frontend.model, "/models/checkpoint");
}

#[test]
fn rejects_missing_or_malformed_required_arguments() {
    assert!(ServeConfig::from_args(["orbitkv-serve"].map(str::to_string)).is_err());
    assert!(
        ServeConfig::from_args(
            [
                "orbitkv-serve",
                "--model",
                "/model",
                "--page-counts",
                "1,bad"
            ]
            .map(str::to_string)
        )
        .is_err()
    );
}

#[test]
fn frontend_assets_can_be_separate_from_model_weights() {
    let config = ServeConfig::from_args(
        [
            "orbitkv-serve",
            "--model",
            "/models/weights",
            "--frontend-model",
            "/models/tokenizer",
            "--served-model",
            "public-model",
            "--page-counts",
            "128,66",
        ]
        .map(str::to_string),
    )
    .unwrap();
    assert_eq!(
        config.engine.model_directory,
        PathBuf::from("/models/weights")
    );
    assert_eq!(config.frontend.model, "/models/tokenizer");
    assert_eq!(config.frontend.served_model_names, ["public-model"]);
}

#[test]
fn decoder_artifact_path_is_forwarded_to_the_engine() {
    let config = ServeConfig::from_args(
        [
            "orbitkv-serve",
            "--model",
            "/models/weights",
            "--decoder-artifact",
            "/artifacts/decoder.json",
            "--page-counts",
            "128,66",
        ]
        .map(str::to_string),
    )
    .unwrap();
    assert_eq!(
        config.engine.decoder_artifact,
        Some(PathBuf::from("/artifacts/decoder.json"))
    );
}
