use super::*;

#[test]
fn persists_decoder_artifact_without_overwriting_an_existing_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("decoder.json");
    let artifact =
        DecoderArtifact::from_bytes(&artifact_fixture(DecoderArtifact::SCHEMA_VERSION)).unwrap();

    persist_decoder_artifact(&path, &artifact).unwrap();
    let first = std::fs::read(&path).unwrap();
    assert_eq!(first, artifact.to_bytes().unwrap());
    assert!(persist_decoder_artifact(&path, &artifact).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), first);
}

#[test]
fn decoder_artifact_rejects_unknown_schema() {
    let unknown_schema = DecoderArtifact::SCHEMA_VERSION + 1;
    let error = DecoderArtifact::from_bytes(&artifact_fixture(unknown_schema)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains(&format!("schema {unknown_schema}"))
    );
}

fn artifact_fixture(schema: u32) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "schema": schema, "identity": "test",
        "schedule": {"dim_buckets": {}, "buckets": []},
        "environment": {
            "target": {"major": 9, "minor": 0},
            "device": {"name": "fixture", "multiprocessors": 1, "total_memory_bytes": 1024},
            "driver_api_version": 12080,
            "nvrtc": {"version": 12080, "options": []},
            "provider_lock_digest": "fixture", "providers": {}, "native_compiler": null,
            "cublaslt_autotune": null
        },
        "cuda_modules": {
            "version": 3,
            "signature": {"target_arch": "sm_90", "nvrtc_options": [], "nvrtc_version": 12080},
            "images": {}
        },
    }))
    .unwrap()
}

#[test]
fn decoder_artifact_reads_valid_input_beyond_the_former_file_ceiling() {
    use std::io::Read;
    // The former 64 MiB ceiling rejected even artifacts emitted by this engine.
    const FORMER_CEILING: u64 = 64 * 1024 * 1024;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("large.json");
    let mut file = std::fs::File::create(&path).unwrap();
    std::io::copy(
        &mut std::io::repeat(b' ').take(FORMER_CEILING + 1),
        &mut file,
    )
    .unwrap();
    let bytes = artifact_fixture(DecoderArtifact::SCHEMA_VERSION);
    file.write_all(&bytes).unwrap();
    drop(file);
    let artifact = read_decoder_artifact(&path).unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&artifact.to_bytes().unwrap()).unwrap(),
        serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()
    );
}
