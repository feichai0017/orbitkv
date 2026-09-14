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
        "cuda_modules": {
            "version": 3,
            "signature": {"target_arch": "sm_90", "nvrtc_options": [], "nvrtc_version": 12080},
            "images": {}
        },
    }))
    .unwrap()
}

#[test]
fn decoder_artifact_read_rejects_oversized_input() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("oversized.json");
    let file = std::fs::File::create(&path).unwrap();
    file.set_len(MAX_DECODER_ARTIFACT_BYTES + 1).unwrap();

    let error = read_decoder_artifact(&path).unwrap_err();
    assert!(error.to_string().contains("decoder artifact exceeds"));
}
