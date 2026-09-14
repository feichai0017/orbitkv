use super::*;

fn fixture() -> serde_json::Value {
    serde_json::json!({
        "schema": DECODER_ARTIFACT_SCHEMA,
        "identity": "format-test",
        "schedule": {"dim_buckets": {}, "buckets": []},
        "cuda_modules": {
            "version": 3,
            "signature": {"target_arch": "sm_90", "nvrtc_options": [], "nvrtc_version": 12080},
            "images": {}
        }
    })
}

#[test]
fn complete_artifact_round_trips() {
    let value = fixture();
    let artifact = DecoderArtifact::from_bytes(&serde_json::to_vec(&value).unwrap()).unwrap();
    assert_eq!(artifact.module_image_count(), 0);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&artifact.to_bytes().unwrap()).unwrap(),
        value
    );
}

#[test]
fn cached_schema_requires_images_and_rejects_unknown_versions() {
    for schema in [DECODER_ARTIFACT_SCHEMA - 1, DECODER_ARTIFACT_SCHEMA + 1, 0] {
        let mut value = fixture();
        value["schema"] = schema.into();
        assert!(DecoderArtifact::from_bytes(&serde_json::to_vec(&value).unwrap()).is_err());
    }
}

#[test]
fn missing_or_null_module_artifact_is_rejected() {
    let mut value = fixture();
    value["cuda_modules"] = serde_json::Value::Null;
    assert!(DecoderArtifact::from_bytes(&serde_json::to_vec(&value).unwrap()).is_err());
    value.as_object_mut().unwrap().remove("cuda_modules");
    assert!(DecoderArtifact::from_bytes(&serde_json::to_vec(&value).unwrap()).is_err());
}
