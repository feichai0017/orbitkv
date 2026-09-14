use super::*;

fn fixture() -> serde_json::Value {
    serde_json::json!({
        "schema": DECODER_ARTIFACT_SCHEMA,
        "identity": "format-test",
        "schedule": {"dim_buckets": {}, "buckets": []},
        "environment": {
            "target": {"major": 9, "minor": 0},
            "device": {"name": "fixture", "multiprocessors": 1, "total_memory_bytes": 1024},
            "driver_api_version": 12080,
            "nvrtc": {"version": 12080, "options": []},
            "provider_lock_digest": "fixture",
            "providers": {},
            "native_compiler": null,
            "cublaslt_autotune": null
        },
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
fn missing_or_null_execution_records_are_rejected() {
    for field in ["cuda_modules", "environment"] {
        let mut value = fixture();
        value[field] = serde_json::Value::Null;
        assert!(DecoderArtifact::from_bytes(&serde_json::to_vec(&value).unwrap()).is_err());
        value.as_object_mut().unwrap().remove(field);
        assert!(DecoderArtifact::from_bytes(&serde_json::to_vec(&value).unwrap()).is_err());
    }
}
