use super::*;

mod gpu;

fn fixture() -> CudaModuleArtifact {
    CudaModuleArtifact {
        signature: ModuleArtifactSignature {
            target_arch: "sm_90".into(),
            nvrtc_options: vec!["--gpu-architecture=sm_90".into()],
            nvrtc_version: Some(12080),
        },
        images: BTreeMap::from([(module_key("kernel"), vec![1, 2, 3])]),
    }
}

fn session(data: CudaModuleArtifact, loading: bool) -> Arc<Mutex<ModuleArtifact>> {
    Arc::new(Mutex::new(ModuleArtifact {
        data,
        loading,
        capturing: false,
    }))
}

#[test]
fn module_artifact_round_trip_is_deterministic() {
    let artifact = fixture();
    let data = serde_json::to_string(&artifact).unwrap();
    let loaded: CudaModuleArtifact = serde_json::from_str(&data).unwrap();
    assert_eq!(artifact, loaded);
    assert_eq!(serde_json::to_string(&loaded).unwrap(), data);
    let mut value: serde_json::Value = serde_json::from_str(&data).unwrap();
    value["version"] = (MODULE_ARTIFACT_VERSION - 1).into();
    assert!(serde_json::from_value::<CudaModuleArtifact>(value).is_err());
}

#[test]
fn module_artifact_rejects_corrupt_or_empty_images_and_invalid_source_keys() {
    let data = serde_json::to_value(fixture()).unwrap();
    for invalid in ["broken base64", "", "BAUG"] {
        let mut value = data.clone();
        value["images"][module_key("kernel")]["data"] = invalid.into();
        assert!(serde_json::from_value::<CudaModuleArtifact>(value).is_err());
    }
    let mut value = data;
    let image = value["images"]
        .as_object_mut()
        .unwrap()
        .remove(&module_key("kernel"))
        .unwrap();
    value["images"]["not-a-source-digest"] = image;
    assert!(serde_json::from_value::<CudaModuleArtifact>(value).is_err());
}

#[test]
fn module_artifact_requires_matching_target_options_and_compiler() {
    let artifact = fixture();
    let mut signatures = vec![artifact.signature.clone(); 3];
    signatures[0].target_arch = "sm_80".into();
    signatures[1].nvrtc_options.push("--use_fast_math".into());
    signatures[2].nvrtc_version = None;
    for signature in signatures {
        assert!(artifact.validate_signature(&signature).is_err());
    }
}

#[test]
fn loaded_artifact_does_not_fall_back_to_compilation() {
    with_module_artifact_session(session(fixture(), true), || {
        assert!(matches!(
            lookup_module_image(&module_key("kernel")),
            ModuleImageLookup::Hit(_)
        ));
        assert!(matches!(
            lookup_module_image(&module_key("changed kernel")),
            ModuleImageLookup::Missing { available: 1 }
        ));
        record_module_image(&module_key("changed kernel"), &[4]);
        assert!(matches!(
            lookup_module_image(&module_key("changed kernel")),
            ModuleImageLookup::Missing { .. }
        ));
    });
}

#[test]
fn nested_sessions_restore_after_panic_and_preserve_ambient_capture() {
    assert!(current_module_artifact_session().is_none());
    let outer = session(fixture(), false);
    with_module_artifact_session(outer.clone(), || {
        let _no_override = ModuleArtifactGuard::enter(None);
        let result = std::panic::catch_unwind(|| {
            with_module_artifact_session(session(fixture(), true), || panic!("unwind session"));
        });
        assert!(result.is_err());
        assert!(Arc::ptr_eq(
            &current_module_artifact_session().unwrap(),
            &outer
        ));
        assert!(matches!(
            lookup_module_image("missing"),
            ModuleImageLookup::Compile
        ));
    });
    assert!(current_module_artifact_session().is_none());
}
