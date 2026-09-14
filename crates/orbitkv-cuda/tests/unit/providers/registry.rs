use super::*;

#[test]
fn lock_separates_toolkit_libraries_from_pinned_source_dependencies() {
    let lock = provider_lock();
    assert!(matches!(
        lock.providers[&ProviderId::CublasLt],
        ProviderOrigin::CudaToolkit { .. }
    ));
    let attention = ProviderId::FlashAttention.source().unwrap();
    let gemm = ProviderId::DeepGemm.source().unwrap();
    assert_ne!(
        attention.dependencies[0].revision,
        gemm.dependencies[0].revision
    );
    assert!(ProviderId::CublasLt.source().is_err());
    assert!("flashmla".parse::<ProviderId>().is_err());
}

#[test]
fn floating_revisions_and_escaping_paths_are_rejected() {
    let original: serde_json::Value =
        serde_json::from_str(include_str!("../../../providers.lock.json")).unwrap();
    for (field, value) in [
        ("revision", "main"),
        ("revision", "12345678"),
        ("url", "file:///tmp/source"),
    ] {
        let mut changed = original.clone();
        changed["providers"]["deepgemm"][field] = value.into();
        assert!(
            ProviderLock::parse(&changed.to_string()).is_err(),
            "{field}"
        );
    }
    let mut changed = original;
    changed["providers"]["flashinfer"]["dependencies"][0]["relative_path"] = "../outside".into();
    assert!(ProviderLock::parse(&changed.to_string()).is_err());
}

#[test]
fn dependency_pin_changes_are_visible_in_lock_identity() {
    let original = provider_lock();
    let mut changed: serde_json::Value = serde_json::to_value(original).unwrap();
    changed["providers"]["deepgemm"]["dependencies"][0]["revision"] =
        "0123456789012345678901234567890123456789".into();
    let changed = ProviderLock::parse(&changed.to_string()).unwrap();
    assert_ne!(original.digest(), changed.digest());
}
